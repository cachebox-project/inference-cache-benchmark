"""Unit tests for lib/lookup_proxy.py helpers — CAC-150 fix 2 + fix 3.

Covers:
  * ``_classify_match_quality`` — trivial/weak/strong bucketing
  * ``await_zmq_ready`` — startup health gate (success and timeout paths)
  * ``retry_silent_subs`` — restarts SUB tasks for silent replicas while
    siblings flow
  * ``LookupProxy.handle_metrics_prom`` — per-replica counter exposed in
    Prometheus text format
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# transformers is needed for LookupProxy construction; skip the module
# if the operator-install dep set is missing.
pytest.importorskip("transformers")

from event_index import EventIndex, BlockStored, EventBatch  # noqa: E402
from lookup_proxy import (  # noqa: E402
    LookupProxy,
    _classify_match_quality,
    await_zmq_ready,
    retry_silent_subs,
)

BS = 16


# ---- match_quality bucketing -------------------------------------------------


@pytest.mark.parametrize(
    "blocks,expected",
    [
        (0, None),
        (-1, None),
        (1, "trivial"),
        (2, "weak"),
        (5, "weak"),
        (7, "weak"),
        (8, "strong"),
        (50, "strong"),
        (300, "strong"),
    ],
)
def test_classify_match_quality(blocks, expected):
    assert _classify_match_quality(blocks) == expected


# ---- await_zmq_ready --------------------------------------------------------


@pytest.mark.asyncio
async def test_await_zmq_ready_fires_when_every_replica_has_events():
    index = EventIndex()
    for rid in ("r0", "r1", "r2"):
        index.add_replica(rid, f"http://up-{rid}")
    ready = asyncio.Event()

    async def feed_after_delay():
        await asyncio.sleep(0.05)
        for rid in ("r0", "r1", "r2"):
            index.apply_batch(rid, EventBatch(events=[
                BlockStored(block_hashes=[1], parent_block_hash=None,
                            token_ids=list(range(BS)), block_size=BS),
            ]))

    feeder = asyncio.create_task(feed_after_delay())
    silent = await await_zmq_ready(
        index, ready, ["r0", "r1", "r2"], timeout_s=2.0, poll_s=0.02,
    )
    await feeder
    assert silent == []
    assert ready.is_set()


@pytest.mark.asyncio
async def test_await_zmq_ready_times_out_with_silent_replicas(caplog):
    index = EventIndex()
    for rid in ("r0", "r1", "r2"):
        index.add_replica(rid, f"http://up-{rid}")
    # Only r0 ever sees an event.
    index.apply_batch("r0", EventBatch(events=[
        BlockStored(block_hashes=[1], parent_block_hash=None,
                    token_ids=list(range(BS)), block_size=BS),
    ]))
    ready = asyncio.Event()

    silent = await await_zmq_ready(
        index, ready, ["r0", "r1", "r2"], timeout_s=0.1, poll_s=0.02,
    )
    assert ready.is_set()
    assert sorted(silent) == ["r1", "r2"]


# ---- retry_silent_subs ------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_silent_subs_restarts_silent_replica():
    """When r1's SUB is silent and r0/r2 are flowing, the retry loop
    should cancel r1's subscriber task and create a fresh one. We capture
    the new task by patching subscribe_replica to a sentinel coroutine."""
    index = EventIndex()
    for rid in ("r0", "r1", "r2"):
        index.add_replica(rid, f"http://up-{rid}")
    # r0 and r2 are flowing; r1 is stuck at 0.
    for rid in ("r0", "r2"):
        index.apply_batch(rid, EventBatch(events=[
            BlockStored(block_hashes=[1], parent_block_hash=None,
                        token_ids=list(range(BS)), block_size=BS),
        ]))

    async def _dummy_long_running():
        await asyncio.sleep(10)

    initial_tasks: Dict[str, asyncio.Task] = {
        rid: asyncio.create_task(_dummy_long_running()) for rid in ("r0", "r1", "r2")
    }
    original_r1 = initial_tasks["r1"]

    new_subs_created: List[str] = []

    async def fake_subscribe_replica(_idx, rid, _ep, **_kwargs):
        new_subs_created.append(rid)
        await asyncio.sleep(10)

    specs: List[Tuple[str, str, str, str]] = [
        ("r0", "tcp://x:1", "http://up-r0", None),
        ("r1", "tcp://x:2", "http://up-r1", None),
        ("r2", "tcp://x:3", "http://up-r2", None),
    ]

    with patch("lookup_proxy.subscribe_replica", side_effect=fake_subscribe_replica):
        retry_task = asyncio.create_task(
            retry_silent_subs(index, initial_tasks, specs, interval_s=0.05)
        )
        # Give the loop a couple of iterations to fire.
        await asyncio.sleep(0.3)
        retry_task.cancel()
        try:
            await retry_task
        except asyncio.CancelledError:
            pass

    assert "r1" in new_subs_created, new_subs_created
    assert "r0" not in new_subs_created  # don't disturb flowing SUBs
    assert "r2" not in new_subs_created
    assert original_r1.cancelled() or original_r1.done()

    # Clean up surviving tasks.
    for rid, t in initial_tasks.items():
        if not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_retry_silent_subs_skips_when_nothing_is_flowing():
    """If EVERY replica is silent, the cluster is idle — don't churn SUBs."""
    index = EventIndex()
    for rid in ("r0", "r1", "r2"):
        index.add_replica(rid, f"http://up-{rid}")

    async def _dummy():
        await asyncio.sleep(10)

    tasks: Dict[str, asyncio.Task] = {
        rid: asyncio.create_task(_dummy()) for rid in ("r0", "r1", "r2")
    }
    specs = [(rid, f"tcp://x:{i}", f"http://up-{rid}", None)
             for i, rid in enumerate(("r0", "r1", "r2"))]

    created: List[str] = []

    async def fake_subscribe_replica(_idx, rid, _ep, **_kwargs):
        created.append(rid)
        await asyncio.sleep(10)

    with patch("lookup_proxy.subscribe_replica", side_effect=fake_subscribe_replica):
        retry_task = asyncio.create_task(
            retry_silent_subs(index, tasks, specs, interval_s=0.05)
        )
        await asyncio.sleep(0.3)
        retry_task.cancel()
        try:
            await retry_task
        except asyncio.CancelledError:
            pass

    assert created == [], "must not restart anything when the cluster is idle"

    for t in tasks.values():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


# ---- /proxy/metrics.prom ----------------------------------------------------


@pytest.mark.asyncio
async def test_handle_metrics_prom_exposes_per_replica_counter():
    """The smoke-test signal: a silent replica shows up as 0 in the
    Prometheus output, distinguishable from flowing ones."""
    index = EventIndex()
    index.add_replica("r0", "http://up-r0")
    index.add_replica("r1", "http://up-r1")
    index.add_replica("r2", "http://up-r2")
    # r0/r2 flowing, r1 silent.
    for rid in ("r0", "r2"):
        index.apply_batch(rid, EventBatch(events=[
            BlockStored(block_hashes=[1], parent_block_hash=None,
                        token_ids=list(range(BS)), block_size=BS),
            BlockStored(block_hashes=[2], parent_block_hash=1,
                        token_ids=list(range(BS, 2 * BS)), block_size=BS),
        ]))

    # Bypass the heavy tokenizer load.
    with patch("lookup_proxy.AutoTokenizer.from_pretrained",
               return_value=MagicMock(__len__=lambda self: 0)):
        proxy = LookupProxy(
            ic_server="ignored:0",
            default_upstream="http://default",
            tokenizer_name="any",
            hash_scheme="vllm",
            event_index=index,
            tenant="t",
        )

    fake_req = MagicMock()
    resp = await proxy.handle_metrics_prom(fake_req)
    body = resp.text
    assert 'lookup_proxy_zmq_events_received_total{replica="r0"} 2' in body
    assert 'lookup_proxy_zmq_events_received_total{replica="r1"} 0' in body
    assert 'lookup_proxy_zmq_events_received_total{replica="r2"} 2' in body
    assert 'lookup_proxy_chain_entries{replica="r0"} 2' in body
    assert 'lookup_proxy_ready 0' in body  # not ready yet
