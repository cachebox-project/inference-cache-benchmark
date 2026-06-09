"""Unit tests for lib/lookup_proxy_legacy.py helpers — CAC-150 fix 2 + fix 3.

The module under test is the deprecated chain-walking proxy retained for one
release as `lookup_proxy_legacy.py` (CAC-152). The production gateway model
lives in `lib/dumb_gateway_client.py`; these tests guard the legacy code path
against regressions until it's removed.

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
from collections import Counter
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# transformers is needed for LookupProxy construction; skip the module
# if the operator-install dep set is missing.
pytest.importorskip("transformers")

from event_index import EventIndex, BlockStored, EventBatch  # noqa: E402
from lookup_proxy_legacy import (  # noqa: E402
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

    with patch("lookup_proxy_legacy.subscribe_replica", side_effect=fake_subscribe_replica):
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

    with patch("lookup_proxy_legacy.subscribe_replica", side_effect=fake_subscribe_replica):
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
    with patch("lookup_proxy_legacy.AutoTokenizer.from_pretrained",
               return_value=MagicMock(__len__=lambda self: 0)):
        proxy = LookupProxy(
            ic_server="ignored:0",
            replicas=["http://up-r0", "http://up-r1", "http://up-r2"],
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


# ---- CAC-154: round-robin fallback on NO_HINT / TIMEOUT ---------------------


def _build_proxy(replicas):
    """Construct a LookupProxy without doing the heavy tokenizer load."""
    index = EventIndex()
    with patch("lookup_proxy_legacy.AutoTokenizer.from_pretrained",
               return_value=MagicMock(__len__=lambda self: 0)):
        return LookupProxy(
            ic_server="ignored:0",
            replicas=replicas,
            tokenizer_name="any",
            hash_scheme="vllm",
            event_index=index,
            tenant="t",
        )


def test_no_hint_fallback_round_robins_across_replicas():
    """30 NO_HINT picks → ~10/10/10 distribution, not 30/0/0.

    Reproduces CAC-154's bug: the old --default-upstream pointed at a single
    pod, so every NO_HINT/TIMEOUT response concentrated on r0. The fix
    replaces it with --replicas (plural) and a round-robin pointer.
    """
    urls = [
        "http://r0.local:38010",
        "http://r1.local:38011",
        "http://r2.local:38012",
    ]
    proxy = _build_proxy(urls)

    counts = Counter()
    reasons = Counter()
    for _ in range(30):
        upstream, reason = proxy._pick_upstream(hint_replica_id=None)
        counts[upstream] += 1
        reasons[reason] += 1

    # Strict equality is achievable because 30 is a multiple of 3 and the
    # pointer is deterministic.
    assert counts == Counter({url: 10 for url in urls}), counts
    assert reasons == Counter({"ROUND_ROBIN": 30}), reasons
    assert proxy.stats["routed_round_robin_fallback"] == 30
    assert proxy.stats["routed_to_hint"] == 0


def test_pick_upstream_prefers_hint_when_replica_id_matches_event_index():
    """A valid hint short-circuits round-robin and routes to the named replica."""
    urls = [
        "http://r0.local:38010",
        "http://r1.local:38011",
        "http://r2.local:38012",
    ]
    proxy = _build_proxy(urls)
    proxy.event_index.add_replica("r1", "http://r1.local:38011")

    upstream, reason = proxy._pick_upstream(hint_replica_id="r1")
    assert reason == "HINT"
    assert upstream == "http://r1.local:38011"
    assert proxy.stats["routed_to_hint"] == 1
    assert proxy.stats["routed_round_robin_fallback"] == 0


def test_pick_upstream_falls_back_to_round_robin_when_hint_is_unknown():
    """An unparseable / unknown hint should still spread, not pin to r0."""
    urls = [
        "http://r0.local:38010",
        "http://r1.local:38011",
        "http://r2.local:38012",
    ]
    proxy = _build_proxy(urls)
    proxy.event_index.add_replica("r1", "http://r1.local:38011")

    counts = Counter()
    for _ in range(30):
        upstream, reason = proxy._pick_upstream(hint_replica_id="not-a-real-pod")
        assert reason == "ROUND_ROBIN"
        counts[upstream] += 1
    assert counts == Counter({url: 10 for url in urls}), counts


def test_lookup_proxy_rejects_empty_replicas():
    with patch("lookup_proxy_legacy.AutoTokenizer.from_pretrained",
               return_value=MagicMock(__len__=lambda self: 0)):
        with pytest.raises(ValueError, match="non-empty"):
            LookupProxy(
                ic_server="ignored:0",
                replicas=[],
                tokenizer_name="any",
                hash_scheme="vllm",
                event_index=EventIndex(),
                tenant="t",
            )
