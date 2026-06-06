"""Integration test: real ZMQ wire, drive the proxy's subscriber pipeline.

Spins up three in-process zmq PUB sockets (one per fake replica), wires the
existing ``subscribe_replica`` coroutine onto each, publishes BlockStored
events that mimic a trivial 1-block chat-template prefix collision across
all three replicas, then calls ``find_best_chain`` from many simulated
requests and asserts the upstream distribution is balanced.

Pre-CAC-150 fix, this test would fail with r0 absorbing every request.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import msgspec
import pytest
import zmq
import zmq.asyncio

from event_index import (
    BlockStored,
    EventBatch,
    EventIndex,
    subscribe_replica,
)


BS = 16
TOPIC = b"kv-events"
REPLICAS = ["r0", "r1", "r2"]


async def _wait_for_events(index: EventIndex, replica_ids, min_each: int, deadline_s: float):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if all(
            index.replicas[rid].events_received >= min_each
            for rid in replica_ids
        ):
            return
        await asyncio.sleep(0.05)
    counts = {
        rid: index.replicas[rid].events_received for rid in replica_ids
    }
    raise AssertionError(
        f"timed out waiting for events on all replicas: {counts}"
    )


@pytest.mark.asyncio
async def test_balanced_routing_over_real_zmq():
    ctx = zmq.asyncio.Context.instance()

    pubs = []
    endpoints = []
    try:
        for _ in REPLICAS:
            pub = ctx.socket(zmq.PUB)
            port = pub.bind_to_random_port("tcp://127.0.0.1")
            pubs.append(pub)
            endpoints.append(f"tcp://127.0.0.1:{port}")

        index = EventIndex()
        for i, rid in enumerate(REPLICAS):
            index.add_replica(rid, f"http://upstream-{rid}")

        sub_tasks = [
            asyncio.create_task(
                subscribe_replica(
                    index, rid, ep,
                    topic=TOPIC.decode(),
                    initial_settle_s=0.1,
                )
            )
            for rid, ep in zip(REPLICAS, endpoints)
        ]

        # Let the SUBs attach. The 0.1s initial_settle_s above is the
        # subscriber's own grace window; we add a little extra to be safe.
        await asyncio.sleep(0.3)

        # Publish the same 1-block prefix to all three pubs. Each replica
        # mints its own block_hash (vLLM's hash is process-local) — so the
        # walk finds three equally-long 1-block chains.
        tokens = list(range(BS))
        for i, pub in enumerate(pubs):
            batch = EventBatch(events=[
                BlockStored(
                    block_hashes=[100 + i],
                    parent_block_hash=None,
                    token_ids=tokens,
                    block_size=BS,
                ),
            ])
            payload = msgspec.msgpack.encode(batch)
            await pub.send_multipart([TOPIC, payload])

        await _wait_for_events(index, REPLICAS, min_each=1, deadline_s=2.0)

        # Drive 30 simulated requests through find_best_chain. With the
        # round-robin tie-break, distribution should be exactly 10/10/10.
        picks = []
        for _ in range(30):
            rid, hashes, _ = index.find_best_chain(tokens)
            assert hashes, "expected a 1-block match"
            picks.append(rid)

        distribution = Counter(picks)
        assert distribution == Counter({"r0": 10, "r1": 10, "r2": 10}), distribution

        for t in sub_tasks:
            t.cancel()
        await asyncio.gather(*sub_tasks, return_exceptions=True)
    finally:
        for p in pubs:
            try:
                p.close()
            except Exception:
                pass
