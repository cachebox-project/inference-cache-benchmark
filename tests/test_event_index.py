"""Unit tests for lib/event_index.py — CAC-150 fix 1.

Focuses on the round-robin tie-break in ``find_best_chain``. Pre-fix, the
walk picked the first replica in dict-iteration order on ties; with
orchestrator-built ``--replica-alias r0=..., r1=..., r2=...`` aliasing,
r0 always won and absorbed ~99% of cache-stress traffic.
"""

from __future__ import annotations

from collections import Counter

import pytest

from event_index import (
    BlockStored,
    EventBatch,
    EventIndex,
)


BS = 16  # vLLM's default block size


def _seed_identical_one_block(index: EventIndex, replica_ids, token_ids):
    """Make every replica observe the same 1-block prefix (chain length 1).

    Simulates the common "trivial chat-template framing" case: all three
    replicas processed the same opening tokens, so the proxy's chain walk
    finds an equal-length 1-block chain on each.
    """
    for i, rid in enumerate(replica_ids):
        index.add_replica(rid, f"http://up-{rid}")
        batch = EventBatch(events=[
            BlockStored(
                block_hashes=[1000 + i],
                parent_block_hash=None,
                token_ids=list(token_ids),
                block_size=BS,
            ),
        ])
        index.apply_batch(rid, batch)


def test_find_best_chain_round_robin_breaks_ties_evenly():
    """The canonical CAC-150 regression: 9 lookups across 3 tied replicas
    must distribute exactly 3/3/3, not 9/0/0."""
    index = EventIndex()
    tokens = list(range(BS))
    _seed_identical_one_block(index, ["r0", "r1", "r2"], tokens)

    picks = []
    for _ in range(9):
        rid, hashes, counts = index.find_best_chain(tokens)
        assert hashes, "expected a chain match for the seeded prefix"
        assert counts == [BS]
        picks.append(rid)

    assert Counter(picks) == Counter({"r0": 3, "r1": 3, "r2": 3}), picks


def test_find_best_chain_sorted_keys_not_insertion_order():
    """Insertion order: r2 first, then r0, then r1. Tie-break must rotate
    by sorted id (r0 → r1 → r2 → r0 …), not by insertion.
    """
    index = EventIndex()
    tokens = list(range(BS))
    _seed_identical_one_block(index, ["r2", "r0", "r1"], tokens)

    first_three = []
    for _ in range(3):
        rid, _, _ = index.find_best_chain(tokens)
        first_three.append(rid)
    assert first_three == ["r0", "r1", "r2"], first_three


def test_find_best_chain_strict_winner_overrides_tie_break():
    """When one replica has a strictly longer chain, the round-robin
    counter must not steal a request from it.
    """
    index = EventIndex()
    for rid in ("r0", "r1", "r2"):
        index.add_replica(rid, f"http://up-{rid}")

    # All three see the root block. Only r1 sees the second block.
    root_tokens = list(range(BS))
    second_tokens = list(range(BS, 2 * BS))
    for i, rid in enumerate(("r0", "r1", "r2")):
        index.apply_batch(rid, EventBatch(events=[
            BlockStored(
                block_hashes=[10 + i],
                parent_block_hash=None,
                token_ids=root_tokens,
                block_size=BS,
            ),
        ]))
    index.apply_batch("r1", EventBatch(events=[
        BlockStored(
            block_hashes=[10 + 1, 20],
            parent_block_hash=None,
            token_ids=root_tokens + second_tokens,
            block_size=BS,
        ),
    ]))

    request_tokens = root_tokens + second_tokens
    for _ in range(5):
        rid, hashes, _ = index.find_best_chain(request_tokens)
        assert rid == "r1", "r1 has the unique 2-block chain"
        assert len(hashes) == 2


def test_find_best_chain_min_blocks_filter():
    """A request shorter than block_size must not match anything."""
    index = EventIndex()
    _seed_identical_one_block(index, ["r0", "r1", "r2"], list(range(BS)))

    rid, hashes, _ = index.find_best_chain(list(range(BS // 2)))
    assert rid is None and hashes == []


def test_find_best_chain_empty_index_returns_none():
    index = EventIndex()
    rid, hashes, counts = index.find_best_chain(list(range(BS * 4)))
    assert rid is None and hashes == [] and counts == []


def test_apply_batch_increments_per_replica_events_received():
    """Per-replica event counter is the load-bearing signal for spotting
    silent-SUB outages (CAC-150 bug 2). Must increment per event, per
    replica, never cross-contaminate."""
    index = EventIndex()
    index.add_replica("r0", "http://up-r0")
    index.add_replica("r1", "http://up-r1")

    index.apply_batch("r0", EventBatch(events=[
        BlockStored(block_hashes=[1], parent_block_hash=None,
                    token_ids=list(range(BS)), block_size=BS),
        BlockStored(block_hashes=[2], parent_block_hash=1,
                    token_ids=list(range(BS, 2 * BS)), block_size=BS),
    ]))

    assert index.replicas["r0"].events_received == 2
    assert index.replicas["r1"].events_received == 0

    index.apply_batch("r1", EventBatch(events=[
        BlockStored(block_hashes=[100], parent_block_hash=None,
                    token_ids=list(range(BS)), block_size=BS),
    ]))
    assert index.replicas["r0"].events_received == 2
    assert index.replicas["r1"].events_received == 1


def test_apply_batch_unknown_replica_is_ignored():
    """A stray event for a replica we don't track shouldn't crash."""
    index = EventIndex()
    index.add_replica("r0", "http://up-r0")
    index.apply_batch("not-a-replica", EventBatch(events=[
        BlockStored(block_hashes=[1], parent_block_hash=None,
                    token_ids=list(range(BS)), block_size=BS),
    ]))
    assert index.replicas["r0"].events_received == 0
