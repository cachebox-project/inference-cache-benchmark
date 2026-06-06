"""
event_index.py — Shadow subscriber for vLLM KV-cache events.

Why
---
vLLM emits per-block hashes via ZMQ when KV blocks are stored, removed, or
cleared. The hashes are produced by vLLM's internal hash function — which in
default config is `builtins.hash()`, *process-local* and not reproducible
across replicas (each pod has its own PYTHONHASHSEED).

A gateway proxy that wants `LookupRoute` to return `PREFIX_MATCH` cannot just
hash a prompt itself. It has to discover what hashes the engine has been
emitting, *per replica*, and query LookupRoute using those exact byte
sequences. This module is the discovery side.

For each replica, we subscribe to its ZMQ PUB endpoint, decode the
`BlockStored` / `BlockRemoved` / `AllBlocksCleared` events, and maintain a
chain table:

    (parent_block_hash, tuple_of_token_ids) → block_hash

When a new request arrives, the proxy tokenizes the prompt, chunks it into
B-token blocks, and walks each replica's chain looking for the longest
leading-prefix match. The replica with the longest hit becomes the routing
hint — and its specific chain of `block_hashes` goes into the LookupRoute
query, ensuring the server's index can match by exact byte equality.

This is "Option B-b" from CAC-135: observe rather than recompute.

Cold-start trade-off
--------------------
Until the proxy has observed a prefix in events, it can't get a PREFIX_MATCH
for that prefix. Real-cluster behavior:
- First request: no chain known → NO_HINT (proxy falls back to default
  upstream); engine processes; events arrive; chain table populates.
- Steady state (after ~5-10 requests with the same shared prefix): every
  request walks an existing chain → PREFIX_MATCH every time.

This is exactly what benchmarks measure (warmup + steady-state), and what a
real gateway integration would see.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

try:
    import msgspec
    import zmq
    import zmq.asyncio
except ImportError as e:  # pragma: no cover
    raise ImportError(
        f"event_index requires pyzmq and msgspec: {e}\n"
        "Install with: make install  (or: pip install pyzmq msgspec)"
    )


# ---------------------------------------------------------------------------
# vLLM KV-event message types (mirrors what the in-cluster subscriber decodes).
# vLLM's events.py: tagged-union over [BlockStored, BlockRemoved, AllBlocksCleared].
# ---------------------------------------------------------------------------


class _Event(msgspec.Struct, array_like=True, omit_defaults=True, tag=True):
    pass


class BlockStored(_Event):
    block_hashes: List[int] = []          # one per block
    parent_block_hash: Optional[int] = None
    token_ids: List[int] = []             # FLAT list: block_size tokens per block, in order
    block_size: int = 0                   # tokens per block
    lora_id: Optional[int] = None


class BlockRemoved(_Event):
    block_hashes: List[int] = []


class AllBlocksCleared(_Event):
    pass


# Typed union so msgspec dispatches each element to the right struct via its
# tag. Without this typing the events come through as raw lists/tuples and
# every isinstance() check downstream fails silently.
_KVEvent = Union[BlockStored, BlockRemoved, AllBlocksCleared]


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float = 0.0
    events: List[_KVEvent] = []


_DECODER = msgspec.msgpack.Decoder(type=EventBatch)


# ---------------------------------------------------------------------------
# Per-replica chain table.
# ---------------------------------------------------------------------------


# Maximum chain-table entries per replica before LRU eviction.
# 100k entries × (parent_hash + 16-token tuple + block_hash) ≈ ~10 MB at worst.
_MAX_ENTRIES_PER_REPLICA = 100_000


@dataclass
class ReplicaIndex:
    """One replica's observed hash chain.

    `chain_table` keys are (parent_hash, tuple_of_token_ids); values are the
    block_hash this replica emitted for that block. Same token sequence at
    different positions in the chain gets a DIFFERENT hash, so the parent is
    part of the key.

    `tokens_index` is a secondary index used by ``find_best_chain`` when the
    root-block lookup misses (the ZMQ SUB attached too late to catch the
    parent=None event for this prefix; CAC-136). It maps a token-tuple to the
    block_hashes whose tokens equal that tuple. Because vLLM's block hash is a
    deterministic function of (parent_hash, tokens, extra_keys), a content
    re-anchor is semantically equivalent to having received the missed root
    event.
    """

    replica_id: str
    upstream_url: str
    chain_table: "OrderedDict[Tuple[Optional[int], Tuple[int, ...]], int]" = field(default_factory=OrderedDict)
    # Reverse index: block_hash → (parent_hash, token_tuple). Used to remove
    # entries on BlockRemoved without scanning the whole table.
    by_hash: Dict[int, Tuple[Optional[int], Tuple[int, ...]]] = field(default_factory=dict)
    # Secondary index: token_tuple → list of block_hashes observed for those
    # tokens (under any parent). Used for root-miss re-anchoring.
    tokens_index: Dict[Tuple[int, ...], List[int]] = field(default_factory=dict)
    last_seen: float = 0.0
    events_received: int = 0

    def store(self, parent_hash: Optional[int], token_tuple: Tuple[int, ...], block_hash: int) -> None:
        key = (parent_hash, token_tuple)
        # If we already had this hash under a different key (shouldn't happen
        # in practice, but defensive), clean up the by_hash entry first.
        if block_hash in self.by_hash and self.by_hash[block_hash] != key:
            old_key = self.by_hash[block_hash]
            self.chain_table.pop(old_key, None)
            # Also clean the old tokens_index entry; the new one overwrites below.
            old_tokens = old_key[1]
            if old_tokens in self.tokens_index:
                try:
                    self.tokens_index[old_tokens].remove(block_hash)
                except ValueError:
                    pass
                if not self.tokens_index[old_tokens]:
                    del self.tokens_index[old_tokens]
        self.chain_table[key] = block_hash
        self.chain_table.move_to_end(key)
        self.by_hash[block_hash] = key
        # Append to tokens_index unless already present.
        bucket = self.tokens_index.setdefault(token_tuple, [])
        if block_hash not in bucket:
            bucket.append(block_hash)
        if len(self.chain_table) > _MAX_ENTRIES_PER_REPLICA:
            evicted_key, evicted_hash = self.chain_table.popitem(last=False)
            self.by_hash.pop(evicted_hash, None)
            evicted_tokens = evicted_key[1]
            if evicted_tokens in self.tokens_index:
                try:
                    self.tokens_index[evicted_tokens].remove(evicted_hash)
                except ValueError:
                    pass
                if not self.tokens_index[evicted_tokens]:
                    del self.tokens_index[evicted_tokens]
        self.last_seen = time.time()

    def lookup(self, parent_hash: Optional[int], token_tuple: Tuple[int, ...]) -> Optional[int]:
        return self.chain_table.get((parent_hash, token_tuple))

    def find_by_tokens(self, token_tuple: Tuple[int, ...]) -> Optional[int]:
        """Return any observed block_hash whose tokens equal token_tuple.

        Used as the content-re-anchor for root-miss in ``find_best_chain``.
        If multiple block_hashes share these tokens (different parents), return
        the most recently inserted one — heuristic for "still warm here."
        """
        bucket = self.tokens_index.get(token_tuple)
        if not bucket:
            return None
        return bucket[-1]

    def remove_hash(self, block_hash: int) -> None:
        key = self.by_hash.pop(block_hash, None)
        if key is not None:
            self.chain_table.pop(key, None)
            tokens = key[1]
            if tokens in self.tokens_index:
                try:
                    self.tokens_index[tokens].remove(block_hash)
                except ValueError:
                    pass
                if not self.tokens_index[tokens]:
                    del self.tokens_index[tokens]

    def clear(self) -> None:
        self.chain_table.clear()
        self.by_hash.clear()
        self.tokens_index.clear()


# ---------------------------------------------------------------------------
# EventIndex — aggregator across replicas.
# ---------------------------------------------------------------------------


class EventIndex:
    def __init__(self) -> None:
        self.replicas: Dict[str, ReplicaIndex] = {}
        # Inferred from BlockStored.block_size. Defaults to 16 (vLLM's default
        # KV block size); updated by the first event we receive.
        self.block_size: int = 16
        # Mapping from server-returned hint id (typically the engine pod name)
        # to the local replica id used as a chain-table key. Empty by default;
        # registered via ``add_alias`` from the proxy spec.
        self.replica_aliases: Dict[str, str] = {}
        # Round-robin counter used to break ties when multiple replicas have
        # equal-length chains in ``find_best_chain``. Without this, dict
        # iteration order (= --replica-alias arg order) wins every tie and
        # the first-declared replica (typically r0) attracts ~99% of traffic
        # whenever the longest chain is a trivial 1-block chat-template match
        # — which is ~70% of requests in practice. See CAC-150.
        self._tie_break_counter: int = 0
        self.stats = {
            "events_received": 0,
            "blocks_stored": 0,
            "blocks_removed": 0,
            "all_blocks_cleared": 0,
            "zmq_errors": 0,
            "decode_errors": 0,
            "root_reanchor_hits": 0,
            "replay_batches_received": 0,
            "tie_break_count": 0,
        }

    def add_replica(self, replica_id: str, upstream_url: str) -> None:
        if replica_id not in self.replicas:
            self.replicas[replica_id] = ReplicaIndex(replica_id=replica_id, upstream_url=upstream_url)

    def add_alias(self, alias: str, target_replica_id: str) -> None:
        """Register that the server may return ``alias`` as a hint that should
        route to the replica known internally as ``target_replica_id``."""
        if alias and target_replica_id:
            self.replica_aliases[alias] = target_replica_id

    def find_best_chain(
        self, token_ids: List[int], min_blocks: int = 1
    ) -> Tuple[Optional[str], List[int], List[int]]:
        """For a request's token sequence, find the replica with the longest
        leading-block chain we've observed. Returns
        ``(replica_id, block_hashes, block_token_counts)``.

        Cold-start handling (CAC-136): on the first block we tolerate a
        root-lookup miss by re-anchoring via the secondary tokens_index. vLLM
        v1 emits the root BlockStored event (parent=None) only once per
        prefix, at the moment the engine first caches it; if our SUB attached
        after that moment the event is lost to ZMQ slow-joiner. Because vLLM's
        block hash is a deterministic function of (parent, tokens, extra_keys),
        a content re-anchor is semantically equivalent to receiving the missed
        root event.

        If no replica has any matching leading-block, returns ``(None, [], [])``.
        """
        bs = self.block_size
        if bs <= 0 or len(token_ids) < bs:
            return None, [], []

        # Collect every replica's best chain so we can break ties deterministically
        # across the full set. Sort replica ids before iterating: dict insertion
        # order reflects argv order, and using it would mean the first-declared
        # replica (typically r0) wins every tie. See CAC-150.
        per_replica: List[Tuple[str, List[int], List[int]]] = []
        max_len = 0
        for rid in sorted(self.replicas.keys()):
            rep = self.replicas[rid]
            hashes: List[int] = []
            counts: List[int] = []
            parent: Optional[int] = None
            pos = 0
            while pos + bs <= len(token_ids):
                block = tuple(token_ids[pos : pos + bs])
                h = rep.lookup(parent, block)
                if h is None and parent is None and not hashes:
                    # Root miss: try content re-anchor (CAC-136).
                    h = rep.find_by_tokens(block)
                    if h is not None:
                        self.stats["root_reanchor_hits"] = (
                            self.stats.get("root_reanchor_hits", 0) + 1
                        )
                if h is None:
                    break
                hashes.append(h)
                counts.append(bs)
                parent = h
                pos += bs
            per_replica.append((rid, hashes, counts))
            if len(hashes) > max_len:
                max_len = len(hashes)

        if max_len < min_blocks:
            return None, [], []

        tied = [entry for entry in per_replica if len(entry[1]) == max_len]
        # Round-robin among the tied set so a 1-block chat-template match
        # (tied across every replica) doesn't pin every request to one upstream.
        picked = tied[self._tie_break_counter % len(tied)]
        self._tie_break_counter += 1
        self.stats["tie_break_count"] = self.stats.get("tie_break_count", 0) + (
            1 if len(tied) > 1 else 0
        )
        return picked

    # ---- batch application — called by subscriber tasks --------------------

    def apply_batch(self, replica_id: str, batch: EventBatch) -> None:
        rep = self.replicas.get(replica_id)
        if rep is None:
            return
        rep.events_received += len(batch.events) if batch.events else 0
        self.stats["events_received"] += len(batch.events) if batch.events else 0
        for ev in batch.events or []:
            if isinstance(ev, BlockStored):
                self._apply_stored(rep, ev)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    rep.remove_hash(h)
                self.stats["blocks_removed"] += len(ev.block_hashes)
            elif isinstance(ev, AllBlocksCleared):
                rep.clear()
                self.stats["all_blocks_cleared"] += 1

    def _apply_stored(self, rep: ReplicaIndex, ev: BlockStored) -> None:
        bs = ev.block_size or self.block_size
        if bs > 0:
            self.block_size = bs
        parent = ev.parent_block_hash
        token_ids = ev.token_ids or []
        # block_hashes and chunks of token_ids must align (block_size tokens
        # per block). If token_ids is empty (some events emit metadata-only),
        # we have nothing usable — skip.
        if not token_ids or not ev.block_hashes:
            self.stats["decode_errors"] += 1
            return
        for i, h in enumerate(ev.block_hashes):
            start = i * bs
            end = start + bs
            if end > len(token_ids):
                # Truncated chunk — bail out of this event rather than store partials.
                self.stats["decode_errors"] += 1
                break
            tok = tuple(token_ids[start:end])
            rep.store(parent, tok, h)
            parent = h
        self.stats["blocks_stored"] += len(ev.block_hashes)


# ---------------------------------------------------------------------------
# ZMQ subscriber task — one per replica.
# ---------------------------------------------------------------------------


async def replay_from_router(
    index: EventIndex,
    replica_id: str,
    router_endpoint: str,
    *,
    from_seq: int = 0,
    timeout_s: float = 8.0,
) -> int:
    """Drain the publisher's replay buffer for this replica.

    vLLM's ZmqEventPublisher exposes a sidecar ROUTER socket (PUB port + 1
    by default) that lets a late-joining subscriber request batches from a
    given sequence number onward. The publisher serves whatever's in its
    bounded ring buffer (``buffer_steps``, default 10 000 batches).

    Protocol (from vllm/examples/features/kv_events/kv_events_subscriber.py):
      REQ → 8-byte big-endian seq
      REP → (seq_bytes, payload) for each buffered batch, then
            (seq_bytes, b"") as an end-of-replay marker.

    This is the load-bearing fix for CAC-136: a freshly-started subscriber
    that hits ROUTER replay at seq=0 recovers the root BlockStored events
    (parent=None) that PUB/SUB dropped during slow-joiner.

    Returns the count of batches successfully applied. Logs and returns 0
    on connection failure — the proxy keeps running with content-re-anchor
    as the fallback.
    """
    ctx = zmq.asyncio.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    applied = 0
    try:
        sock.connect(router_endpoint)
        await sock.send(from_seq.to_bytes(8, "big"))
        logging.info(
            "zmq_replay.request replica=%s endpoint=%s from_seq=%d",
            replica_id, router_endpoint, from_seq,
        )
        while True:
            try:
                frames = await asyncio.wait_for(
                    sock.recv_multipart(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                logging.warning(
                    "zmq_replay.timeout replica=%s applied=%d", replica_id, applied
                )
                break
            if len(frames) < 2:
                break
            seq_bytes, payload = frames[0], frames[1]
            if not payload:
                # End-of-replay marker
                logging.info(
                    "zmq_replay.done replica=%s endpoint=%s applied=%d",
                    replica_id, router_endpoint, applied,
                )
                break
            try:
                batch = _DECODER.decode(payload)
            except Exception:
                index.stats["decode_errors"] += 1
                continue
            index.apply_batch(replica_id, batch)
            index.stats["replay_batches_received"] += 1
            applied += 1
    except Exception as e:
        logging.warning(
            "zmq_replay.error replica=%s endpoint=%s err=%s",
            replica_id, router_endpoint, e,
        )
    finally:
        sock.close()
    return applied


async def subscribe_replica(
    index: EventIndex,
    replica_id: str,
    endpoint: str,
    topic: str = "kv-events",
    *,
    reconnect_backoff_s: float = 2.0,
    router_endpoint: Optional[str] = None,
    initial_settle_s: float = 0.5,
) -> None:
    """Long-running task: subscribe to one replica's vLLM ZMQ event stream.

    If ``router_endpoint`` is provided, requests a one-shot replay from
    seq=0 against that endpoint after the SUB attaches but before the
    main receive loop starts. This recovers events that ZMQ PUB dropped
    while our subscriber was still completing its TCP+SUBSCRIBE handshake
    (the CAC-136 slow-joiner trap).

    Reconnects the SUB on error after a brief backoff. Cancellation-safe.
    """
    ctx = zmq.asyncio.Context.instance()
    first_attach = True
    while True:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(endpoint)
            sock.setsockopt_string(zmq.SUBSCRIBE, topic)
            logging.info(
                "zmq_subscribe.connected replica=%s endpoint=%s topic=%s",
                replica_id, endpoint, topic,
            )
            # Give the TCP + subscription handshake time to settle so the PUB
            # doesn't drop early events. ROUTER replay covers anything we
            # still miss in this window.
            await asyncio.sleep(initial_settle_s)
            if first_attach and router_endpoint:
                await replay_from_router(index, replica_id, router_endpoint)
            first_attach = False
            while True:
                frames = await sock.recv_multipart()
                payload = frames[-1] if frames else b""
                try:
                    batch = _DECODER.decode(payload)
                except Exception:
                    index.stats["decode_errors"] += 1
                    continue
                index.apply_batch(replica_id, batch)
        except asyncio.CancelledError:
            sock.close()
            raise
        except Exception as e:
            index.stats["zmq_errors"] += 1
            logging.warning(
                "zmq_subscribe.error replica=%s endpoint=%s err=%s — reconnecting in %.1fs",
                replica_id, endpoint, e, reconnect_backoff_s,
            )
            try:
                sock.close()
            except Exception:
                pass
            await asyncio.sleep(reconnect_backoff_s)
