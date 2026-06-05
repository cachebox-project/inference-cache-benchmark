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
from typing import Dict, List, Optional, Tuple

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


_KVEvent = "BlockStored | BlockRemoved | AllBlocksCleared"


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float = 0.0
    events: list = []   # decoded as tagged-union of the three above


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
    """

    replica_id: str
    upstream_url: str
    chain_table: "OrderedDict[Tuple[Optional[int], Tuple[int, ...]], int]" = field(default_factory=OrderedDict)
    # Reverse index: block_hash → (parent_hash, token_tuple). Used to remove
    # entries on BlockRemoved without scanning the whole table.
    by_hash: Dict[int, Tuple[Optional[int], Tuple[int, ...]]] = field(default_factory=dict)
    last_seen: float = 0.0
    events_received: int = 0

    def store(self, parent_hash: Optional[int], token_tuple: Tuple[int, ...], block_hash: int) -> None:
        key = (parent_hash, token_tuple)
        # If we already had this hash under a different key (shouldn't happen
        # in practice, but defensive), clean up the by_hash entry first.
        if block_hash in self.by_hash and self.by_hash[block_hash] != key:
            old_key = self.by_hash[block_hash]
            self.chain_table.pop(old_key, None)
        self.chain_table[key] = block_hash
        self.chain_table.move_to_end(key)
        self.by_hash[block_hash] = key
        if len(self.chain_table) > _MAX_ENTRIES_PER_REPLICA:
            evicted_key, evicted_hash = self.chain_table.popitem(last=False)
            self.by_hash.pop(evicted_hash, None)
        self.last_seen = time.time()

    def lookup(self, parent_hash: Optional[int], token_tuple: Tuple[int, ...]) -> Optional[int]:
        return self.chain_table.get((parent_hash, token_tuple))

    def remove_hash(self, block_hash: int) -> None:
        key = self.by_hash.pop(block_hash, None)
        if key is not None:
            self.chain_table.pop(key, None)

    def clear(self) -> None:
        self.chain_table.clear()
        self.by_hash.clear()


# ---------------------------------------------------------------------------
# EventIndex — aggregator across replicas.
# ---------------------------------------------------------------------------


class EventIndex:
    def __init__(self) -> None:
        self.replicas: Dict[str, ReplicaIndex] = {}
        # Inferred from BlockStored.block_size. Defaults to 16 (vLLM's default
        # KV block size); updated by the first event we receive.
        self.block_size: int = 16
        self.stats = {
            "events_received": 0,
            "blocks_stored": 0,
            "blocks_removed": 0,
            "all_blocks_cleared": 0,
            "zmq_errors": 0,
            "decode_errors": 0,
        }

    def add_replica(self, replica_id: str, upstream_url: str) -> None:
        if replica_id not in self.replicas:
            self.replicas[replica_id] = ReplicaIndex(replica_id=replica_id, upstream_url=upstream_url)

    def find_best_chain(
        self, token_ids: List[int], min_blocks: int = 1
    ) -> Tuple[Optional[str], List[int], List[int]]:
        """For a request's token sequence, find the replica with the longest
        leading-block chain we've observed. Returns
        ``(replica_id, block_hashes, block_token_counts)``.

        If no replica has even the first block, returns ``(None, [], [])``.
        """
        bs = self.block_size
        if bs <= 0 or len(token_ids) < bs:
            return None, [], []

        best_rid: Optional[str] = None
        best_hashes: List[int] = []
        best_counts: List[int] = []
        for rid, rep in self.replicas.items():
            hashes: List[int] = []
            counts: List[int] = []
            parent: Optional[int] = None
            pos = 0
            while pos + bs <= len(token_ids):
                block = tuple(token_ids[pos : pos + bs])
                h = rep.lookup(parent, block)
                if h is None:
                    break
                hashes.append(h)
                counts.append(bs)
                parent = h
                pos += bs
            if len(hashes) > len(best_hashes):
                best_rid, best_hashes, best_counts = rid, hashes, counts
        if len(best_hashes) < min_blocks:
            return None, [], []
        return best_rid, best_hashes, best_counts

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


async def subscribe_replica(
    index: EventIndex,
    replica_id: str,
    endpoint: str,
    topic: str = "kv-events",
    *,
    reconnect_backoff_s: float = 2.0,
) -> None:
    """Long-running task: subscribe to one replica's vLLM ZMQ event stream.

    Reconnects on error after a brief backoff. Cancellation-safe.
    """
    ctx = zmq.asyncio.Context.instance()
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
