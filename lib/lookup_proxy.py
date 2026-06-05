"""
lookup_proxy.py — LookupRoute-aware HTTP proxy in front of vLLM engine pods.

Architecture (CAC-135 Option B-b — observe rather than recompute)
-----------------------------------------------------------------
genai-bench expects a single OpenAI-compatible HTTP endpoint. inference-cache's
routing model is "ask the server which replica is warm, then route there."
This proxy bridges the two:

1. Subscribe to each replica's vLLM ZMQ event stream and observe the engine's
   own BlockStored events. Maintain a per-replica chain table mapping
   (parent_hash, tuple_of_token_ids) → block_hash.

2. On an incoming HTTP request:
   a. Tokenize the prompt using the configured HF tokenizer.
   b. Walk each replica's chain table to find the longest leading-block
      match for the request's token sequence.
   c. Send LookupRoute with that replica's exact block_hashes + token_counts
      chain. Because those hashes came from the replica's own events, the
      server's index has them and returns PREFIX_MATCH.
   d. Forward the HTTP request to the hinted replica's upstream URL.
   e. Fall back to round-robin / default upstream on NO_HINT / TIMEOUT / error.

Why this approach rather than recomputing vLLM's hash in Python: vLLM's default
hash function is `builtins.hash()` — process-local with PYTHONHASHSEED
randomization, not reproducible across pods. Reimplementation would silently
break with every vLLM upgrade. Observing the engine's emitted hashes is
robust to upstream changes.

Failure semantics: fail-open. If LookupRoute errors / times out / returns
NO_HINT, the proxy forwards to the default upstream — same as the gateway
contract.

Usage
-----
    python3 lookup_proxy.py \
      --listen 0.0.0.0:18100 \
      --ic-server localhost:38002 \
      --default-upstream http://localhost:38000 \
      --tokenizer hf-internal-testing/llama-tokenizer \
      --replica 'replica-0|tcp://localhost:15001|http://localhost:38010' \
      --replica 'replica-1|tcp://localhost:15002|http://localhost:38011' \
      --replica 'replica-2|tcp://localhost:15003|http://localhost:38012' \
      --log /tmp/lookup_proxy.log

Each --replica arg is "<id>|<zmq_endpoint>|<http_upstream_url>". Port-forward
each replica's :5557 (ZMQ) and :8000 (HTTP) to local ports first.

Requires the inference-cache proto stubs on sys.path
(PYTHONPATH=$INFERENCE_CACHE_PROTO_DIR).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Optional, Tuple

import aiohttp
from aiohttp import web

try:
    import grpc

    # protoc emits stubs under the proto's package path:
    #   proto/inferencecache/v1alpha1/inferencecache_pb2.py
    # PYTHONPATH must include the `proto/` root (NOT the v1alpha1/ leaf).
    from inferencecache.v1alpha1 import inferencecache_pb2 as pb
    from inferencecache.v1alpha1 import inferencecache_pb2_grpc as pb_grpc
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        f"Could not import inference-cache proto stubs: {e}\n"
        "Generate them with `make proto` (sibling inference-cache repo expected)\n"
        "or set PYTHONPATH to the directory containing inferencecache/v1alpha1/.\n"
    )
    sys.exit(2)

try:
    from event_index import EventIndex, subscribe_replica
except ImportError:
    # When invoked as `python3 lib/lookup_proxy.py`, the lib/ dir is the script
    # parent; otherwise the absolute import works.
    from lib.event_index import EventIndex, subscribe_replica  # type: ignore

try:
    from transformers import AutoTokenizer
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        f"Could not import transformers: {e}\nInstall with: make install\n"
    )
    sys.exit(2)


# MUST match the engine's kvevent-subscriber `--hash-scheme` flag.
DEFAULT_HASH_SCHEME = "vllm"

# Client-side LookupRoute timeout — fail-open on overrun. Must exceed
# CachePolicy.lookupTimeoutMs + network RTT + serialization. Block-hash chains
# of 80+ blocks take longer than the proxy old 50ms budget could allow.
LOOKUP_TIMEOUT_S = 0.5


class LookupProxy:
    def __init__(
        self,
        ic_server: str,
        default_upstream: str,
        tokenizer_name: str,
        hash_scheme: str,
        event_index: EventIndex,
    ):
        self.ic_server = ic_server
        self.default_upstream = default_upstream.rstrip("/")
        self.hash_scheme = hash_scheme
        self.event_index = event_index
        self._channel: Optional["grpc.aio.Channel"] = None
        self._stub: Optional[pb_grpc.InferenceCacheStub] = None
        self._client_session: Optional[aiohttp.ClientSession] = None

        logging.info("loading tokenizer: %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info("tokenizer loaded: vocab_size=%d", len(self.tokenizer))

        # Round-robin counter for fallback-mode replica selection.
        self._rr_counter = 0
        self.stats = {
            "requests_total": 0,
            "lookup_prefix_match": 0,
            "lookup_no_hint": 0,
            "lookup_timeout": 0,
            "lookup_errors": 0,
            "lookup_skipped_no_chain": 0,
            "routed_to_hint": 0,
            "routed_to_default": 0,
            "routed_round_robin_fallback": 0,
            "lookup_latency_us_sum": 0.0,
        }

    async def setup(self) -> None:
        self._channel = grpc.aio.insecure_channel(self.ic_server)
        self._stub = pb_grpc.InferenceCacheStub(self._channel)
        self._client_session = aiohttp.ClientSession()

    async def teardown(self) -> None:
        if self._channel:
            await self._channel.close()
        if self._client_session:
            await self._client_session.close()

    # ---- core: tokenize, chain-walk, lookup, route -------------------------

    def _extract_prompt_tokens(self, payload: dict) -> list:
        """Convert OpenAI chat-completion or completion payload → token IDs.

        We match vLLM's tokenization path so the resulting block hashes
        agree with what the engine emits in ZMQ events.

        OpenAI-style payloads come in two shapes:
          /v1/chat/completions  → {"messages": [{"role":..., "content":...}]}
          /v1/completions       → {"prompt": "raw text"}

        The legacy `/v1/completions` doesn't apply a chat template; vLLM
        tokenizes the raw `prompt` field as-is. genai-bench's text-to-text
        backend defaults to this path.
        """
        # 1) /v1/completions style — prompt as a string (or list of strings)
        prompt = payload.get("prompt")
        if prompt is not None:
            if isinstance(prompt, list):
                # OpenAI allows batch prompts; we concatenate for token-budget purposes
                prompt = "\n".join(str(p) for p in prompt)
            return list(self.tokenizer.encode(str(prompt), add_special_tokens=True))
        # 2) /v1/chat/completions style — apply the chat template
        messages = payload.get("messages") or []
        if messages:
            try:
                # Render the template to a string first, then tokenize. Some
                # tokenizer versions return a BatchEncoding from
                # apply_chat_template(tokenize=True) — a 2-element view of
                # [input_ids, attention_mask] rather than a flat int list,
                # which silently collapses to len=2 downstream. Going through
                # the text path avoids that ambiguity.
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # The template already includes its own special tokens.
                return list(self.tokenizer.encode(rendered, add_special_tokens=False))
            except Exception as e:
                logging.warning("apply_chat_template failed (%s); falling back to concat", e)
                text = "\n".join(m.get("content", "") for m in messages)
                return list(self.tokenizer.encode(text, add_special_tokens=True))
        logging.warning(
            "request body has neither 'prompt' nor 'messages'; keys=%s",
            list(payload.keys()),
        )
        return []

    async def _lookup_with_chain(
        self,
        tenant: str,
        model: str,
        block_hashes: list,
        block_token_counts: list,
    ) -> Tuple[Optional[str], str]:
        """Call LookupRoute with a block-hash chain.

        Returns (replica_id_hint, reason_code). replica_id_hint is the
        server's top pick (may be None on NO_HINT/TIMEOUT/error).
        """
        assert self._stub is not None
        t0 = time.perf_counter()
        try:
            req = pb.LookupRouteRequest(
                tenant_id=tenant,
                model_id=model,
                hash_scheme=self.hash_scheme,
                block_hashes=[_int_to_be8(h) for h in block_hashes],
                block_token_counts=block_token_counts,
            )
            resp = await asyncio.wait_for(self._stub.LookupRoute(req), timeout=LOOKUP_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.stats["lookup_timeout"] += 1
            return None, "TIMEOUT"
        except grpc.aio.AioRpcError as e:
            self.stats["lookup_errors"] += 1
            logging.warning("lookup_route_rpc_error: %s", e)
            return None, "ERROR"
        finally:
            self.stats["lookup_latency_us_sum"] += (time.perf_counter() - t0) * 1e6

        rc = resp.reason_code or "NO_HINT"
        if rc == "PREFIX_MATCH" and resp.replica_scores:
            self.stats["lookup_prefix_match"] += 1
            return resp.replica_scores[0].replica_id, rc
        if rc == "NO_HINT":
            self.stats["lookup_no_hint"] += 1
        return None, rc

    def _pick_upstream(self, hint_replica_id: Optional[str]) -> Tuple[str, str]:
        """Map a hint to an upstream URL. Returns (upstream_url, routing_reason).

        Order of preference:
          1. Hint matches a known replica → use that replica's upstream
          2. Hint is set but replica unknown to proxy → use default_upstream
          3. No hint → round-robin among known replicas if available
             (this is what the harness wants: even on NO_HINT, fan out across
             replicas so different prefixes land on different pods and the
             chain table populates faster)
          4. No known replicas → default_upstream
        """
        if hint_replica_id is not None:
            rep = self.event_index.replicas.get(hint_replica_id)
            if rep is not None:
                self.stats["routed_to_hint"] += 1
                return rep.upstream_url, "HINT"
        if self.event_index.replicas:
            replicas = list(self.event_index.replicas.values())
            chosen = replicas[self._rr_counter % len(replicas)]
            self._rr_counter += 1
            self.stats["routed_round_robin_fallback"] += 1
            return chosen.upstream_url, "ROUND_ROBIN"
        self.stats["routed_to_default"] += 1
        return self.default_upstream, "DEFAULT"

    # ---- HTTP handlers ----------------------------------------------------

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        self.stats["requests_total"] += 1
        body = await request.read()
        try:
            payload = json.loads(body)
        except Exception:
            return web.Response(status=400, text="malformed JSON")

        model = payload.get("model", "")
        token_ids = self._extract_prompt_tokens(payload)

        replica_id_local, hashes, counts = self.event_index.find_best_chain(token_ids)
        if hashes:
            hint_replica, reason = await self._lookup_with_chain(
                tenant="default",
                model=model,
                block_hashes=hashes,
                block_token_counts=counts,
            )
        else:
            self.stats["lookup_skipped_no_chain"] += 1
            hint_replica, reason = None, "NO_CHAIN_OBSERVED"

        upstream_url, route_reason = self._pick_upstream(hint_replica)

        # Inject a header so genai-bench / log analysis can see the proxy's
        # routing decision per request.
        log_extra = {
            "tokens": len(token_ids),
            "chain_blocks": len(hashes),
            "local_pick": replica_id_local,
            "lookup_reason": reason,
            "route_reason": route_reason,
            "upstream": upstream_url,
        }
        logging.info("route_decision %s", log_extra)

        try:
            assert self._client_session is not None
            fwd_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in {"host", "content-length"}
            }
            async with self._client_session.post(
                f"{upstream_url}/v1/chat/completions",
                data=body,
                headers=fwd_headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as upstream_resp:
                stream = web.StreamResponse(
                    status=upstream_resp.status, headers=upstream_resp.headers
                )
                stream.headers["X-Cache-Lookup-Reason"] = reason
                stream.headers["X-Cache-Route-Reason"] = route_reason
                await stream.prepare(request)
                async for chunk in upstream_resp.content.iter_chunked(8192):
                    await stream.write(chunk)
                await stream.write_eof()
                return stream
        except Exception as e:
            logging.exception("forward failed")
            return web.Response(status=502, text=f"upstream error: {e}")

    async def handle_metrics(self, _: web.Request) -> web.Response:
        out = {
            "proxy": self.stats,
            "event_index": self.event_index.stats,
            "replicas": {
                rid: {
                    "upstream_url": rep.upstream_url,
                    "chain_entries": len(rep.chain_table),
                    "events_received": rep.events_received,
                    "last_seen_s_ago": (time.time() - rep.last_seen) if rep.last_seen else None,
                }
                for rid, rep in self.event_index.replicas.items()
            },
        }
        return web.json_response(out)


def _int_to_be8(h: int) -> bytes:
    """Normalize a vLLM-emitted integer hash to 8-byte big-endian bytes.

    Matches the inference-cache adapter (pkg/adapters/engine/events.go
    `decodeHashes`): "either binary (used as-is) or an integer (vLLM's int hash
    variant, normalized to 8-byte big-endian)."

    Python's builtins.hash() returns a signed 64-bit int; mask to unsigned
    before serializing.
    """
    return (h & 0xFFFFFFFFFFFFFFFF).to_bytes(8, byteorder="big", signed=False)


def _parse_replica_spec(spec: str) -> Tuple[str, str, str, Optional[str]]:
    """Parse `id|zmq_sub_endpoint|http_url[|zmq_router_endpoint]`.

    Pipe is the separator because both URLs contain colons (`tcp://host:port`,
    `http://host:port`), which makes a colon-separated tuple ambiguous. The id
    cannot contain a pipe.

    The fourth field (ROUTER endpoint for replay-on-subscribe) is optional;
    when present, the subscriber requests a seq=0 replay at startup to
    recover events lost to ZMQ slow-joiner (CAC-136). vLLM's publisher binds
    the ROUTER on `pub_port + 1` by default.
    """
    parts = spec.split("|")
    if len(parts) not in (3, 4) or not all(parts[:3]):
        raise argparse.ArgumentTypeError(
            f"--replica spec must be id|zmq_sub|http_url[|zmq_router] (three "
            f"or four pipe-separated parts), got: {spec!r}"
        )
    rid = parts[0]
    zmq_endpoint = parts[1]
    http_url = parts[2]
    zmq_router = parts[3] if len(parts) == 4 and parts[3] else None
    if "://" not in zmq_endpoint or "://" not in http_url:
        raise argparse.ArgumentTypeError(
            f"--replica spec parts 2 and 3 must be URLs with a scheme, got: "
            f"zmq={zmq_endpoint!r} http={http_url!r}"
        )
    if zmq_router and "://" not in zmq_router:
        raise argparse.ArgumentTypeError(
            f"--replica spec part 4 (zmq_router) must be a URL with a scheme, "
            f"got: {zmq_router!r}"
        )
    return rid, zmq_endpoint, http_url, zmq_router


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    index = EventIndex()
    for rid, zmq_ep, http_url, _zmq_router in args.replica:
        index.add_replica(rid, http_url)

    proxy = LookupProxy(
        ic_server=args.ic_server,
        default_upstream=args.default_upstream,
        tokenizer_name=args.tokenizer,
        hash_scheme=args.hash_scheme,
        event_index=index,
    )
    await proxy.setup()

    # Spin up subscriber tasks. When the spec carries a ROUTER endpoint
    # (4th field), the subscriber will request seq=0 replay at startup
    # to recover slow-joiner-lost events (CAC-136).
    subscriber_tasks = [
        asyncio.create_task(
            subscribe_replica(index, rid, zmq_ep, router_endpoint=zmq_router)
        )
        for rid, zmq_ep, _http, zmq_router in args.replica
    ]

    app = web.Application()
    app.router.add_post("/v1/chat/completions", proxy.handle_chat)
    # genai-bench's text-to-text backend hits /v1/completions by default;
    # route both through the same handler (handle_chat picks the right
    # payload shape).
    app.router.add_post("/v1/completions", proxy.handle_chat)
    app.router.add_get("/proxy/metrics", proxy.handle_metrics)
    app.router.add_get("/health", lambda _r: web.Response(status=200))

    host, port = args.listen.rsplit(":", 1)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, int(port))
    await site.start()
    logging.info(
        "lookup_proxy listening on %s; ic_server=%s; default_upstream=%s; "
        "hash_scheme=%s; replicas=%d",
        args.listen, args.ic_server, args.default_upstream,
        args.hash_scheme, len(args.replica),
    )

    try:
        await asyncio.Event().wait()
    finally:
        for t in subscriber_tasks:
            t.cancel()
        await proxy.teardown()
        await runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:18100")
    ap.add_argument("--ic-server", required=True,
                    help="inference-cache-server gRPC endpoint, e.g. localhost:38002")
    ap.add_argument("--default-upstream", required=True,
                    help="vLLM fallback URL when no hint / no replicas")
    ap.add_argument("--upstream", default=None,
                    help="DEPRECATED alias for --default-upstream")
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer id, e.g. hf-internal-testing/llama-tokenizer")
    ap.add_argument(
        "--replica",
        action="append",
        type=_parse_replica_spec,
        default=[],
        help="Replica config: id|zmq_sub|http_url[|zmq_router]. Repeat per replica. "
             "The optional 4th field is vLLM's ROUTER replay endpoint (PUB port + 1 "
             "by default); when present, the proxy requests seq=0 replay at startup "
             "to recover slow-joiner-lost events. Example: "
             "'r0|tcp://localhost:15001|http://localhost:38010|tcp://localhost:15101'",
    )
    ap.add_argument("--hash-scheme", default=DEFAULT_HASH_SCHEME)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if args.upstream and not args.default_upstream:
        args.default_upstream = args.upstream

    if not args.replica:
        sys.stderr.write(
            "WARNING: no --replica specified. Without per-replica ZMQ subscriptions, "
            "the proxy cannot observe block hashes, so LookupRoute will always return "
            "NO_HINT and every request will route to --default-upstream. This is the "
            "broken behavior CAC-135 fixes; pass --replica args to enable it.\n"
        )

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
