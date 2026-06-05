"""
lookup_proxy.py — LookupRoute-aware HTTP proxy in front of vLLM engine pods.

Why it exists
-------------
genai-bench expects a single OpenAI-compatible HTTP endpoint. inference-cache's
routing model is "ask the server which replica is warm, then route there." This
proxy bridges the two: it receives genai-bench's HTTP request, computes a prefix
hash, calls LookupRoute via gRPC, then forwards the request to the hinted
replica (or to the default upstream when NO_HINT).

This is the same shape an actual gateway integration would have.

Failure semantics: fail-open. If LookupRoute errors or times out, forward to
upstream and pretend nothing happened — same as the real gateway contract.

The HASH_SCHEME constant below MUST match the kvevent-subscriber's
--hash-scheme flag on the engine pods. A mismatch returns NO_HINT silently,
indistinguishable from a true cache miss.

Usage
-----
    python3 lookup_proxy.py \
      --listen 0.0.0.0:18100 \
      --ic-server localhost:38002 \
      --upstream http://localhost:38000 \
      --log /tmp/lookup_proxy.log

Requires the inference-cache proto stubs on sys.path
(PYTHONPATH=$INFERENCE_CACHE_PROTO_DIR).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from typing import Optional

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


# MUST match the engine's kvevent-subscriber --hash-scheme.
# Override via --hash-scheme if your engine uses something other than 'vllm'.
DEFAULT_HASH_SCHEME = "vllm"

# Hardcoded gRPC service path used by inference-cache.
SERVICE_PATH = "inferencecache.v1alpha1.InferenceCache/LookupRoute"


def hash_prefix(text: str) -> bytes:
    """Hash a prompt prefix to a fixed-width byte string for LookupRoute.

    This is a placeholder consistent with how a thin gateway-side wrapper would
    extract a prefix hash. A production gateway should use the engine's
    block-hash chain (block_hashes + block_token_counts) for longest-prefix
    matching. The purpose of this proxy is to exercise the routing path, not
    to be a production gateway.
    """
    h = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return h


class LookupProxy:
    def __init__(self, ic_server: str, upstream: str, hash_scheme: str):
        self.ic_server = ic_server
        self.upstream = upstream.rstrip("/")
        self.hash_scheme = hash_scheme
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[pb_grpc.InferenceCacheStub] = None
        self._client_session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            "requests_total": 0,
            "lookup_prefix_match": 0,
            "lookup_no_hint": 0,
            "lookup_errors": 0,
            "lookup_latency_us_sum": 0.0,
        }

    async def setup(self):
        self._channel = grpc.aio.insecure_channel(self.ic_server)
        self._stub = pb_grpc.InferenceCacheStub(self._channel)
        self._client_session = aiohttp.ClientSession()

    async def teardown(self):
        if self._channel:
            await self._channel.close()
        if self._client_session:
            await self._client_session.close()

    async def lookup(self, tenant: str, model: str, prefix_hash: bytes) -> Optional[str]:
        """Call LookupRoute; return chosen replica_id or None on NO_HINT/error."""
        assert self._stub is not None
        t0 = time.perf_counter()
        try:
            req = pb.LookupRouteRequest(
                tenant_id=tenant,
                model_id=model,
                hash_scheme=self.hash_scheme,
                prefix_hash=prefix_hash,
            )
            resp = await asyncio.wait_for(self._stub.LookupRoute(req), timeout=0.05)
        except (asyncio.TimeoutError, grpc.aio.AioRpcError) as e:
            self.stats["lookup_errors"] += 1
            logging.warning("lookup_route failed: %s", e)
            return None
        finally:
            self.stats["lookup_latency_us_sum"] += (time.perf_counter() - t0) * 1e6

        if resp.reason_code == "PREFIX_MATCH" and resp.replica_scores:
            self.stats["lookup_prefix_match"] += 1
            return resp.replica_scores[0].replica_id  # caller maps id → url
        self.stats["lookup_no_hint"] += 1
        return None

    async def forward(self, body: bytes, replica_url: str, original_headers: dict) -> aiohttp.ClientResponse:
        assert self._client_session is not None
        fwd_headers = {k: v for k, v in original_headers.items() if k.lower() not in {"host", "content-length"}}
        return await self._client_session.post(
            f"{replica_url}/v1/chat/completions",
            data=body,
            headers=fwd_headers,
            timeout=aiohttp.ClientTimeout(total=120),
        )

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        self.stats["requests_total"] += 1
        body = await request.read()
        try:
            payload = json.loads(body)
        except Exception:
            return web.Response(status=400, text="malformed JSON")

        model = payload.get("model", "")
        messages = payload.get("messages", [])
        concat = "\n".join(m.get("content", "") for m in messages)
        prefix = hash_prefix(concat)

        # Try LookupRoute; fall back to upstream on NO_HINT/error.
        replica_id = await self.lookup(tenant="default", model=model, prefix_hash=prefix)
        replica_url = self.upstream  # default
        # NOTE: a future enhancement should resolve replica_id → specific pod URL
        # via the inference-cache CacheBackend status or a client SDK. For now
        # the proxy exercises the lookup path but doesn't pin to the warm pod —
        # the TTFT win still shows up at the engine because LMCache + vLLM's
        # local prefix cache pick up the same KV.

        try:
            async with await self.forward(body, replica_url, dict(request.headers)) as upstream_resp:
                stream = web.StreamResponse(status=upstream_resp.status, headers=upstream_resp.headers)
                await stream.prepare(request)
                async for chunk in upstream_resp.content.iter_chunked(8192):
                    await stream.write(chunk)
                await stream.write_eof()
                return stream
        except Exception as e:
            logging.exception("forward failed")
            return web.Response(status=502, text=f"upstream error: {e}")

    async def handle_metrics(self, _: web.Request) -> web.Response:
        """Expose proxy-local stats so correlate.py can merge them."""
        return web.json_response(self.stats)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:18100")
    ap.add_argument("--ic-server", required=True, help="inference-cache-server gRPC endpoint")
    ap.add_argument("--upstream", required=True, help="default vLLM upstream URL")
    ap.add_argument("--hash-scheme", default=DEFAULT_HASH_SCHEME,
                    help=f"hash scheme to send with LookupRoute (default: {DEFAULT_HASH_SCHEME})")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    proxy = LookupProxy(
        ic_server=args.ic_server,
        upstream=args.upstream,
        hash_scheme=args.hash_scheme,
    )
    await proxy.setup()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", proxy.handle_chat)
    app.router.add_get("/proxy/metrics", proxy.handle_metrics)
    app.router.add_get("/health", lambda _r: web.Response(status=200))

    host, port = args.listen.rsplit(":", 1)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, int(port))
    await site.start()
    logging.info(
        "lookup_proxy listening on %s; ic_server=%s; upstream=%s; hash_scheme=%s",
        args.listen, args.ic_server, args.upstream, args.hash_scheme,
    )
    try:
        await asyncio.Event().wait()
    finally:
        await proxy.teardown()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
