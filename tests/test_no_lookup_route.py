"""Unit tests for --no-lookup-route mode (CAC-153).

When the flag is set the proxy must:

* Skip the tokenizer load (it's expensive and unused in this mode).
* Skip the gRPC channel/stub creation (--ic-server is unused).
* Skip ZMQ subscriber tasks (the chain table is unused).
* Round-robin every request across the configured ``--replicas`` pool,
  regardless of prompt content.

Without these properties the harness's no-hint mode falls back to a single
upstream and the Phase-2 attribution is wrong.
"""

from __future__ import annotations

from collections import Counter
from typing import List
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

from event_index import EventIndex  # noqa: E402
from lookup_proxy import LookupProxy  # noqa: E402


REPLICA_URLS = ("http://up-r0", "http://up-r1", "http://up-r2")


def _make_proxy(no_lookup_route: bool, replicas=REPLICA_URLS):
    index = EventIndex()
    # ZMQ-side event index entries aren't strictly needed for round-robin
    # (CAC-154 round-robins over self.replicas, not event_index.replicas),
    # but populate them so /proxy/metrics looks consistent in either mode.
    for i, _url in enumerate(replicas):
        index.add_replica(f"r{i}", _url)
    with patch("lookup_proxy.AutoTokenizer.from_pretrained") as p:
        proxy = LookupProxy(
            ic_server="ignored:0",
            replicas=list(replicas),
            tokenizer_name="any-tokenizer",
            hash_scheme="vllm",
            event_index=index,
            tenant="t",
            no_lookup_route=no_lookup_route,
        )
        proxy._tokenizer_loader = p  # type: ignore[attr-defined]
    return proxy


def test_constructor_skips_tokenizer_when_no_lookup_route():
    proxy = _make_proxy(no_lookup_route=True)
    assert proxy.tokenizer is None
    proxy._tokenizer_loader.assert_not_called()


def test_constructor_loads_tokenizer_when_lookup_enabled():
    proxy = _make_proxy(no_lookup_route=False)
    assert proxy.tokenizer is not None
    proxy._tokenizer_loader.assert_called_once_with("any-tokenizer")


@pytest.mark.asyncio
async def test_setup_skips_grpc_channel_when_no_lookup_route():
    proxy = _make_proxy(no_lookup_route=True)
    with patch("lookup_proxy.grpc.aio.insecure_channel") as mock_chan:
        await proxy.setup()
        try:
            mock_chan.assert_not_called()
            assert proxy._channel is None
            assert proxy._stub is None
            # The aiohttp session is still needed to forward requests.
            assert proxy._client_session is not None
        finally:
            await proxy.teardown()


@pytest.mark.asyncio
async def test_setup_opens_grpc_channel_when_lookup_enabled():
    proxy = _make_proxy(no_lookup_route=False)
    with patch("lookup_proxy.grpc.aio.insecure_channel") as mock_chan:
        mock_chan.return_value = MagicMock()
        await proxy.setup()
        try:
            mock_chan.assert_called_once_with("ignored:0")
        finally:
            # The MagicMock channel.close() returns a MagicMock, not an
            # awaitable; null it out before teardown to avoid the await.
            proxy._channel = None
            await proxy.teardown()


@pytest.mark.asyncio
async def test_no_lookup_route_round_robins_evenly_across_replicas():
    """Drive 30 requests through handle_chat via a real aiohttp test client.
    Expect 10/10/10 distribution across the three --replicas URLs."""
    proxy = _make_proxy(no_lookup_route=True, replicas=REPLICA_URLS)
    await proxy.setup()

    # No LookupRoute stub should exist; if handle_chat ever tries to call it
    # the test will NPE — but we also assert the stat counters below.
    assert proxy._stub is None

    upstreams_hit: List[str] = []

    class _FakeContent:
        async def iter_chunked(self, _n):
            yield b'{"ok":true}'

    class _FakeResp:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}
            self.content = _FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

    class _FakeSession:
        def post(self, url, **_kw):
            upstreams_hit.append(url.rsplit("/v1/", 1)[0])
            return _FakeResp()

        async def close(self):
            pass

    proxy._client_session = _FakeSession()  # type: ignore[assignment]

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    app.router.add_post("/v1/chat/completions", proxy.handle_chat)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        for i in range(30):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "prompt": f"req-{i}"},
            )
            assert resp.status == 200
            await resp.read()
    finally:
        await client.close()
        await proxy.teardown()

    distribution = Counter(upstreams_hit)
    assert distribution == Counter({
        "http://up-r0": 10,
        "http://up-r1": 10,
        "http://up-r2": 10,
    }), distribution

    # LOOKUP_DISABLED bookkeeping ticked once per request.
    assert proxy.stats["lookup_disabled"] == 30
    assert proxy.stats["routed_round_robin_fallback"] == 30
    # The "lookup_*" code path was not taken.
    assert proxy.stats["lookup_prefix_match"] == 0
    assert proxy.stats["lookup_no_hint"] == 0
    assert proxy.stats["lookup_skipped_no_chain"] == 0
