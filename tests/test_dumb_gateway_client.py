"""Unit + integration-mock tests for lib/dumb_gateway_client.py (CAC-152).

The new dumb client has no chain table, no ZMQ subscription, and no per-replica
state. Tests focus on:

  * hash-block construction (content-addressed, deterministic, full-block-only)
  * replica spec parser (both id=url and legacy id|zmq|http_url|router)
  * routing semantics — PREFIX_MATCH honored, NO_HINT/TIMEOUT/TENANT_HOT/unknown
    hint all fall through to round-robin (CAC-154 spread, not pinning)
  * round-robin distribution is even and per-process deterministic
  * --routing-mode=round-robin short-circuits the RPC entirely
  * end-to-end HTTP integration with a fake aiohttp upstream + mocked stub
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# The full client pulls in transformers (heavy) + the proto stubs (shimmed by
# tests/conftest.py). Skip if transformers isn't installed.
pytest.importorskip("transformers")

from dumb_gateway_client import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DumbGatewayClient,
    _classify_match_quality,
    _hash_blocks,
    _parse_replica,
)


# ---- _hash_blocks ----------------------------------------------------------


def test_hash_blocks_returns_empty_for_short_prompt():
    """Anything shorter than one full block is dropped (engine never indexes it)."""
    hashes, counts = _hash_blocks(list(range(DEFAULT_BLOCK_SIZE - 1)), DEFAULT_BLOCK_SIZE)
    assert hashes == []
    assert counts == []


def test_hash_blocks_one_full_block():
    hashes, counts = _hash_blocks(list(range(DEFAULT_BLOCK_SIZE)), DEFAULT_BLOCK_SIZE)
    assert len(hashes) == 1
    assert len(hashes[0]) == 8
    assert counts == [DEFAULT_BLOCK_SIZE]


def test_hash_blocks_drops_partial_trailing_block():
    """4-block-worth of tokens + 5 extra → 4 blocks, no trailing partial."""
    tokens = list(range(DEFAULT_BLOCK_SIZE * 4 + 5))
    hashes, counts = _hash_blocks(tokens, DEFAULT_BLOCK_SIZE)
    assert len(hashes) == 4
    assert counts == [DEFAULT_BLOCK_SIZE] * 4


def test_hash_blocks_chain_is_content_addressed():
    """Same prompt → same chain. Different prompt → different chain at the
    first divergent block onward."""
    a = list(range(DEFAULT_BLOCK_SIZE * 3))
    b = list(range(DEFAULT_BLOCK_SIZE * 3))
    c = list(range(DEFAULT_BLOCK_SIZE * 3))
    c[-1] = 999_999

    ha, _ = _hash_blocks(a, DEFAULT_BLOCK_SIZE)
    hb, _ = _hash_blocks(b, DEFAULT_BLOCK_SIZE)
    hc, _ = _hash_blocks(c, DEFAULT_BLOCK_SIZE)
    assert ha == hb
    assert ha[:2] == hc[:2]  # shared prefix matches
    assert ha[2] != hc[2]    # diverges in the last block


def test_hash_blocks_chains_parent_into_child():
    """block_hash[i] = sha256(block_hash[i-1] || token_bytes_block_i)[:8].

    Validates the chain explicitly so a future change to the algorithm
    (e.g. swapping in a different digest) is loud.
    """
    tokens = list(range(DEFAULT_BLOCK_SIZE * 2))
    hashes, _ = _hash_blocks(tokens, DEFAULT_BLOCK_SIZE)
    expected0_bytes = b"".join(
        (t & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        for t in range(DEFAULT_BLOCK_SIZE)
    )
    expected0 = hashlib.sha256(b"\x00" * 8 + expected0_bytes).digest()[:8]
    assert hashes[0] == expected0
    expected1_bytes = b"".join(
        (t & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        for t in range(DEFAULT_BLOCK_SIZE, 2 * DEFAULT_BLOCK_SIZE)
    )
    expected1 = hashlib.sha256(expected0 + expected1_bytes).digest()[:8]
    assert hashes[1] == expected1


# ---- _parse_replica --------------------------------------------------------


def test_parse_replica_id_equals_url():
    assert _parse_replica("r0=http://localhost:38010") == ("r0", "http://localhost:38010")


def test_parse_replica_strips_trailing_slash():
    assert _parse_replica("r0=http://x/") == ("r0", "http://x")


def test_parse_replica_legacy_pipe_form():
    """Drop-in support for LOOKUP_PROXY_REPLICAS encoded as id|zmq|http_url[|router]."""
    spec = "r0|tcp://localhost:15001|http://localhost:38010"
    assert _parse_replica(spec) == ("r0", "http://localhost:38010")


def test_parse_replica_legacy_with_router_endpoint():
    spec = "r0|tcp://localhost:15001|http://localhost:38010|tcp://localhost:15101"
    assert _parse_replica(spec) == ("r0", "http://localhost:38010")


@pytest.mark.parametrize("bad", [
    "no-equals-or-pipe",
    "=missing-id",
    "r0=",
    "r0=no-scheme",
    "|missing-id|http://x",
])
def test_parse_replica_rejects_malformed(bad):
    import argparse as ap
    with pytest.raises(ap.ArgumentTypeError):
        _parse_replica(bad)


# ---- _classify_match_quality ----------------------------------------------


@pytest.mark.parametrize("blocks,expected", [
    (0, None), (-1, None),
    (1, "trivial"),
    (2, "weak"), (5, "weak"), (7, "weak"),
    (8, "strong"), (50, "strong"),
])
def test_classify_match_quality_buckets(blocks, expected):
    assert _classify_match_quality(blocks) == expected


# ---- DumbGatewayClient construction ---------------------------------------


REPLICAS: List[Tuple[str, str]] = [
    ("r0", "http://r0.local:38010"),
    ("r1", "http://r1.local:38011"),
    ("r2", "http://r2.local:38012"),
]


def _build(routing_mode: str = "lookup", replicas=REPLICAS) -> DumbGatewayClient:
    with patch("dumb_gateway_client.AutoTokenizer.from_pretrained",
               return_value=MagicMock(__len__=lambda self: 0)):
        return DumbGatewayClient(
            ic_server="ignored:0",
            replicas=replicas,
            tokenizer_name="any",
            tenant="t",
            routing_mode=routing_mode,
        )


def test_constructor_rejects_empty_replicas():
    with pytest.raises(ValueError):
        _build(replicas=[])


def test_constructor_rejects_unknown_routing_mode():
    with pytest.raises(ValueError):
        _build(routing_mode="bogus")


def test_constructor_skips_tokenizer_in_round_robin_mode():
    client = _build(routing_mode="round-robin")
    assert client.tokenizer is None


def test_constructor_loads_tokenizer_in_lookup_mode():
    with patch("dumb_gateway_client.AutoTokenizer.from_pretrained") as load:
        load.return_value = MagicMock(__len__=lambda self: 0)
        DumbGatewayClient(
            ic_server="ignored:0",
            replicas=REPLICAS,
            tokenizer_name="some-model",
            tenant="t",
            routing_mode="lookup",
        )
        load.assert_called_once_with("some-model")


# ---- _pick_upstream + round-robin -----------------------------------------


def test_pick_upstream_honors_prefix_match_hint():
    client = _build()
    url, reason = client._pick_upstream("r1")
    assert reason == "HINT"
    assert url == "http://r1.local:38011"
    assert client.stats["routed_to_hint"] == 1
    assert client.stats["routed_round_robin_fallback"] == 0


def test_pick_upstream_unknown_hint_falls_back_to_round_robin():
    """An unparseable / unknown replica id must spread, never pin."""
    client = _build()
    counts = Counter()
    for _ in range(30):
        url, reason = client._pick_upstream("not-a-real-pod")
        assert reason == "ROUND_ROBIN"
        counts[url] += 1
    assert counts == Counter({u: 10 for _, u in REPLICAS})
    assert client.stats["lookup_unknown_hint"] == 30
    assert client.stats["routed_round_robin_fallback"] == 30


def test_round_robin_pointer_is_deterministic_and_even():
    """CAC-154: 30 picks across 3 replicas → 10/10/10 exact."""
    client = _build()
    counts = Counter()
    for _ in range(30):
        url, reason = client._pick_upstream(None)
        assert reason == "ROUND_ROBIN"
        counts[url] += 1
    assert counts == Counter({u: 10 for _, u in REPLICAS})


# ---- _lookup_route ---------------------------------------------------------


class _FakeReplicaScore:
    def __init__(self, rid: str):
        self.replica_id = rid


class _FakeLookupResponse:
    def __init__(self, reason: str, scores=None):
        self.reason_code = reason
        self.replica_scores = scores or []


async def _async_return(value):
    return value


def _install_fake_stub(client: DumbGatewayClient, returns) -> MagicMock:
    """Wire a fake LookupRoute stub. ``returns`` is a list of responses or
    exception instances to play through, one per call.

    Also assigns the channel field so teardown doesn't NPE.
    """
    stub = MagicMock()
    calls = list(returns)

    async def _lookup(_req):
        result = calls.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    stub.LookupRoute = _lookup
    client._stub = stub
    client._channel = None
    return stub


@pytest.mark.asyncio
async def test_lookup_route_prefix_match_returns_top_replica_id():
    client = _build()
    _install_fake_stub(client, [_FakeLookupResponse(
        reason="PREFIX_MATCH",
        scores=[_FakeReplicaScore("r1"), _FakeReplicaScore("r2")],
    )])
    hint, reason = await client._lookup_route("m", [b"\x00" * 8], [16])
    assert reason == "PREFIX_MATCH"
    assert hint == "r1"
    assert client.stats["lookup_prefix_match"] == 1


@pytest.mark.asyncio
async def test_lookup_route_no_hint_returns_none():
    client = _build()
    _install_fake_stub(client, [_FakeLookupResponse(reason="NO_HINT")])
    hint, reason = await client._lookup_route("m", [b"\x00" * 8], [16])
    assert hint is None
    assert reason == "NO_HINT"
    assert client.stats["lookup_no_hint"] == 1


@pytest.mark.asyncio
async def test_lookup_route_tenant_hot_returns_none():
    """TENANT_HOT must round-robin, not pin — production gateway contract."""
    client = _build()
    _install_fake_stub(client, [_FakeLookupResponse(reason="TENANT_HOT")])
    hint, reason = await client._lookup_route("m", [b"\x00" * 8], [16])
    assert hint is None
    assert reason == "TENANT_HOT"
    assert client.stats["lookup_tenant_hot"] == 1


@pytest.mark.asyncio
async def test_lookup_route_timeout_returns_none_and_increments_stat():
    """Sleep-longer-than-timeout exercises asyncio.wait_for's TimeoutError."""
    client = _build()
    client.lookup_timeout_s = 0.01

    async def _slow(_req):
        await asyncio.sleep(1.0)

    stub = MagicMock()
    stub.LookupRoute = _slow
    client._stub = stub
    client._channel = None

    hint, reason = await client._lookup_route("m", [b"\x00" * 8], [16])
    assert hint is None
    assert reason == "TIMEOUT"
    assert client.stats["lookup_timeout"] == 1


# ---- end-to-end via aiohttp test client ------------------------------------


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
    def __init__(self):
        self.calls: List[str] = []

    def post(self, url, **_kw):
        self.calls.append(url)
        return _FakeResp()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_round_robin_mode_skips_lookup_and_spreads_evenly():
    """30 requests in --routing-mode=round-robin → 10/10/10 across replica URLs,
    LookupRoute never called, tokenizer never loaded."""
    client = _build(routing_mode="round-robin")
    await client.setup()
    assert client._stub is None
    fake = _FakeSession()
    client._client_session = fake  # type: ignore[assignment]

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    app.router.add_post("/v1/chat/completions", client.handle_chat)

    server = TestServer(app)
    http = TestClient(server)
    await http.start_server()
    try:
        for i in range(30):
            r = await http.post(
                "/v1/chat/completions",
                json={"model": "m", "prompt": f"req-{i}"},
            )
            assert r.status == 200
            await r.read()
    finally:
        await http.close()
        await client.teardown()

    forwarded = Counter(url.rsplit("/v1/", 1)[0] for url in fake.calls)
    assert forwarded == Counter({u: 10 for _, u in REPLICAS})
    assert client.stats["lookup_disabled"] == 30
    assert client.stats["routed_round_robin_fallback"] == 30
    assert client.stats["lookup_prefix_match"] == 0


@pytest.mark.asyncio
async def test_lookup_mode_prefix_match_routes_to_hinted_replica():
    """Drive a real-shaped HTTP request through handle_chat with a mocked
    tokenizer + stub. Expect the response to land at r1's URL when the
    server returns PREFIX_MATCH→r1."""
    client = _build(routing_mode="lookup")
    # Make the tokenizer produce enough tokens to fill exactly 2 blocks.
    client.tokenizer = MagicMock()
    client.tokenizer.encode.return_value = list(range(DEFAULT_BLOCK_SIZE * 2))
    client.tokenizer.apply_chat_template.return_value = "rendered"

    _install_fake_stub(client, [_FakeLookupResponse(
        reason="PREFIX_MATCH",
        scores=[_FakeReplicaScore("r1")],
    )] * 10)

    fake = _FakeSession()
    client._client_session = fake  # type: ignore[assignment]

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    app.router.add_post("/v1/chat/completions", client.handle_chat)

    server = TestServer(app)
    http = TestClient(server)
    await http.start_server()
    try:
        for i in range(10):
            r = await http.post(
                "/v1/chat/completions",
                json={"model": "m", "prompt": f"req-{i}"},
            )
            assert r.status == 200
            await r.read()
    finally:
        await http.close()
        await client.teardown()

    forwarded = Counter(url.rsplit("/v1/", 1)[0] for url in fake.calls)
    assert forwarded == Counter({"http://r1.local:38011": 10})
    assert client.stats["lookup_prefix_match"] == 10
    assert client.stats["routed_to_hint"] == 10
    assert client.stats["routed_round_robin_fallback"] == 0
    assert client.stats["match_quality_weak"] == 10  # 2-block chain → "weak"


@pytest.mark.asyncio
async def test_lookup_mode_no_hint_falls_back_to_round_robin():
    """Server returns NO_HINT for every call → 30 requests spread 10/10/10."""
    client = _build(routing_mode="lookup")
    client.tokenizer = MagicMock()
    client.tokenizer.encode.return_value = list(range(DEFAULT_BLOCK_SIZE * 2))
    client.tokenizer.apply_chat_template.return_value = "rendered"

    _install_fake_stub(client, [_FakeLookupResponse(reason="NO_HINT")] * 30)

    fake = _FakeSession()
    client._client_session = fake  # type: ignore[assignment]

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    app.router.add_post("/v1/chat/completions", client.handle_chat)

    server = TestServer(app)
    http = TestClient(server)
    await http.start_server()
    try:
        for i in range(30):
            r = await http.post(
                "/v1/chat/completions",
                json={"model": "m", "prompt": f"req-{i}"},
            )
            assert r.status == 200
            await r.read()
    finally:
        await http.close()
        await client.teardown()

    forwarded = Counter(url.rsplit("/v1/", 1)[0] for url in fake.calls)
    assert forwarded == Counter({u: 10 for _, u in REPLICAS})
    assert client.stats["lookup_no_hint"] == 30
    assert client.stats["routed_round_robin_fallback"] == 30
    assert client.stats["routed_to_hint"] == 0
