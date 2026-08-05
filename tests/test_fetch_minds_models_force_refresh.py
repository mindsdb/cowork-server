"""force_refresh bypasses a cached success but still honors a cached failure.

The Settings model picker fetches on every dropdown open so a wallet top-up
made in an external tab isn't masked by the cached `enabled` map for the full
5-minute TTL. That's what `force_refresh` is for.

The negative cache is deliberately NOT bypassed: `_MINDS_MODELS_FAIL_TTL`
exists so an unreachable MindsHub isn't re-probed on every call, and the picker
opens on demand. Without this, every open during an outage would pay the full
HTTP timeout, and the dropdown would sit unopenable for that long each click.
"""
import asyncio

import cowork.services.providers as providers
from cowork.services.providers import fetch_minds_models

_URL = "https://api.mindshub.ai"


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _client_factory(status_code, body, calls):
    """An httpx.AsyncClient stand-in that counts requests into `calls`."""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            calls.append(url)
            return _Resp(status_code, body)

    return _FakeClient


_OK_BODY = {"data": [{"id": "mindshub_air", "enabled": True, "label": "MindsHub Air"}]}


def test_cached_success_is_reused_without_force_refresh(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    asyncio.run(fetch_minds_models(_URL, "mdb_test"))

    assert len(calls) == 1, "second plain call should have been served from cache"


def test_force_refresh_bypasses_a_cached_success(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    listing = asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True))
    ids, enabled, labels = listing.ids, listing.enabled, listing.labels

    assert len(calls) == 2, "force_refresh must re-hit the network"
    # The refreshed answer is returned, not the cached tuple.
    assert ids == ["mindshub_air"]
    assert enabled == {"mindshub_air": True}
    assert labels == {"mindshub_air": "MindsHub Air"}


def test_force_refresh_result_is_cached_for_the_next_plain_call(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True))
    asyncio.run(fetch_minds_models(_URL, "mdb_test"))

    assert len(calls) == 1, "force_refresh skips the cache read, not the write"


def test_force_refresh_honors_a_cached_failure(monkeypatch):
    # A down/undeployed MindsHub is cached for _MINDS_MODELS_FAIL_TTL. Opening
    # the picker again inside that window must NOT re-probe: otherwise every
    # open pays the full httpx timeout for the length of the outage.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    first = asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    second = asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True))

    assert first == providers._EMPTY_LISTING
    assert second == providers._EMPTY_LISTING
    assert len(calls) == 1, "cached failure must be honored even under force_refresh"


def test_expired_failure_is_retried(monkeypatch):
    # The negative cache is a short coalescing window, not a lockout: once
    # _MINDS_MODELS_FAIL_TTL has passed the next call probes again.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    # Age the cached failure past its TTL.
    ts, val = providers._minds_models_cache[providers.minds_chat_base_url(_URL)]
    providers._minds_models_cache[providers.minds_chat_base_url(_URL)] = (
        ts - providers._MINDS_MODELS_FAIL_TTL - 1,
        val,
    )
    asyncio.run(fetch_minds_models(_URL, "mdb_test"))

    assert len(calls) == 2
