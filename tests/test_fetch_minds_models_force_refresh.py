"""The listing cache's TTLs and how force_refresh bypasses them.

A top-up must reach the picker within seconds (ENG-4147): a success is fresh for
15s for MindsHub (300s for a custom endpoint); force_refresh skips it, and
MindsHub callers also retry a failure after 5s.
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

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))

    assert len(calls) == 1, "second plain call should have been served from cache"


def test_force_refresh_bypasses_a_cached_success(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    listing = asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True, mindshub=True))
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

    asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True, mindshub=True))
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))

    assert len(calls) == 1, "force_refresh skips the cache read, not the write"


def _age(seconds):
    """Backdate the single cache entry by ``seconds``."""
    (key, entry), = providers._minds_models_cache.items()
    providers._minds_models_cache[key] = tuple(t - seconds if i != 1 else t for i, t in enumerate(entry))


def test_success_expires_after_its_short_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    _age(providers._MINDS_MODELS_TTL - 1)
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    assert len(calls) == 1, "a success is reused inside its TTL"

    _age(2)
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    assert len(calls) == 2, "a success past its TTL is fetched again"
    assert providers._MINDS_MODELS_TTL == 15.0


def test_plain_read_honors_a_cached_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    _age(providers._MINDS_MODELS_FAIL_TTL - 1)
    second = asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))

    assert second == providers._empty_listing()
    assert len(calls) == 1, "a plain read keeps the failure for its full TTL"


def test_force_refresh_retries_a_cached_failure_past_the_retry_window(monkeypatch):
    # A failure right after a top-up must not hide the unlock for 30s.
    calls = []
    factory = _client_factory(503, {}, calls)
    monkeypatch.setattr(providers.httpx, "AsyncClient", factory)
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    _age(providers._MINDS_MODELS_FORCED_RETRY_S + 1)
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    listing = asyncio.run(
        fetch_minds_models(_URL, "mdb_test", force_refresh=True, mindshub=True)
    )

    assert len(calls) == 2
    assert listing.enabled == {"mindshub_air": True}


def test_custom_endpoint_force_refresh_honors_a_cached_failure(monkeypatch):
    # The BYO openai-compatible overlay: an outage must not cost every refresh a fetch.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    _age(providers._MINDS_MODELS_FORCED_RETRY_S + 1)
    second = asyncio.run(fetch_minds_models(_URL, "mdb_test", force_refresh=True))

    assert second == providers._empty_listing()
    assert len(calls) == 1


def test_force_refresh_honors_a_fresh_cached_failure(monkeypatch):
    # Inside the retry window an outage is not re-probed on every picker open.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    second = asyncio.run(
        fetch_minds_models(_URL, "mdb_test", force_refresh=True, mindshub=True)
    )

    assert second == providers._empty_listing()
    assert len(calls) == 1


def test_a_failure_keeps_the_last_success_for_cached_reads(monkeypatch):
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    _age(providers._MINDS_MODELS_TTL + 1)

    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    failed = asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    again = asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))

    assert failed == providers._empty_listing()
    assert again == providers._empty_listing(), "inside the fail TTL the failure is served"
    assert len(calls) == 2
    assert providers.cached_minds_models(_URL, api_key="mdb_test").ids == ["mindshub_air"]


def test_expired_failure_is_retried(monkeypatch):
    # Once _MINDS_MODELS_FAIL_TTL has passed, a plain call probes again.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    # Age the cached failure past its TTL.
    key = providers._listing_cache_key(
        base_url=providers.minds_chat_base_url(_URL), api_key="mdb_test", tenant_key=None, user_id=None
    )
    ts, val = providers._minds_models_cache[key]
    providers._minds_models_cache[key] = (
        ts - providers._MINDS_MODELS_FAIL_TTL - 1,
        val,
    )
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))

    assert len(calls) == 2


def test_a_late_failure_is_aged_by_its_own_time_not_its_old_success(monkeypatch):
    # A failure near the end of the retain window must still be served for the fail TTL.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    _age(providers._MINDS_MODELS_RETAIN_S - 1)

    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(503, {}, calls))
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    key = next(iter(providers._minds_models_cache))
    success_at, listing, failed_at = providers._minds_models_cache[key]
    providers._minds_models_cache[key] = (success_at - 2, listing, failed_at)
    providers._store_listing(("other",), providers._empty_listing())  # triggers a sweep

    assert key in providers._minds_models_cache, "the fresh failure must survive the sweep"
    asyncio.run(fetch_minds_models(_URL, "mdb_test", mindshub=True))
    assert len(calls) == 2, "the failure is served inside its TTL, not refetched"


def test_a_custom_endpoint_success_keeps_the_long_ttl(monkeypatch):
    # A BYO endpoint isn't refetched on every mount and settings save.
    calls = []
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_factory(200, _OK_BODY, calls))
    providers._minds_models_cache.clear()

    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    _age(providers._MINDS_MODELS_TTL + 1)
    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    assert len(calls) == 1

    _age(providers._CUSTOM_MODELS_TTL)
    asyncio.run(fetch_minds_models(_URL, "mdb_test"))
    assert len(calls) == 2


def test_the_org_catalog_fetches_as_mindshub(monkeypatch):
    seen = {}

    async def fake_fetch(url, key, **kw):
        seen.update(kw)
        return providers._empty_listing()

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    asyncio.run(providers.fetch_org_model_catalog(
        org_id="o1", user_id="u1", bearer_token="jwt", refresh=True
    ))

    assert seen["mindshub"] is True and seen["force_refresh"] is True
