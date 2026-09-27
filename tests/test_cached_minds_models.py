from __future__ import annotations

import asyncio
import time

import pytest

from cowork.services import providers


@pytest.fixture(autouse=True)
def clear_cache():
    providers._minds_models_cache.clear()
    yield
    providers._minds_models_cache.clear()


def _listing(ids):
    return providers.MindsModelListing(
        ids=ids, efforts={"gpt": {"efforts": ["low", "max"], "default": "low"}} if ids else {},
        enabled={}, labels={}, providers={}, families={}, role_defaults={},
        disabled_reasons={},
    )


def _key(url, *, api_key="mdb_test", tenant_key=None, user_id=None):
    return providers._listing_cache_key(
        base_url=providers.minds_chat_base_url(url), api_key=api_key,
        tenant_key=tenant_key, user_id=user_id,
    )


def test_the_cached_listing_is_read_without_a_fetch_even_when_stale() -> None:
    url = "https://api.mindshub.ai/v1"
    providers._minds_models_cache[_key(url)] = (time.monotonic() - 10_000, _listing(["gpt"]))

    listing = providers.cached_minds_models(url, api_key="mdb_test")

    assert listing is not None
    assert listing.efforts["gpt"]["efforts"] == ["low", "max"]


def test_nothing_cached_or_a_cached_failure_reads_as_unknown() -> None:
    url = "https://api.mindshub.ai/v1"
    assert providers.cached_minds_models(url, api_key="mdb_test") is None
    assert providers.cached_minds_models("", api_key="mdb_test") is None

    providers._minds_models_cache[_key(url)] = (time.monotonic(), _listing(None))
    assert providers.cached_minds_models(url, api_key="mdb_test") is None


def test_the_read_is_tenant_scoped_like_the_fetch() -> None:
    url = "https://api.mindshub.ai/v1"
    providers._minds_models_cache[_key(url, tenant_key="org-1")] = (time.monotonic(), _listing(["gpt"]))

    assert providers.cached_minds_models(url, api_key="mdb_test") is None
    assert providers.cached_minds_models(url, tenant_key="org-1") is not None


def test_the_read_is_caller_scoped_like_the_org_catalog_fetch() -> None:
    url = "https://api.mindshub.ai/v1"
    key = _key(url, tenant_key="org-1", user_id="user-1")
    providers._minds_models_cache[key] = (time.monotonic(), _listing(["gpt"]))

    assert providers.cached_minds_models(url, tenant_key="org-1") is None
    assert providers.cached_minds_models(url, tenant_key="org-1", user_id="user-2") is None
    assert providers.cached_minds_models(url, tenant_key="org-1", user_id="user-1") is not None


# ── Bounded and credential-keyed ────────────────────────────────────────────
#
# The cache holds one listing per (base, org, member) in org mode, and was only
# ever overwritten, so it grew with every member a pod served. On a desktop it
# had no credential in the key, so an account switch in the running process was
# served the previous account's rows for the TTL.

_URL = "https://api.mindshub.ai/v1"


def _serve(monkeypatch, calls: list[str]) -> None:
    """Answer every /models fetch; ``sonnet`` is restricted for a key naming account A."""

    async def fake_get(url, headers=None, **kw):
        bearer = headers["Authorization"]
        calls.append(bearer)
        restricted = "account-a" in bearer

        class R:
            status_code = 200

            def json(self):
                sonnet = {"id": "sonnet", "enabled": not restricted}
                if restricted:
                    sonnet["disabled_reason"] = "model_restricted"
                return {"data": [{"id": "mindshub_air", "enabled": True}, sonnet]}

        return R()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        get = staticmethod(fake_get)

    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda *a, **k: FakeClient())


def _fetch(api_key: str):
    return asyncio.run(providers.fetch_minds_models(_URL, api_key))


def test_two_desktop_credentials_do_not_share_a_listing(monkeypatch) -> None:
    calls: list[str] = []
    _serve(monkeypatch, calls)
    account_a = "mdb_account-a_raw_secret_value"
    account_b = "mdb_account-b_raw_secret_value"

    first = _fetch(account_a)
    second = _fetch(account_b)

    assert first.disabled_reasons == {"sonnet": "model_restricted"}
    assert second.enabled["sonnet"] is True
    assert second.disabled_reasons == {}
    assert len(calls) == 2, "the second account must be fetched, not served the first one's entry"
    assert providers.cached_minds_models(_URL, api_key=account_a) == first
    assert providers.cached_minds_models(_URL, api_key=account_b) == second
    for key in providers._minds_models_cache:
        assert account_a not in repr(key) and account_b not in repr(key)
        assert len(key.credential) == 16


def test_an_org_entry_is_keyed_without_the_credential() -> None:
    # The org catalog's bearer rotates; the member's user id names the entry.
    key = _key(_URL, api_key="rotating-bearer", tenant_key="org-1", user_id="u-1")
    assert key.credential is None


def test_expired_entries_are_dropped_on_the_next_write(monkeypatch) -> None:
    _serve(monkeypatch, [])
    expired = _key(_URL, api_key="departed")
    fresh = _key(_URL, api_key="still-here")
    past_every_ttl = time.monotonic() - providers._MINDS_MODELS_MAX_TTL_S - 1
    providers._minds_models_cache[expired] = (past_every_ttl, _listing(["gpt"]))
    providers._minds_models_cache[fresh] = (time.monotonic(), _listing(["gpt"]))

    _fetch("newcomer")

    assert expired not in providers._minds_models_cache
    assert fresh in providers._minds_models_cache
    assert _key(_URL, api_key="newcomer") in providers._minds_models_cache


def test_the_cache_holds_at_most_the_cap_and_drops_the_least_recently_used(monkeypatch) -> None:
    calls: list[str] = []
    _serve(monkeypatch, calls)
    monkeypatch.setattr(providers, "_MINDS_MODELS_CACHE_MAX", 3)

    for api_key in ("first", "second", "third"):
        _fetch(api_key)
    _fetch("first")  # a cache hit makes it the most recently used
    _fetch("fourth")

    assert len(providers._minds_models_cache) == 3
    assert _key(_URL, api_key="second") not in providers._minds_models_cache
    for api_key in ("first", "third", "fourth"):
        assert _key(_URL, api_key=api_key) in providers._minds_models_cache
    assert len(calls) == 4, "the repeat of the first key was served from the cache"
