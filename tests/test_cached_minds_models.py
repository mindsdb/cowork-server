from __future__ import annotations

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
    )


def test_the_cached_listing_is_read_without_a_fetch_even_when_stale() -> None:
    url = "https://api.mindshub.ai/v1"
    key = (providers.minds_chat_base_url(url), None)
    providers._minds_models_cache[key] = (time.monotonic() - 10_000, _listing(["gpt"]))

    listing = providers.cached_minds_models(url)

    assert listing is not None
    assert listing.efforts["gpt"]["efforts"] == ["low", "max"]


def test_nothing_cached_or_a_cached_failure_reads_as_unknown() -> None:
    url = "https://api.mindshub.ai/v1"
    assert providers.cached_minds_models(url) is None
    assert providers.cached_minds_models("") is None

    providers._minds_models_cache[(providers.minds_chat_base_url(url), None)] = (time.monotonic(), _listing(None))
    assert providers.cached_minds_models(url) is None


def test_the_read_is_tenant_scoped_like_the_fetch() -> None:
    url = "https://api.mindshub.ai/v1"
    providers._minds_models_cache[(providers.minds_chat_base_url(url), "org-1")] = (time.monotonic(), _listing(["gpt"]))

    assert providers.cached_minds_models(url) is None
    assert providers.cached_minds_models(url, tenant_key="org-1") is not None
