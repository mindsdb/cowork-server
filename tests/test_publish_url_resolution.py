"""Publishing resolves to the prod publish host, not to the retired 4nton.ai.

view.mindshub.ai serves the same publishing API as 4nton.ai, and 4nton.ai is
being retired for publishing. A prod (or unknown) provider endpoint falls back
to view.mindshub.ai, and a 4nton.ai URL from the environment or a saved
setting is replaced by it. Non-prod endpoints keep their own api host.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from cowork.services import publish
from cowork.services.providers import (
    LEGACY_PUBLISH_HOST,
    PUBLISH_FAILSAFE_URL,
    is_mindshub_publish_url,
    normalize_publish_url,
    publish_url_for_endpoint,
)


def _settings(*, publish_url: str = "", minds_url: str = "https://api.mindshub.ai/v1"):
    return SimpleNamespace(
        openai_base_url="",
        minds_url=minds_url,
        minds_api_key=SecretStr("mdb_key"),
        publish_url=publish_url,
    )


def test_prod_publish_host_is_the_view_host():
    assert PUBLISH_FAILSAFE_URL == "https://view.mindshub.ai"
    assert LEGACY_PUBLISH_HOST == "4nton.ai"


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("https://api.mindshub.ai/v1", "https://view.mindshub.ai"),
        ("https://mdb.ai", "https://view.mindshub.ai"),
        ("", "https://view.mindshub.ai"),
        (None, "https://view.mindshub.ai"),
        ("https://api.staging.mindshub.ai/v1", "https://api.staging.mindshub.ai"),
        ("https://api.dev.mindshub.ai/v1", "https://api.dev.mindshub.ai"),
    ],
)
def test_publish_url_for_endpoint(endpoint, expected):
    assert publish_url_for_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "url",
    ["https://4nton.ai", "https://4nton.ai/", "http://4nton.ai", "https://4NTON.AI", "https://4nton.ai."],
)
def test_the_legacy_host_is_replaced(url):
    assert normalize_publish_url(url) == "https://view.mindshub.ai"


@pytest.mark.parametrize(
    "url",
    [
        "https://view.mindshub.ai",
        "https://api.staging.mindshub.ai",
        "http://publisher-api:8081",
        # Lookalikes and subdomains are not the publish host.
        "https://4nton.ai.customer.example",
        "https://cw-abc.4nton.ai",
        "",
        # `.hostname` raises on an unbalanced bracket.
        "http://[publisher",
    ],
)
def test_other_urls_are_kept(url):
    assert normalize_publish_url(url) == url


def test_prod_endpoint_without_settings_publishes_on_the_view_host(monkeypatch):
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)
    assert publish._resolve_publish_endpoint(_settings()) == ("https://view.mindshub.ai", "mdb_key")


def test_a_saved_legacy_setting_is_replaced(monkeypatch):
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)
    url, _key = publish._resolve_publish_endpoint(_settings(publish_url="https://4nton.ai"))
    assert url == "https://view.mindshub.ai"


def test_a_legacy_env_override_is_replaced(monkeypatch):
    monkeypatch.setenv("ANTON_PUBLISH_URL", "https://4nton.ai")
    url, _key = publish._resolve_publish_endpoint(_settings())
    assert url == "https://view.mindshub.ai"


def test_an_operator_override_is_kept(monkeypatch):
    monkeypatch.setenv("ANTON_PUBLISH_URL", "http://publisher-api:8081")
    url, _key = publish._resolve_publish_endpoint(_settings(publish_url="https://4nton.ai"))
    assert url == "http://publisher-api:8081"


def test_a_staging_endpoint_keeps_its_api_host(monkeypatch):
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)
    url, _key = publish._resolve_publish_endpoint(_settings(minds_url="https://api.staging.mindshub.ai/v1"))
    assert url == "https://api.staging.mindshub.ai"


@pytest.mark.parametrize("url", ["https://view.mindshub.ai", "https://4nton.ai", "https://4nton.ai."])
def test_both_publish_hosts_count_as_mindshub(url):
    assert is_mindshub_publish_url(url) is True


@pytest.mark.parametrize(
    "url", ["http://publisher-api:8081", "https://cw-abc.4nton.ai", "https://4nton.ai.customer.example", "", None]
)
def test_other_publishers_do_not_count_as_mindshub(url):
    assert is_mindshub_publish_url(url) is False
