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
from cowork.services.providers import normalize_publish_url, publish_url_for_endpoint


def _settings(*, publish_url: str = "", minds_url: str = "https://api.mindshub.ai/v1"):
    return SimpleNamespace(
        openai_base_url="",
        minds_url=minds_url,
        minds_api_key=SecretStr("mdb_key"),
        publish_url=publish_url,
    )


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


# (ANTON_PUBLISH_URL, publish_url setting, provider endpoint) -> publish URL.
# An empty ANTON_PUBLISH_URL reads as unset.
@pytest.mark.parametrize(
    "env_url, setting_url, minds_url, expected",
    [
        pytest.param("", "", "https://api.mindshub.ai/v1", "https://view.mindshub.ai", id="prod-default"),
        pytest.param("", "https://4nton.ai", "https://api.mindshub.ai/v1", "https://view.mindshub.ai", id="saved-legacy-setting"),
        pytest.param("https://4nton.ai", "", "https://api.mindshub.ai/v1", "https://view.mindshub.ai", id="legacy-env"),
        pytest.param(
            "http://publisher-api:8081", "https://4nton.ai", "https://api.mindshub.ai/v1", "http://publisher-api:8081",
            id="operator-override",
        ),
        pytest.param("", "", "https://api.staging.mindshub.ai/v1", "https://api.staging.mindshub.ai", id="staging-endpoint"),
    ],
)
def test_resolve_publish_endpoint(monkeypatch, env_url, setting_url, minds_url, expected):
    monkeypatch.setenv("ANTON_PUBLISH_URL", env_url)
    settings = _settings(publish_url=setting_url, minds_url=minds_url)
    assert publish._resolve_publish_endpoint(settings) == (expected, "mdb_key")
