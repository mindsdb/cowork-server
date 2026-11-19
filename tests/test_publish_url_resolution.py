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
    _publish_token(monkeypatch, None)
    settings = _settings(publish_url=setting_url, minds_url=minds_url)
    assert publish._resolve_publish_endpoint(settings) == (expected, "mdb_key")


MINDSHUB = "https://api.mindshub.ai/v1"
ROUTED = "http://model-route:8000/v1"
# The key each provider sends without COWORK_PUBLISH_TOKEN: a MindsHub endpoint
# sends its own model key, any other endpoint the MindsHub slot's key.
KEY_WITHOUT_TOKEN = {MINDSHUB: "mindshub_model_key", ROUTED: "mdb_key"}


def _provider_settings(openai_base_url: str):
    return SimpleNamespace(
        openai_base_url=openai_base_url,
        openai_compatible_api_key=SecretStr("mindshub_model_key"),
        openai_api_key=None,
        minds_url="https://minds.invalid",
        minds_api_key=SecretStr("mdb_key"),
        publish_url="",
    )


def _publish_token(monkeypatch, token: SecretStr | None) -> None:
    """AppSettings.publish_token, which COWORK_PUBLISH_TOKEN sets; get_app_settings is cached."""
    monkeypatch.setattr(publish, "get_app_settings", lambda: SimpleNamespace(publish_token=token))


@pytest.mark.parametrize("openai_base_url", [MINDSHUB, ROUTED], ids=["mindshub", "routed"])
def test_the_publish_token_replaces_the_provider_key(monkeypatch, openai_base_url):
    monkeypatch.setenv("ANTON_PUBLISH_URL", "http://publisher-api:8081")
    _publish_token(monkeypatch, SecretStr(" mdb_publish_token\n"))

    assert publish._resolve_publish_endpoint(_provider_settings(openai_base_url)) == (
        "http://publisher-api:8081", "mdb_publish_token",
    )


@pytest.mark.parametrize("token", [None, "", "  "], ids=["unset", "empty", "blank"])
@pytest.mark.parametrize("openai_base_url", [MINDSHUB, ROUTED], ids=["mindshub", "routed"])
def test_without_a_publish_token_the_provider_key_is_sent(monkeypatch, openai_base_url, token):
    monkeypatch.setenv("ANTON_PUBLISH_URL", "http://publisher-api:8081")
    _publish_token(monkeypatch, None if token is None else SecretStr(token))

    assert publish._resolve_publish_endpoint(_provider_settings(openai_base_url)) == (
        "http://publisher-api:8081", KEY_WITHOUT_TOKEN[openai_base_url],
    )
