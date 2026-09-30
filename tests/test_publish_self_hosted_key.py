"""A self-hosted install publishes with its own key from the environment.

`ANTON_PUBLISH_API_KEY` lets a desktop with no MindsHub key still publish
artifacts, against an explicit publish URL only — never against a host derived
from the active provider, which could be a MindsDB one. `_resolve_publish_endpoint`
is the single function every publish entry point (publish, update, unpublish,
delete, list, versions, activate) goes through, so it is exercised directly here.
"""
from __future__ import annotations

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import UserSettings
from cowork.services.publish import _resolve_publish_endpoint, desktop_publish_credential

SELF_HOSTED_URL = "http://publisher-api:8081"


def _settings(**overrides):
    base = dict(minds_api_key=None, minds_url="https://api.mindshub.ai")
    base.update(overrides)
    return UserSettings.model_validate(base)


@pytest.fixture(autouse=True)
def _reset_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_env_key_and_url_publish_against_the_explicit_url(monkeypatch):
    monkeypatch.setenv("ANTON_PUBLISH_API_KEY", "self-hosted-secret")
    monkeypatch.setenv("ANTON_PUBLISH_URL", SELF_HOSTED_URL)

    url, key = _resolve_publish_endpoint(_settings())

    assert url == SELF_HOSTED_URL
    assert key == "self-hosted-secret"


def test_env_key_falls_back_to_the_stored_publish_url_setting(monkeypatch):
    monkeypatch.setenv("ANTON_PUBLISH_API_KEY", "self-hosted-secret")
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)

    url, key = _resolve_publish_endpoint(_settings(publish_url=SELF_HOSTED_URL))

    assert url == SELF_HOSTED_URL
    assert key == "self-hosted-secret"


def test_env_key_without_any_explicit_url_is_refused(monkeypatch):
    """No fallback to a host derived from the active provider — that host can
    be a MindsDB one, and this key must never reach it."""
    monkeypatch.setenv("ANTON_PUBLISH_API_KEY", "self-hosted-secret")
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)

    url, key = _resolve_publish_endpoint(_settings(minds_url="https://api.mindshub.ai"))

    assert (url, key) == ("", "")
    with pytest.raises(ValueError):
        desktop_publish_credential()


def test_unset_env_key_leaves_resolution_unchanged(monkeypatch):
    monkeypatch.delenv("ANTON_PUBLISH_API_KEY", raising=False)
    monkeypatch.delenv("ANTON_PUBLISH_URL", raising=False)

    url, key = _resolve_publish_endpoint(_settings(minds_api_key="mdb_test"))

    assert key == "mdb_test"
    assert url  # derived from minds_url, as today


def test_org_mode_ignores_the_env_key_and_resolves_as_before(monkeypatch):
    """A cluster env var must not reach every tenant's publish flow."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    monkeypatch.setenv("ANTON_PUBLISH_API_KEY", "self-hosted-secret")
    monkeypatch.setenv("ANTON_PUBLISH_URL", SELF_HOSTED_URL)

    url, key = _resolve_publish_endpoint(_settings(minds_api_key="mdb_org_key"))

    # ANTON_PUBLISH_URL is still an operator override read unconditionally
    # today (pre-existing, unchanged) — only the new env key is gated.
    assert url == SELF_HOSTED_URL
    assert key == "mdb_org_key"
