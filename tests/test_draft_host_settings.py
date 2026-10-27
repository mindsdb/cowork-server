"""Draft-link settings: all six values or no draft links."""
from __future__ import annotations

import json

import pytest

from cowork.common.settings.app_settings import AppSettings, default_draft_env_label

from draft_host_support import DRAFT_HOST, KEYS, draft_env  # noqa: F401  (fixture)


def test_all_six_values_in_org_mode_turn_draft_links_on(draft_env):
    settings = AppSettings()

    assert settings.draft_links_enabled() is True
    assert settings.draft_host_names() == frozenset({DRAFT_HOST})
    assert settings.draft_link_keyring() == KEYS
    assert settings.draft_port == 9011


@pytest.mark.parametrize(
    "name, value",
    [
        ("COWORK_TENANCY_MODE", "local"),
        ("COWORK_DRAFT_HOSTS", ""),
        ("COWORK_DRAFT_ORIGIN", ""),
        ("COWORK_DRAFT_ORIGIN", "draft-staging.artifacts.test"),
        ("COWORK_WEB_ORIGIN", ""),
        ("COWORK_WEB_ORIGIN", "cowork.staging.mindshub.ai"),
        ("COWORK_DRAFT_LINK_KID", "k2"),
        ("COWORK_DRAFT_LINK_KEYS", "not json"),
        ("COWORK_DRAFT_LINK_KEYS", json.dumps(["k1"])),
        ("COWORK_DRAFT_PORT", "0"),
        ("COWORK_DRAFT_PORT", ""),
    ],
)
def test_any_missing_value_leaves_draft_links_off(draft_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert AppSettings().draft_links_enabled() is False


@pytest.mark.parametrize(
    "name, value",
    [
        # A link to a host the listener answers 404 for.
        ("COWORK_DRAFT_ORIGIN", "https://draft.other.test"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test.evil.test"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test/preview"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test?x=1"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test#top"),
        ("COWORK_DRAFT_ORIGIN", "https://user@draft-staging.artifacts.test"),
        ("COWORK_DRAFT_ORIGIN", "https:///draft-staging.artifacts.test"),
        ("COWORK_DRAFT_ORIGIN", "ftp://draft-staging.artifacts.test"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test//"),
        # Sources that would widen or end the CSP's frame-ancestors list.
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai https://evil.test"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai;"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai; script-src *"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai,https://evil.test"),
        ("COWORK_WEB_ORIGIN", "https://*.mindshub.ai"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai/app"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai:port"),
        ("COWORK_WEB_ORIGIN", "https://"),
    ],
)
def test_an_origin_that_is_not_a_bare_origin_leaves_draft_links_off(draft_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert AppSettings().draft_links_enabled() is False


@pytest.mark.parametrize(
    "name, value",
    [
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test/"),
        ("COWORK_DRAFT_ORIGIN", "https://Draft-Staging.Artifacts.Test"),
        ("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test:8443"),
        ("COWORK_DRAFT_ORIGIN", "http://draft-staging.artifacts.test"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai/"),
        ("COWORK_WEB_ORIGIN", "http://localhost:5173"),
        # A mounted secret often ends in a newline; the scheme is case-insensitive.
        ("COWORK_DRAFT_ORIGIN", " https://draft-staging.artifacts.test"),
        ("COWORK_DRAFT_ORIGIN", "HTTPS://draft-staging.artifacts.test"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai\n"),
        ("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai/ "),
    ],
)
def test_a_bare_origin_keeps_draft_links_on(draft_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert AppSettings().draft_links_enabled() is True


def test_the_origins_are_read_as_bare_lower_case_origins(draft_env, monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_ORIGIN", " HTTPS://Draft-Staging.Artifacts.Test/\n")
    monkeypatch.setenv("COWORK_WEB_ORIGIN", "https://Cowork.Staging.Mindshub.AI:8443/")
    settings = AppSettings()

    assert settings.draft_link_origin() == "https://draft-staging.artifacts.test"
    assert settings.draft_frame_origin() == "https://cowork.staging.mindshub.ai:8443"


def test_an_origin_that_is_not_bare_reads_as_none(draft_env, monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_ORIGIN", "https://draft-staging.artifacts.test/preview")
    monkeypatch.setenv("COWORK_WEB_ORIGIN", "https://cowork.staging.mindshub.ai; script-src *")
    settings = AppSettings()

    assert settings.draft_link_origin() == ""
    assert settings.draft_frame_origin() == ""


def test_the_draft_port_defaults_off_and_an_empty_value_is_off(monkeypatch):
    monkeypatch.delenv("COWORK_DRAFT_PORT", raising=False)
    assert AppSettings().draft_port == 0

    # An empty value from a values file must not stop the server at boot.
    monkeypatch.setenv("COWORK_DRAFT_PORT", " ")
    assert AppSettings().draft_port == 0


def test_host_names_are_normalized(monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_HOSTS", " Draft.Mindshub.Co. ,draft-dev.artifacts.test,, ")

    assert AppSettings().draft_host_names() == frozenset(
        {"draft.mindshub.co", "draft-dev.artifacts.test"}
    )


def test_the_keyring_drops_entries_that_are_not_strings(monkeypatch):
    secret = "a" * 32
    monkeypatch.setenv(
        "COWORK_DRAFT_LINK_KEYS", json.dumps({"k1": secret, "k2": 5, "": secret, "k3": ""})
    )

    assert AppSettings().draft_link_keyring() == {"k1": secret}


def test_the_keyring_drops_secrets_shorter_than_32_characters(monkeypatch):
    # The token is readable by the artifact's own script and by anyone who can
    # read an access log, so a short secret could be brute-forced offline.
    monkeypatch.setenv(
        "COWORK_DRAFT_LINK_KEYS", json.dumps({"short": "a" * 31, "exact": "b" * 32, "long": "c" * 64})
    )

    assert AppSettings().draft_link_keyring() == {"exact": "b" * 32, "long": "c" * 64}


def test_a_short_secret_for_the_active_kid_leaves_draft_links_off(draft_env, monkeypatch):
    monkeypatch.setenv("COWORK_DRAFT_LINK_KEYS", json.dumps({"k1": "a" * 31}))

    assert AppSettings().draft_links_enabled() is False


@pytest.mark.parametrize(
    "env, namespace, expected",
    [
        ("staging", "staging", "staging"),
        ("dev", "", "dev"),
        ("prod", "prod", "prod"),
        ("", "", "prod"),
        ("development", "pr-cowork-server-42", "pr-cowork-server-42"),
        ("staging", "pr-7", "pr-7"),
    ],
)
def test_the_token_environment_follows_the_namespace_then_env(monkeypatch, env, namespace, expected):
    monkeypatch.setenv("ENV", env)
    monkeypatch.delenv("NAMESPACE", raising=False)
    monkeypatch.setenv("POD_NAMESPACE", namespace)

    assert default_draft_env_label() == expected


def test_the_namespace_fallback_is_read_too(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("POD_NAMESPACE", raising=False)
    monkeypatch.setenv("NAMESPACE", "pr-auth-110")

    assert default_draft_env_label() == "pr-auth-110"
