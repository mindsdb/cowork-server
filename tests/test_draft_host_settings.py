"""Draft-link settings: all six values or no draft links."""
from __future__ import annotations

import json

import pytest

from cowork.common.settings.app_settings import AppSettings, default_draft_env_label

CONFIGURED = {
    "COWORK_TENANCY_MODE": "org",
    "COWORK_DRAFT_HOSTS": "draft-staging.artifacts.test",
    "COWORK_DRAFT_ORIGIN": "https://draft-staging.artifacts.test",
    "COWORK_DRAFT_LINK_KEYS": json.dumps({"k1": "draft-secret"}),
    "COWORK_DRAFT_LINK_KID": "k1",
    "COWORK_WEB_ORIGIN": "https://cowork.staging.mindshub.ai",
    "COWORK_DRAFT_PORT": "9011",
}


@pytest.fixture
def configured(monkeypatch):
    for name, value in CONFIGURED.items():
        monkeypatch.setenv(name, value)


def test_all_six_values_in_org_mode_turn_draft_links_on(configured):
    settings = AppSettings()

    assert settings.draft_links_enabled() is True
    assert settings.draft_host_names() == frozenset({"draft-staging.artifacts.test"})
    assert settings.draft_link_keyring() == {"k1": "draft-secret"}
    assert settings.draft_port == 9011


@pytest.mark.parametrize(
    "name, value",
    [
        ("COWORK_TENANCY_MODE", "local"),
        ("COWORK_DRAFT_HOSTS", ""),
        ("COWORK_DRAFT_ORIGIN", ""),
        ("COWORK_DRAFT_ORIGIN", "draft-staging.artifacts.test"),
        ("COWORK_WEB_ORIGIN", ""),
        ("COWORK_DRAFT_LINK_KID", "k2"),
        ("COWORK_DRAFT_LINK_KEYS", "not json"),
        ("COWORK_DRAFT_LINK_KEYS", json.dumps(["k1"])),
        ("COWORK_DRAFT_PORT", "0"),
        ("COWORK_DRAFT_PORT", ""),
    ],
)
def test_any_missing_value_leaves_draft_links_off(configured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert AppSettings().draft_links_enabled() is False


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
    monkeypatch.setenv(
        "COWORK_DRAFT_LINK_KEYS", json.dumps({"k1": "a", "k2": 5, "": "b", "k3": ""})
    )

    assert AppSettings().draft_link_keyring() == {"k1": "a"}


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
