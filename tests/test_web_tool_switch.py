"""ANTON_WEB_*_ENABLED in the server's env turns anton's web tools off for every session."""

import pytest

from cowork.common.settings.user_settings import Provider
from cowork.services import providers


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    monkeypatch.delenv("ANTON_WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("ANTON_WEB_FETCH_ENABLED", raising=False)


def test_defaults_are_kept_without_the_switch():
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {}


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "OFF"])
def test_switch_turns_web_search_off(monkeypatch, value):
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", value)
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {"web_search_enabled": False}


def test_switch_turns_both_off(monkeypatch):
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", "false")
    monkeypatch.setenv("ANTON_WEB_FETCH_ENABLED", "false")
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {
        "web_search_enabled": False, "web_fetch_enabled": False,
    }


def test_true_never_turns_on_what_the_provider_turned_off(monkeypatch):
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setattr(providers, "openai_compatible_agent_api", lambda: "responses")
    assert providers.web_tool_kwargs_for(Provider.OPENAI_COMPATIBLE) == {
        "web_search_enabled": False, "web_fetch_enabled": False,
    }
