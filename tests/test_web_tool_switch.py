"""ANTON_WEB_*_ENABLED and ANTON_INTERNET_RESEARCH_ENABLED in the server's env turn
anton's web tools or internet research off for every session."""

import pytest

from cowork.common.settings.user_settings import Provider
from cowork.services import providers


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    monkeypatch.delenv("ANTON_WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("ANTON_WEB_FETCH_ENABLED", raising=False)
    monkeypatch.delenv("ANTON_INTERNET_RESEARCH_ENABLED", raising=False)


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


def test_switch_turns_internet_research_off(monkeypatch):
    # Independent of the anton version the lock pins.
    monkeypatch.setattr(providers, "_anton_config_takes", lambda field: True)
    monkeypatch.setenv("ANTON_INTERNET_RESEARCH_ENABLED", "false")
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {"internet_research_enabled": False}


def test_a_switch_the_installed_anton_cannot_take_is_dropped(monkeypatch):
    """An older anton refuses an unknown ChatSessionConfig keyword, which would fail every turn."""
    import dataclasses

    from anton.core.session import ChatSessionConfig

    known = [f for f in dataclasses.fields(ChatSessionConfig) if f.name != "internet_research_enabled"]
    monkeypatch.setattr(dataclasses, "fields", lambda cls: known if cls is ChatSessionConfig else ())
    monkeypatch.setattr(providers, "_warned_config_fields", set())
    monkeypatch.setenv("ANTON_INTERNET_RESEARCH_ENABLED", "false")
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", "false")
    # Recorded at the call, not through caplog: the suite's logging setup can stop propagation.
    warnings: list[str] = []
    monkeypatch.setattr(providers.logger, "warning", lambda msg, *args: warnings.append(msg % args))
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {"web_search_enabled": False}
    assert len(warnings) == 1 and "internet_research_enabled" in warnings[0]
