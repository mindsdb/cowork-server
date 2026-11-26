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


def test_zdr_turns_both_off_on_mindshub():
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD, zdr_on=True) == {
        "web_search_enabled": False, "web_fetch_enabled": False,
    }


def test_zdr_leaves_another_providers_web_tools_alone():
    # MindsHub drops native web tools under ZDR; a BYOK provider is not affected.
    assert providers.web_tool_kwargs_for(Provider.ANTHROPIC, zdr_on=True) == {}


def test_switch_turns_internet_research_off(monkeypatch):
    # Independent of the anton version the lock pins.
    monkeypatch.setattr(providers, "supported_kwargs", lambda cls, **kwargs: kwargs)
    monkeypatch.setenv("ANTON_INTERNET_RESEARCH_ENABLED", "false")
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {"internet_research_enabled": False}


def test_a_switch_the_installed_anton_cannot_take_is_dropped(monkeypatch):
    """An older anton refuses an unknown ChatSessionConfig keyword, which would fail every turn."""
    import dataclasses

    from anton.core.session import ChatSessionConfig

    known = [f for f in dataclasses.fields(ChatSessionConfig) if f.name != "internet_research_enabled"]
    monkeypatch.setattr(dataclasses, "fields", lambda cls: known if cls is ChatSessionConfig else ())
    monkeypatch.setenv("ANTON_INTERNET_RESEARCH_ENABLED", "false")
    monkeypatch.setenv("ANTON_WEB_SEARCH_ENABLED", "false")
    assert providers.web_tool_kwargs_for(Provider.MINDS_CLOUD) == {"web_search_enabled": False}


@pytest.mark.parametrize(("planning", "read", "zdr_on"), [
    ("minds_cloud", True, True),
    ("minds_cloud", False, False),
    ("minds_cloud", RuntimeError("down"), True),
    ("anthropic", True, False),
])
async def test_the_probe_reads_zdr_only_when_planning_on_mindshub(monkeypatch, planning, read, zdr_on):
    from cowork.common.settings.user_settings import UserSettings
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.handlers.probe import ProbeHandler
    from cowork.services import mindshub_zdr

    asked = []

    async def zdr_on_for(**kwargs):
        asked.append(kwargs["org_bearer"])
        if isinstance(read, Exception):
            raise read
        return read

    monkeypatch.setattr(mindshub_zdr, "zdr_on_for", zdr_on_for)
    settings = UserSettings.model_validate({"planning_provider": planning, "minds_cloud_api_key": "k"})

    assert await ProbeHandler(scope=LOCAL_SCOPE, mindshub_bearer="b")._zdr_on(settings) is zdr_on
    assert asked == ([] if planning == "anthropic" else ["b"])
