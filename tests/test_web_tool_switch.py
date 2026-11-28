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


@pytest.mark.parametrize(("planning", "coding", "read", "known_on"), [
    ("minds_cloud", "minds_cloud", "on", True),
    ("minds_cloud", "minds_cloud", "off", False),
    # A failed or unreadable answer keeps web search: only a known ZDR turns it off.
    ("minds_cloud", "minds_cloud", "unknown", False),
    ("minds_cloud", "minds_cloud", RuntimeError("down"), False),
    # The scratchpad's web_search() runs on the coding model, so MindsHub coding is read too.
    ("anthropic", "minds_cloud", "on", True),
    ("anthropic", "anthropic", "on", False),
])
async def test_the_probe_reads_zdr_only_when_a_role_is_on_mindshub(monkeypatch, planning, coding, read, known_on):
    from types import SimpleNamespace

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.handlers.probe import ProbeHandler
    from cowork.services import mindshub_zdr

    asked = []

    async def zdr_state_for(**kwargs):
        asked.append(kwargs["org_bearer"])
        if isinstance(read, Exception):
            raise read
        return mindshub_zdr.ZdrState(read)

    monkeypatch.setattr(mindshub_zdr, "zdr_state_for", zdr_state_for)
    settings = SimpleNamespace(resolved_planning_provider=Provider(planning), resolved_coding_provider=Provider(coding))

    assert await ProbeHandler(scope=LOCAL_SCOPE, mindshub_bearer="b")._zdr_known_on(settings) is known_on
    assert asked == ([] if "minds_cloud" not in (planning, coding) else ["b"])


async def _probe_config(monkeypatch, *, scratchpad_web_search_off: bool) -> dict:
    """The probe's ChatSessionConfig keywords; it stops where its session would be built."""
    import anton.core.session as session_module

    from cowork.services.connectors import probe as probe_module

    calls: list[dict] = []
    real = session_module.ChatSessionConfig

    class RecordingChatSessionConfig(real):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            super().__init__(**kwargs)

    def build(_config):
        raise RuntimeError("stop before the session starts")

    monkeypatch.setattr(session_module, "ChatSessionConfig", RecordingChatSessionConfig)
    monkeypatch.setattr(probe_module, "build_chat_session", build)
    probe = probe_module.CredentialProbe(
        engine="postgres", credentials={}, llm_client=object(), workspace=None,
        scratchpad_web_search_off=scratchpad_web_search_off,
    )
    [event async for event in probe.run()]
    return calls[-1]


@pytest.mark.parametrize("off", [True, False])
async def test_the_probe_turns_off_the_scratchpads_web_search_when_asked(monkeypatch, off):
    import anton.core.settings as anton_settings

    class CoreSettings:
        # Stands in for an anton that has the switch.
        scratchpad_web_search = "on"

    monkeypatch.setattr(anton_settings, "CoreSettings", CoreSettings)
    config = await _probe_config(monkeypatch, scratchpad_web_search_off=off)

    if off:
        assert config["settings"].scratchpad_web_search == "off"
    else:
        assert "settings" not in config


async def test_a_probe_on_an_anton_without_the_switch_keeps_its_settings(monkeypatch):
    import anton.core.settings as anton_settings

    class CoreSettings:
        pass

    monkeypatch.setattr(anton_settings, "CoreSettings", CoreSettings)
    config = await _probe_config(monkeypatch, scratchpad_web_search_off=True)

    assert "settings" not in config
