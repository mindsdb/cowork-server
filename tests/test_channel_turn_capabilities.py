"""A channel turn's session gets only what its binding's switches allow.

Builds the real harness config and a real anton ChatSession from it, then reads
what the model is offered and what a same-round tool call can reach.
"""
from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from anton.core.session import ChatSession
from pydantic import SecretStr

from cowork import code_sandbox
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.harnesses.base import ChannelContext
from cowork.services.conversations import ConversationService

CONNECTOR_TOOLS = {"connect_new_datasource", "lookup_connector", "request_credentials", "label_connection"}
CODE_TOOLS = {"scratchpad", "read_text_file", "read_image", "select_path", "publish_or_preview"}


@pytest.fixture(params=["sandbox off", "sandbox on"])
def sandbox(request, monkeypatch):
    """Run each case with and without the code sandbox.

    The switch is patched where the harness and the cell sandbox read it; the
    cached app settings stay untouched, since a fresh copy loses what the
    session fixtures resolved on it."""
    on = request.param == "sandbox on"
    if on:
        request.getfixturevalue("landlock_kernel")
    monkeypatch.setattr(harness, "code_sandbox_on", lambda: on)
    monkeypatch.setattr(code_sandbox, "code_sandbox_on", lambda: on)
    return request.param


@pytest.fixture
def project_env_and_live_apps(monkeypatch):
    """A project .env with a connection value, and live apps on for every turn,
    so a turn that drops them is seen dropping them."""
    live_app_host = object()
    monkeypatch.setattr(harness, "live_app_host_for_turn", lambda **_kwargs: live_app_host)
    created: list[Path] = []

    def write(project_path: str) -> None:
        env_file = Path(project_path) / ".anton" / ".env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("DS_POSTGRES_PROD__PASSWORD=owner-only\n")
        created.append(env_file)

    yield SimpleNamespace(host=live_app_host, write=write)
    for env_file in created:
        env_file.unlink(missing_ok=True)


@asynccontextmanager
async def _channel_session(monkeypatch, channel: ChannelContext | None, *, before_build=None):
    """The config the harness builds for this turn and a real ChatSession from it."""
    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI,
        planning_model="p", coding_model="c", router_model="r",
        openai_api_key=SecretStr("test-only-key"),
        episodic_memory=False,
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    # Capture the config instead of building through the org-mode guard.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(topic="channel-caps")
        if before_build is not None:
            before_build(conversation.project.path)
        built = await harness.AntonHarness()._build_chat_session(conversation, channel_context=channel)
        config = built.session
        session = ChatSession(config)
        try:
            yield config, session, built.turn_scratch_dir
        finally:
            await session.close()
            if built.turn_scratch_dir:
                shutil.rmtree(built.turn_scratch_dir, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()


def _offered(session: ChatSession) -> set[str]:
    return {tool["name"] for tool in session._build_tools()}


def _registered(session: ChatSession, name: str) -> bool:
    """Whether a tool call by this name would reach a handler; anton raises for an unknown name."""
    try:
        session.tool_registry._find_tool(name)
    except ValueError:
        return False
    return True


def _unlock_connectors(session: ChatSession) -> None:
    """What a recall_skill('connect-datasource') call does in the middle of a round,
    after the round's first build has filled the deferred bundles."""
    session._build_tools()
    session._register_tool_bundle("connect-datasource")


async def test_default_channel_turn_is_plain_chat(monkeypatch, sandbox, project_env_and_live_apps):
    async with _channel_session(
        monkeypatch, ChannelContext(channel_type="telegram"), before_build=project_env_and_live_apps.write,
    ) as (config, session, scratch):
        assert _offered(session) <= {"recall_skill", "recall_history"}
        assert "recall_skill" in _offered(session)

        _unlock_connectors(session)
        assert not any(_registered(session, name) for name in CONNECTOR_TOOLS | CODE_TOOLS | {"memorize"})
        assert _offered(session) <= {"recall_skill", "recall_history"}

        assert config.elicitor is None
        assert getattr(config, "live_app_host", None) is None
        assert (config.web_search_enabled, config.web_fetch_enabled) == (False, False)
        assert config.background_memory is False
        assert not getattr(config, "workspace_env_overlay", None)
        assert config.data_vault.list_connections() == []
        assert Path(config.data_vault._dir).resolve().is_relative_to(scratch.resolve())
        assert Path(config.data_vault._dir).resolve() != (Path.home() / ".anton" / "data_vault").resolve()
        assert {tool.name for tool in config.tools}.isdisjoint(CONNECTOR_TOOLS | {"create_skill_draft"})


async def test_connectors_without_the_scratchpad_stay_off(monkeypatch, sandbox):
    channel = ChannelContext(channel_type="telegram", allow_connectors=True)
    async with _channel_session(monkeypatch, channel) as (config, session, _scratch):
        _unlock_connectors(session)

        assert not any(_registered(session, name) for name in CONNECTOR_TOOLS)
        assert config.data_vault.list_connections() == []
        assert _offered(session) <= {"recall_skill", "recall_history"}


async def test_scratchpad_switch_brings_code_tools_and_the_web_but_not_connectors(monkeypatch, sandbox):
    channel = ChannelContext(channel_type="telegram", allow_scratchpad=True)
    async with _channel_session(monkeypatch, channel) as (config, session, _scratch):
        offered = _offered(session)
        _unlock_connectors(session)

        assert {"scratchpad", "read_text_file"} <= offered
        assert "publish_or_preview" in {tool.name for tool in config.tools}
        assert config.tool_allowlist is None
        assert (config.web_search_enabled, config.web_fetch_enabled) != (False, False)
        assert not any(_registered(session, name) for name in CONNECTOR_TOOLS)
        assert config.data_vault.list_connections() == []
        assert config.elicitor is None
        assert config.background_memory is False


async def test_scratchpad_and_connectors_switches_bring_connector_tools(monkeypatch, sandbox):
    channel = ChannelContext(channel_type="telegram", allow_scratchpad=True, allow_connectors=True)
    async with _channel_session(monkeypatch, channel) as (config, session, _scratch):
        _unlock_connectors(session)

        assert _registered(session, "lookup_connector")
        assert "connect_new_datasource" in {tool.name for tool in config.tools}


async def test_memory_switch_lets_the_turn_write_memory(monkeypatch, sandbox):
    off = ChannelContext(channel_type="telegram", allow_scratchpad=True)
    on = ChannelContext(channel_type="telegram", allow_scratchpad=True, allow_memory_writes=True)
    async with _channel_session(monkeypatch, off) as (config_off, _session, _scratch):
        tools_off = {tool.name for tool in config_off.tools}
        background_off = config_off.background_memory
        mode_off = config_off.cortex.mode
    async with _channel_session(monkeypatch, on) as (config_on, _session, _scratch):
        tools_on = {tool.name for tool in config_on.tools}
        background_on = config_on.background_memory
        mode_on = config_on.cortex.mode
    async with _channel_session(monkeypatch, None) as (config_ui, _session, _scratch):
        mode_ui = config_ui.cortex.mode

    assert "create_skill_draft" not in tools_off and "create_skill_draft" in tools_on
    assert background_off is False
    assert background_on is harness._learns_this_turn(None, True)
    # With the scratchpad on `memorize` is registered; the cortex mode is what stops it writing.
    assert mode_ui != "off"
    assert (mode_off, mode_on) == ("off", mode_ui)


async def test_ui_turn_keeps_every_capability(monkeypatch, project_env_and_live_apps):
    async with _channel_session(
        monkeypatch, None, before_build=project_env_and_live_apps.write,
    ) as (config, session, _scratch):
        assert config.workspace_env_overlay == {"DS_POSTGRES_PROD__PASSWORD": "owner-only"}
        assert config.live_app_host is project_env_and_live_apps.host
        assert config.tool_allowlist is None
        assert "scratchpad" in _offered(session)
        assert {"publish_or_preview", "create_skill_draft", *CONNECTOR_TOOLS} <= {tool.name for tool in config.tools}
        assert config.elicitor is not None


async def test_a_same_round_call_to_an_unlocked_connector_tool_finds_nothing(monkeypatch):
    async with _channel_session(monkeypatch, ChannelContext(channel_type="telegram")) as (_c, session, _s):
        _unlock_connectors(session)

        with pytest.raises(ValueError, match="lookup_connector not found"):
            await session.tool_registry.dispatch_tool(session, "lookup_connector", {"query": "postgres"})
