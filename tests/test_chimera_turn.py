"""Chimera on a turn (cowork.services.chimera) and the harness that runs it.

The pseudo-model `chimera` names a pair only MindsHub serves; chimera_enabled
turns it on for every turn; openai-compatible ignores both. On a Chimera turn
the harness hands anton a ChimeraConfig and the sidekick the last turn left,
and saves the sidekick the turn leaves. Off, or on an anton without Chimera,
the turn is built exactly as before.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import _chimera_anton
from cowork.common.settings.app_settings import CHIMERA_MODEL_ID, get_app_settings
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.services import chimera
from cowork.services.conversations import ConversationService
from cowork.services.sidekick_contexts import SidekickContextService


def _minds(**kw) -> UserSettings:
    return UserSettings(**{
        "_env_file": None,
        "planning_provider": Provider.MINDS_CLOUD,
        "coding_provider": Provider.MINDS_CLOUD,
        "router_provider": Provider.MINDS_CLOUD,
        "sidekick_provider": Provider.MINDS_CLOUD,
        "episodic_memory": False,
        "minds_api_key": SecretStr("mdb-key"),
        "minds_url": "https://api.mindshub.ai",
        **kw,
    })


def _openai_compatible(**kw) -> UserSettings:
    return UserSettings(**{
        "_env_file": None,
        "planning_provider": Provider.OPENAI_COMPATIBLE,
        "coding_provider": Provider.OPENAI_COMPATIBLE,
        "router_provider": Provider.OPENAI_COMPATIBLE,
        "sidekick_provider": Provider.OPENAI_COMPATIBLE,
        "planning_model": "local", "coding_model": "local",
        "openai_compatible_api_key": SecretStr("sk-compat"),
        "openai_base_url": "http://localhost:8000/v1",
        "minds_api_key": None, "anthropic_api_key": None, "openai_api_key": None,
        **kw,
    })


# ── chimera_turn ─────────────────────────────────────────────────────

def test_the_pseudo_model_runs_its_pair_on_mindshub():
    turn = chimera.chimera_turn(CHIMERA_MODEL_ID, _minds())
    assert turn == chimera.ChimeraTurn(True, "opus", "gpt-luna-6")


def test_the_pseudo_model_follows_the_configured_pair():
    settings = _minds(chimera_main_model="sonnet", chimera_sidekick_model="kimi")
    assert chimera.chimera_turn(CHIMERA_MODEL_ID, settings) == chimera.ChimeraTurn(True, "sonnet", "kimi")


def test_the_pseudo_model_is_ignored_on_openai_compatible():
    # Never a model id the endpoint is sent: the account's models run instead.
    assert chimera.chimera_turn(CHIMERA_MODEL_ID, _openai_compatible()) == chimera.ChimeraTurn(False, None)


def test_the_pseudo_model_is_ignored_on_a_direct_provider():
    settings = UserSettings(
        _env_file=None, planning_provider=Provider.ANTHROPIC, anthropic_api_key=SecretStr("sk-ant"),
    )
    assert chimera.chimera_turn(CHIMERA_MODEL_ID, settings) == chimera.ChimeraTurn(False, None)


@pytest.mark.parametrize("model", [None, "sonnet"])
def test_chimera_enabled_keeps_the_turns_own_planning_pick(model):
    turn = chimera.chimera_turn(model, _minds(chimera_enabled=True))
    assert turn == chimera.ChimeraTurn(True, model, None)


def test_chimera_enabled_is_ignored_on_openai_compatible():
    assert chimera.chimera_turn(None, _openai_compatible(chimera_enabled=True)).on is False


def test_chimera_enabled_is_ignored_with_an_openai_compatible_sidekick():
    settings = _minds(
        chimera_enabled=True,
        sidekick_provider=Provider.OPENAI_COMPATIBLE,
        openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url="http://localhost:8000/v1",
    )
    assert chimera.chimera_turn(None, settings).on is False


@pytest.mark.parametrize("model", [None, "sonnet"])
def test_off_by_default_the_pick_passes_through(model):
    assert chimera.chimera_turn(model, _minds()) == chimera.ChimeraTurn(False, model)


def test_off_by_default_reads_no_provider():
    # The common path touches nothing but the switch, so a settings stand-in
    # without resolvers still builds a turn.
    assert chimera.chimera_turn("sonnet", SimpleNamespace(chimera_enabled=False)).model == "sonnet"


def test_a_hosted_turn_runs_the_pairs_main_model_alone():
    assert chimera.remote_turn_model(CHIMERA_MODEL_ID, _minds()) == "opus"
    assert chimera.remote_turn_model("sonnet", _minds()) == "sonnet"
    assert chimera.remote_turn_model(None, _minds()) is None


# ── What the installed anton can do ──────────────────────────────────

def test_an_anton_without_chimera_offers_none(monkeypatch):
    # uv.lock pins an anton that predates Chimera; nothing is offered or passed.
    _chimera_anton.hide(monkeypatch)
    assert chimera.anton_supports_chimera() is False
    assert chimera.build_chimera_config() is None


def test_a_chimera_anton_is_recognised(monkeypatch):
    _chimera_anton.install(monkeypatch)
    assert chimera.anton_supports_chimera() is True


@pytest.mark.parametrize("missing", ["config", "client", "session"])
def test_every_piece_is_required(monkeypatch, missing):
    _chimera_anton.install(monkeypatch, **{missing: False})
    assert chimera.anton_supports_chimera() is False


def test_the_config_is_enabled_and_takes_antons_limits(monkeypatch):
    _chimera_anton.install(monkeypatch)
    anton_settings = SimpleNamespace(chimera_max_rounds=12, chimera_jev_every=0, chimera_jev_threshold=0.9)
    config = chimera.build_chimera_config(anton_settings)
    assert config == _chimera_anton.ChimeraConfig(enabled=True, max_rounds=12, jev_every=0, jev_threshold=0.9)
    # Without the knobs, the contract's defaults.
    assert chimera.build_chimera_config(SimpleNamespace()) == _chimera_anton.ChimeraConfig(enabled=True)


def test_the_planning_cache_lifetime():
    assert chimera.planning_cache_ttl(SimpleNamespace(), chimera=False) is None
    assert chimera.planning_cache_ttl(SimpleNamespace(), chimera=True) == "1h"
    assert chimera.planning_cache_ttl(SimpleNamespace(main_cache_ttl="5m"), chimera=True) == "5m"
    assert chimera.planning_cache_ttl(SimpleNamespace(main_cache_ttl="1h"), chimera=False) == "1h"


# ── chimera_available ────────────────────────────────────────────────

def test_available_on_mindshub_with_a_chimera_anton(monkeypatch):
    _chimera_anton.install(monkeypatch)
    monkeypatch.setenv("COWORK_CHIMERA_PICKER", "1")
    assert chimera.chimera_available(_minds()) is True


def test_not_offered_until_the_rollout_switch_is_on(monkeypatch):
    _chimera_anton.install(monkeypatch)
    monkeypatch.delenv("COWORK_CHIMERA_PICKER", raising=False)
    assert chimera.chimera_available(_minds()) is False


def test_not_available_on_openai_compatible(monkeypatch):
    _chimera_anton.install(monkeypatch)
    monkeypatch.setenv("COWORK_CHIMERA_PICKER", "1")
    assert chimera.chimera_available(_openai_compatible()) is False


def test_not_available_without_a_chimera_anton(monkeypatch):
    _chimera_anton.hide(monkeypatch)
    assert chimera.chimera_available(_minds()) is False


def test_not_available_where_turns_run_on_a_pod(monkeypatch):
    _chimera_anton.install(monkeypatch)
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    assert chimera.chimera_available(_minds()) is False


def test_the_picker_endpoint_advertises_it(monkeypatch):
    from _fakes import FakeRequest

    from cowork.api.v1.endpoints import settings as settings_endpoint

    async def _no_catalog(*_args, **_kwargs):
        from cowork.services.providers import MindsModelListing

        return MindsModelListing(None, {}, {}, {}, {}, {}, {}, {})

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", _no_catalog)
    seen = []
    monkeypatch.setattr(
        settings_endpoint, "chimera_available", lambda settings: seen.append(settings) or True
    )
    with get_open_session() as session:
        result = asyncio.run(settings_endpoint.recommended_models(FakeRequest(), session, LOCAL_SCOPE))
    assert result["chimera_available"] is True
    assert isinstance(seen[0], UserSettings)


# ── The harness ──────────────────────────────────────────────────────

def _load_sidekick(conversation_id) -> dict | None:
    with get_open_session() as db:
        return SidekickContextService(ScopedSession(db, LOCAL_SCOPE)).load(conversation_id)


_STATE = {
    "sidekick_id": "sk-1",
    "model": "gpt-sol",
    "messages": [{"role": "user", "content": "Count the rows"}],
    "summary": "Counted rows.",
    "status": "idle",
    "last_call_at": "2026-11-04T10:00:00+00:00",
    "signals": 0,
}


async def _session_config(monkeypatch, settings: UserSettings, *, model=None, saved: dict | None = None):
    """One turn's ChatSessionConfig, built by the real harness on a new
    conversation, with ``saved`` as the sidekick an earlier turn left."""
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: settings)
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    with get_open_session() as db:
        scoped = ScopedSession(db, LOCAL_SCOPE)
        conversation = ConversationService(scoped).create_conversation(topic="chimera")
        if saved is not None:
            SidekickContextService(scoped).save(conversation.id, saved)
        return (await harness.AntonHarness()._build_chat_session(conversation, model=model)).session


@pytest.mark.asyncio
async def test_a_pseudo_model_turn_hands_anton_chimera_and_the_saved_sidekick(monkeypatch):
    _chimera_anton.install(monkeypatch)

    config = await _session_config(monkeypatch, _minds(), model=CHIMERA_MODEL_ID, saved=_STATE)

    assert config.chimera == _chimera_anton.ChimeraConfig(enabled=True)
    assert config.sidekick_state == _STATE
    assert config.llm_client.planning_model == "opus"
    assert config.llm_client.sidekick_model == "gpt-luna-6"
    assert config.llm_client.main_cache_ttl == "1h"


@pytest.mark.asyncio
async def test_a_chimera_turn_without_a_saved_sidekick_starts_one_fresh(monkeypatch):
    _chimera_anton.install(monkeypatch)

    config = await _session_config(monkeypatch, _minds(chimera_enabled=True))

    assert config.chimera.enabled is True
    assert config.sidekick_state is None
    assert config.llm_client.sidekick_model == _minds().resolved_sidekick_model


@pytest.mark.asyncio
async def test_a_turn_without_chimera_is_built_as_before(monkeypatch):
    _chimera_anton.install(monkeypatch)

    config = await _session_config(monkeypatch, _minds(), saved=_STATE)

    assert config.chimera is None
    assert config.sidekick_state is None
    assert config.llm_client.has_sidekick is False
    assert config.llm_client.main_cache_ttl is None


@pytest.mark.asyncio
async def test_an_anton_without_chimera_runs_the_pseudo_model_as_its_main_model(monkeypatch):
    _chimera_anton.hide(monkeypatch)

    config = await _session_config(monkeypatch, _minds(), model=CHIMERA_MODEL_ID)

    assert not hasattr(config, "chimera")
    assert config.llm_client.planning_model == "opus"


class _Session:
    def __init__(self, sidekick_state):
        self.sidekick_state = sidekick_state

    async def turn_stream(self, *_args, **_kwargs):
        return
        yield


def _run_turn(monkeypatch, conversation, session) -> None:
    from cowork.services import task_objects

    async def _fake_build(self, conversation, **_kwargs):
        return harness.BuiltChatSession(session=session, turn_scratch_dir=None, seed_info=None, learns=True)

    async def _drain():
        return [
            event
            async for event in harness.AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}],
            )
        ]

    monkeypatch.setattr(harness.AntonHarness, "_build_chat_session", _fake_build)
    monkeypatch.setattr(
        task_objects, "turn_artifact_changes",
        lambda *_a, **_k: task_objects.ArtifactChanges(created=[], touched=set()),
    )
    asyncio.run(_drain())


def test_the_turn_saves_the_sidekick_it_leaves(monkeypatch):
    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(topic="save")
        left = {**_STATE, "status": "open", "messages": [*_STATE["messages"], {"role": "assistant", "content": "done"}]}

        _run_turn(monkeypatch, conversation, _Session(left))

    assert _load_sidekick(conversation.id) == left


def test_a_turn_with_no_sidekick_saves_nothing(monkeypatch):
    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(topic="none")

        _run_turn(monkeypatch, conversation, _Session(None))

    assert _load_sidekick(conversation.id) is None


def test_a_failed_save_never_fails_the_turn(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("database gone")

    monkeypatch.setattr(harness, "_save_sidekick_state", _boom)
    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(topic="boom")

        _run_turn(monkeypatch, conversation, _Session(dict(_STATE)))


@pytest.fixture(autouse=True)
def _fresh_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()
