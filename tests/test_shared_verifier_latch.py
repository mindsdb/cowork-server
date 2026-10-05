"""Chat turns share anton's completion-verifier latch across messages.

The harness builds a new ChatSession for every message, so a latch kept on the
session never reaches its threshold and a verifier that always fails diagnoses
on every message. The chat config asks anton to keep the latch per coding
endpoint and model instead. An anton that predates the field gets no kwarg, so
the turn still builds.
"""
from __future__ import annotations

import dataclasses

import pytest
from anton.core.session import ChatSessionConfig
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.services.conversations import ConversationService


def _settings() -> UserSettings:
    """Minds-cloud settings for every role, with no file or env input."""
    return UserSettings(
        _env_file=None,
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
        episodic_memory=False,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
    )


def _config_without_shared_latch():
    """A ChatSessionConfig class as an anton without the shared latch declares it."""
    fields = []
    for f in dataclasses.fields(ChatSessionConfig):
        if f.name == "shared_verifier_latch":
            continue
        if f.default is not dataclasses.MISSING:
            spec = dataclasses.field(default=f.default)
        elif f.default_factory is not dataclasses.MISSING:
            spec = dataclasses.field(default_factory=f.default_factory)
        else:
            spec = dataclasses.field()
        fields.append((f.name, f.type, spec))
    return dataclasses.make_dataclass("ChatSessionConfig", fields)


async def _chat_session_config(monkeypatch):
    """One chat turn's ChatSessionConfig, built by the real harness.

    Side effects: creates a conversation in the test database.
    """
    settings = _settings()
    monkeypatch.setattr(
        "cowork.common.settings.user_settings.get_user_settings", lambda: settings
    )
    # Capture the config rather than a session: no scratchpad, no connectors.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    with get_open_session() as db:
        conversation = ConversationService(
            ScopedSession(db, LOCAL_SCOPE)
        ).create_conversation(topic="verifier latch")
        config, _, _ = await harness.AntonHarness()._build_chat_session(conversation)
    return config


@pytest.mark.asyncio
async def test_a_chat_turn_shares_the_verifier_latch(monkeypatch):
    config = await _chat_session_config(monkeypatch)

    assert config.shared_verifier_latch is True


@pytest.mark.asyncio
async def test_an_anton_without_the_field_still_builds_the_turn(monkeypatch):
    monkeypatch.setattr(
        "anton.core.session.ChatSessionConfig", _config_without_shared_latch()
    )

    config = await _chat_session_config(monkeypatch)

    assert not hasattr(config, "shared_verifier_latch")
    assert config.session_id
