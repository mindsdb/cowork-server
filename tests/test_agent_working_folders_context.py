"""The agent is told about a chat's working folders, and only when it should be."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine, get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.harnesses.anton_harness.harness import (
    _file_access_rules,
    _working_folder_paths,
    _working_folders_context,
)
from cowork.harnesses.base import ChannelContext
from cowork.models.conversation import Conversation
from cowork.models.conversation_folder import ConversationFolder
from cowork.models.project import Project
from cowork.services.conversation_folders import ConversationFolderService
from cowork.services.conversations import ConversationService


@pytest.fixture()
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _chat_with_folders(session: Session, tmp_path: Path, name: str, folders: list[Path]):
    project_dir = tmp_path / "user" / f"{name}-project"
    project_dir.mkdir(parents=True)
    project = Project(name=name, path=str(project_dir))
    session.add(project)
    session.commit()
    conversation = Conversation(topic="t", project_id=project.id)
    session.add(conversation)
    session.commit()
    added = datetime.now(timezone.utc)
    for i, folder in enumerate(folders):
        session.add(
            ConversationFolder(
                conversation_id=conversation.id,
                path=str(folder),
                created_at=added + timedelta(seconds=i),
            )
        )
    session.commit()
    session.refresh(conversation)
    return conversation


def _folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / "user" / name
    folder.mkdir(parents=True)
    return folder.resolve()


def test_without_folders_the_access_rules_are_todays_text():
    assert _file_access_rules([]) == (
        "The only other files that you are allowed to access are any items that are attached to the conversation."
        "Access to any files not attached to the conversation or located outside the project is strictly forbidden."
    )
    assert _working_folders_context([]) == ""


def test_with_folders_the_rules_allow_them_and_the_context_lists_them():
    paths = ["/Users/x/docs", "/Users/x/reports"]

    rules = _file_access_rules(paths)
    context = _working_folders_context(paths)

    assert "the working folders" in rules
    assert "outside the project is strictly forbidden" not in rules
    assert "  - /Users/x/docs\n  - /Users/x/reports" in context
    assert "ask the user for permission first" in context
    assert "do not use select_path" in context


def test_a_chats_folders_reach_the_agent(local_mode, tmp_path):
    docs = _folder(tmp_path, "docs")
    reports = _folder(tmp_path, "reports")
    with Session(get_engine(get_app_settings().database.uri)) as session:
        conversation = _chat_with_folders(session, tmp_path, "ctx-folders", [docs, reports])

        assert _working_folder_paths(conversation, None) == [str(docs), str(reports)]


def test_a_channel_turn_is_offered_no_folders(local_mode, tmp_path):
    docs = _folder(tmp_path, "docs")
    with Session(get_engine(get_app_settings().database.uri)) as session:
        conversation = _chat_with_folders(session, tmp_path, "ctx-channel", [docs])

        assert _working_folder_paths(conversation, ChannelContext(channel_type="slack")) == []


def test_a_folder_that_is_gone_is_left_out(local_mode, tmp_path):
    docs = _folder(tmp_path, "docs")
    gone = _folder(tmp_path, "gone")
    gone.rmdir()
    with Session(get_engine(get_app_settings().database.uri)) as session:
        conversation = _chat_with_folders(session, tmp_path, "ctx-gone", [docs, gone])

        assert _working_folder_paths(conversation, None) == [str(docs)]


async def _suffix(monkeypatch, conversation_id, channel_context=None) -> str:
    """The system prompt suffix one turn of this chat is built with, by the real harness."""
    settings = UserSettings(
        _env_file=None,
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
        episodic_memory=False,
        minds_api_key=SecretStr("mdb-key"),
        minds_url="https://api.mindshub.ai",
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: settings)
    # Capture the config rather than a session: no scratchpad, no connectors.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).get_conversation(
            conversation_id
        )
        config, _, _ = await harness.AntonHarness()._build_chat_session(
            conversation, channel_context=channel_context
        )
    return config.system_prompt_context.suffix


def _chat_in_general(tmp_path: Path, folders: list[Path]):
    with get_open_session() as db:
        scoped = ScopedSession(db, LOCAL_SCOPE)
        conversation = ConversationService(scoped).create_conversation(topic="folders prompt")
        for folder in folders:
            ConversationFolderService(scoped).add_folder(conversation.id, str(folder))
        return conversation.id


@pytest.mark.asyncio
async def test_the_built_prompt_grants_each_folder(local_mode, monkeypatch, tmp_path):
    docs = _folder(tmp_path, "docs")
    reports = _folder(tmp_path, "reports")
    chat = _chat_in_general(tmp_path, [docs, reports])

    suffix = await _suffix(monkeypatch, chat)

    assert f"  - {docs}\n  - {reports}" in suffix
    assert "do not use select_path" in suffix
    # A file named without its folder was searched for in one folder only, and
    # the agent then answered about a different file.
    assert "search the project and every working folder" in suffix
    assert "never answer about a different file instead" in suffix
    assert "located outside the project is strictly forbidden" not in suffix


@pytest.mark.asyncio
async def test_without_folders_the_built_prompt_is_unchanged(local_mode, monkeypatch, tmp_path):
    chat = _chat_in_general(tmp_path, [])

    suffix = await _suffix(monkeypatch, chat)
    monkeypatch.setattr(harness, "_working_folder_paths", lambda *_a, **_k: [])
    without_lookup = await _suffix(monkeypatch, chat)

    assert suffix == without_lookup
    assert "located outside the project is strictly forbidden" in suffix
    assert "working folder" not in suffix


@pytest.mark.asyncio
async def test_a_channel_turn_prompt_carries_no_folders(local_mode, monkeypatch, tmp_path):
    docs = _folder(tmp_path, "docs")
    chat = _chat_in_general(tmp_path, [docs])

    suffix = await _suffix(monkeypatch, chat, ChannelContext(channel_type="slack"))

    assert str(docs) not in suffix
    assert "working folder" not in suffix
