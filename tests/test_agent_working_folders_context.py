"""The agent is told about a chat's working folders, and only when it should be."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
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


def test_the_session_builder_passes_the_channel_and_appends_the_context():
    source = inspect.getsource(harness.AntonHarness._build_chat_session)
    assert "_working_folder_paths(conversation, channel_context)" in source
    assert "+ _file_access_rules(folder_paths)" in source
    assert "+ _working_folders_context(folder_paths)" in source
