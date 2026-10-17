"""Builds one in-process turn's session config, for tests of what a turn hands anton."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from uuid import uuid4

from pydantic import SecretStr
from sqlalchemy import delete

from cowork.common.settings.user_settings import Provider, UserSettings, use_settings_scope
from cowork.db.scoped import TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.models.conversation import Conversation
from cowork.models.project import Project


class ScopeOrg(Enum):
    """built_turn's default project org: the scope's partition where the
    scope is personal, else none."""

    FROM_SCOPE = "from-scope"


@dataclass(frozen=True)
class BuiltTurn:
    config: object
    scratch: Path | None
    project: Project


def stub_turn_settings(monkeypatch) -> None:
    """Saved settings a turn can build with, and build_chat_session returning
    the config instead of a session."""
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
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")


async def close_turn(config: object | None, scratch: Path | None) -> None:
    """Remove the turn's scratch folder, as stream_response does, and close
    its model clients."""
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    if config is not None:
        client = config.llm_client
        for provider in (client.planning_provider, client.coding_provider, client.router_provider):
            await provider._client.close()


@asynccontextmanager
async def built_turn(
    *,
    monkeypatch,
    tmp_path: Path,
    scope: TenantScope | None,
    disabled_connections: list[dict] | None = None,
    project_org: str | None | ScopeOrg = ScopeOrg.FROM_SCOPE,
    before_build: Callable[[Path], None] | None = None,
    project: Project | None = None,
) -> AsyncIterator[BuiltTurn]:
    """Build one turn under ``scope``, with build_chat_session returning the
    config instead of a session.

    The turn runs in a new project under ``tmp_path``, whose row carries
    ``project_org`` (by default the org id ``scope`` gives its rows), or in
    ``project``, an existing one whose row and folder the caller keeps.
    ``before_build`` runs on the project folder before the turn is built, for
    files the turn must find there. The conversation carries ``scope``'s org
    id. Afterwards the turn's scratch folder is removed, as stream_response
    removes it, and every row made here is deleted, a failed build's included.
    """
    stub_turn_settings(monkeypatch)
    org_id = scope.org_id if scope is not None and scope.personal else None
    user_id = scope.user_id if scope is not None else None
    made_project: Project | None = None
    with get_open_session() as db:
        if project is None:
            project_path = tmp_path / "projects" / f"p-{uuid4().hex[:6]}"
            project_path.mkdir(parents=True)
            made_project = Project(
                name=project_path.name,
                path=str(project_path),
                org_id=org_id if project_org is ScopeOrg.FROM_SCOPE else project_org,
                created_by=user_id,
            )
            db.add(made_project)
            db.commit()
            project = made_project
        conversation = Conversation(
            topic="managed", project_id=project.id, org_id=org_id, created_by=user_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        if made_project is not None:
            db.refresh(made_project)
        config = scratch = None
        try:
            if before_build is not None:
                before_build(Path(project.path))
            with use_settings_scope(scope) if scope is not None else nullcontext():
                config, scratch, _ = await harness.AntonHarness()._build_chat_session(
                    conversation, disabled_connections=disabled_connections
                )
            yield BuiltTurn(config=config, scratch=scratch, project=project)
        finally:
            await close_turn(config, scratch)
            db.exec(delete(Conversation).where(Conversation.id == conversation.id))
            if made_project is not None:
                db.exec(delete(Project).where(Project.id == made_project.id))
            db.commit()


class ConnectedInChat(NamedTuple):
    # What the tool told the agent.
    answer: str
    # The vault the session holds once the tool returns.
    session_vault: object


async def connect_in_chat(turn: BuiltTurn, monkeypatch, *, known_variables: dict[str, str]) -> ConnectedInChat:
    """Run the turn's own connect_new_datasource tool as the agent calls it
    with postgres credentials pasted in chat, its connection test passing."""

    async def connection_test_passes(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(
        "anton.commands.datasource.verify.run_connection_test", connection_test_passes
    )
    [tool] = [tool for tool in turn.config.tools if tool.name == "connect_new_datasource"]
    session = SimpleNamespace(
        _data_vault=turn.config.data_vault,
        _console=None,
        _settings=SimpleNamespace(analytics_enabled=False),
        _scratchpads=None,
        _active_datasource=None,
    )
    answer = await tool.handler(session, {"engine": "postgres", "known_variables": known_variables})
    return ConnectedInChat(answer=answer, session_vault=session._data_vault)
