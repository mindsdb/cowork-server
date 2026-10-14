"""Builds one in-process turn's session config, for tests of what a turn hands anton."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr
from sqlalchemy import delete

from cowork.common.settings.user_settings import Provider, UserSettings, use_settings_scope
from cowork.db.scoped import TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.models.conversation import Conversation
from cowork.models.project import Project


@dataclass(frozen=True)
class BuiltTurn:
    config: object
    scratch: Path | None
    project: Project


@asynccontextmanager
async def built_turn(
    *,
    monkeypatch,
    tmp_path: Path,
    scope: TenantScope | None,
    disabled_connections: list[dict] | None = None,
) -> AsyncIterator[BuiltTurn]:
    """Build one turn in a new project under ``scope``, with
    build_chat_session returning the config instead of a session. The rows
    carry ``scope``'s org id, the turn's scratch folder is removed afterwards
    as stream_response removes it, and the rows are deleted."""
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

    org_id = scope.org_id if scope is not None and scope.personal else None
    user_id = scope.user_id if scope is not None else None
    project_path = tmp_path / "projects" / f"p-{uuid4().hex[:6]}"
    project_path.mkdir(parents=True)
    with get_open_session() as db:
        project = Project(name=project_path.name, path=str(project_path), org_id=org_id, created_by=user_id)
        db.add(project)
        db.commit()
        conversation = Conversation(
            topic="managed", project_id=project.id, org_id=org_id, created_by=user_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        db.refresh(project)
        with use_settings_scope(scope) if scope is not None else nullcontext():
            config, scratch, _ = await harness.AntonHarness()._build_chat_session(
                conversation, disabled_connections=disabled_connections
            )
        try:
            yield BuiltTurn(config=config, scratch=scratch, project=project)
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()
            db.exec(delete(Conversation).where(Conversation.id == conversation.id))
            db.exec(delete(Project).where(Project.id == project.id))
            db.commit()
