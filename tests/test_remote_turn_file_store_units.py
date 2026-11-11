"""A remote turn's workspace staging, its seed and skills reads and the memory
saves its pod reports work on the shared file store (COWORK_HOME), so each runs
as one unit on the file store's threads (run_efs_unit), as does the SKILL.md
read that decides whether an always-on skill reaches a turn. A slow file store
then holds file store slots, which EFS_SLOTS and the caller's EFS_SHARE count,
and a busy one refuses the work in POOL_TIMEOUT instead of holding a database
unit's thread outside them.

The guard tests run the real staging, the real memory save, and a turn start
and remote web turn with an always-on skill, in org tenancy, where the
blocking guard (tests/_blocking_guard.py) records file store work on any
thread run_efs did not start.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import _blocking_guard as blocking_guard
import cowork.db.session as db_session
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, scope_from_principal
from cowork.db.units import FilesystemBusy
from cowork.handlers import responses as responses_mod
from cowork.handlers.responses import ResponsesHandler, save_turn_memory, stage_remote_workspace
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.principal import Principal
from cowork.server import create_app
from test_run_efs import holding

ORG = "5d6c1f0e-2b7a-4c3d-8e9f-0a1b2c3d4e72"
USER = "6e7d2a1f-3c8b-4d4e-9f0a-1b2c3d4e5f72"
MEMBER = {"X-User-Id": USER, "X-Organization-Id": ORG}
PRINCIPAL = Principal(user_id=USER, org_id=ORG)
MEMORY_WARNING = "[responses] no database connection or file store slot freed in time to apply a turn's memory"


class OrgConversation(NamedTuple):
    id: UUID
    project_path: Path
    scope: TenantScope


@pytest.fixture
def org_conversation(org_deployment) -> OrgConversation:
    """A member's conversation in a project of their own, made through the
    app, so the project's folder is under COWORK_HOME."""
    client = TestClient(create_app())
    project = client.post("/api/v1/projects/", json={"name": f"staging-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert project.status_code == 201, project.text
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": project.json()["id"]}, headers=MEMBER,
    )
    assert conversation.status_code == 201, conversation.text
    return OrgConversation(
        id=UUID(conversation.json()["id"]),
        project_path=Path(project.json()["path"]),
        scope=scope_from_principal(PRINCIPAL),
    )


@pytest.fixture
def one_second_wait(monkeypatch):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)


def test_a_remote_turns_staging_runs_on_the_file_store_threads(org_conversation):
    """The real staging, unstubbed: it makes the project's artifacts base and
    copies the project's instructions into the conversation's workspace, and
    the guard finds no file store work off run_efs."""
    instructions = org_conversation.project_path / ".anton" / "anton.md"
    instructions.parent.mkdir(parents=True, exist_ok=True)
    instructions.write_text("Answer in charts.")

    with blocking_guard.watching() as found:
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert sorted(found) == []
    assert (org_conversation.project_path / ".anton" / "artifacts").is_dir()
    staged = org_conversation.project_path / "conversations" / str(org_conversation.id) / ".anton" / "anton.md"
    assert staged.read_text() == "Answer in charts."


def test_a_remote_turns_memory_save_runs_on_the_file_store_threads(org_conversation):
    """The real memory save, apply_turn_memory included: the entry lands in
    the member's memory store, and the guard finds no file store work off
    run_efs."""
    text = f"The team reads charts, not tables ({uuid4().hex[:8]})."

    with blocking_guard.watching() as found:
        asyncio.run(save_turn_memory(
            conv_id=org_conversation.id, entries=[{"text": text, "kind": "always"}],
            principal=PRINCIPAL, scope=org_conversation.scope, request_id="corr-1",
        ))

    assert sorted(found) == []
    root = GlobalMemoryStore(scope=org_conversation.scope).root
    assert any(text in path.read_text() for path in root.rglob("*") if path.is_file())


async def test_an_always_on_skill_is_read_on_the_file_store_threads_at_a_turns_start_and_in_its_pod_request(
    org_deployment, monkeypatch,
):
    """An organization whose store holds one skill shared with everyone and
    always on, with its row. The turn start reads it as handle() does: the
    first unit, a database unit for a turn whose project is named by id, sees
    the always-on row, and the SKILL.md that decides whether it reaches the
    turn is parsed in a unit of its own (_settle_always_on). The remote web
    turn's producer then reads the pod's seed, project memory included, and
    builds its skills block with the real remote_turn_skills, which reads the
    skill's folder. The guard finds no site the allowlist doesn't list, and
    the pod's request carries the skill always on.

    A new organization, whose store also holds a skill from before sharing
    with no row, so its first turn backfills that row (backfill_legacy_access),
    which lists the store, as an organization's first turn after the rollout
    does."""
    from functools import partial

    from anton.core.tools.skill_format import dump_skill

    from cowork.common.settings.user_settings import use_turn_settings
    from cowork.db.scoped import ScopedSession
    from cowork.db.session import get_open_session
    from cowork.db.units import run_db
    from cowork.models.skill import Skill
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.skill_access import access_rows, stage_access
    from cowork.services.skill_sharing import SkillAccessRule
    from cowork.services.skills import SkillService
    from tests._org_sharing import MEMBER_BEARER, mindshub_sharing_answers

    org_id, user_id = str(uuid4()), str(uuid4())
    principal = Principal(user_id=user_id, org_id=org_id)
    scope = scope_from_principal(principal)
    member = {"X-User-Id": user_id, "X-Organization-Id": org_id}
    client = TestClient(create_app())
    project = client.post("/api/v1/projects/", json={"name": "charts"}, headers=member)
    assert project.status_code == 201, project.text
    project_id = UUID(project.json()["id"])
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": str(project_id)}, headers=member,
    )
    assert conversation.status_code == 201, conversation.text
    conv_id = UUID(conversation.json()["id"])

    slug, legacy = "house-style", "legacy-notes"
    for name in (slug, legacy):
        folder = SkillService(scope).root / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            dump_skill(Skill(name=name, description=f"About {name}.", instructions="Answer in charts."))
        )
    session = get_open_session()
    try:
        stage_access(ScopedSession(session, scope), slug, SkillAccessRule(org_allowed=True, always_on=True))
        session.commit()
    finally:
        session.close()
    mindshub_sharing_answers(monkeypatch, teams=())
    sent: list[dict] = []

    async def pod(**kwargs):
        sent.append(kwargs)
        yield "turn_delta", {"text": "Here is the chart."}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    handler = ResponsesHandler(principal=principal, mindshub_bearer=MEMBER_BEARER)
    request = ResponsesRequest(input="What is 12 times 7?", conversation=str(conv_id), project_id=project_id)
    buffer = _Frames()

    with blocking_guard.watching() as found:
        await handler._resolve_skill_viewer(scope)
        start = await run_db(
            partial(handler._read_turn_start, request=request, has_disabled_connections=False), scope=scope,
        )
        assert start.always_on_unread
        start = await handler._settle_always_on(start)
        with use_turn_settings(scope, start.settings):
            await handler._produce_remote(
                conv_id=conv_id, input_text="What is 12 times 7?", original_content="What is 12 times 7?",
                model="anton", harness_id="anton", buffer=buffer,
            )

    assert blocking_guard.unlisted(found) == []
    assert (start.always_on_skills, start.gate_rows) == (True, [])
    assert buffer.closed == ["completed"]
    [job] = sent
    assert job["skills"][slug].always_on
    assert job["skills"][slug].files["SKILL.md"].startswith("---\n")
    assert not job["skills"][legacy].always_on
    session = get_open_session()
    try:
        backfilled = access_rows(ScopedSession(session, scope))[legacy]
        assert (backfilled.org_allowed, backfilled.always_on) == (True, False)
    finally:
        session.close()


class _Frames:
    """A turn's buffer that keeps only how it closed."""

    def __init__(self) -> None:
        self.closed: list[str] = []
        self.is_closed = False

    async def append(self, kind, record) -> None:
        pass

    async def close(self, reason) -> None:
        self.closed.append(reason)
        self.is_closed = True


async def test_a_busy_file_store_refuses_a_remote_turns_staging(monkeypatch, one_second_wait):
    """With every file store slot held, staging is refused with
    FilesystemBusy, so the turn is refused rather than run without what the
    user sent, and the staging never runs."""
    staged: list[UUID] = []
    monkeypatch.setattr(
        ResponsesHandler, "_stage_remote_workspace_files",
        staticmethod(lambda session, conv_id: staged.append(conv_id)),
    )

    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
        with pytest.raises(FilesystemBusy):
            await stage_remote_workspace(
                conv_id=uuid4(), scope=TenantScope(org_mode=True, org_id=ORG, user_id=USER),
            )

    assert staged == []


async def test_a_busy_file_store_drops_a_memory_save_with_a_warning(monkeypatch, one_second_wait, caplog):
    """With every file store slot held, the memory save is dropped with one
    WARNING naming the conversation and the request, and raises nothing, so
    the turn goes on to its reply."""
    applied: list[list[dict]] = []
    monkeypatch.setattr(
        ResponsesHandler, "_persist_turn_memory",
        staticmethod(lambda session, conv_id, entries, principal: applied.append(entries)),
    )
    conv_id = uuid4()

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
            await save_turn_memory(
                conv_id=conv_id, entries=[{"text": "remember", "kind": "always"}], principal=PRINCIPAL,
                scope=TenantScope(org_mode=True, org_id=ORG, user_id=USER), request_id="corr-1",
            )

    assert applied == []
    [record] = [r for r in caplog.records if r.name == responses_mod.logger.name]
    assert (record.levelname, record.getMessage()) == ("WARNING", MEMORY_WARNING)
    assert (record.conversation_id, record.request_id) == (str(conv_id), "corr-1")
