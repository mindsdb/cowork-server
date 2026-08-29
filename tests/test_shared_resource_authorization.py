from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import shutil
from threading import Barrier, Event
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import project_files, projects, skills
from cowork.api.v1.endpoints.compat import stubs as compat_stubs
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.models.project import Project
from cowork.models.skill import META_PROJECTS
from cowork.models.shared_resource import (
    SharedResourceAttribution,
    SharedResourceMutation,
)
from cowork.principal import ORG_MANAGE_ROLE, Principal
from cowork.schemas.memory import MemoryResponse, MemoryScope
from cowork.schemas.projects import ProjectCreateRequest, ProjectUpdateRequest
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.memory import MemoryService, apply_turn_memory, build_turn_memory
from cowork.harnesses.memory.store import ProjectMemoryStore
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService, attachment_purpose
from cowork.services.projects import GENERAL_PROJECT, ProjectService
from cowork.services.shared_resources import (
    PROJECT,
    PROJECT_INSTRUCTIONS,
    PROJECT_MEMORY,
    SKILL,
    SharedResourceAccess,
    project_memory_resource_key,
    project_resource_key,
)
from cowork.services.skills import (
    BUILTIN_SKILLS_DIR,
    BUILTIN_SKILL_SLUGS,
    SkillService,
)


ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OTHER_ORG = "7ca7b810-9dad-11d1-80b4-00c04fd430c8"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


def _principal(user_id: str, *, admin: bool = False) -> Principal:
    return Principal(
        user_id=user_id,
        org_id=ORG,
        email=f"{user_id[0]}@example.com",
        roles=frozenset({ORG_MANAGE_ROLE}) if admin else frozenset(),
    )


def _scoped(engine, user_id: str) -> ScopedSession:
    return ScopedSession(
        Session(engine),
        TenantScope(org_mode=True, org_id=ORG, user_id=user_id),
    )


def test_openapi_advertises_shared_resource_contracts():
    from cowork.server import create_app

    schema = create_app().openapi()
    paths = schema["paths"]
    assert paths["/api/v1/projects/"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ProjectResponse")
    assert paths["/api/v1/skills/{skill_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/SkillResponse")
    assert paths["/api/v1/projects/{project_name}/instructions"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ProjectInstructionsResponse"
    )
    attribution = schema["components"]["schemas"]["ResourceAttribution"]
    assert set(attribution["properties"]) == {
        "createdBy",
        "lastModifiedBy",
        "lastModifiedAt",
    }
    assert set(attribution["required"]) == set(attribution["properties"])
    skill_response = schema["components"]["schemas"]["SkillResponse"]
    assert {"attribution", "capabilities", "isBuiltin"} <= set(
        skill_response["required"]
    )
    memory_response = schema["components"]["schemas"]["MemoryResponse"]
    assert {"attribution", "capabilities"} <= set(memory_response["required"])


def test_shared_resource_schema_capability_defaults_fail_closed():
    with pytest.raises(ValidationError):
        MemoryResponse(scope="global", category="rules", content="")

    with pytest.raises(ValidationError):
        SkillResponse(
            id="safe-default",
            label="safe-default",
            name="safe-default",
            description=None,
            instructions="",
            created_at=None,
            updated_at=None,
            enabled=True,
            projects=[],
        )


def test_pending_claims_hide_attribution_and_fail_capabilities_closed(org_engine):
    alice = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(alice, _principal(ALICE))

    project_id = uuid4()
    project_claim, project_token = access.reserve_claim(
        PROJECT,
        project_resource_key(project_id),
    )
    assert project_claim is not None and project_token is not None
    project = ProjectService(alice).create_project(
        "pending-response-project",
        project_id=project_id,
    )
    project_response = jsonable_encoder(projects._project_response(project, access))
    assert project_response["attribution"] == {
        "createdBy": None,
        "lastModifiedBy": None,
        "lastModifiedAt": None,
    }
    assert project_response["capabilities"] == {
        "canRename": False,
        "canDelete": False,
        "canEditInstructions": False,
    }

    instruction_claim, instruction_token = access.reserve_claim(
        PROJECT_INSTRUCTIONS,
        project_resource_key(project.id),
    )
    assert instruction_claim is not None and instruction_token is not None
    instruction_response = jsonable_encoder(
        project_files.get_project_instructions(
            project.name,
            alice,
            _principal(ALICE),
        )["file"]
    )
    assert instruction_response["attribution"] == {
        "createdBy": None,
        "lastModifiedBy": None,
        "lastModifiedAt": None,
    }
    assert instruction_response["capabilities"] == {
        "canEdit": False,
        "canDelete": False,
    }

    skill_service = SkillService(alice.scope)
    skill_slug = "pending-response-skill"
    skill_claim, skill_token = access.reserve_claim(SKILL, skill_slug)
    assert skill_claim is not None and skill_token is not None
    skill_service.create_skill(
        label=skill_slug,
        instructions="pending bytes",
    )
    skill_response = skills.get_skill(skill_slug, alice, _principal(ALICE))
    assert skill_response["attribution"] == {
        "createdBy": None,
        "lastModifiedBy": None,
        "lastModifiedAt": None,
    }
    assert skill_response["capabilities"] == {
        "canEdit": False,
        "canDelete": False,
        "canDisable": False,
    }


@pytest.fixture()
def org_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


def test_audit_service_keeps_delete_event_and_drops_current_attribution(org_engine):
    scoped = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(scoped, _principal(ALICE))

    access.register(SKILL, "quarterly-report")
    row = scoped.exec(scoped.select(SharedResourceAttribution)).one()
    assert row.created_by_id == ALICE
    assert row.created_by_email == "1@example.com"

    member = _scoped(org_engine, BOB)
    member_access = SharedResourceAccess(member, _principal(BOB))
    with pytest.raises(HTTPException) as denied:
        member_access.require_change(
            row.created_by_id,
            detail="creator or admin required",
        )
    assert denied.value.status_code == 403
    assert len(member.exec(member.select(SharedResourceMutation)).all()) == 1

    access.record_update(SKILL, "quarterly-report", action="update")
    access.record_delete(SKILL, "quarterly-report")

    assert scoped.exec(scoped.select(SharedResourceAttribution)).all() == []
    events = scoped.exec(scoped.select(SharedResourceMutation)).all()
    assert sorted(event.action for event in events) == ["create", "delete", "update"]
    assert {event.actor_id for event in events} == {ALICE}

    # A second claimant re-reads the first author's row instead of taking it.
    alice_claim, created = access.claim(PROJECT_MEMORY, "project:rules")
    assert created is True
    bob_claim, created = member_access.claim(PROJECT_MEMORY, "project:rules")
    assert created is False
    assert bob_claim is not None and bob_claim.created_by_id == ALICE
    assert member_access.can_change(bob_claim.created_by_id) is False


def test_same_actor_updates_strictly_advance_last_modified_at(
    org_engine,
    monkeypatch,
):
    scoped = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(scoped, _principal(ALICE))
    access.register(SKILL, "timestamped-skill")
    frozen = datetime(2042, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "cowork.services.shared_resources._utc_now",
        lambda: frozen,
    )

    access.record_update(SKILL, "timestamped-skill", action="update")
    first = access.attribution(SKILL, "timestamped-skill").model_dump(by_alias=True)[
        "lastModifiedAt"
    ]
    access.record_update(SKILL, "timestamped-skill", action="update")
    second = access.attribution(SKILL, "timestamped-skill").model_dump(by_alias=True)[
        "lastModifiedAt"
    ]

    assert first == frozen
    assert second == frozen + timedelta(microseconds=1)
    assert [
        event.action
        for event in scoped.exec(scoped.select(SharedResourceMutation)).all()
    ] == ["create", "update", "update"]


def test_mismatched_principal_cannot_use_scope_identity(org_engine):
    scoped = _scoped(org_engine, ALICE)
    mismatched = SharedResourceAccess(scoped, _principal(BOB))

    assert mismatched.can_change(ALICE) is False
    with pytest.raises(HTTPException) as denied:
        mismatched.register(SKILL, "forged-owner")
    assert denied.value.status_code == 403
    assert scoped.exec(scoped.select(SharedResourceMutation)).all() == []


def test_project_creator_admin_and_general_policy(org_engine):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("reports")
    SharedResourceAccess(alice, _principal(ALICE)).register(
        "project",
        str(project.id),
        creator_id=project.created_by,
        creator_email="1@example.com",
    )

    bob = _scoped(org_engine, BOB)
    foreign_skill = skills.create_skill(
        SkillCreateRequest(
            label="Foreign linked skill",
            instructions="Keep the project link current",
            projects=[project.name],
        ),
        bob,
        _principal(BOB),
    )
    skill_service = SkillService(alice.scope)
    skill_service.ensure_builtin_skills()
    builtin_slug = next(
        entry.name
        for entry in sorted(BUILTIN_SKILLS_DIR.iterdir())
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    )
    builtin = skill_service.get_skill(builtin_slug)
    builtin.metadata = {**builtin.metadata, META_PROJECTS: project.name}
    skill_service._write(builtin)
    with pytest.raises(HTTPException) as denied:
        projects.update_project(
            project.id,
            ProjectUpdateRequest(name="stolen"),
            bob,
            _principal(BOB),
        )
    assert denied.value.status_code == 403

    before_events = len(alice.exec(alice.select(SharedResourceMutation)).all())
    selected = projects.update_project(
        project.id,
        ProjectUpdateRequest(name=f"  {project.name}  ", is_active=True),
        bob,
        _principal(BOB),
    )
    assert selected["is_active"] is True
    assert len(alice.exec(alice.select(SharedResourceMutation)).all()) == before_events
    selected_attribution = selected["attribution"].model_dump(by_alias=True)
    assert selected_attribution["lastModifiedBy"]["userId"] == ALICE

    renamed = projects.update_project(
        project.id,
        ProjectUpdateRequest(name="monthly-reports"),
        alice,
        _principal(ALICE),
    )
    assert renamed["name"] == "monthly-reports"
    assert renamed["capabilities"].can_rename is True
    attribution = renamed["attribution"].model_dump(by_alias=True)
    assert attribution["createdBy"]["userId"] == ALICE
    assert attribution["lastModifiedBy"]["email"] == "1@example.com"
    encoded_project = jsonable_encoder(renamed)
    assert set(encoded_project["capabilities"]) == {
        "canRename",
        "canDelete",
        "canEditInstructions",
    }
    linked = skills.get_skill(
        foreign_skill["id"],
        alice,
        _principal(ALICE),
    )
    assert linked["projects"] == ["monthly-reports"]
    assert linked["attribution"]["lastModifiedBy"]["userId"] == BOB
    assert skill_service.get_skill(builtin_slug).projects == ["monthly-reports"]

    legacy = Project(
        name="legacy",
        path=str(Path(project.path).parent / "legacy"),
        org_id=ORG,
    )
    Path(legacy.path).mkdir()
    with Session(org_engine) as raw:
        raw.add(legacy)
        raw.commit()
        raw.refresh(legacy)
        legacy_id = legacy.id
    assert legacy.created_by is None
    with pytest.raises(HTTPException) as denied_legacy:
        projects.update_project(
            legacy_id,
            ProjectUpdateRequest(name="member-renamed"),
            bob,
            _principal(BOB),
        )
    assert denied_legacy.value.status_code == 403
    admin = _scoped(org_engine, ADMIN)
    updated = projects.update_project(
        legacy_id,
        ProjectUpdateRequest(name="admin-renamed"),
        admin,
        _principal(ADMIN, admin=True),
    )
    assert updated["name"] == "admin-renamed"
    assert updated["attribution"].created_by is None
    assert updated["attribution"].last_modified_by.user_id == ADMIN

    general = ProjectService(admin).ensure_general_for_scope()
    assert general is not None
    with pytest.raises(HTTPException) as immutable:
        projects.update_project(
            general.id,
            ProjectUpdateRequest(name="not-general"),
            admin,
            _principal(ADMIN, admin=True),
        )
    assert immutable.value.status_code == 403


def test_project_collision_normalized_noop_stays_member_selectable(org_engine):
    alice = _scoped(org_engine, ALICE)
    first = projects.create_project(
        ProjectCreateRequest(name="collision"),
        alice,
        _principal(ALICE),
    )
    second = projects.create_project(
        ProjectCreateRequest(name="collision"),
        alice,
        _principal(ALICE),
    )
    assert first["name"] == "collision"
    assert second["name"] == "collision-2"
    before = [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]

    bob = _scoped(org_engine, BOB)
    selected = projects.update_project(
        second["id"],
        ProjectUpdateRequest(name="  collision  ", is_active=True),
        bob,
        _principal(BOB),
    )

    assert selected["name"] == "collision-2"
    assert selected["is_active"] is True
    assert [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ] == before


def test_project_rename_restores_directory_and_all_skill_bytes_on_repair_failure(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="atomic-rename"),
        alice,
        _principal(ALICE),
    )
    old_path = Path(project["path"])
    (old_path / "ordinary.txt").write_text("keep me", encoding="utf-8")
    created_skills = [
        skills.create_skill(
            SkillCreateRequest(
                label=label,
                instructions=f"{label} content",
                projects=[project["name"]],
            ),
            alice,
            _principal(ALICE),
        )
        for label in ("atomic alpha", "atomic beta")
    ]
    skill_service = SkillService(alice.scope)
    skill_files = {
        item["id"]: skill_service._skill_dir(item["id"]) / "SKILL.md"
        for item in created_skills
    }
    before_skill_bytes = {slug: path.read_bytes() for slug, path in skill_files.items()}
    before_events = [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]
    original_write = SkillService._write
    attempts = 0

    def fail_second_repair(self, skill, **kwargs):
        nonlocal attempts
        if skill.name in skill_files:
            attempts += 1
            if attempts == 2:
                raise RuntimeError("second skill repair failed")
        return original_write(self, skill, **kwargs)

    monkeypatch.setattr(SkillService, "_write", fail_second_repair)

    with pytest.raises(RuntimeError, match="second skill repair failed"):
        projects.update_project(
            project["id"],
            ProjectUpdateRequest(name="atomic-renamed"),
            alice,
            _principal(ALICE),
        )

    current = ProjectService(alice).get_project(project["id"])
    assert current.name == "atomic-rename"
    assert Path(current.path) == old_path
    assert (old_path / "ordinary.txt").read_text(encoding="utf-8") == "keep me"
    assert not old_path.with_name("atomic-renamed").exists()
    assert {
        slug: path.read_bytes() for slug, path in skill_files.items()
    } == before_skill_bytes
    assert [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ] == before_events


@pytest.mark.parametrize("failure_point", ["audit", "commit"])
def test_project_rename_compensates_audit_and_db_commit_failures(
    org_engine,
    monkeypatch,
    failure_point,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name=f"{failure_point}-rename"),
        alice,
        _principal(ALICE),
    )
    old_path = Path(project["path"])
    instructions = old_path / ".anton" / "anton.md"
    instructions.parent.mkdir(parents=True)
    instructions.write_text("move with project", encoding="utf-8")
    access = SharedResourceAccess(alice, _principal(ALICE))
    access.register(
        PROJECT_INSTRUCTIONS,
        project_resource_key(project["id"]),
    )
    created_skill = skills.create_skill(
        SkillCreateRequest(
            label=f"{failure_point} linked skill",
            instructions="keep exact bytes",
            projects=[project["name"]],
        ),
        alice,
        _principal(ALICE),
    )
    skill_service = SkillService(alice.scope)
    skill_file = skill_service._skill_dir(created_skill["id"]) / "SKILL.md"
    before_skill_bytes = skill_file.read_bytes()
    before_events = [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]

    if failure_point == "audit":

        def fail_audit(*_args, **_kwargs):
            raise RuntimeError("project audit failed")

        monkeypatch.setattr(SharedResourceAccess, "stage_update", fail_audit)
        failure = "project audit failed"
    else:

        def fail_commit():
            raise RuntimeError("project commit failed")

        monkeypatch.setattr(alice, "commit", fail_commit)
        failure = "project commit failed"

    with pytest.raises(RuntimeError, match=failure):
        projects.update_project(
            project["id"],
            ProjectUpdateRequest(name=f"{failure_point}-renamed"),
            alice,
            _principal(ALICE),
        )

    current = ProjectService(alice).get_project(project["id"])
    assert current.name == f"{failure_point}-rename"
    assert Path(current.path) == old_path
    assert instructions.read_text(encoding="utf-8") == "move with project"
    assert not old_path.with_name(f"{failure_point}-renamed").exists()
    assert skill_file.read_bytes() == before_skill_bytes
    assert [
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ] == before_events


def test_project_rename_does_not_surface_post_commit_refresh_failure(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="refresh-safe-rename"),
        alice,
        _principal(ALICE),
    )
    old_path = Path(project["path"])

    def fail_refresh(_row):
        raise RuntimeError("refresh unavailable after commit")

    monkeypatch.setattr(alice, "refresh", fail_refresh)
    renamed = projects.update_project(
        project["id"],
        ProjectUpdateRequest(name="refresh-safe-renamed"),
        alice,
        _principal(ALICE),
    )

    assert renamed["name"] == "refresh-safe-renamed"
    assert Path(renamed["path"]).is_dir()
    assert not old_path.exists()
    current = ProjectService(alice).get_project(project["id"])
    assert current.name == "refresh-safe-renamed"
    events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == PROJECT,
            SharedResourceMutation.resource_key == str(project["id"]),
        )
    ).all()
    assert [event.action for event in events] == ["create", "rename"]


def test_project_create_does_not_surface_post_commit_refresh_failure(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)

    def fail_refresh(_row):
        raise RuntimeError("refresh unavailable after commit")

    monkeypatch.setattr(alice, "refresh", fail_refresh)
    created = projects.create_project(
        ProjectCreateRequest(name="refresh-safe-create"),
        alice,
        _principal(ALICE),
    )

    assert created["name"] == "refresh-safe-create"
    assert Path(created["path"]).is_dir()
    assert ProjectService(alice).get_project(created["id"]).name == created["name"]
    events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == PROJECT,
            SharedResourceMutation.resource_key == str(created["id"]),
        )
    ).all()
    assert [event.action for event in events] == ["create"]


def test_project_rename_rejects_a_pending_instruction_first_write(org_engine):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="instruction-race"),
        alice,
        _principal(ALICE),
    )
    access = SharedResourceAccess(alice, _principal(ALICE))
    claim, token = access.reserve_claim(
        PROJECT_INSTRUCTIONS,
        project_resource_key(project["id"]),
    )
    assert claim is not None and token is not None

    with pytest.raises(HTTPException) as conflict:
        projects.update_project(
            project["id"],
            ProjectUpdateRequest(name="instruction-race-renamed"),
            alice,
            _principal(ALICE),
        )

    assert conflict.value.status_code == 409
    current = ProjectService(alice).get_project(project["id"])
    assert current.name == "instruction-race"
    assert Path(current.path).is_dir()
    access.release_claim(claim, claim_token=token)


def test_project_rename_waits_for_an_existing_instruction_writer(
    org_engine,
    monkeypatch,
    tmp_path,
):
    race_engine = create_engine(
        f"sqlite:///{tmp_path / 'instruction-rename-race.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(race_engine)
    alice = _scoped(race_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="instruction-lock-race"),
        alice,
        _principal(ALICE),
    )
    project_files.write_project_file(
        project["name"],
        project_files._validated_project_path(".anton/anton.md"),
        project_files._FileWriteRequest(content="writer's completed bytes"),
        alice,
        _principal(ALICE),
    )
    rename_reached_instruction_guard = Event()
    original_guard = projects._guard_project_instructions_for_move

    def observe_guard(*args, **kwargs):
        rename_reached_instruction_guard.set()
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(
        projects,
        "_guard_project_instructions_for_move",
        observe_guard,
    )

    def rename_project():
        scoped = _scoped(race_engine, ALICE)
        try:
            return projects.update_project(
                project["id"],
                ProjectUpdateRequest(name="instruction-lock-renamed"),
                scoped,
                _principal(ALICE),
            )
        finally:
            scoped.close()

    access = SharedResourceAccess(alice, _principal(ALICE))
    key = project_resource_key(project["id"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        with access.mutation_lock(
            PROJECT_INSTRUCTIONS,
            key,
            resource_exists=lambda: (
                Path(project["path"]) / ".anton" / "anton.md"
            ).is_file(),
        ):
            rename_future = pool.submit(rename_project)
            assert rename_reached_instruction_guard.wait(timeout=5)
            assert rename_future.done() is False
        renamed = rename_future.result(timeout=5)

    renamed_instructions = Path(renamed["path"]) / ".anton" / "anton.md"
    assert renamed["name"] == "instruction-lock-renamed"
    assert renamed_instructions.read_text(encoding="utf-8") == (
        "writer's completed bytes"
    )
    assert not Path(project["path"]).exists()


def test_project_rename_waits_for_memory_writer_and_audit_failure_stays_converged(
    org_engine,
    monkeypatch,
    tmp_path,
):
    race_engine = create_engine(
        f"sqlite:///{tmp_path / 'memory-rename-race.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(race_engine)
    alice = _scoped(race_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="memory-lock-race"),
        alice,
        _principal(ALICE),
    )
    MemoryService(alice, _principal(ALICE))._write(
        MemoryScope.project,
        MemorySlot.RULES,
        "initial memory bytes",
        project["id"],
    )
    old_path = Path(project["path"])
    writer_entered = Event()
    release_writer = Event()
    rename_reached_memory_guard = Event()
    original_write = ProjectMemoryStore.write

    def blocking_write(store, slot, content):
        if content == "writer's completed bytes":
            writer_entered.set()
            assert release_writer.wait(timeout=5)
        return original_write(store, slot, content)

    monkeypatch.setattr(ProjectMemoryStore, "write", blocking_write)
    original_memory_guard = projects._guard_project_memory_for_move

    def observe_memory_guard(*args, **kwargs):
        rename_reached_memory_guard.set()
        return original_memory_guard(*args, **kwargs)

    monkeypatch.setattr(
        projects,
        "_guard_project_memory_for_move",
        observe_memory_guard,
    )
    original_stage_update = SharedResourceAccess.stage_update

    def fail_project_rename(access, kind, key, **kwargs):
        if kind == PROJECT and kwargs.get("action") == "rename":
            raise RuntimeError("rename audit failed")
        return original_stage_update(access, kind, key, **kwargs)

    monkeypatch.setattr(SharedResourceAccess, "stage_update", fail_project_rename)

    def write_memory():
        scoped = _scoped(race_engine, ALICE)
        try:
            MemoryService(scoped, _principal(ALICE))._write(
                MemoryScope.project,
                MemorySlot.RULES,
                "writer's completed bytes",
                project["id"],
            )
        finally:
            scoped.close()

    def rename_project():
        scoped = _scoped(race_engine, ALICE)
        try:
            return projects.update_project(
                project["id"],
                ProjectUpdateRequest(name="memory-lock-renamed"),
                scoped,
                _principal(ALICE),
            )
        finally:
            scoped.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(write_memory)
        assert writer_entered.wait(timeout=5)
        rename = pool.submit(rename_project)
        assert rename_reached_memory_guard.wait(timeout=0.1) is False
        release_writer.set()
        writer.result(timeout=5)
        with pytest.raises(RuntimeError, match="rename audit failed"):
            rename.result(timeout=5)

    alice.rollback()
    current = ProjectService(alice).get_project(project["id"])
    assert current.name == "memory-lock-race"
    assert Path(current.path) == old_path
    assert ProjectMemoryStore(old_path).read(MemorySlot.RULES).strip() == (
        "writer's completed bytes"
    )
    assert not old_path.with_name("memory-lock-renamed").exists()
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert (PROJECT_MEMORY, "update") in {
        (event.resource_kind, event.action) for event in events
    }
    assert (PROJECT, "rename") not in {
        (event.resource_kind, event.action) for event in events
    }
    race_engine.dispose()


def test_project_rename_serializes_and_rejects_a_stale_skill_project_add(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="reference-race"),
        alice,
        _principal(ALICE),
    )
    skill = skills.create_skill(
        SkillCreateRequest(label="reference race skill", instructions="content"),
        alice,
        _principal(ALICE),
    )
    rename_holds_reference_lock = Event()
    finish_rename = Event()
    skill_validation_entered = Event()
    original_prepare = SkillService.prepare_project_reference_rewrites
    original_require_projects = skills._require_known_projects

    def pause_rename(self, old_name, new_name, **kwargs):
        if old_name == project["name"]:
            rename_holds_reference_lock.set()
            assert finish_rename.wait(timeout=5)
        return original_prepare(self, old_name, new_name, **kwargs)

    monkeypatch.setattr(
        SkillService,
        "prepare_project_reference_rewrites",
        pause_rename,
    )

    def observe_skill_validation(scoped, project_names):
        skill_validation_entered.set()
        return original_require_projects(scoped, project_names)

    monkeypatch.setattr(skills, "_require_known_projects", observe_skill_validation)

    def rename_project():
        scoped = _scoped(org_engine, ALICE)
        try:
            return projects.update_project(
                project["id"],
                ProjectUpdateRequest(name="reference-race-renamed"),
                scoped,
                _principal(ALICE),
            )
        finally:
            scoped.close()

    def add_old_reference():
        scoped = _scoped(org_engine, ALICE)
        try:
            try:
                skills.update_skill(
                    skill["id"],
                    SkillUpdateRequest(projects=[project["name"]]),
                    scoped,
                    _principal(ALICE),
                )
            except HTTPException as exc:
                return exc.status_code
            return 200
        finally:
            scoped.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rename_future = pool.submit(rename_project)
        assert rename_holds_reference_lock.wait(timeout=5)
        add_future = pool.submit(add_old_reference)
        assert skill_validation_entered.wait(timeout=0.1) is False
        finish_rename.set()
        renamed = rename_future.result(timeout=5)
        add_status = add_future.result(timeout=5)

    assert renamed["name"] == "reference-race-renamed"
    assert add_status == 409
    assert skill_validation_entered.is_set()
    assert SkillService(alice.scope).get_skill(skill["id"]).projects == []


def test_project_delete_and_general_matrix(org_engine):
    alice = _scoped(org_engine, ALICE)
    bob = _scoped(org_engine, BOB)
    admin = _scoped(org_engine, ADMIN)

    creator_target = projects.create_project(
        ProjectCreateRequest(name="creator-delete"),
        alice,
        _principal(ALICE),
    )
    with pytest.raises(HTTPException) as peer_denied:
        projects.delete_project(creator_target["id"], bob, _principal(BOB))
    assert peer_denied.value.status_code == 403
    projects.delete_project(creator_target["id"], alice, _principal(ALICE))

    admin_target = projects.create_project(
        ProjectCreateRequest(name="admin-delete"),
        alice,
        _principal(ALICE),
    )
    projects.delete_project(
        admin_target["id"],
        admin,
        _principal(ADMIN, admin=True),
    )

    foreign = ScopedSession(
        Session(org_engine),
        TenantScope(org_mode=True, org_id=OTHER_ORG, user_id=BOB),
    )
    foreign_target = ProjectService(foreign).create_project("other-org")
    for hidden_id in (uuid4(), foreign_target.id):
        with pytest.raises(HTTPException) as hidden:
            projects.delete_project(hidden_id, alice, _principal(ALICE))
        assert hidden.value.status_code == 404

    general = ProjectService(alice).ensure_general_for_scope()
    assert general is not None
    for scoped, principal in (
        (alice, _principal(ALICE)),
        (bob, _principal(BOB)),
        (admin, _principal(ADMIN, admin=True)),
    ):
        with pytest.raises(HTTPException) as rename_denied:
            projects.update_project(
                general.id,
                ProjectUpdateRequest(name="replace-general"),
                scoped,
                principal,
            )
        assert rename_denied.value.status_code == 403
        with pytest.raises(HTTPException) as delete_denied:
            projects.delete_project(general.id, scoped, principal)
        assert delete_denied.value.status_code == 403

    selected_general = projects.update_project(
        general.id,
        ProjectUpdateRequest(name=GENERAL_PROJECT, is_active=True),
        bob,
        _principal(BOB),
    )
    assert selected_general["name"] == GENERAL_PROJECT

    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert [event.action for event in events].count("delete") == 2


@pytest.mark.asyncio
async def test_skill_and_project_memory_creator_gates(org_engine):
    alice = _scoped(org_engine, ALICE)
    created = skills.create_skill(
        SkillCreateRequest(label="Quarterly report", instructions="Draft it"),
        alice,
        _principal(ALICE),
    )
    assert created["capabilities"]["canEdit"] is True
    assert created["attribution"]["createdBy"]["userId"] == ALICE
    encoded_skill = jsonable_encoder(created)
    assert set(encoded_skill["attribution"]) == {
        "createdBy",
        "lastModifiedBy",
        "lastModifiedAt",
    }
    assert set(encoded_skill["capabilities"]) == {
        "canEdit",
        "canDelete",
        "canDisable",
    }

    bob = _scoped(org_engine, BOB)
    with pytest.raises(HTTPException) as denied_skill:
        skills.update_skill(
            created["id"],
            SkillUpdateRequest(enabled=False),
            bob,
            _principal(BOB),
        )
    assert denied_skill.value.status_code == 403

    admin = _scoped(org_engine, ADMIN)
    disabled = skills.update_skill(
        created["id"],
        SkillUpdateRequest(enabled=False),
        admin,
        _principal(ADMIN, admin=True),
    )
    assert disabled["enabled"] is False
    assert disabled["attribution"]["createdBy"]["userId"] == ALICE
    assert disabled["attribution"]["lastModifiedBy"]["userId"] == ADMIN

    edited = skills.update_skill(
        created["id"],
        SkillUpdateRequest(instructions="Creator revision"),
        alice,
        _principal(ALICE),
    )
    assert edited["declarative"] == "Creator revision"
    with pytest.raises(HTTPException) as denied_delete:
        skills.delete_skill(created["id"], bob, _principal(BOB))
    assert denied_delete.value.status_code == 403
    skills.delete_skill(created["id"], alice, _principal(ALICE))

    admin_delete = skills.create_skill(
        SkillCreateRequest(label="Admin delete", instructions="Temporary"),
        alice,
        _principal(ALICE),
    )
    skills.delete_skill(
        admin_delete["id"],
        admin,
        _principal(ADMIN, admin=True),
    )
    skill_events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == SKILL
        )
    ).all()
    assert {event.action for event in skill_events} >= {
        "create",
        "disable",
        "update",
        "delete",
    }

    SkillService(alice.scope).create_skill(
        label="Legacy skill",
        instructions="No attribution row",
    )
    with pytest.raises(HTTPException) as legacy_peer_denied:
        skills.update_skill(
            "legacy-skill",
            SkillUpdateRequest(instructions="Peer edit"),
            bob,
            _principal(BOB),
        )
    assert legacy_peer_denied.value.status_code == 403
    legacy_skill = skills.update_skill(
        "legacy-skill",
        SkillUpdateRequest(instructions="Admin edit"),
        admin,
        _principal(ADMIN, admin=True),
    )
    assert legacy_skill["attribution"]["createdBy"] is None
    assert legacy_skill["attribution"]["lastModifiedBy"]["userId"] == ADMIN

    project = ProjectService(alice).create_project("memory-project")
    memory = MemoryService(alice, _principal(ALICE))
    untouched = await MemoryService(bob, _principal(BOB)).update_memory(
        scope="project",
        category=MemorySlot.LESSONS,
        content="",
        project_id=project.id,
    )
    untouched_key = project_memory_resource_key(project.id, MemorySlot.LESSONS.value)
    assert untouched.attribution.created_by is None
    assert (
        SharedResourceAccess(bob, _principal(BOB)).has_attribution(
            PROJECT_MEMORY,
            untouched_key,
        )
        is False
    )
    assert not (Path(project.path) / ".anton" / "memory" / "lessons.md").exists()
    # The first writer owns this slot.
    authored = await memory.update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Deploy only on green",
        project_id=project.id,
    )
    assert authored.attribution.created_by.user_id == ALICE

    with pytest.raises(HTTPException) as denied_memory:
        await MemoryService(bob, _principal(BOB)).update_memory(
            scope="project",
            category=MemorySlot.RULES,
            content="Ignore CI",
            project_id=project.id,
        )
    assert denied_memory.value.status_code == 403

    admin_result = await MemoryService(
        admin, _principal(ADMIN, admin=True)
    ).update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Deploy after approval",
        project_id=project.id,
    )
    assert admin_result.attribution.last_modified_by.user_id == ADMIN

    with pytest.raises(HTTPException) as denied_clear:
        await MemoryService(bob, _principal(BOB)).delete_memory(
            scope="project",
            category=MemorySlot.RULES,
            project_id=project.id,
        )
    assert denied_clear.value.status_code == 403

    await memory.delete_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=project.id,
    )
    cleared = await memory.get_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=project.id,
    )
    assert cleared.content == ""
    assert cleared.attribution.created_by.user_id == ALICE
    assert cleared.attribution.last_modified_by.user_id == ALICE
    with pytest.raises(HTTPException):
        await MemoryService(bob, _principal(BOB)).update_memory(
            scope="project",
            category=MemorySlot.RULES,
            content="claim after clear",
            project_id=project.id,
        )

    memory_events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == PROJECT_MEMORY,
            SharedResourceMutation.resource_key
            == project_memory_resource_key(project.id, MemorySlot.RULES.value),
        )
    ).all()
    assert {event.action for event in memory_events} == {"create", "update", "clear"}


def test_two_members_cannot_both_create_the_same_skill(
    org_engine,
    tmp_path,
):
    race_engine = create_engine(
        f"sqlite:///{tmp_path / 'skill-race.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(race_engine)
    barrier = Barrier(2)

    def attempt(user_id: str, instructions: str):
        scoped = _scoped(race_engine, user_id)
        try:
            barrier.wait(timeout=5)
            response = skills.create_skill(
                SkillCreateRequest(
                    label="race-skill",
                    instructions=instructions,
                ),
                scoped,
                _principal(user_id),
            )
        except HTTPException as exc:
            return ("error", user_id, instructions, exc.status_code)
        finally:
            scoped.close()
        return ("created", user_id, instructions, response)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda args: attempt(*args),
                [(ALICE, "Alice's bytes"), (BOB, "Bob's bytes")],
            )
        )

    created = [result for result in results if result[0] == "created"]
    rejected = [result for result in results if result[0] == "error"]
    assert len(created) == 1
    assert len(rejected) == 1
    assert rejected[0][3] == 409

    winner_id = created[0][1]
    winner_bytes = created[0][2]
    assert (
        SkillService(_scoped(race_engine, winner_id).scope)
        .get_skill("race-skill")
        .instructions
        == winner_bytes
    )
    check = _scoped(race_engine, winner_id)
    attribution = check.exec(check.select(SharedResourceAttribution)).one()
    events = check.exec(check.select(SharedResourceMutation)).all()
    assert attribution.created_by_id == winner_id
    assert [(event.action, event.actor_id) for event in events] == [
        ("create", winner_id)
    ]
    check.close()
    race_engine.dispose()


def test_packaged_skills_are_immutable_for_every_org_role(org_engine):
    alice = _scoped(org_engine, ALICE)
    SkillService(alice.scope).ensure_builtin_skills()
    assert {
        entry.name
        for entry in BUILTIN_SKILLS_DIR.iterdir()
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    } == BUILTIN_SKILL_SLUGS
    victim = next(
        entry.name
        for entry in sorted(BUILTIN_SKILLS_DIR.iterdir())
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    )

    callers = [
        (alice, _principal(ALICE)),
        (_scoped(org_engine, BOB), _principal(BOB)),
        (_scoped(org_engine, ADMIN), _principal(ADMIN, admin=True)),
    ]
    for scoped, principal in callers:
        response = skills.get_skill(victim, scoped, principal)
        assert response["isBuiltin"] is True
        assert response["capabilities"]["canEdit"] is False
        assert response["capabilities"]["canDelete"] is False
        assert response["capabilities"]["canDisable"] is False
        with pytest.raises(HTTPException) as edit_denied:
            skills.update_skill(
                victim,
                SkillUpdateRequest(enabled=False),
                scoped,
                principal,
            )
        assert edit_denied.value.status_code == 403
        with pytest.raises(HTTPException) as delete_denied:
            skills.delete_skill(victim, scoped, principal)
        assert delete_denied.value.status_code == 403


def test_builtin_identity_fails_closed_when_package_assets_are_missing(
    org_engine,
    monkeypatch,
    tmp_path,
):
    alice = _scoped(org_engine, ALICE)
    SkillService(alice.scope).ensure_builtin_skills()
    victim = "documents"
    assert SkillService(alice.scope).get_skill(victim).name == victim
    monkeypatch.setattr(
        "cowork.services.skills.BUILTIN_SKILLS_DIR",
        tmp_path / "missing-package-assets",
    )

    admin = _scoped(org_engine, ADMIN)
    response = skills.get_skill(victim, admin, _principal(ADMIN, admin=True))
    assert response["isBuiltin"] is True
    assert response["capabilities"]["canEdit"] is False
    with pytest.raises(HTTPException) as denied:
        skills.update_skill(
            victim,
            SkillUpdateRequest(instructions="replace packaged content"),
            admin,
            _principal(ADMIN, admin=True),
        )
    assert denied.value.status_code == 403


def test_local_mode_keeps_project_and_builtin_mutations(org_engine):
    local = ScopedSession(Session(org_engine), LOCAL_SCOPE)
    project = projects.create_project(
        ProjectCreateRequest(name="local-project"),
        local,
        None,
    )
    renamed = projects.update_project(
        project["id"],
        ProjectUpdateRequest(name="local-renamed"),
        local,
        None,
    )
    assert renamed["name"] == "local-renamed"
    assert renamed["capabilities"].can_delete is True

    applied = apply_turn_memory(
        local.scope,
        renamed["path"],
        [
            {
                "text": "Remember this desktop project rule",
                "kind": "always",
                "scope": "project",
            }
        ],
    )
    assert applied == 1
    assert (
        "Remember this desktop project rule"
        in build_turn_memory(
            local.scope,
            renamed["path"],
        )["project"]["rules"]
    )

    service = SkillService(LOCAL_SCOPE)
    victim = next(
        entry.name
        for entry in sorted(BUILTIN_SKILLS_DIR.iterdir())
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    )
    target = service.root / victim
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUILTIN_SKILLS_DIR / victim, target)
    response = skills.get_skill(victim, local, None)
    assert response["isBuiltin"] is True
    assert response["capabilities"]["canEdit"] is True
    skills.update_skill(
        victim,
        SkillUpdateRequest(enabled=False),
        local,
        None,
    )
    skills.delete_skill(victim, local, None)
    assert not target.exists()

    project_path = Path(renamed["path"])
    projects.delete_project(renamed["id"], local, None)
    assert local.get(Project, renamed["id"]) is None
    assert not project_path.exists()


def test_memory_claim_event_waits_for_success_and_finalized_claim_is_safe(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(alice, _principal(ALICE))
    claim, token = access.reserve_claim(PROJECT_MEMORY, "project:rules")
    assert claim is not None and token is not None
    assert alice.exec(alice.select(SharedResourceMutation)).all() == []

    finalized = access.finalize_claim(
        claim,
        token,
        action="create",
    )
    assert finalized is not None
    # A delayed failure cleanup cannot delete a reservation after another
    # successful path has finalized it.
    assert access.release_claim(claim, claim_token=token) is False
    assert alice.exec(alice.select(SharedResourceAttribution)).one().id == claim.id
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert [(event.action, event.actor_id) for event in events] == [("create", ALICE)]


def test_durable_audit_does_not_surface_a_post_commit_refresh_failure(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(alice, _principal(ALICE))
    claim, token = access.reserve_claim(PROJECT_MEMORY, "project:refresh-safe")
    assert claim is not None and token is not None

    def fail_refresh(_row):
        raise RuntimeError("refresh unavailable after commit")

    monkeypatch.setattr(alice, "refresh", fail_refresh)

    assert access.finalize_claim(claim, token, action="create") is not None
    assert (
        access.record_update(
            PROJECT_MEMORY,
            "project:refresh-safe",
            action="update",
        )
        is not None
    )
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert [event.action for event in events] == ["create", "update"]


def test_claim_helpers_do_not_refresh_after_committed_outcomes(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(alice, _principal(ALICE))

    def fail_refresh(_row):
        raise RuntimeError("refresh unavailable after commit")

    monkeypatch.setattr(alice, "refresh", fail_refresh)
    claimed, created = access.claim(SKILL, "committed-claim")
    assert claimed is not None and created is True
    reserved, token = access.reserve_claim(SKILL, "committed-reservation")
    assert reserved is not None and token is not None
    placeholder, created = access.ensure_mutation_identity(
        PROJECT_MEMORY,
        "committed-placeholder",
    )
    assert created is True

    assert access.release_claim(reserved, claim_token=token) is True
    assert access.release_pristine_identity(placeholder) is True
    assert access._find(SKILL, "committed-claim") is not None
    assert access._find(SKILL, "committed-reservation") is None
    assert access._find(PROJECT_MEMORY, "committed-placeholder") is None


@pytest.mark.asyncio
async def test_failed_first_memory_write_releases_the_claim(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("failed-memory")
    original_write = ProjectMemoryStore.write
    attempts = 0

    def fail_once(store, slot, content):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("volume unavailable")
        return original_write(store, slot, content)

    monkeypatch.setattr(ProjectMemoryStore, "write", fail_once)
    with pytest.raises(OSError, match="volume unavailable"):
        await MemoryService(alice, _principal(ALICE)).update_memory(
            scope="project",
            category=MemorySlot.RULES,
            content="first attempt",
            project_id=project.id,
        )

    key = project_memory_resource_key(project.id, MemorySlot.RULES.value)
    alice_access = SharedResourceAccess(alice, _principal(ALICE))
    assert alice_access.has_attribution(PROJECT_MEMORY, key) is False
    assert alice.exec(alice.select(SharedResourceMutation)).all() == []

    bob = _scoped(org_engine, BOB)
    result = await MemoryService(bob, _principal(BOB)).update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="second attempt",
        project_id=project.id,
    )
    assert result.attribution.created_by.user_id == BOB


@pytest.mark.asyncio
async def test_empty_existing_memory_slots_are_claimable_by_first_writer(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    bob = _scoped(org_engine, BOB)

    direct_project = ProjectService(alice).create_project("empty-direct-memory")
    direct_store = ProjectMemoryStore(Path(direct_project.path))
    direct_store.write(MemorySlot.RULES, "   ")
    before = await MemoryService(bob, _principal(BOB)).get_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=direct_project.id,
    )
    assert before.content.strip() == ""
    assert before.capabilities.can_edit is True

    direct = await MemoryService(bob, _principal(BOB)).update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Bob is the first non-empty writer",
        project_id=direct_project.id,
    )
    assert direct.attribution.created_by.user_id == BOB

    remote_project = ProjectService(alice).create_project("empty-remote-memory")
    remote_store = ProjectMemoryStore(Path(remote_project.path))
    remote_store.write(MemorySlot.RULES, "\n\t")
    applied = apply_turn_memory(
        alice.scope,
        remote_project.path,
        [
            {
                "text": "Alice is the first remote writer",
                "kind": "always",
                "scope": "project",
            }
        ],
        access=SharedResourceAccess(alice, _principal(ALICE)),
        project_id=remote_project.id,
    )
    assert applied == 1
    remote_key = project_memory_resource_key(
        remote_project.id,
        MemorySlot.RULES.value,
    )
    remote_attribution = alice.exec(
        alice.select(SharedResourceAttribution).where(
            SharedResourceAttribution.resource_kind == PROJECT_MEMORY,
            SharedResourceAttribution.resource_key == remote_key,
        )
    ).one()
    assert remote_attribution.created_by_id == ALICE


@pytest.mark.asyncio
async def test_unreadable_legacy_memory_fails_closed_for_list_and_write(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("unreadable-memory")
    store = ProjectMemoryStore(Path(project.path))
    store.write(MemorySlot.RULES, "Legacy secret")
    original_read_checked = ProjectMemoryStore.read_checked

    def unreadable(store, slot):
        if MemorySlot(slot) == MemorySlot.RULES:
            raise OSError("shared volume read failed")
        return original_read_checked(store, slot)

    monkeypatch.setattr(ProjectMemoryStore, "read_checked", unreadable)
    service = MemoryService(alice, _principal(ALICE))
    listed = await service.list_memory(project_id=project.id)
    rules = next(
        item
        for item in listed
        if item.scope.value == "project" and item.category == MemorySlot.RULES
    )
    assert rules.content == ""
    assert rules.capabilities.can_edit is False
    assert rules.capabilities.can_delete is False

    with pytest.raises(OSError, match="shared volume read failed"):
        await service.update_memory(
            scope="project",
            category=MemorySlot.RULES,
            content="Overwrite unreadable legacy bytes",
            project_id=project.id,
        )

    assert store.read(MemorySlot.RULES).strip() == "Legacy secret"
    key = project_memory_resource_key(project.id, MemorySlot.RULES.value)
    assert (
        SharedResourceAccess(alice, _principal(ALICE)).has_attribution(
            PROJECT_MEMORY,
            key,
        )
        is False
    )
    assert alice.exec(alice.select(SharedResourceMutation)).all() == []


@pytest.mark.asyncio
async def test_invalid_utf8_memory_reads_fail_closed_with_safe_capabilities(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("invalid-utf8-memory")
    store = ProjectMemoryStore(Path(project.path))
    store.write(MemorySlot.RULES, "create the slot")
    (store.root / "rules.md").write_bytes(b"\xff\xfe")

    service = MemoryService(alice, _principal(ALICE))
    direct = await service.get_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=project.id,
    )
    assert direct.content == ""
    assert direct.capabilities.can_edit is False
    assert direct.capabilities.can_delete is False

    listed = await service.list_memory(project_id=project.id)
    rules = next(
        item
        for item in listed
        if item.scope == MemoryScope.project and item.category == MemorySlot.RULES
    )
    assert rules.content == ""
    assert rules.capabilities.can_edit is False
    assert rules.capabilities.can_delete is False


@pytest.mark.asyncio
async def test_pending_memory_claim_blocks_clear_without_changing_bytes(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("pending-memory-clear")
    store = ProjectMemoryStore(Path(project.path))
    key = project_memory_resource_key(project.id, MemorySlot.RULES.value)
    access = SharedResourceAccess(alice, _principal(ALICE))
    claim, token = access.reserve_claim(PROJECT_MEMORY, key)
    assert claim is not None and token is not None
    store.write(MemorySlot.RULES, "Pending writer bytes")

    with pytest.raises(HTTPException) as blocked:
        await MemoryService(alice, _principal(ALICE)).delete_memory(
            scope="project",
            category=MemorySlot.RULES,
            project_id=project.id,
        )
    assert blocked.value.status_code == 409
    assert store.read(MemorySlot.RULES).strip() == "Pending writer bytes"
    assert alice.exec(alice.select(SharedResourceMutation)).all() == []


@pytest.mark.asyncio
async def test_pending_memory_is_not_authorship_and_expired_empty_claim_reopens_slot(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("pending-memory-response")
    key = project_memory_resource_key(project.id, MemorySlot.RULES.value)
    access = SharedResourceAccess(alice, _principal(ALICE))
    claim, token = access.reserve_claim(PROJECT_MEMORY, key)
    assert claim is not None and token is not None

    bob = _scoped(org_engine, BOB)
    pending = await MemoryService(bob, _principal(BOB)).get_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=project.id,
    )
    assert pending.attribution.created_by is None
    assert pending.attribution.last_modified_by is None
    assert pending.attribution.last_modified_at is None
    assert pending.capabilities.can_edit is False
    assert pending.capabilities.can_delete is False

    claim.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    alice.add(claim)
    alice.commit()
    bob.rollback()
    reopened = await MemoryService(bob, _principal(BOB)).get_memory(
        scope="project",
        category=MemorySlot.RULES,
        project_id=project.id,
    )
    assert reopened.attribution.created_by is None
    assert reopened.capabilities.can_edit is True
    assert reopened.capabilities.can_delete is True
    assert (
        SharedResourceAccess(bob, _principal(BOB)).has_attribution(
            PROJECT_MEMORY,
            key,
        )
        is False
    )


def test_stale_memory_claim_releases_empty_or_recovers_surviving_bytes(org_engine):
    alice = _scoped(org_engine, ALICE)
    access = SharedResourceAccess(alice, _principal(ALICE))

    empty, empty_token = access.reserve_claim(PROJECT_MEMORY, "stale:empty")
    assert empty is not None and empty_token is not None
    empty.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    alice.add(empty)
    alice.commit()
    assert (
        access.recover_stale_claim(
            PROJECT_MEMORY,
            "stale:empty",
            resource_exists=lambda: False,
        )
        is True
    )
    assert access.has_attribution(PROJECT_MEMORY, "stale:empty") is False

    surviving, surviving_token = access.reserve_claim(
        PROJECT_MEMORY,
        "stale:surviving",
    )
    assert surviving is not None and surviving_token is not None
    surviving.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(
        seconds=1
    )
    alice.add(surviving)
    alice.commit()
    assert (
        access.recover_stale_claim(
            PROJECT_MEMORY,
            "stale:surviving",
            resource_exists=lambda: True,
        )
        is True
    )
    recovered = access._find(PROJECT_MEMORY, "stale:surviving")
    assert recovered is not None
    assert recovered.pending_claim_token is None
    assert recovered.created_by_id == ALICE
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert [(event.action, event.actor_id) for event in events] == [("create", ALICE)]


@pytest.mark.asyncio
async def test_first_memory_finalize_failure_restores_bytes_and_metadata(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("failed-memory-finalize")
    store = ProjectMemoryStore(Path(project.path))
    store.write(MemorySlot.RULES, "   ")

    def fail_finalize(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(SharedResourceAccess, "finalize_claim", fail_finalize)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        await MemoryService(alice, _principal(ALICE)).update_memory(
            scope="project",
            category=MemorySlot.RULES,
            content="Must be rolled back",
            project_id=project.id,
        )

    assert store.read(MemorySlot.RULES).strip() == ""
    key = project_memory_resource_key(project.id, MemorySlot.RULES.value)
    assert (
        SharedResourceAccess(alice, _principal(ALICE)).has_attribution(
            PROJECT_MEMORY,
            key,
        )
        is False
    )
    assert alice.exec(alice.select(SharedResourceMutation)).all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["direct", "remote"])
@pytest.mark.parametrize("prior", [b"foo\n\n", b"foo\r\n\r\n"])
async def test_existing_memory_audit_failure_restores_exact_prior_bytes(
    org_engine,
    monkeypatch,
    writer,
    prior,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project(f"exact-{writer}-restore")
    service = MemoryService(alice, _principal(ALICE))
    await service.update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="establish attribution",
        project_id=project.id,
    )
    store = ProjectMemoryStore(Path(project.path))
    slot_path = store.root / "rules.md"
    slot_path.write_bytes(prior)
    before_events = {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    }

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("memory audit failed")

    monkeypatch.setattr(SharedResourceAccess, "record_update", fail_audit)
    with pytest.raises(RuntimeError, match="memory audit failed"):
        if writer == "direct":
            await service.update_memory(
                scope="project",
                category=MemorySlot.RULES,
                content="replacement bytes",
                project_id=project.id,
            )
        else:
            apply_turn_memory(
                alice.scope,
                project.path,
                [
                    {
                        "text": "remote replacement",
                        "kind": "always",
                        "scope": "project",
                    }
                ],
                access=SharedResourceAccess(alice, _principal(ALICE)),
                project_id=project.id,
            )

    assert slot_path.read_bytes() == prior
    assert {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    } == before_events


def test_mutation_lock_serializes_the_entire_resource_window(org_engine):
    alice = _scoped(org_engine, ALICE)
    SharedResourceAccess(alice, _principal(ALICE)).register(SKILL, "locked-skill")
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def hold_first():
        scoped = _scoped(org_engine, ALICE)
        try:
            with SharedResourceAccess(scoped, _principal(ALICE)).mutation_lock(
                SKILL,
                "locked-skill",
            ):
                first_entered.set()
                assert release_first.wait(timeout=5)
        finally:
            scoped.close()

    def enter_second():
        scoped = _scoped(org_engine, ALICE)
        try:
            with SharedResourceAccess(scoped, _principal(ALICE)).mutation_lock(
                SKILL,
                "locked-skill",
            ):
                second_entered.set()
        finally:
            scoped.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(hold_first)
        assert first_entered.wait(timeout=5)
        second = pool.submit(enter_second)
        assert second_entered.wait(timeout=0.1) is False
        release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert second_entered.is_set()


def test_stale_recovery_evaluates_existence_callback_inside_coordination_lock(
    org_engine,
    tmp_path,
):
    race_engine = create_engine(
        f"sqlite:///{tmp_path / 'stale-recovery-lock.db'}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(race_engine)
    owner = _scoped(race_engine, ALICE)
    access = SharedResourceAccess(owner, _principal(ALICE))
    key = "locked-stale-recovery"
    claim, token = access.reserve_claim(PROJECT_MEMORY, key)
    assert claim is not None and token is not None
    claim.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    owner.add(claim)
    owner.commit()
    callback_evaluated = Event()

    def recover():
        scoped = _scoped(race_engine, BOB)
        try:
            return SharedResourceAccess(scoped, _principal(BOB)).recover_stale_claim(
                PROJECT_MEMORY,
                key,
                resource_exists=lambda: callback_evaluated.set() or False,
            )
        finally:
            scoped.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with access.coordination_lock(PROJECT_MEMORY, key):
            future = pool.submit(recover)
            assert callback_evaluated.wait(timeout=0.1) is False
        assert future.result(timeout=5) is True

    assert callback_evaluated.is_set()
    race_engine.dispose()


def test_nested_postgres_resource_locks_share_one_dedicated_connection(
    org_engine,
    monkeypatch,
):
    class FakeConnection:
        def __init__(self):
            self.calls = []
            self.closed = 0

        def execute(self, statement, parameters):
            self.calls.append((str(statement), parameters))

        def close(self):
            self.closed += 1

    connection = FakeConnection()

    class FakeBind:
        dialect = SimpleNamespace(name="postgresql")

        def __init__(self):
            self.connects = 0

        def connect(self):
            self.connects += 1
            return connection

    bind = FakeBind()
    scoped = _scoped(org_engine, ALICE)
    monkeypatch.setattr(scoped, "get_bind", lambda: bind)
    access = SharedResourceAccess(scoped, _principal(ALICE))

    with access._database_locks.lock(ORG, PROJECT, "project-id"):
        with access._database_locks.lock(
            ORG,
            PROJECT_INSTRUCTIONS,
            "project-id",
        ):
            assert bind.connects == 1
            assert connection.closed == 0

    operations = [
        "unlock" if "advisory_unlock" in statement else "lock"
        for statement, _parameters in connection.calls
    ]
    assert operations == ["lock", "lock", "unlock", "unlock"]
    assert bind.connects == 1
    assert connection.closed == 1


def test_stale_skill_waiter_cannot_leave_a_ghost_attribution(org_engine):
    alice = _scoped(org_engine, ALICE)
    created = skills.create_skill(
        SkillCreateRequest(label="reusable slug", instructions="First version"),
        alice,
        _principal(ALICE),
    )
    slug = created["id"]
    stale_service = SkillService(alice.scope)
    assert stale_service.get_skill(slug).name == slug
    skills.delete_skill(slug, alice, _principal(ALICE))

    admin = _scoped(org_engine, ADMIN)
    with pytest.raises(HTTPException) as gone:
        with SharedResourceAccess(
            admin,
            _principal(ADMIN, admin=True),
        ).mutation_lock(
            SKILL,
            slug,
            resource_exists=lambda: stale_service._skill_dir(slug).is_dir(),
        ):
            pytest.fail("a stale waiter must not enter the mutation window")
    assert gone.value.status_code == 404
    assert admin.exec(admin.select(SharedResourceAttribution)).all() == []

    bob = _scoped(org_engine, BOB)
    recreated = skills.create_skill(
        SkillCreateRequest(label="reusable slug", instructions="Second version"),
        bob,
        _principal(BOB),
    )
    assert recreated["attribution"]["createdBy"]["userId"] == BOB


def _leave_expired_incomplete_skill_claim(scoped: ScopedSession, slug: str) -> None:
    access = SharedResourceAccess(scoped, _principal(scoped.scope.user_id))
    claim, token = access.reserve_claim(SKILL, slug)
    assert claim is not None and token is not None
    claim.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    scoped.add(claim)
    scoped.commit()
    service = SkillService(scoped.scope)
    service._skill_dir(slug).mkdir(parents=True)


def _leave_expired_complete_skill_claim(
    scoped: ScopedSession,
    slug: str,
) -> None:
    access = SharedResourceAccess(scoped, _principal(scoped.scope.user_id))
    claim, token = access.reserve_claim(SKILL, slug)
    assert claim is not None and token is not None
    SkillService(scoped.scope).create_skill(slug, "Survived creator crash")
    claim.pending_claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    scoped.add(claim)
    scoped.commit()


def test_stale_incomplete_skill_claim_is_cleaned_before_create(org_engine):
    alice = _scoped(org_engine, ALICE)
    slug = "recovered-create"
    _leave_expired_incomplete_skill_claim(alice, slug)
    service = SkillService(alice.scope)
    (service._skill_dir(slug) / "SKILL.md").write_text(
        "partial file without frontmatter",
        encoding="utf-8",
    )

    created = skills.create_skill(
        SkillCreateRequest(label=slug, instructions="Recovered content"),
        alice,
        _principal(ALICE),
    )

    assert created["id"] == slug
    assert service.has_complete_skill(slug) is True
    assert service.get_skill(slug).instructions == "Recovered content"
    actions = [
        event.action
        for event in alice.exec(alice.select(SharedResourceMutation)).all()
        if event.resource_kind == SKILL and event.resource_key == slug
    ]
    assert actions == ["create"]


def test_stale_complete_skill_claim_is_recovered_before_mutations(org_engine):
    alice = _scoped(org_engine, ALICE)
    update_slug = "recovered-before-update"
    _leave_expired_complete_skill_claim(alice, update_slug)

    updated = skills.update_skill(
        update_slug,
        SkillUpdateRequest(instructions="Updated after recovery"),
        alice,
        _principal(ALICE),
    )

    assert updated["declarative"] == "Updated after recovery"
    update_events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == SKILL,
            SharedResourceMutation.resource_key == update_slug,
        )
    ).all()
    assert [(event.action, event.actor_id) for event in update_events] == [
        ("create", ALICE),
        ("update", ALICE),
    ]

    delete_slug = "recovered-before-delete"
    _leave_expired_complete_skill_claim(alice, delete_slug)
    admin = _scoped(org_engine, ADMIN)

    skills.delete_skill(delete_slug, admin, _principal(ADMIN, admin=True))

    assert not SkillService(alice.scope)._skill_dir(delete_slug).exists()
    delete_events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == SKILL,
            SharedResourceMutation.resource_key == delete_slug,
        )
    ).all()
    assert [(event.action, event.actor_id) for event in delete_events] == [
        ("create", ALICE),
        ("delete", ADMIN),
    ]


@pytest.mark.asyncio
async def test_stale_incomplete_skill_claim_is_cleaned_before_import(org_engine):
    alice = _scoped(org_engine, ALICE)
    slug = "recovered-import"
    _leave_expired_incomplete_skill_claim(alice, slug)
    upload = UploadFile(
        filename="recovered.md",
        file=io.BytesIO(
            b"---\n"
            b"name: recovered-import\n"
            b"description: Imported after crash recovery\n"
            b"---\n"
            b"Recovered import content\n"
        ),
    )

    created = await skills.upload_skill(upload, alice, _principal(ALICE))

    service = SkillService(alice.scope)
    assert created["id"] == slug
    assert service.has_complete_skill(slug) is True
    assert service.get_skill(slug).instructions == "Recovered import content\n"
    actions = [
        event.action
        for event in alice.exec(alice.select(SharedResourceMutation)).all()
        if event.resource_kind == SKILL and event.resource_key == slug
    ]
    assert actions == ["create"]


def test_skill_noop_semantic_actions_and_delete_compensation(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    created = skills.create_skill(
        SkillCreateRequest(label="audit skill", instructions="Initial"),
        alice,
        _principal(ALICE),
    )
    slug = created["id"]
    service = SkillService(alice.scope)
    skill_file = service._skill_dir(slug) / "SKILL.md"
    before_bytes = skill_file.read_bytes()
    before_updated = created["updatedAt"]
    before_events = alice.exec(alice.select(SharedResourceMutation)).all()

    noop = skills.update_skill(
        slug,
        SkillUpdateRequest(instructions="Initial", enabled=True),
        alice,
        _principal(ALICE),
    )
    assert noop["updatedAt"] == before_updated
    assert skill_file.read_bytes() == before_bytes
    assert len(alice.exec(alice.select(SharedResourceMutation)).all()) == len(
        before_events
    )

    changed = skills.update_skill(
        slug,
        SkillUpdateRequest(
            label="audit skill renamed",
            instructions="Changed",
            enabled=False,
        ),
        alice,
        _principal(ALICE),
    )
    renamed_slug = changed["id"]
    assert renamed_slug != slug
    actions = [
        event.action for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]
    assert actions == ["create", "rename", "update", "disable"]

    reenabled = skills.update_skill(
        renamed_slug,
        SkillUpdateRequest(enabled=True),
        alice,
        _principal(ALICE),
    )
    assert reenabled["enabled"] is True
    actions = [
        event.action for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]
    assert actions == ["create", "rename", "update", "disable", "enable"]

    def fail_delete(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(SharedResourceAccess, "record_delete", fail_delete)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        skills.delete_skill(renamed_slug, alice, _principal(ALICE))
    assert service.get_skill(renamed_slug).instructions == "Changed"
    attribution = SharedResourceAccess(alice, _principal(ALICE))._find(
        SKILL,
        renamed_slug,
    )
    assert attribution is not None and attribution.created_by_id == ALICE
    assert "delete" not in [
        event.action for event in alice.exec(alice.select(SharedResourceMutation)).all()
    ]


def test_skill_audit_failure_restores_noncanonical_bytes_and_metadata(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    created = skills.create_skill(
        SkillCreateRequest(
            label="byte exact rollback",
            instructions="Initial canonical body",
        ),
        alice,
        _principal(ALICE),
    )
    slug = created["id"]
    service = SkillService(alice.scope)
    skill_file = service._skill_dir(slug) / "SKILL.md"
    original_bytes = (
        b"---   \n"
        b"# Preserve this hand-authored ordering and comment.\n"
        b"description: 'Byte exact rollback'\n"
        b"name: byte-exact-rollback\n"
        b"metadata: {enabled: 'true', custom: ' spaced value '}\n"
        b"---    \n"
        b"Original body with trailing spaces.   \n\n"
    )
    skill_file.write_bytes(original_bytes)
    assert service.get_skill(slug).instructions.endswith("   \n\n")

    access = SharedResourceAccess(alice, _principal(ALICE))
    before_attribution = access._find(SKILL, slug)
    assert before_attribution is not None
    before_attribution_state = (
        before_attribution.id,
        before_attribution.resource_key,
        before_attribution.created_by_id,
        before_attribution.updated_by_id,
        before_attribution.modified_at,
    )
    before_events = {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    }

    def stage_then_fail(access, kind, key, **kwargs):
        access.stage_updates(kind, key, **kwargs)
        raise RuntimeError("skill audit failed")

    monkeypatch.setattr(SharedResourceAccess, "record_updates", stage_then_fail)
    with pytest.raises(RuntimeError, match="skill audit failed"):
        skills.update_skill(
            slug,
            SkillUpdateRequest(instructions="Uncommitted replacement"),
            alice,
            _principal(ALICE),
        )

    assert skill_file.read_bytes() == original_bytes
    restored_attribution = SharedResourceAccess(alice, _principal(ALICE))._find(
        SKILL,
        slug,
    )
    assert restored_attribution is not None
    assert (
        restored_attribution.id,
        restored_attribution.resource_key,
        restored_attribution.created_by_id,
        restored_attribution.updated_by_id,
        restored_attribution.modified_at,
    ) == before_attribution_state
    assert {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    } == before_events


@pytest.mark.asyncio
async def test_project_delete_audits_and_cleans_protected_children(org_engine):
    alice = _scoped(org_engine, ALICE)
    bob = _scoped(org_engine, BOB)
    project = projects.create_project(
        ProjectCreateRequest(name="cascade-delete"),
        alice,
        _principal(ALICE),
    )
    project_id = project["id"]
    project_name = project["name"]
    direct_conversation = ConversationService(alice).create_conversation(
        "directly deleted chat",
        project_id=project_id,
    )
    direct_purpose = attachment_purpose(str(direct_conversation.id))
    direct_owner_attachment = FileService(alice).create_file_from_bytes(
        filename="owner-direct.txt",
        content_type="text/plain",
        data=b"owner direct attachment",
        purpose=direct_purpose,
    )
    direct_peer_attachment = FileService(bob).create_file_from_bytes(
        filename="peer-direct.txt",
        content_type="text/plain",
        data=b"peer direct attachment",
        purpose=direct_purpose,
    )
    direct_owner_path = Path(direct_owner_attachment.path)
    direct_peer_path = Path(direct_peer_attachment.path)

    assert ConversationService(alice).delete_conversation(direct_conversation.id)
    bob.rollback()
    assert FileService(alice).list_file_rows(direct_purpose) == []
    assert [row.id for row in FileService(bob).list_file_rows(direct_purpose)] == [
        direct_peer_attachment.id
    ]
    assert not direct_owner_path.exists()
    assert direct_peer_path.exists()

    project_files.write_project_file(
        project_name,
        project_files._validated_project_path(".anton/anton.md"),
        project_files._FileWriteRequest(content="Protected instructions"),
        alice,
        _principal(ALICE),
    )
    await MemoryService(alice, _principal(ALICE)).update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Protected memory",
        project_id=project_id,
    )
    foreign_conversation = ConversationService(bob).create_conversation(
        "another member's chat",
        project_id=project_id,
    )
    foreign_conversation_id = foreign_conversation.id
    foreign_attachment_purpose = attachment_purpose(str(foreign_conversation_id))
    foreign_attachment = FileService(bob).create_file_from_bytes(
        filename="foreign.txt",
        content_type="text/plain",
        data=b"foreign attachment",
        purpose=foreign_attachment_purpose,
    )
    parent_owner_attachment = FileService(alice).create_file_from_bytes(
        filename="owner-parent.txt",
        content_type="text/plain",
        data=b"owner attachment on peer conversation",
        purpose=foreign_attachment_purpose,
    )
    foreign_attachment_path = Path(foreign_attachment.path)
    parent_owner_attachment_path = Path(parent_owner_attachment.path)

    projects.delete_project(project_id, alice, _principal(ALICE))

    keys = {
        (PROJECT, project_resource_key(project_id)),
        (PROJECT_INSTRUCTIONS, project_resource_key(project_id)),
        (
            PROJECT_MEMORY,
            project_memory_resource_key(project_id, MemorySlot.RULES.value),
        ),
    }
    remaining = alice.exec(alice.select(SharedResourceAttribution)).all()
    assert {
        (row.resource_kind, row.resource_key)
        for row in remaining
        if (row.resource_kind, row.resource_key) in keys
    } == set()
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    actions_by_resource = {
        (event.resource_kind, event.resource_key, event.action) for event in events
    }
    assert (PROJECT, project_resource_key(project_id), "delete") in actions_by_resource
    assert (
        PROJECT_INSTRUCTIONS,
        project_resource_key(project_id),
        "delete",
    ) in actions_by_resource
    assert (
        PROJECT_MEMORY,
        project_memory_resource_key(project_id, MemorySlot.RULES.value),
        "clear",
    ) in actions_by_resource
    bob.rollback()
    with pytest.raises(ValueError, match="not found"):
        ConversationService(bob).get_conversation(foreign_conversation_id)
    assert FileService(bob).list_file_rows(foreign_attachment_purpose) == []
    assert FileService(alice).list_file_rows(foreign_attachment_purpose) == []
    assert not foreign_attachment_path.exists()
    assert not parent_owner_attachment_path.exists()


def test_project_delete_restores_row_directory_and_attribution_on_audit_failure(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="delete-rollback"),
        alice,
        _principal(ALICE),
    )
    project_id = project["id"]
    project_path = Path(project["path"])
    ordinary = project_path / "ordinary.txt"
    ordinary.write_text("must survive", encoding="utf-8")
    instructions = project_path / ".anton" / "anton.md"
    instructions.parent.mkdir(parents=True)
    instructions.write_text("protected", encoding="utf-8")
    conversation = ConversationService(alice).create_conversation(
        "must survive",
        project_id=project_id,
    )
    attachment = FileService(alice).create_file_from_bytes(
        filename="evidence.txt",
        content_type="text/plain",
        data=b"attachment must survive",
        purpose=attachment_purpose(str(conversation.id)),
    )
    attachment_path = Path(attachment.path)
    access = SharedResourceAccess(alice, _principal(ALICE))
    access.register(
        PROJECT_INSTRUCTIONS,
        project_resource_key(project_id),
    )
    before_attribution = {
        row.id for row in alice.exec(alice.select(SharedResourceAttribution)).all()
    }
    before_events = {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    }

    def fail_delete_audit(*_args, **_kwargs):
        raise RuntimeError("delete audit failed")

    monkeypatch.setattr(
        SharedResourceAccess,
        "stage_deletes",
        fail_delete_audit,
    )

    with pytest.raises(RuntimeError, match="delete audit failed"):
        projects.delete_project(project_id, alice, _principal(ALICE))

    current = ProjectService(alice).get_project(project_id)
    assert Path(current.path) == project_path
    assert ordinary.read_text(encoding="utf-8") == "must survive"
    assert instructions.read_text(encoding="utf-8") == "protected"
    assert (
        ConversationService(alice).get_conversation(conversation.id).id
        == conversation.id
    )
    assert FileService(alice).list_file_rows(attachment_purpose(str(conversation.id)))
    assert attachment_path.read_bytes() == b"attachment must survive"
    assert {
        row.id for row in alice.exec(alice.select(SharedResourceAttribution)).all()
    } == before_attribution
    assert {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    } == before_events
    assert not list(project_path.parent.glob(f".delete-{project_id}-*"))


def test_project_delete_cleans_a_legacy_memory_identity_cleared_before_lock(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="legacy-memory-delete-race"),
        alice,
        _principal(ALICE),
    )
    project_id = project["id"]
    store = ProjectMemoryStore(Path(project["path"]))
    store.write(MemorySlot.RULES, "Legacy bytes without attribution")
    memory_key = project_memory_resource_key(project_id, MemorySlot.RULES.value)
    original_ensure = SharedResourceAccess.ensure_mutation_identity

    def clear_after_identity(access, kind, key):
        row, created = original_ensure(access, kind, key)
        if kind == PROJECT_MEMORY and key == memory_key and created:
            store.delete(MemorySlot.RULES)
            row.updated_by_id = ADMIN
            row.updated_by_email = "admin@example.com"
            access.session.add(row)
            access.session.commit()
        return row, created

    monkeypatch.setattr(
        SharedResourceAccess,
        "ensure_mutation_identity",
        clear_after_identity,
    )
    admin = _scoped(org_engine, ADMIN)
    projects.delete_project(project_id, admin, _principal(ADMIN, admin=True))

    assert (
        SharedResourceAccess(admin, _principal(ADMIN, admin=True))._find(
            PROJECT_MEMORY,
            memory_key,
        )
        is None
    )
    events = admin.exec(
        admin.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == PROJECT_MEMORY,
            SharedResourceMutation.resource_key == memory_key,
        )
    ).all()
    assert [event.action for event in events] == ["clear"]


@pytest.mark.asyncio
async def test_remote_peer_keeps_personal_memory_but_drops_shared_project_write(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("remote-memory")
    await MemoryService(alice, _principal(ALICE)).update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Keep the shared rule",
        project_id=project.id,
    )

    bob = _scoped(org_engine, BOB)
    applied = apply_turn_memory(
        bob.scope,
        project.path,
        [
            {"text": "Reply briefly", "kind": "always", "scope": "global"},
            {"text": "Replace the shared rule", "kind": "always", "scope": "project"},
        ],
        access=SharedResourceAccess(bob, _principal(BOB)),
        project_id=project.id,
    )
    assert applied == 1
    payload = build_turn_memory(bob.scope, project.path)
    assert "Reply briefly" in payload["global"]["rules"]
    assert "Keep the shared rule" in payload["project"]["rules"]
    assert "Replace the shared rule" not in payload["project"]["rules"]


def test_instructions_gate_does_not_gate_ordinary_project_files(org_engine):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("context-project")
    instructions_path = project_files._validated_project_path(".anton/anton.md")
    request = project_files._FileWriteRequest(content="Be concise")

    result = project_files.write_project_file(
        project.name,
        instructions_path,
        request,
        alice,
        _principal(ALICE),
    )
    assert result["attribution"].created_by.user_id == ALICE
    assert result["capabilities"].can_edit is True
    special = project_files.get_project_instructions(
        project.name,
        alice,
        _principal(ALICE),
    )["file"]
    direct = project_files.read_project_file(
        project.name,
        ".anton/anton.md",
        alice,
        _principal(ALICE),
    )
    assert special["attribution"].created_by.user_id == ALICE
    assert direct["attribution"].created_by.user_id == ALICE
    assert direct["capabilities"].can_edit is True
    normalized_alias = project_files.read_project_file(
        project.name,
        "notes/../.anton\\anton.md",
        alice,
        _principal(ALICE),
    )
    assert normalized_alias["attribution"].created_by.user_id == ALICE
    assert normalized_alias["capabilities"].can_edit is True
    assert set(jsonable_encoder(direct)["capabilities"]) == {
        "canEdit",
        "canDelete",
    }

    bob = _scoped(org_engine, BOB)
    with pytest.raises(HTTPException) as denied:
        project_files.write_project_file(
            project.name,
            instructions_path,
            project_files._FileWriteRequest(content="Take over"),
            bob,
            _principal(BOB),
        )
    assert denied.value.status_code == 403

    admin = _scoped(org_engine, ADMIN)
    admin_edit = project_files.write_project_file(
        project.name,
        instructions_path,
        project_files._FileWriteRequest(content="Admin revision"),
        admin,
        _principal(ADMIN, admin=True),
    )
    assert admin_edit["attribution"].created_by.user_id == ALICE
    assert admin_edit["attribution"].last_modified_by.user_id == ADMIN

    cleared = project_files.write_project_file(
        project.name,
        instructions_path,
        project_files._FileWriteRequest(content=""),
        alice,
        _principal(ALICE),
    )
    assert cleared["attribution"].created_by.user_id == ALICE
    instruction_events = alice.exec(
        alice.select(SharedResourceMutation).where(
            SharedResourceMutation.resource_kind == "project_instructions",
            SharedResourceMutation.resource_key == str(project.id),
        )
    ).all()
    assert {event.action for event in instruction_events} >= {
        "create",
        "update",
        "clear",
    }

    with pytest.raises(HTTPException) as peer_delete_denied:
        project_files.delete_project_file(
            project.name,
            instructions_path,
            bob,
            _principal(BOB),
        )
    assert peer_delete_denied.value.status_code == 403
    project_files.delete_project_file(
        project.name,
        instructions_path,
        alice,
        _principal(ALICE),
    )
    deleted = project_files.get_project_instructions(
        project.name,
        alice,
        _principal(ALICE),
    )["file"]
    assert deleted["synthetic"] is True
    assert deleted["attribution"].created_by is None

    ordinary = project_files.write_project_file(
        project.name,
        project_files._validated_project_path("notes/member.txt"),
        project_files._FileWriteRequest(content="Member-owned content"),
        bob,
        _principal(BOB),
    )
    assert ordinary["path"] == "notes/member.txt"

    legacy = ProjectService(alice).create_project("legacy-instructions")
    legacy_path = Path(legacy.path) / ".anton" / "anton.md"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("Historical author unknown")
    legacy_edit = project_files.write_project_file(
        legacy.name,
        instructions_path,
        project_files._FileWriteRequest(content="Admin-maintained legacy"),
        admin,
        _principal(ADMIN, admin=True),
    )
    assert legacy_edit["attribution"].created_by is None
    assert legacy_edit["attribution"].last_modified_by.user_id == ADMIN
    creator_edit = project_files.write_project_file(
        legacy.name,
        instructions_path,
        project_files._FileWriteRequest(content="Creator-maintained legacy"),
        alice,
        _principal(ALICE),
    )
    assert creator_edit["attribution"].created_by is None
    assert creator_edit["attribution"].last_modified_by.user_id == ALICE


@pytest.mark.asyncio
async def test_generic_project_files_cannot_bypass_canonical_memory_contract(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = projects.create_project(
        ProjectCreateRequest(name="reserved-memory-files"),
        alice,
        _principal(ALICE),
    )
    service = MemoryService(alice, _principal(ALICE))
    await service.update_memory(
        scope="project",
        category=MemorySlot.RULES,
        content="Canonical rules bytes",
        project_id=project["id"],
    )
    await service.update_memory(
        scope="project",
        category=MemorySlot.LESSONS,
        content="Canonical lessons bytes",
        project_id=project["id"],
    )
    store = ProjectMemoryStore(Path(project["path"]))
    before_bytes = {
        slot: store.read(slot) for slot in (MemorySlot.RULES, MemorySlot.LESSONS)
    }
    before_events = {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    }

    actors = [
        (alice, _principal(ALICE)),
        (_scoped(org_engine, ADMIN), _principal(ADMIN, admin=True)),
        (_scoped(org_engine, BOB), _principal(BOB)),
    ]
    for scoped, principal in actors:
        for slot in ("rules", "lessons"):
            path = project_files._validated_project_path(f".anton/memory/{slot}.md")
            with pytest.raises(HTTPException) as put_denied:
                project_files.write_project_file(
                    project["name"],
                    path,
                    project_files._FileWriteRequest(content="bypass bytes"),
                    scoped,
                    principal,
                )
            assert put_denied.value.status_code == 409
            assert "/api/v1/memory/" in put_denied.value.detail
            with pytest.raises(HTTPException) as delete_denied:
                project_files.delete_project_file(
                    project["name"],
                    path,
                    scoped,
                    principal,
                )
            assert delete_denied.value.status_code == 409
            assert "/api/v1/memory/" in delete_denied.value.detail

    alice.rollback()
    assert {
        slot: store.read(slot) for slot in (MemorySlot.RULES, MemorySlot.LESSONS)
    } == before_bytes
    assert {
        event.id for event in alice.exec(alice.select(SharedResourceMutation)).all()
    } == before_events


def test_generic_project_files_reject_protected_namespace_squatting_but_not_desktop(
    org_engine,
):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("reserved-namespace")
    before = set(Path(project.path).rglob("*"))
    for raw in (
        ".anton",
        ".anton/memory",
        ".anton/anton.md/child",
        ".anton/memory/rules.md/child",
        ".anton/memory/lessons.md/child",
    ):
        path = project_files._validated_project_path(raw)
        with pytest.raises(HTTPException) as put_denied:
            project_files.write_project_file(
                project.name,
                path,
                project_files._FileWriteRequest(content="squat"),
                alice,
                _principal(ALICE),
            )
        assert put_denied.value.status_code == 409
        with pytest.raises(HTTPException) as delete_denied:
            project_files.delete_project_file(
                project.name,
                path,
                alice,
                _principal(ALICE),
            )
        assert delete_denied.value.status_code == 409
    assert set(Path(project.path).rglob("*")) == before

    local = ScopedSession(Session(org_engine), LOCAL_SCOPE)
    local_project = ProjectService(local).create_project("desktop-memory-file")
    local_path = project_files._validated_project_path(".anton/memory/rules.md")
    written = project_files.write_project_file(
        local_project.name,
        local_path,
        project_files._FileWriteRequest(content="desktop bytes"),
        local,
        None,
    )
    assert written["path"] == ".anton/memory/rules.md"
    assert (Path(local_project.path) / written["path"]).read_text() == "desktop bytes"


def test_compat_move_rejects_reserved_anton_without_consuming_attachment(org_engine):
    alice = _scoped(org_engine, ALICE)
    project = ProjectService(alice).create_project("reserved-compat-move")
    attachment = FileService(alice).create_file_from_bytes(
        filename=".anton",
        content_type="application/octet-stream",
        data=b"must remain an attachment",
        purpose=attachment_purpose(str(uuid4())),
    )
    source = Path(attachment.path)

    with pytest.raises(HTTPException) as denied:
        compat_stubs.move_attachment_to_project(
            project.name,
            "session",
            attachment.id,
            alice,
        )

    assert denied.value.status_code == 409
    assert source.read_bytes() == b"must remain an attachment"
    assert FileService(alice).get_file_row(attachment.id).id == attachment.id
    assert not (Path(project.path) / ".anton").exists()
