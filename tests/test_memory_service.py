from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import memory as memory_router
from cowork.common.settings.app_settings import (
    AppSettings,
    MemorySettings,
    get_app_settings,
)
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import ProjectMemoryStore, GlobalMemoryStore
from cowork.models.project import Project
from cowork.principal import Principal
from cowork.schemas.memory import MemoryScope
from cowork.services import memory as memory_service_module
from cowork.services.memory import MemoryService, apply_turn_memory, build_turn_memory
from cowork.services.shared_resources import SharedResourceAccess


@pytest.fixture
def memory_root(tmp_path):
    return tmp_path / "memory"


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    # MemoryService now takes a ScopedSession. In local mode (LOCAL_SCOPE) it
    # neither filters nor stamps, so add/commit/refresh and the service behave
    # exactly as with the raw session — the wrap is transparent here.
    with Session(engine) as raw:
        yield ScopedSession(raw, LOCAL_SCOPE)


@pytest.fixture
def project(tmp_path, session):
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    project = Project(name="My Project", path=str(project_dir), is_active=True)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@pytest.fixture
def memory_settings(memory_root, monkeypatch):
    def _settings() -> AppSettings:
        return AppSettings(memory=MemorySettings(root_dir=str(memory_root)))

    monkeypatch.setattr("cowork.harnesses.memory.store.get_app_settings", _settings)


@pytest.fixture
def client(engine, memory_settings):
    app = FastAPI()
    app.include_router(memory_router.router, prefix="/memory")

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_list_memory_returns_all_global_slots(session, memory_root, memory_settings):
    store = GlobalMemoryStore(root=memory_root)
    store.write(MemorySlot.PROFILE, "user prefs")

    items = await MemoryService(session).list_memory()
    global_slots = {item.category for item in items if item.scope == MemoryScope.global_}

    assert global_slots == {MemorySlot.PROFILE, MemorySlot.RULES, MemorySlot.LESSONS}
    profile = next(item for item in items if item.category == MemorySlot.PROFILE)
    assert profile.content.strip() == "user prefs"


@pytest.mark.asyncio
async def test_update_global_memory(session, memory_root, memory_settings):
    service = MemoryService(session)
    await service.update_memory(
        scope=MemoryScope.global_,
        category=MemorySlot.RULES,
        content="Always use TypeScript",
    )

    store = GlobalMemoryStore(root=memory_root)
    assert store.read(MemorySlot.RULES).strip() == "Always use TypeScript"


@pytest.mark.asyncio
async def test_delete_global_memory(session, memory_root, memory_settings):
    store = GlobalMemoryStore(root=memory_root)
    store.write(MemorySlot.LESSONS, "lesson one")

    await MemoryService(session).delete_memory(
        scope=MemoryScope.global_,
        category=MemorySlot.LESSONS,
    )

    assert store.read(MemorySlot.LESSONS) == ""


@pytest.mark.asyncio
async def test_project_memory_rules_and_lessons(session, project, memory_settings):
    service = MemoryService(session)

    await service.update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        content="project rule",
        project_id=project.id,
    )

    store = ProjectMemoryStore(Path(project.path))
    assert store.read(MemorySlot.RULES).strip() == "project rule"


@pytest.mark.asyncio
async def test_profile_rejected_for_project_scope(session, project, memory_settings):
    with pytest.raises(ValueError, match="not supported for project-scoped memory"):
        await MemoryService(session).update_memory(
            scope=MemoryScope.project,
            category=MemorySlot.PROFILE,
            content="should fail",
            project_id=project.id,
        )


def test_get_list_endpoint(client, memory_root, memory_settings):
    GlobalMemoryStore(root=memory_root).write(MemorySlot.RULES, "global rule")

    response = client.get("/memory/")
    assert response.status_code == 200

    items = response.json()
    categories = {item["category"] for item in items if item["scope"] == "global"}
    assert categories == {"profile", "rules", "lessons"}
    rules = next(item for item in items if item["category"] == "rules")
    assert rules["content"].strip() == "global rule"
    assert rules["attribution"] == {
        "createdBy": None,
        "lastModifiedBy": None,
        "lastModifiedAt": None,
    }
    assert rules["capabilities"] == {"canEdit": True, "canDelete": True}


def test_put_endpoint(client, memory_root, memory_settings):
    response = client.put(
        "/memory/",
        json={
            "scope": "global",
            "category": "lessons",
            "content": "shared lesson",
        },
    )
    assert response.status_code == 200
    assert response.json()["category"] == "lessons"

    store = GlobalMemoryStore(root=memory_root)
    assert store.read(MemorySlot.LESSONS).strip() == "shared lesson"


def test_delete_endpoint(client, memory_root, memory_settings):
    GlobalMemoryStore(root=memory_root).write(MemorySlot.PROFILE, "profile data")

    response = client.request(
        "DELETE",
        "/memory/",
        json={"scope": "global", "category": "profile"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert GlobalMemoryStore(root=memory_root).read(MemorySlot.PROFILE) == ""


def test_put_project_profile_returns_400(client, session, project, memory_settings):
    response = client.put(
        "/memory/",
        json={
            "scope": "project",
            "category": "profile",
            "content": "nope",
            "project_id": str(project.id),
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_memory_filters_by_project_id(session, tmp_path, memory_settings):
    project_dir = tmp_path / "filtered-project"
    project_dir.mkdir()
    project = Project(name="Filtered", path=str(project_dir), is_active=True)
    session.add(project)
    other_dir = tmp_path / "other-project"
    other_dir.mkdir()
    other = Project(name="Other", path=str(other_dir), is_active=True)
    session.add(other)
    session.commit()
    session.refresh(project)
    session.refresh(other)

    service = MemoryService(session)
    await service.update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.LESSONS,
        content="project lesson",
        project_id=project.id,
    )
    await service.update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.LESSONS,
        content="other lesson",
        project_id=other.id,
    )

    filtered = await service.list_memory(project_id=project.id)
    project_items = [item for item in filtered if item.scope == MemoryScope.project]
    assert len(project_items) == 2
    assert all(item.project_id == project.id for item in project_items)
    lessons = next(item for item in project_items if item.category == MemorySlot.LESSONS)
    assert lessons.content.strip() == "project lesson"


# org mode: a project slot is the team's and keeps its first author, while global
# memory stays one member's own

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def org_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
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


def _org_scope(user_id: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=ORG, user_id=user_id)


def _org_principal(user_id: str) -> Principal:
    return Principal(user_id=user_id, org_id=ORG, email="member@example.com")


def _org_session(engine, user_id: str) -> ScopedSession:
    return ScopedSession(Session(engine), _org_scope(user_id))


def _org_memory(engine, user_id: str) -> MemoryService:
    return MemoryService(_org_session(engine, user_id), _org_principal(user_id))


def _org_access(engine, user_id: str) -> SharedResourceAccess:
    return SharedResourceAccess(_org_session(engine, user_id), _org_principal(user_id))


def _org_project(engine, path: Path) -> UUID:
    (path / ".anton" / "memory").mkdir(parents=True, exist_ok=True)
    scoped = _org_session(engine, ALICE)
    project = Project(name=path.name, path=str(path), is_active=True)
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    return project.id


@pytest.mark.asyncio
async def test_deleting_a_symlinked_project_slot_unlinks_the_link(org_engine, tmp_path):
    """A link planted over a slot is unlinked for its author and refused to
    everyone else. It is never followed, and never a failed request."""
    project_dir = tmp_path / "squatted-project"
    project_id = _org_project(org_engine, project_dir)
    await _org_memory(org_engine, ALICE).update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        content="Deploy on green only",
        project_id=project_id,
    )

    outsider = tmp_path / "another-tenant-rules.md"
    outsider.write_text("another tenant's bytes")
    slot = project_dir / ".anton" / "memory" / "rules.md"
    slot.unlink()
    slot.symlink_to(outsider)

    squatted = await _org_memory(org_engine, BOB).get_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        project_id=project_id,
    )
    assert squatted.content == ""
    assert squatted.capabilities.can_delete is False

    with pytest.raises(HTTPException) as denied:
        await _org_memory(org_engine, BOB).delete_memory(
            scope=MemoryScope.project,
            category=MemorySlot.RULES,
            project_id=project_id,
        )
    assert denied.value.status_code == 403
    assert slot.is_symlink()

    await _org_memory(org_engine, ALICE).delete_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        project_id=project_id,
    )
    assert not slot.is_symlink()
    assert not slot.exists()
    assert outsider.read_text() == "another tenant's bytes"


def test_turn_memory_keeps_global_entries_when_a_project_entry_is_refused(
    org_engine,
    tmp_path,
):
    project_dir = tmp_path / "refused-project"
    project_id = _org_project(org_engine, project_dir)
    _org_memory(org_engine, ALICE).update_memory_sync(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        content="Alice's shared rule",
        project_id=project_id,
    )

    bob = _org_scope(BOB)
    applied = apply_turn_memory(
        bob,
        str(project_dir),
        [
            {"text": "Replace the shared rule", "kind": "always", "scope": "project"},
            {"text": "Reply briefly", "kind": "always", "scope": "global"},
        ],
        access=_org_access(org_engine, BOB),
        project_id=project_id,
    )

    assert applied == 1
    payload = build_turn_memory(bob, str(project_dir))
    assert "Reply briefly" in payload["global"]["rules"]
    assert "Alice's shared rule" in payload["project"]["rules"]
    assert "Replace the shared rule" not in payload["project"]["rules"]


def test_turn_memory_keeps_global_entries_when_the_project_half_fails(
    org_engine,
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "failing-project"
    project_id = _org_project(org_engine, project_dir)

    def fail_setup(*_args, **_kwargs):
        raise RuntimeError("project memory setup failed")

    monkeypatch.setattr(
        memory_service_module,
        "_project_slot_coordination",
        fail_setup,
    )

    alice = _org_scope(ALICE)
    with pytest.raises(RuntimeError, match="project memory setup failed"):
        apply_turn_memory(
            alice,
            str(project_dir),
            [
                {
                    "text": "Shared rule that never lands",
                    "kind": "always",
                    "scope": "project",
                },
                {
                    "text": "Personal rule that survives",
                    "kind": "always",
                    "scope": "global",
                },
            ],
            access=_org_access(org_engine, ALICE),
            project_id=project_id,
        )

    assert "Personal rule that survives" in build_turn_memory(alice)["global"]["rules"]


@pytest.mark.asyncio
async def test_clearing_a_project_slot_keeps_its_first_author(org_engine, tmp_path):
    project_dir = tmp_path / "cleared-project"
    project_id = _org_project(org_engine, project_dir)
    await _org_memory(org_engine, ALICE).update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        content="Deploy on green only",
        project_id=project_id,
    )

    emptied = await _org_memory(org_engine, ALICE).update_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        content="",
        project_id=project_id,
    )
    assert emptied.attribution.created_by.user_id == ALICE
    with pytest.raises(HTTPException) as denied_after_empty:
        await _org_memory(org_engine, BOB).update_memory(
            scope=MemoryScope.project,
            category=MemorySlot.RULES,
            content="Bob claims the emptied slot",
            project_id=project_id,
        )
    assert denied_after_empty.value.status_code == 403

    await _org_memory(org_engine, ALICE).delete_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        project_id=project_id,
    )
    deleted = await _org_memory(org_engine, BOB).get_memory(
        scope=MemoryScope.project,
        category=MemorySlot.RULES,
        project_id=project_id,
    )
    assert deleted.content == ""
    assert deleted.attribution.created_by.user_id == ALICE
    assert deleted.capabilities.can_edit is False
    with pytest.raises(HTTPException) as denied_after_delete:
        await _org_memory(org_engine, BOB).update_memory(
            scope=MemoryScope.project,
            category=MemorySlot.RULES,
            content="Bob claims the deleted slot",
            project_id=project_id,
        )
    assert denied_after_delete.value.status_code == 403

    # A cloud turn's automatic write answers to the same author.
    assert (
        apply_turn_memory(
            _org_scope(BOB),
            str(project_dir),
            [{"text": "Bob's turn rule", "kind": "always", "scope": "project"}],
            access=_org_access(org_engine, BOB),
            project_id=project_id,
        )
        == 0
    )
    assert not (project_dir / ".anton" / "memory" / "rules.md").exists()
