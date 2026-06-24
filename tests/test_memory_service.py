from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import memory as memory_router
from cowork.common.settings.app_settings import AppSettings, MemorySettings
from cowork.db.session import get_session
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import ProjectMemoryStore, SharedMemoryStore
from cowork.models.project import Project
from cowork.schemas.memory import MemoryScope
from cowork.services.memory import MemoryService


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
    with Session(engine) as session:
        yield session


@pytest.fixture
def project(tmp_path, session):
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    project = Project(name="My Project", path=str(project_dir))
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
    store = SharedMemoryStore(root=memory_root)
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

    store = SharedMemoryStore(root=memory_root)
    assert store.read(MemorySlot.RULES).strip() == "Always use TypeScript"


@pytest.mark.asyncio
async def test_delete_global_memory(session, memory_root, memory_settings):
    store = SharedMemoryStore(root=memory_root)
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
    SharedMemoryStore(root=memory_root).write(MemorySlot.RULES, "global rule")

    response = client.get("/memory/")
    assert response.status_code == 200

    items = response.json()
    categories = {item["category"] for item in items if item["scope"] == "global"}
    assert categories == {"profile", "rules", "lessons"}
    rules = next(item for item in items if item["category"] == "rules")
    assert rules["content"].strip() == "global rule"


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

    store = SharedMemoryStore(root=memory_root)
    assert store.read(MemorySlot.LESSONS).strip() == "shared lesson"


def test_delete_endpoint(client, memory_root, memory_settings):
    SharedMemoryStore(root=memory_root).write(MemorySlot.PROFILE, "profile data")

    response = client.request(
        "DELETE",
        "/memory/",
        json={"scope": "global", "category": "profile"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert SharedMemoryStore(root=memory_root).read(MemorySlot.PROFILE) == ""


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
    project = Project(name="Filtered", path=str(project_dir))
    session.add(project)
    other_dir = tmp_path / "other-project"
    other_dir.mkdir()
    other = Project(name="Other", path=str(other_dir))
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
