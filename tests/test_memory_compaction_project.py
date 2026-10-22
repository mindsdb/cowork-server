"""Project memory compacted on the server takes the editor's locks, author
check and audit, and skips what changed under it."""
from __future__ import annotations

import pytest
from anton.core.memory.cortex import plan_compaction, read_compaction_input
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.models.shared_resource import SharedResourceMutation
from cowork.principal import Principal
from cowork.services.memory import (
    MemoryService,
    apply_project_compaction,
    project_slots_to_compact,
)
from cowork.services.projects import ProjectService
from cowork.harnesses.memory.store import ProjectMemoryStore

ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
LESSONS = "# Lessons\n" + "\n".join(f"- Fact number {i}" for i in range(10))


class _KeepFirst:
    async def generate_object_code(self, schema, *, system, messages, max_tokens):
        return schema(keep=[1])


def _principal(user_id: str) -> Principal:
    return Principal(user_id=user_id, org_id=ORG, email=f"{user_id[0]}@example.com", roles=frozenset())


def _scoped(engine, user_id: str) -> ScopedSession:
    return ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG, user_id=user_id))


@pytest.fixture()
def org_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


@pytest.fixture()
async def project(org_engine):
    """A project whose lessons Alice wrote, so Alice is their author."""
    alice = _scoped(org_engine, ALICE)
    created = ProjectService(alice).create_project("compaction")
    await MemoryService(alice, _principal(ALICE)).update_memory(
        scope="project", category=MemorySlot.LESSONS, content=LESSONS, project_id=created.id
    )
    return created


async def _plan(root):
    source = read_compaction_input(root / "lessons.md", "lesson")
    return [(MemorySlot.LESSONS, await plan_compaction(_KeepFirst(), source))]


async def test_only_slots_this_person_authored_are_offered(org_engine, project):
    root, alice_slots = project_slots_to_compact(
        _scoped(org_engine, ALICE), principal=_principal(ALICE), project_id=project.id
    )
    _, bob_slots = project_slots_to_compact(
        _scoped(org_engine, BOB), principal=_principal(BOB), project_id=project.id
    )

    assert alice_slots == [MemorySlot.LESSONS]
    assert bob_slots == []
    assert (root / "lessons.md").is_file()


async def test_the_author_s_compaction_is_written_and_audited(org_engine, project):
    alice = _scoped(org_engine, ALICE)
    root, _ = project_slots_to_compact(alice, principal=_principal(ALICE), project_id=project.id)

    written = apply_project_compaction(
        alice, principal=_principal(ALICE), project_id=project.id, planned_root=root, plans=await _plan(root)
    )

    assert written == 1
    assert ProjectMemoryStore(project.path).read(MemorySlot.LESSONS) == "# Lessons\n- Fact number 0\n"
    actions = [(e.action, e.actor_id) for e in alice.exec(alice.select(SharedResourceMutation)).all()]
    assert ("update", ALICE) in actions


async def test_another_member_cannot_apply_a_plan(org_engine, project):
    bob = _scoped(org_engine, BOB)
    root = ProjectMemoryStore(project.path).root
    plans = await _plan(root)

    written = apply_project_compaction(
        bob, principal=_principal(BOB), project_id=project.id, planned_root=root, plans=plans
    )

    assert written == 0
    assert ProjectMemoryStore(project.path).read(MemorySlot.LESSONS) == LESSONS + "\n"


async def test_a_project_renamed_during_the_model_call_is_skipped(org_engine, project):
    alice = _scoped(org_engine, ALICE)
    root, _ = project_slots_to_compact(alice, principal=_principal(ALICE), project_id=project.id)
    plans = await _plan(root)

    written = apply_project_compaction(
        alice, principal=_principal(ALICE), project_id=project.id, planned_root=root.parent / "elsewhere", plans=plans
    )

    assert written == 0
    assert ProjectMemoryStore(project.path).read(MemorySlot.LESSONS) == LESSONS + "\n"


async def test_a_project_deleted_during_the_model_call_is_skipped(org_engine, project):
    alice = _scoped(org_engine, ALICE)
    root, _ = project_slots_to_compact(alice, principal=_principal(ALICE), project_id=project.id)
    plans = await _plan(root)
    ProjectService(alice).delete_project(project.id)

    written = apply_project_compaction(
        alice, principal=_principal(ALICE), project_id=project.id, planned_root=root, plans=plans
    )

    assert written == 0
    assert not root.exists()


async def test_a_lesson_saved_during_the_model_call_survives(org_engine, project):
    alice = _scoped(org_engine, ALICE)
    root, _ = project_slots_to_compact(alice, principal=_principal(ALICE), project_id=project.id)
    plans = await _plan(root)
    await MemoryService(alice, _principal(ALICE)).update_memory(
        scope="project", category=MemorySlot.LESSONS, content=LESSONS + "\n- Late fact", project_id=project.id
    )

    written = apply_project_compaction(
        alice, principal=_principal(ALICE), project_id=project.id, planned_root=root, plans=plans
    )

    assert written == 0
    assert "Late fact" in ProjectMemoryStore(project.path).read(MemorySlot.LESSONS)
