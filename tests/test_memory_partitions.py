"""The memory API and the memory reset inside personal partitions.

A person's global tier is ``<shared>/<partition>/memory/users/<sub>/`` and each
of their project tiers sits under it at ``projects/<project id>/``, whether or
not projects are personal. The Memories listing returns only those tiers.
``reset_person_memory`` removes the whole folder. Memory left shared keeps
today's base tiers; the desktop and hosted org mode are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.principal import personal_partition_id
from cowork.schemas.memory import MemoryScope
from cowork.services.memory import (
    MemoryService,
    PersonMemoryResetError,
    reset_person_memory,
)

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
SUB_B = "8a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
WITHOUT_PROJECTS = ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS}
WITHOUT_MEMORY = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}


@pytest.fixture()
def env(tmp_path, monkeypatch, personal_deployment):
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "base-memory"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    import cowork.models.conversation  # noqa: F401
    import cowork.models.message  # noqa: F401
    import cowork.models.message_event  # noqa: F401
    import cowork.models.project  # noqa: F401

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    yield engine, tmp_path
    get_app_settings.cache_clear()


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


def _service(*, engine, scope: TenantScope) -> MemoryService:
    return MemoryService(ScopedSession(Session(engine), scope))


def _project(*, engine, scope: TenantScope, path: Path) -> UUID:
    from cowork.models.project import Project

    path.mkdir(parents=True, exist_ok=True)
    scoped = ScopedSession(Session(engine), scope)
    project = Project(name=path.name, path=str(path))
    scoped.add(project)
    scoped.commit()
    scoped.refresh(project)
    return project.id


def _person_tier(*, root: Path, partition: str, sub: str) -> Path:
    return root / "shared" / partition / "memory" / "users" / sub


def _general(*, engine, scope: TenantScope, root: Path) -> UUID:
    """The person's own ``general``, where their partition keeps it."""
    return _project(
        engine=engine, scope=scope, path=root / "shared" / scope.org_id / "projects" / "general"
    )


def _contents(items) -> set[str]:
    return {item.content.strip() for item in items if item.content.strip()}


async def _remember(*, engine, scope, project_id, rule: str, lesson: str) -> None:
    service = _service(engine=engine, scope=scope)
    await service.update_memory(scope=MemoryScope.global_, category=MemorySlot.RULES, content=rule)
    await service.update_memory(
        scope=MemoryScope.project, category=MemorySlot.LESSONS, content=lesson, project_id=project_id
    )


async def test_the_listing_holds_only_the_callers_own_tiers(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    bob = _person(partition=PARTITION_B, sub=SUB_B)
    alice_general = _general(engine=engine, scope=alice, root=root)
    bob_general = _general(engine=engine, scope=bob, root=root)
    await _remember(
        engine=engine, scope=alice, project_id=alice_general, rule="A: answer in pallets", lesson="A: P-1111"
    )
    await _remember(
        engine=engine, scope=bob, project_id=bob_general, rule="B: answer in cases", lesson="B: P-2222"
    )

    alice_items = _service(engine=engine, scope=alice).list_memory_sync()
    bob_items = _service(engine=engine, scope=bob).list_memory_sync()
    assert _contents(alice_items) == {"A: answer in pallets", "A: P-1111"}
    assert _contents(bob_items) == {"B: answer in cases", "B: P-2222"}

    tier = _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    assert (tier / "rules.md").read_text().strip() == "A: answer in pallets"
    assert (tier / "projects" / str(alice_general) / "lessons.md").read_text().strip() == "A: P-1111"
    # The project's own .anton/memory is not a person's tier.
    assert not (root / "shared" / PARTITION_A / "projects" / "general" / ".anton" / "memory").exists()


MEMORY_ONLY = frozenset({PersonalResource.MEMORY})


async def test_memory_only_keeps_each_persons_memory_their_own(env):
    """AC7: with memory the only personal resource and everything else shared,
    each person's turn reads and writes only its own memory/users/<sub> tier,
    and its Memories listing holds only its own items, even for a project both
    share."""
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=MEMORY_ONLY)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=MEMORY_ONLY)
    shared_project = _project(engine=engine, scope=alice, path=root / "projects" / "team")
    await _remember(engine=engine, scope=alice, project_id=shared_project, rule="A: rule", lesson="A: lesson")
    await _remember(engine=engine, scope=bob, project_id=shared_project, rule="B: rule", lesson="B: lesson")

    assert _contents(_service(engine=engine, scope=alice).list_memory_sync()) == {"A: rule", "A: lesson"}
    assert _contents(_service(engine=engine, scope=bob).list_memory_sync()) == {"B: rule", "B: lesson"}

    a_tier = _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    b_tier = _person_tier(root=root, partition=PARTITION_B, sub=SUB_B)
    assert (a_tier / "rules.md").read_text().strip() == "A: rule"
    assert (b_tier / "rules.md").read_text().strip() == "B: rule"
    assert (a_tier / "projects" / str(shared_project) / "lessons.md").read_text().strip() == "A: lesson"
    assert (b_tier / "projects" / str(shared_project) / "lessons.md").read_text().strip() == "B: lesson"
    # The shared project's own folder holds nobody's memory.
    assert not (root / "projects" / "team" / ".anton" / "memory").exists()


async def test_a_shared_projects_memory_is_still_per_person(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_PROJECTS)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_PROJECTS)
    shared_project = _project(engine=engine, scope=alice, path=root / "projects" / "team")
    await _remember(engine=engine, scope=alice, project_id=shared_project, rule="A: rule", lesson="A: lesson")

    bob_items = _service(engine=engine, scope=bob).list_memory_sync()
    assert _contents(bob_items) == set()
    assert shared_project in {item.project_id for item in bob_items}
    assert _contents(_service(engine=engine, scope=alice).list_memory_sync()) == {"A: rule", "A: lesson"}
    tier = _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    lessons = tier / "projects" / str(shared_project) / "lessons.md"
    assert lessons.read_text().strip() == "A: lesson"
    assert not (root / "projects" / "team" / ".anton" / "memory").exists()


async def test_shared_memory_keeps_the_base_tiers(env):
    """Memory left off the personal list stays shared, as on today's stack."""
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_MEMORY)
    project = _general(engine=engine, scope=alice, root=root)
    await _remember(
        engine=engine, scope=alice, project_id=project, rule="shared rule", lesson="shared lesson"
    )

    assert (root / "base-memory" / "rules.md").read_text().strip() == "shared rule"
    lessons = root / "shared" / PARTITION_A / "projects" / "general" / ".anton" / "memory" / "lessons.md"
    assert lessons.read_text().strip() == "shared lesson"
    assert "shared rule" in _contents(_service(engine=engine, scope=bob).list_memory_sync())


async def test_the_operator_view_reads_the_base_tiers(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    project = _general(engine=engine, scope=alice, root=root)
    await _remember(engine=engine, scope=alice, project_id=project, rule="A: rule", lesson="A: lesson")
    (root / "base-memory").mkdir(parents=True, exist_ok=True)
    (root / "base-memory" / "rules.md").write_text("deployment rule\n")

    # LOCAL_SCOPE's tiers are the base ones; a person's memory is not among them.
    assert _contents(_service(engine=engine, scope=LOCAL_SCOPE).list_memory_sync()) == {"deployment rule"}


async def test_reset_removes_one_persons_whole_memory(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    bob = _person(partition=PARTITION_B, sub=SUB_B)
    alice_general = _general(engine=engine, scope=alice, root=root)
    bob_general = _general(engine=engine, scope=bob, root=root)
    await _remember(engine=engine, scope=alice, project_id=alice_general, rule="A: rule", lesson="A: lesson")
    await _remember(engine=engine, scope=bob, project_id=bob_general, rule="B: rule", lesson="B: lesson")

    result = reset_person_memory(alice)

    assert result.removed is True
    assert result.root == _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    assert not result.root.exists()
    assert _contents(_service(engine=engine, scope=alice).list_memory_sync()) == set()
    assert _contents(_service(engine=engine, scope=bob).list_memory_sync()) == {"B: rule", "B: lesson"}
    assert reset_person_memory(alice).removed is False


async def test_reset_unlinks_a_planted_link_without_following_it(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    bob = _person(partition=PARTITION_B, sub=SUB_B)
    bob_general = _general(engine=engine, scope=bob, root=root)
    await _remember(engine=engine, scope=bob, project_id=bob_general, rule="B: rule", lesson="B: lesson")
    alice_tier = _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    alice_tier.parent.mkdir(parents=True)
    bob_tier = _person_tier(root=root, partition=PARTITION_B, sub=SUB_B)
    alice_tier.symlink_to(bob_tier, target_is_directory=True)

    assert reset_person_memory(alice).removed is True
    assert not alice_tier.exists() and not alice_tier.is_symlink()
    assert _contents(_service(engine=engine, scope=bob).list_memory_sync()) == {"B: rule", "B: lesson"}


@pytest.mark.parametrize(
    "scope",
    [
        LOCAL_SCOPE,
        TenantScope(
            personal=True, org_id=PARTITION_A, user_id=None, personal_resources=ALL_PERSONAL_RESOURCES
        ),
        TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=WITHOUT_MEMORY),
    ],
    ids=["operator-view", "no-user", "memory-shared"],
)
def test_reset_refuses_memory_that_is_not_one_persons(env, scope):
    engine, root = env
    (root / "base-memory").mkdir(parents=True)
    (root / "base-memory" / "rules.md").write_text("everyone's rule\n")

    with pytest.raises(PersonMemoryResetError):
        reset_person_memory(scope)
    assert (root / "base-memory" / "rules.md").exists()


async def test_reset_in_org_mode_removes_only_the_members_global_tier(env, tmp_path):
    engine, root = env
    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    member = TenantScope(org_mode=True, org_id=org, user_id="alice")
    project_path = tmp_path / "org-project"
    (project_path / ".anton" / "memory").mkdir(parents=True)
    (project_path / ".anton" / "memory" / "lessons.md").write_text("team lesson\n")
    await _service(engine=engine, scope=member).update_memory(
        scope=MemoryScope.global_, category=MemorySlot.RULES, content="alice's rule"
    )

    result = reset_person_memory(member)

    assert result.removed is True and result.root == root / "shared" / org / "memory" / "users" / "alice"
    assert (project_path / ".anton" / "memory" / "lessons.md").read_text() == "team lesson\n"
