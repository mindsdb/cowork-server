"""The memory API and the memory reset inside personal partitions.

A person's global tier is ``<shared>/<partition>/memory/users/<sub>/`` and each
of their project tiers sits under it at ``projects/<project id>/``, whether or
not projects are personal. The Memories listing returns only those tiers.
``reset_person_memory`` removes the whole folder. Memory left shared is one
deployment store: the base global tier, with each project's tier under it at
``projects/<project id>/``, and everyone's listing holds every tier there by
id alone. With memory off the listing holds nothing and stored bytes stay as
they are. The operator view reads that store where memory is shared, and each
project's own ``.anton/memory`` where it is per person. The desktop and hosted
org mode are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import (
    NestedProjectMemoryStore,
    ProjectMemoryStore,
    project_memory_store,
    project_tier_ids,
)
from cowork.principal import personal_partition_id
from cowork.schemas.memory import MemoryScope
from cowork.services.memory import (
    MemoryService,
    PersonMemoryResetError,
    clear_deployment_memory,
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


def _bytes_under(folder: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


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


async def test_shared_memory_is_one_deployment_store(env, monkeypatch):
    """Memory left off the personal list is the deployment's base tiers, with
    each project's tier under the base global one, never in the project
    folder's .anton/memory, which the agent's code can write."""
    engine, root = env
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_MEMORY)
    project = _general(engine=engine, scope=alice, root=root)
    await _remember(
        engine=engine, scope=alice, project_id=project, rule="shared rule", lesson="shared lesson"
    )

    assert (root / "base-memory" / "rules.md").read_text().strip() == "shared rule"
    lessons = root / "base-memory" / "projects" / str(project) / "lessons.md"
    assert lessons.read_text().strip() == "shared lesson"
    assert not (root / "shared" / PARTITION_A / "projects" / "general" / ".anton" / "memory").exists()
    assert not (root / "shared" / PARTITION_A / "memory").exists()
    # Bob reads both, Alice's private project's tier included.
    assert _contents(_service(engine=engine, scope=bob).list_memory_sync()) == {
        "shared rule",
        "shared lesson",
    }


def _shared_tier(root: Path, project_id: UUID | str) -> Path:
    return root / "base-memory" / "projects" / str(project_id)


async def test_the_shared_listing_holds_every_project_tier_by_id_alone(env, monkeypatch):
    """Bob's listing holds the tier of Alice's private project, which he can't
    open, by its id: no project row of hers is read, no name or path reaches
    the response, and he gets no capability on it. A folder in projects/ that
    is not a project id, or a link named like one, is no tier."""
    engine, root = env
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_MEMORY)
    private = _project(engine=engine, scope=alice, path=root / "shared" / PARTITION_A / "projects" / "secret-plans")
    bobs = _general(engine=engine, scope=bob, root=root)
    await _remember(engine=engine, scope=alice, project_id=private, rule="shared rule", lesson="pallets of 40")
    await _remember(engine=engine, scope=bob, project_id=bobs, rule="shared rule", lesson="ships on Fridays")
    (root / "base-memory" / "projects" / "notes").mkdir()
    (root / "base-memory" / "projects" / "notes" / "lessons.md").write_text("not a tier\n")
    outside = root / "outside" / "lessons.md"
    outside.parent.mkdir()
    outside.write_text("followed a link\n")
    linked = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
    _shared_tier(root, linked).symlink_to(outside.parent, target_is_directory=True)
    # Another spelling of a project id is no tier either.
    odd = uuid4()
    for spelling in (str(odd).upper(), odd.hex, "{" + str(odd) + "}"):
        _shared_tier(root, spelling).mkdir()
        (_shared_tier(root, spelling) / "lessons.md").write_text(f"spelled {spelling}\n")

    service = _service(engine=engine, scope=bob)

    def never(*_args, **_kwargs):
        raise AssertionError("the shared listing resolved a project row")

    monkeypatch.setattr(service, "_resolve_project", never)
    items = service.list_memory_sync()

    assert _contents(items) == {"shared rule", "pallets of 40", "ships on Fridays"}
    tiers = {item.project_id for item in items if item.scope == MemoryScope.project}
    assert tiers == {private, bobs}
    assert not any("spelled" in content for content in _contents(items))
    [alice_lesson] = [
        item for item in items if item.project_id == private and item.category == MemorySlot.LESSONS
    ]
    assert alice_lesson.capabilities.can_edit is False and alice_lesson.capabilities.can_delete is False
    assert alice_lesson.attribution.created_by is None
    [bob_lesson] = [
        item for item in items if item.project_id == bobs and item.category == MemorySlot.LESSONS
    ]
    assert bob_lesson.capabilities.can_edit is True
    dumped = repr([item.model_dump() for item in items])
    assert "secret-plans" not in dumped and str(root) not in dumped

    # The project_id filter keeps the global tier and that one project's tier.
    filtered = _service(engine=engine, scope=bob).list_memory_sync(project_id=private)
    assert {item.project_id for item in filtered} == {None, private}
    assert _contents(filtered) == {"shared rule", "pallets of 40"}


def test_a_project_tier_is_named_by_a_canonical_project_id_alone(tmp_path):
    canonical = uuid4()
    projects = tmp_path / "projects"
    # Each other spelling names a different id, so a filesystem that ignores
    # letter case keeps them apart.
    for name in (
        str(canonical),
        str(uuid4()).upper(),
        uuid4().hex,
        "{" + str(uuid4()) + "}",
        f"urn:uuid:{uuid4()}",
        "notes",
    ):
        (projects / name).mkdir(parents=True)

    assert project_tier_ids(global_root=tmp_path) == [canonical]
    assert project_tier_ids(global_root=tmp_path / "missing") == []


async def test_the_shared_listing_gives_the_caller_empty_slots_for_its_projects_with_no_tier(
    env, monkeypatch
):
    """The tiers on disk and the caller's own projects together: Bob gets empty,
    editable slots for a project of his that has no tier yet, and listing them
    makes no folder."""
    engine, root = env
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_MEMORY)
    alices = _general(engine=engine, scope=alice, root=root)
    await _remember(engine=engine, scope=alice, project_id=alices, rule="shared rule", lesson="pallets")
    fresh = _project(engine=engine, scope=bob, path=root / "shared" / PARTITION_B / "projects" / "fresh")
    assert not _shared_tier(root, fresh).exists()

    items = _service(engine=engine, scope=bob).list_memory_sync()

    mine = [item for item in items if item.project_id == fresh]
    assert sorted(item.category for item in mine) == sorted([MemorySlot.RULES, MemorySlot.LESSONS])
    assert all(item.content == "" and item.capabilities.can_edit is True for item in mine)
    assert alices in {item.project_id for item in items}
    assert not _shared_tier(root, fresh).exists()
    filtered = _service(engine=engine, scope=bob).list_memory_sync(project_id=fresh)
    assert {item.project_id for item in filtered} == {None, fresh}
    assert _contents(filtered) == {"shared rule"}


async def test_a_link_in_place_of_the_shared_project_tiers_is_refused(env, monkeypatch):
    """The listing and the clear both refuse a link where projects/ belongs,
    and the clear removes nothing before it refuses."""
    engine, root = env
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=WITHOUT_MEMORY)
    base = root / "base-memory"
    base.mkdir()
    (base / "rules.md").write_text("shared rule\n")
    outside = root / "outside"
    (outside / str(uuid4())).mkdir(parents=True)
    next(outside.iterdir()).joinpath("lessons.md").write_text("behind a link\n")
    (base / "projects").symlink_to(outside, target_is_directory=True)
    before = _bytes_under(outside)

    with pytest.raises(OSError):
        _service(engine=engine, scope=bob).list_memory_sync()
    with pytest.raises(OSError):
        clear_deployment_memory()

    assert (base / "rules.md").read_text() == "shared rule\n"
    assert (base / "projects").is_symlink()
    assert _bytes_under(outside) == before


async def test_memory_off_lists_nothing_and_leaves_every_tier_as_it_was(env, monkeypatch):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    alice_general = _general(engine=engine, scope=alice, root=root)
    await _remember(engine=engine, scope=alice, project_id=alice_general, rule="A: rule", lesson="A: lesson")
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    shared_alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    await _remember(
        engine=engine, scope=shared_alice, project_id=alice_general, rule="shared rule", lesson="shared lesson"
    )
    assert (_shared_tier(root, alice_general) / "lessons.md").read_text().strip() == "shared lesson"
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(ALL_PERSONAL_RESOURCES)))
    get_app_settings.cache_clear()
    before = _bytes_under(root)

    monkeypatch.setenv("MEMORY_ENABLED", "false")
    for scope in (alice, shared_alice, LOCAL_SCOPE):
        assert _service(engine=engine, scope=scope).list_memory_sync() == []
    assert _bytes_under(root) == before

    monkeypatch.setenv("MEMORY_ENABLED", "true")
    assert _contents(_service(engine=engine, scope=alice).list_memory_sync()) == {"A: rule", "A: lesson"}
    assert _bytes_under(root) == before


async def test_switching_scope_moves_and_merges_nothing(env, monkeypatch):
    """Per person, then shared, then per person again: each store keeps its
    own bytes, and each listing reads only the store its scope names."""
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    project = _general(engine=engine, scope=alice, root=root)
    await _remember(engine=engine, scope=alice, project_id=project, rule="A: rule", lesson="A: pallets")
    personal_tier = _person_tier(root=root, partition=PARTITION_A, sub=SUB_A)
    personal_bytes = _bytes_under(personal_tier)

    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    shared_alice = _person(partition=PARTITION_A, sub=SUB_A, resources=WITHOUT_MEMORY)
    assert _contents(_service(engine=engine, scope=shared_alice).list_memory_sync()) == set()
    await _remember(engine=engine, scope=shared_alice, project_id=project, rule="P-1111", lesson="shared")
    assert _contents(_service(engine=engine, scope=shared_alice).list_memory_sync()) == {"P-1111", "shared"}
    assert _bytes_under(personal_tier) == personal_bytes
    shared_bytes = _bytes_under(root / "base-memory")

    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(ALL_PERSONAL_RESOURCES)))
    get_app_settings.cache_clear()
    assert _contents(_service(engine=engine, scope=alice).list_memory_sync()) == {"A: rule", "A: pallets"}
    assert _bytes_under(personal_tier) == personal_bytes
    assert _bytes_under(root / "base-memory") == shared_bytes


async def test_hosted_org_mode_lists_stored_memory_whatever_memory_enabled_says(env, monkeypatch, tmp_path):
    """Hosted keeps its behavior: the off listing belongs to the deployment
    that trusts identity headers, and a project tier stays in the project's
    own folder."""
    engine, root = env
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    get_app_settings.cache_clear()
    org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    member = TenantScope(org_mode=True, org_id=org, user_id="alice")
    project = _project(engine=engine, scope=member, path=tmp_path / "org-project")
    (tmp_path / "org-project" / ".anton" / "memory").mkdir(parents=True)
    (tmp_path / "org-project" / ".anton" / "memory" / "lessons.md").write_text("team lesson\n")
    await _service(engine=engine, scope=member).update_memory(
        scope=MemoryScope.global_, category=MemorySlot.RULES, content="alice's rule"
    )

    items = _service(engine=engine, scope=member).list_memory_sync()

    assert _contents(items) == {"alice's rule", "team lesson"}
    assert project in {item.project_id for item in items}


async def test_the_operator_view_reads_the_base_tiers(env):
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    project = _general(engine=engine, scope=alice, root=root)
    await _remember(engine=engine, scope=alice, project_id=project, rule="A: rule", lesson="A: lesson")
    (root / "base-memory").mkdir(parents=True, exist_ok=True)
    (root / "base-memory" / "rules.md").write_text("deployment rule\n")

    # LOCAL_SCOPE's tiers are the base ones; a person's memory is not among them.
    assert _contents(_service(engine=engine, scope=LOCAL_SCOPE).list_memory_sync()) == {"deployment rule"}


async def test_the_operator_view_keeps_the_projects_own_folder_where_memory_is_per_person(env):
    """Memory per person: the operator view's project tier is the project's
    .anton/memory, as before memory could be shared. A folder in the
    deployment's projects/ is no tier of its, and Alice's listing holds none of
    the operator's lessons."""
    engine, root = env
    alice = _person(partition=PARTITION_A, sub=SUB_A)
    project_path = root / "operator-project"
    project = _project(engine=engine, scope=LOCAL_SCOPE, path=project_path)
    _general(engine=engine, scope=alice, root=root)

    store = project_memory_store(scope=LOCAL_SCOPE, project_path=project_path, project_id=project)
    assert type(store) is ProjectMemoryStore
    assert store.root == project_path / ".anton" / "memory"
    await _service(engine=engine, scope=LOCAL_SCOPE).update_memory(
        scope=MemoryScope.project, category=MemorySlot.LESSONS, content="operator lesson", project_id=project
    )
    assert (project_path / ".anton" / "memory" / "lessons.md").read_text().strip() == "operator lesson"
    items = _service(engine=engine, scope=LOCAL_SCOPE).list_memory_sync()
    assert "operator lesson" in _contents(items)
    assert not (root / "base-memory" / "projects").exists()

    stray = _shared_tier(root, uuid4())
    stray.mkdir(parents=True)
    (stray / "lessons.md").write_text("stray lesson\n")
    items = _service(engine=engine, scope=LOCAL_SCOPE).list_memory_sync()
    assert "stray lesson" not in _contents(items)
    assert "operator lesson" not in _contents(_service(engine=engine, scope=alice).list_memory_sync())


async def test_the_operator_view_reads_the_deployment_tier_where_memory_is_shared(env, monkeypatch):
    engine, root = env
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(WITHOUT_MEMORY)))
    get_app_settings.cache_clear()
    project_path = root / "operator-project"
    project = _project(engine=engine, scope=LOCAL_SCOPE, path=project_path)

    store = project_memory_store(scope=LOCAL_SCOPE, project_path=project_path, project_id=project)
    assert type(store) is NestedProjectMemoryStore and store.root == _shared_tier(root, project)
    await _service(engine=engine, scope=LOCAL_SCOPE).update_memory(
        scope=MemoryScope.project, category=MemorySlot.LESSONS, content="operator lesson", project_id=project
    )

    assert (_shared_tier(root, project) / "lessons.md").read_text().strip() == "operator lesson"
    assert not (project_path / ".anton" / "memory").exists()
    items = _service(engine=engine, scope=LOCAL_SCOPE).list_memory_sync()
    assert "operator lesson" in _contents(items)
    assert project in {item.project_id for item in items}


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
