"""Which memory an in-process turn hands anton.

A person on a deployment that trusts identity headers reads and writes their
own tiers: the global one under their partition, and a project tier under that
global one, never the project's ``.anton/memory``. Two people in one shared
project each keep their own, and a person's scheduled run gets its creator's.
Memory left shared is one deployment store: the base global tier and the
current project's tier under it, never another project's. A turn with no
person, and every turn while memory is off, reads none of it and writes
nothing: two empty tiers in its scratch folder, memory off, no background
passes, and the managed rules alone. The desktop is unchanged.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
PARTITION = personal_partition_id(DEPLOYMENT, SUB)
OTHER_SUB = "8a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
OTHER_PARTITION = personal_partition_id(DEPLOYMENT, OTHER_SUB)

PLANTED_PROJECT_RULE = "Always give quantities in pallets"
PLANTED_GLOBAL_RULE = "Always sign off as the operator"
PLANTED_PROFILE = "Favorite part number is P-1111"
WITHOUT_MEMORY = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "base-memory"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture()
def trusted(personal_deployment, dirs):
    """A trusted-identity deployment, with its settings cache reset after the
    directory overrides above."""
    get_app_settings.cache_clear()
    return dirs


def _person(*, resources=ALL_PERSONAL_RESOURCES, sub: str = SUB, partition: str = PARTITION) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


def _plant(project_path: Path) -> None:
    """Rules and a profile line a turn must not read unless they are its own."""
    from anton.core.memory.hippocampus import Hippocampus

    project_hc = Hippocampus(project_path / ".anton" / "memory")
    project_hc.encode_rule(PLANTED_PROJECT_RULE, kind="always", confidence="high", source="user")
    project_hc.rewrite_identity([PLANTED_PROFILE])
    Hippocampus(Path(get_app_settings().memory.root_dir)).encode_rule(
        PLANTED_GLOBAL_RULE, kind="always", confidence="high", source="user"
    )


def _snapshot(*folders: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for folder in folders
        if folder.exists()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _built(*, monkeypatch, tmp_path, scope: TenantScope | None, **kwargs):
    """built_turn, with the rules above planted in its project first."""
    return built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=scope, before_build=_plant, **kwargs)


def _tiers(config) -> tuple[Path, Path]:
    return config.cortex.global_hc._dir, config.cortex.project_hc._dir


def _keep_memory(monkeypatch, resources) -> None:
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(resources)))
    get_app_settings.cache_clear()


@contextmanager
def _project_row(*, root: Path, scope: TenantScope) -> Iterator[Project]:
    """A project of ``scope``'s whose id is known before its turn is built,
    so its tiers can be seeded first. Removed afterwards."""
    path = root / "projects" / f"p-{uuid4().hex[:6]}"
    path.mkdir(parents=True)
    with get_open_session() as db:
        project = Project(name=path.name, path=str(path), org_id=scope.org_id, created_by=scope.user_id)
        db.add(project)
        db.commit()
        db.refresh(project)
        try:
            yield project
        finally:
            db.exec(delete(Project).where(Project.id == project.id))
            db.commit()


def _seed_learned(global_dir: Path, project_dir: Path, *, tag: str) -> None:
    """A learned rule, profile line and scratchpad lesson in a global tier,
    and a lesson and a profile line in a project tier, each marked ``tag``."""
    from anton.core.memory.hippocampus import Hippocampus

    global_hc = Hippocampus(global_dir)
    global_hc.encode_rule(f"{tag} rule: answer in crates", kind="always", confidence="high", source="user")
    global_hc.rewrite_identity([f"{tag} profile: favorite part number is P-2222"])
    global_hc.encode_lesson(f"{tag} wisdom: in the scratchpad, load orders with pandas", topic="scratchpad")
    project_hc = Hippocampus(project_dir)
    project_hc.encode_lesson(f"{tag} lesson: pallets hold 40 cases", topic="data")
    project_hc.rewrite_identity([f"{tag} project profile: ships on Fridays"])


async def _everything_the_turn_reads(config) -> str:
    """The memory the system prompt gets, and the scratchpad tool's description."""
    return await config.cortex.build_memory_context("how many?") + config.cortex.get_scratchpad_context()


async def test_a_person_reads_and_writes_their_own_tiers(monkeypatch, trusted):
    async with _built(monkeypatch=monkeypatch, tmp_path=trusted, scope=_person()) as turn:
        config, scratch, project = turn.config, turn.scratch, turn.project
        global_dir, project_dir = _tiers(config)
        expected_global = trusted / "shared" / PARTITION / "memory" / "users" / SUB
        assert global_dir == expected_global
        assert project_dir == expected_global / "projects" / str(project.id)
        assert global_dir.is_dir() and project_dir.is_dir()
        assert config.cortex.mode == "autopilot"
        assert config.background_memory is True
        # The turn's scratch folder holds only the read-only root of the
        # skills others share with the person: none of its memory.
        assert scratch is not None and {path.name for path in scratch.iterdir()} == {"shared-skills"}
        # Neither the project's own tier nor the base tier reaches the prompt.
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE not in context
        assert PLANTED_GLOBAL_RULE not in context


async def test_a_persons_project_tier_stays_theirs_when_projects_are_shared(monkeypatch, trusted):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS}
    built = _built(
        monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(resources=resources), project_org=None
    )
    async with built as turn:
        config, project = turn.config, turn.project
        global_dir, project_dir = _tiers(config)
        assert global_dir == trusted / "shared" / PARTITION / "memory" / "users" / SUB
        assert project_dir == global_dir / "projects" / str(project.id)
        assert PLANTED_PROJECT_RULE not in await config.cortex.build_memory_context("how many?")


async def test_shared_memory_reads_the_deployment_store_and_the_current_projects_tier(monkeypatch, trusted):
    """The shared global tier is the base one, and the project tier is the
    current project's under it: never the project folder's .anton/memory,
    which the agent's code can write, and never another project's tier."""
    from anton.core.memory.hippocampus import Hippocampus

    _keep_memory(monkeypatch, WITHOUT_MEMORY)
    person = _person(resources=WITHOUT_MEMORY)
    shared = trusted / "base-memory"
    with _project_row(root=trusted, scope=person) as project:
        Hippocampus(shared / "projects" / str(project.id)).encode_lesson(
            "This project's pallets hold 40 cases", topic="data"
        )
        Hippocampus(shared / "projects" / str(uuid4())).encode_lesson(
            "Another project's plant ships on Fridays", topic="data"
        )
        built = built_turn(
            monkeypatch=monkeypatch, tmp_path=trusted, scope=person, project=project, before_build=_plant
        )
        async with built as turn:
            config = turn.config
            global_dir, project_dir = _tiers(config)
            assert global_dir == shared
            assert project_dir == shared / "projects" / str(project.id)
            assert config.cortex.mode == "autopilot"
            assert config.background_memory is True
            context = await _everything_the_turn_reads(config)
    assert PLANTED_GLOBAL_RULE in context
    assert "This project's pallets hold 40 cases" in context
    assert "Another project's plant ships on Fridays" not in context
    assert PLANTED_PROJECT_RULE not in context and PLANTED_PROFILE not in context
    assert not (Path(project.path) / ".anton" / "memory" / "lessons.md").exists()


async def test_two_people_share_one_store_but_not_each_others_project_lessons(monkeypatch, trusted):
    """With memory shared, what A's turn learns globally, profile included,
    reaches B's next turn in another project; A's project lesson stays with
    A's project."""
    from anton.core.memory.base import Engram

    _keep_memory(monkeypatch, WITHOUT_MEMORY)
    alice = _person(resources=WITHOUT_MEMORY)
    bob = _person(resources=WITHOUT_MEMORY, sub=OTHER_SUB, partition=OTHER_PARTITION)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=alice) as a:
        await a.config.cortex.encode([
            Engram(text="Favorite part number is P-3333", kind="profile", scope="global"),
            Engram(text="Answer in crates", kind="always", scope="global", confidence="high", source="user"),
            Engram(text="The plant ships on Fridays", kind="lesson", scope="project", source="llm"),
        ])
        alice_project = a.project.id
    async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=bob) as b:
        assert b.project.id != alice_project
        context = await _everything_the_turn_reads(b.config)
    assert "P-3333" in context and "crates" in context
    assert "Fridays" not in context
    lessons = trusted / "base-memory" / "projects" / str(alice_project) / "lessons.md"
    assert "The plant ships on Fridays" in lessons.read_text()


@pytest.mark.parametrize("resources", [ALL_PERSONAL_RESOURCES, WITHOUT_MEMORY], ids=["personal", "shared"])
async def test_memory_off_reads_and_writes_no_learned_memory(
    monkeypatch, trusted, managed_dir, caplog, resources
):
    """MEMORY_ENABLED false: the turn gets two empty tiers in its scratch
    folder, memory off and no background passes. The operator's managed rule
    still reaches it, and no learned rule, lesson, profile line or project
    profile does, from the person's tiers or the shared store, in the prompt
    or in the scratchpad's description. Every stored byte stays as it was, and
    the turn, which has a person, logs no warning that it has none."""
    from tests._fakes import MANAGED_RULE

    _keep_memory(monkeypatch, resources)
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    person = _person(resources=resources)
    personal_tier = trusted / "shared" / PARTITION / "memory" / "users" / SUB
    shared = trusted / "base-memory"
    with _project_row(root=trusted, scope=person) as project:
        _seed_learned(personal_tier, personal_tier / "projects" / str(project.id), tag="personal")
        _seed_learned(shared, shared / "projects" / str(project.id), tag="shared")
        _plant(Path(project.path))
        project_memory = Path(project.path) / ".anton" / "memory"
        before = _snapshot(personal_tier, shared, project_memory)
        with caplog.at_level(logging.WARNING, logger=harness.__name__):
            async with built_turn(
                monkeypatch=monkeypatch, tmp_path=trusted, scope=person, project=project
            ) as turn:
                config, scratch = turn.config, turn.scratch
                global_dir, project_dir = _tiers(config)
                assert scratch is not None
                assert global_dir.parent.parent == scratch and project_dir.parent.parent == scratch
                assert config.cortex.mode == "off"
                assert config.background_memory is False
                context = await _everything_the_turn_reads(config)
                assert list(global_dir.iterdir()) == [] and list(project_dir.iterdir()) == []
                after = _snapshot(personal_tier, shared, project_memory)
    assert MANAGED_RULE in context
    for learned in ("personal ", "shared ", PLANTED_GLOBAL_RULE, PLANTED_PROJECT_RULE, PLANTED_PROFILE):
        assert learned not in context
    assert after == before
    assert not any(
        "has no person" in record.getMessage()
        for record in caplog.records
        if record.name == harness.__name__ and record.levelno == logging.WARNING
    )


@pytest.mark.parametrize("resources", [ALL_PERSONAL_RESOURCES, WITHOUT_MEMORY], ids=["personal", "shared"])
async def test_memory_off_makes_no_memory_folder(monkeypatch, trusted, resources):
    """The empty tiers are picked before any real tier's folder is made."""
    _keep_memory(monkeypatch, resources)
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(resources=resources)) as turn:
        assert turn.config.cortex.mode == "off"
    assert not (trusted / "shared" / PARTITION / "memory").exists()
    assert not (trusted / "base-memory").exists()


@pytest.mark.parametrize("resources", [ALL_PERSONAL_RESOURCES, WITHOUT_MEMORY], ids=["personal", "shared"])
async def test_a_turn_with_no_person_learns_nothing(monkeypatch, trusted, caplog, resources):
    _keep_memory(monkeypatch, resources)
    base_memory = trusted / "base-memory"
    with caplog.at_level(logging.WARNING, logger=harness.__name__):
        async with _built(monkeypatch=monkeypatch, tmp_path=trusted, scope=LOCAL_SCOPE) as turn:
            config, scratch, project = turn.config, turn.scratch, turn.project
            project_memory = Path(project.path) / ".anton" / "memory"
            before = _snapshot(project_memory, base_memory)

            global_dir, project_dir = _tiers(config)
            assert scratch is not None and scratch.parent == harness._vault_scratch_dir()
            assert global_dir.parent.parent == scratch and project_dir.parent.parent == scratch
            assert list(global_dir.iterdir()) == [] and list(project_dir.iterdir()) == []
            assert config.cortex.mode == "off"
            assert config.background_memory is False

            context = await config.cortex.build_memory_context("how many?")
            assert PLANTED_PROJECT_RULE not in context
            assert PLANTED_GLOBAL_RULE not in context
            assert PLANTED_PROFILE not in context
            # The profile move and every other write left the real tiers alone.
            assert _snapshot(project_memory, base_memory) == before
            assert PLANTED_PROFILE in (project_memory / "profile.md").read_text()
        assert not scratch.exists()

    warnings = [
        record for record in caplog.records
        if record.name == harness.__name__ and record.levelno == logging.WARNING
    ]
    assert any("has no person" in record.getMessage() for record in warnings)


async def test_the_desktop_keeps_the_base_tiers(monkeypatch, dirs):
    async with _built(monkeypatch=monkeypatch, tmp_path=dirs, scope=None) as turn:
        config, scratch, project = turn.config, turn.scratch, turn.project
        global_dir, project_dir = _tiers(config)
        assert global_dir == dirs / "base-memory"
        assert project_dir == Path(project.path) / ".anton" / "memory"
        assert config.cortex.mode == "autopilot"
        assert config.background_memory is True
        assert scratch is None
        # Control for the no-person case: the same planted rules do reach a
        # turn that owns these tiers.
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE in context
        assert PLANTED_GLOBAL_RULE in context


async def test_the_desktop_with_memory_disabled_still_reads_the_base_tiers(monkeypatch, dirs):
    """The desktop keeps its own disabled-memory behavior: memory mode off over
    the base tiers, which the turn still reads."""
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    async with _built(monkeypatch=monkeypatch, tmp_path=dirs, scope=None) as turn:
        config, scratch, project = turn.config, turn.scratch, turn.project
        global_dir, project_dir = _tiers(config)
        assert global_dir == dirs / "base-memory"
        assert project_dir == Path(project.path) / ".anton" / "memory"
        assert config.cortex.mode == "off"
        assert config.background_memory is True
        assert scratch is None
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE in context
        assert PLANTED_GLOBAL_RULE in context


def test_person_turn_memory_needs_a_person_only_when_identity_is_trusted(trusted, tmp_path):
    from cowork.harnesses.memory.turn_layout import empty_turn_memory, person_turn_memory

    project = tmp_path / "proj"
    assert person_turn_memory(scope=LOCAL_SCOPE, project_path=project, project_id=uuid4()) is None
    assert person_turn_memory(scope=None, project_path=project, project_id=uuid4()) is None
    no_user = TenantScope(personal=True, org_id=PARTITION, personal_resources=ALL_PERSONAL_RESOURCES)
    assert person_turn_memory(scope=no_user, project_path=project, project_id=uuid4()) is None

    empty = empty_turn_memory(tmp_path / "scratch")
    assert empty.learns is False and empty.global_dir.is_dir() and empty.project_dir.is_dir()


async def test_two_people_in_one_shared_project_keep_their_own_memory(monkeypatch, trusted):
    """What A's turn learns in a shared project, B's turn there never reads:
    each person's project tier is their own, whatever the project."""
    from anton.core.memory.base import Engram

    resources = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS})
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(resources)))
    get_app_settings.cache_clear()
    alice = _person(resources=resources)
    bob = _person(resources=resources, sub=OTHER_SUB, partition=OTHER_PARTITION)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=alice, project_org=None) as a:
        await a.config.cortex.encode([
            Engram(text="Favorite part number is P-2222", kind="profile", scope="global"),
            Engram(text="Answer in crates", kind="always", scope="global", confidence="high", source="user"),
            Engram(text="The plant ships on Fridays", kind="lesson", scope="project", source="llm"),
        ])
        assert "crates" in await a.config.cortex.build_memory_context("how many?")

        async with built_turn(monkeypatch=monkeypatch, tmp_path=trusted, scope=bob, project=a.project) as b:
            assert b.project.id == a.project.id
            context = (
                await b.config.cortex.build_memory_context("how many?")
                + b.config.cortex.get_scratchpad_context()
            )
    for learned in ("P-2222", "crates", "Fridays"):
        assert learned not in context


@pytest.mark.parametrize("self_hosted", [False, True])
async def test_a_persons_scheduled_run_gets_its_creators_tiers(
    monkeypatch, trusted, granted_product_permissions, self_hosted
):
    """A personal schedule runs as its creator: the scheduler builds the
    principal from the row, ResponsesHandler._produce binds its scope around
    the turn, and the turn's Cortex reads and writes the creator's tiers.
    Drop the binding and the run reads and learns nothing, which only this
    whole chain shows."""
    import cowork.handlers.responses as responses_mod
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.deployment_identity import initialize_deployment_identity
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService
    from tests._turns import close_turn, stub_turn_settings

    if self_hosted:
        pack = trusted / "pack.json"
        pack.write_text('{"name":"Scheduled Workbench"}', encoding="utf-8")
        monkeypatch.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
        monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(pack))
        get_app_settings.cache_clear()
        initialize_deployment_identity(get_app_settings(), remote_turns=False)

    stub_turn_settings(monkeypatch)
    seen: dict[str, object] = {}

    async def no_events():
        return
        yield

    class ScheduledRun(responses_mod.ResponsesHandler):
        """The real handler's _produce around a turn that only builds."""

        async def handle(self, request):
            await self._produce(conversation_id=UUID(request.conversation))
            return no_events()

        async def _run_turn(self, *, conversation_id: UUID) -> None:
            with get_open_session() as db:
                conversation = db.get(Conversation, conversation_id)
                config, scratch, _ = await harness.AntonHarness()._build_chat_session(conversation)
                try:
                    seen["tiers"] = _tiers(config)
                    seen["mode"] = config.cortex.mode
                    seen["identity"] = getattr(config, "deployment_identity", None)
                    seen["memory_scope"] = getattr(config, "learned_memory_scope", None)
                finally:
                    await close_turn(config, scratch)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", ScheduledRun)

    person = _person()
    raw = get_open_session()
    try:
        scoped = ScopedSession(raw, person)
        project_path = trusted / "projects" / f"p-{uuid4().hex[:6]}"
        project_path.mkdir(parents=True)
        project = scoped.add(Project(name=project_path.name, path=str(project_path)))
        scoped.commit()
        scoped.refresh(project)
        project_id = project.id
        schedule = ScheduleService(scoped).create_schedule(
            title="daily", prompt="do it", cadence="daily",
            next_run_at=datetime(2027, 1, 1, 9, 0, tzinfo=timezone.utc),
            model="default", project_id=project_id,
        )
        schedule_id = schedule.id
        assert (schedule.org_id, schedule.created_by) == (PARTITION, SUB)
    finally:
        raw.close()

    try:
        await execute_schedule(schedule_id, is_manual=True)

        global_dir, project_dir = seen["tiers"]
        expected_global = trusted / "shared" / PARTITION / "memory" / "users" / SUB
        assert global_dir == expected_global
        assert project_dir == expected_global / "projects" / str(project_id)
        assert seen["mode"] == "autopilot"
        if self_hosted:
            assert seen["identity"].product_name == "Scheduled Workbench"
            assert seen["memory_scope"] == "personal"
        else:
            assert seen["identity"] is None
            assert seen["memory_scope"] is None
    finally:
        with get_open_session() as db:
            ScheduleService(ScopedSession(db, SYSTEM_SCOPE)).delete_schedule(schedule_id)
            db.exec(delete(Conversation).where(Conversation.project_id == project_id))
            db.exec(delete(Project).where(Project.id == project_id))
            db.commit()
