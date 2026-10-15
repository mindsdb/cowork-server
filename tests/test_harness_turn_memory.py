"""Which memory an in-process turn hands anton.

A person on a deployment that trusts identity headers reads and writes their
own tiers: the global one under their partition, and a project tier under that
global one, never the project's ``.anton/memory``. Two people in one shared
project each keep their own, and a person's scheduled run gets its creator's.
Memory left shared keeps today's base tiers. A turn with no person reads none
of it and writes nothing: two empty tiers in its scratch folder, memory off, no
background passes. The desktop is unchanged.
"""
from __future__ import annotations

import logging
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
        assert scratch is None
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


async def test_shared_memory_keeps_the_base_tiers(monkeypatch, trusted):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}
    async with _built(monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(resources=resources)) as turn:
        config, project = turn.config, turn.project
        global_dir, project_dir = _tiers(config)
        assert global_dir == trusted / "base-memory"
        assert project_dir == Path(project.path) / ".anton" / "memory"
        assert config.cortex.mode == "autopilot"
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE in context
        assert PLANTED_GLOBAL_RULE in context


async def test_a_turn_with_no_person_learns_nothing(monkeypatch, trusted, caplog):
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


async def test_a_persons_scheduled_run_gets_its_creators_tiers(
    monkeypatch, trusted, granted_product_permissions
):
    """A personal schedule runs as its creator: the scheduler builds the
    principal from the row, ResponsesHandler._produce binds its scope around
    the turn, and the turn's Cortex reads and writes the creator's tiers.
    Drop the binding and the run reads and learns nothing, which only this
    whole chain shows."""
    import cowork.handlers.responses as responses_mod
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.scheduler import execute_schedule
    from cowork.services.schedules import ScheduleService
    from tests._turns import close_turn, stub_turn_settings

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
    finally:
        with get_open_session() as db:
            ScheduleService(ScopedSession(db, SYSTEM_SCOPE)).delete_schedule(schedule_id)
            db.exec(delete(Conversation).where(Conversation.project_id == project_id))
            db.exec(delete(Project).where(Project.id == project_id))
            db.commit()
