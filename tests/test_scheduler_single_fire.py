"""One cron slot starts one run, however many replicas poll it.

Every replica runs the poll loop against the same database. The run's insert
names its slot, and a unique index makes that insert the claim, so a second
replica's run for the slot does nothing. A live run renews its lease; a run
whose lease runs out belonged to a process that died, and any replica's poll
reaps it. Its slot fires again when its turn never started, and is consumed
when the turn may have run, so a restart neither skips nor repeats a fire.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlmodel import Session, select

from cowork import scheduler
from cowork.common.datetime_utils import ensure_utc
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.project import Project
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService, SlotAlreadyClaimed
from tests.integration.prereq import missing_prerequisite

# A slot that came due a minute ago: one missed occurrence stays due, where
# an older one would be fast-forwarded past by the poll's missed-run pass.
SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)


@pytest.fixture(autouse=True)
def _no_scheduler_left(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())


@pytest.fixture
def schedule_id():
    """A daily schedule whose slot is due, removed afterwards with its runs."""
    title = f"single fire {uuid4()}"
    session = get_open_session()
    schedule = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
        title=title, prompt="do the thing", cadence="daily", next_run_at=SLOT,
        model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
    )
    created = schedule.id
    session.close()
    yield created
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    cleanup = get_open_session()
    scoped = ScopedSession(cleanup, SYSTEM_SCOPE)
    for conversation_id in cleanup.exec(select(Conversation.id).where(Conversation.topic == title)).all():
        ConversationService(scoped).delete_conversation(conversation_id)
    ScheduleService(scoped).delete_schedule(created)
    cleanup.close()


@pytest.fixture
def turns(monkeypatch):
    """A turn handler that counts the turns it starts and holds each one until
    ``release`` is set."""
    import cowork.handlers.responses as responses_mod

    started: list[str] = []
    release = asyncio.Event()

    class CountingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            started.append(request.conversation)

            async def stream():
                await release.wait()
                yield "event: response.completed\ndata: {}\n\n"

            return stream()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", CountingHandler)
    return started, release


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    session = get_open_session()
    try:
        return ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).list_runs(schedule_id)
    finally:
        session.close()


def _schedule(schedule_id: UUID) -> Schedule:
    session = get_open_session()
    try:
        return ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).get_schedule(schedule_id)
    finally:
        session.close()


def _poll() -> list[scheduler.DueSlot]:
    session = get_open_session()
    try:
        return scheduler._poll(ScopedSession(session, SYSTEM_SCOPE))
    finally:
        session.close()


def _left_running(schedule_id: UUID, *, linked_conversation: bool) -> UUID:
    """A run another replica started for SLOT and then died during, so its
    lease ran out a minute ago."""
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        run = ScheduleRunService(scoped).create_run(schedule_id, slot_at=SLOT, lease_seconds=-60)
        if linked_conversation:
            conversation = ConversationService(scoped).create_conversation(
                topic=_schedule(schedule_id).title, project_id=GENERAL_PROJECT_ID,
            )
            ScheduleRunService(scoped).set_run_conversation(run.id, conversation.id)
        return run.id
    finally:
        session.close()


async def test_two_replicas_that_find_one_slot_due_start_one_run(schedule_id, turns):
    started, release = turns

    first = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    second = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    async with asyncio.timeout(10):
        while not started:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

    assert len(started) == 1
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.success
    assert ensure_utc(run.slot_at) == SLOT
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1)


async def test_a_live_run_renews_its_lease_so_no_poll_reaps_it(schedule_id, turns, monkeypatch):
    started, release = turns
    monkeypatch.setattr(scheduler, "_RUN_LEASE_SECONDS", 0.5)
    monkeypatch.setattr(scheduler, "_LEASE_RENEW_SECONDS", 0.1)

    run = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    async with asyncio.timeout(10):
        while not started:
            await asyncio.sleep(0.01)
    for _ in range(4):
        await asyncio.sleep(0.4)
        _poll()
    # Checked before the run ends: its own finish would overwrite a reap.
    (live,) = _runs(schedule_id)
    assert live.status == RunStatus.running
    release.set()
    await asyncio.wait_for(run, timeout=10)

    (finished,) = _runs(schedule_id)
    assert finished.status == RunStatus.success
    assert finished.error is None
    assert finished.lease_expires_at is None


async def test_a_dead_run_whose_turn_never_started_gives_its_slot_back(schedule_id, turns):
    started, release = turns
    release.set()
    dead = _left_running(schedule_id, linked_conversation=False)

    due = _poll()

    assert scheduler.DueSlot(schedule_id=schedule_id, slot_at=SLOT) in due
    (reaped,) = _runs(schedule_id)
    assert reaped.id == dead
    assert reaped.status == RunStatus.failed
    assert reaped.error == "Run was interrupted: the server running it stopped before it finished."
    assert reaped.slot_at is None
    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)
    assert len(started) == 1
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1)


async def test_a_dead_run_whose_turn_may_have_run_consumes_its_slot(schedule_id, turns):
    started, _ = turns
    _left_running(schedule_id, linked_conversation=True)

    due = _poll()

    assert schedule_id not in [slot.schedule_id for slot in due]
    (reaped,) = _runs(schedule_id)
    assert reaped.status == RunStatus.failed
    assert ensure_utc(reaped.slot_at) == SLOT
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1)
    assert started == []


async def test_a_run_reaped_before_its_turn_starts_never_starts_it(schedule_id, turns, monkeypatch):
    """The lease ran out while the run waited, say on a full pool, and another
    replica reaped it and may now run the slot. This run must not start its
    turn, and must leave the schedule for that replica."""
    import cowork.services.product_permissions as permissions

    started, release = turns
    release.set()

    async def reaped_meanwhile(scope, permission):
        session = get_open_session()
        try:
            scoped = ScopedSession(session, SYSTEM_SCOPE)
            ScheduleRunService(scoped).reap_lapsed_runs(
                now=datetime.now(timezone.utc) + timedelta(hours=1), error="reaped",
            )
            scoped.commit()
        finally:
            session.close()

    monkeypatch.setattr(permissions, "require_product_permission", reaped_meanwhile)

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)

    assert started == []
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.slot_at is None
    schedule = _schedule(schedule_id)
    assert ensure_utc(schedule.next_run_at) == SLOT
    assert schedule.last_error is None


async def test_a_run_that_fails_before_its_turn_gives_its_slot_back_for_the_next_poll(
    schedule_id, turns, monkeypatch,
):
    """A permission service outage clears on its own and the agent did
    nothing, so the slot stays due, and the next poll can claim it."""
    import cowork.services.product_permissions as permissions

    started, release = turns
    release.set()
    outage = [True]
    real = permissions.require_product_permission

    async def unavailable_once(scope, permission):
        if outage.pop() if outage else False:
            raise permissions.ProductPermissionUnavailable()
        return await real(scope, permission)

    monkeypatch.setattr(permissions, "require_product_permission", unavailable_once)

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)
    (refused,) = _runs(schedule_id)
    assert refused.slot_at is None
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)
    assert len(started) == 1


def _pause(schedule_id: UUID) -> None:
    session = get_open_session()
    try:
        ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).pause_schedule(schedule_id)
    finally:
        session.close()


def _resume(schedule_id: UUID) -> None:
    session = get_open_session()
    try:
        ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).resume_schedule(schedule_id)
    finally:
        session.close()


async def test_a_paused_schedule_whose_run_failed_runs_the_slot_it_left_due_once_resumed(schedule_id, monkeypatch):
    """The user pauses the schedule while its turn runs, and the turn then
    fails. The pause keeps the slot due, so the run gives it back: a resumed
    schedule's next poll can claim it, instead of meeting the failed run's
    claim on every poll until the next occurrence is skipped."""
    import cowork.handlers.responses as responses_mod

    started: list[str] = []

    class PausedThenFailingHandler:
        def __init__(self, *, principal=None, interactive=True):
            pass

        async def handle(self, request):
            if not started:
                started.append("failed")
                _pause(schedule_id)
                raise RuntimeError("the agent failed")
            started.append(request.conversation)

            async def stream():
                yield "event: response.completed\ndata: {}\n\n"

            return stream()

    monkeypatch.setattr(responses_mod, "ResponsesHandler", PausedThenFailingHandler)

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)
    (failed,) = _runs(schedule_id)
    assert failed.status == RunStatus.failed
    assert failed.slot_at is None
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT

    _resume(schedule_id)
    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)

    assert len(started) == 2
    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1)


async def test_a_paused_schedule_whose_server_died_mid_turn_resumes_at_its_next_occurrence(schedule_id, turns):
    """The reaper uses up the dead run's slot, since its turn may have run.
    It moves a paused schedule on too: left on that slot, a resumed
    schedule's every poll would meet the dead run's claim."""
    started, _ = turns
    _left_running(schedule_id, linked_conversation=True)
    _pause(schedule_id)

    _poll()

    assert ensure_utc(_schedule(schedule_id).next_run_at) == SLOT + timedelta(days=1)
    _resume(schedule_id)
    assert schedule_id not in [slot.schedule_id for slot in _poll()]
    assert started == []


async def test_a_run_reaped_after_its_check_never_links_or_starts_its_turn(schedule_id, turns, monkeypatch):
    """Another replica reaps the run after its claim check passed and before
    its conversation is linked. The link refuses a run that is no longer
    running, so the turn never starts, and the conversation made for it is
    deleted again."""
    from cowork.models.conversation import Conversation

    started, release = turns
    release.set()
    real_still_running = ScheduleRunService.still_running

    def reaped_right_after_the_check(self, run_id):
        alive = real_still_running(self, run_id)
        session = get_open_session()
        try:
            ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).reap_lapsed_runs(
                now=datetime.now(timezone.utc) + timedelta(hours=1), error="reaped",
            )
            session.commit()
        finally:
            session.close()
        return alive

    monkeypatch.setattr(ScheduleRunService, "still_running", reaped_right_after_the_check)

    await scheduler.execute_schedule(schedule_id, slot_at=SLOT)

    assert started == []
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.conversation_id is None
    session = get_open_session()
    try:
        title = _schedule(schedule_id).title
        assert session.exec(select(Conversation.id).where(Conversation.topic == title)).all() == []
    finally:
        session.close()


async def test_a_run_another_replica_reaped_stops_renewing_and_says_so_once(schedule_id, turns, monkeypatch, caplog):
    started, release = turns
    monkeypatch.setattr(scheduler, "_LEASE_RENEW_SECONDS", 0.05)
    # The migration tests' alembic env calls fileConfig, which disables every
    # logger that existed before it for the rest of the session.
    monkeypatch.setattr(scheduler.logger, "disabled", False)
    caplog.set_level(logging.WARNING, logger=scheduler.logger.name)

    run = asyncio.create_task(scheduler.execute_schedule(schedule_id, slot_at=SLOT))
    async with asyncio.timeout(10):
        while not started:
            await asyncio.sleep(0.01)
    session = get_open_session()
    try:
        ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).reap_lapsed_runs(
            now=datetime.now(timezone.utc) + timedelta(hours=1), error="reaped",
        )
        session.commit()
    finally:
        session.close()
    await asyncio.sleep(0.4)
    release.set()
    await asyncio.wait_for(run, timeout=10)

    lost = [
        record for record in caplog.records
        if record.name == scheduler.logger.name
        and record.levelno == logging.WARNING
        and record.getMessage() == "A scheduled run lost its lease to another replica"
    ]
    assert len(lost) == 1


def test_a_boot_reap_gives_the_slot_back_so_a_restarted_desktop_runs_it():
    """The desktop reaps every run a quit left running when it starts again,
    and the slot those runs left due fires, as it did before runs claimed
    slots."""
    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        schedule = ScheduleService(scoped).create_schedule(
            title=f"boot reap {uuid4()}", prompt="p", cadence="daily", next_run_at=SLOT,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        )
        runs = ScheduleRunService(scoped)
        runs.create_run(schedule.id, slot_at=SLOT, lease_seconds=60)

        runs.reap_orphaned_runs()

        runs.create_run(schedule.id, slot_at=SLOT, lease_seconds=60)
        assert [run.status for run in runs.list_runs(schedule.id)].count(RunStatus.running) == 1
        ScheduleService(scoped).delete_schedule(schedule.id)
    finally:
        session.close()


def test_a_second_insert_for_a_claimed_slot_raises_slot_already_claimed():
    session = get_open_session()
    try:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        schedule = ScheduleService(scoped).create_schedule(
            title=f"claim {uuid4()}", prompt="p", cadence="daily", next_run_at=SLOT,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        )
        runs = ScheduleRunService(scoped)
        runs.create_run(schedule.id, slot_at=SLOT)

        with pytest.raises(SlotAlreadyClaimed):
            runs.create_run(schedule.id, slot_at=SLOT)
        runs.create_run(schedule.id, is_manual=True)
        runs.create_run(schedule.id, slot_at=SLOT + timedelta(days=1))
        assert len(runs.list_runs(schedule.id)) == 3
        ScheduleService(scoped).delete_schedule(schedule.id)
    finally:
        session.close()


@dataclass(frozen=True)
class PostgresSchedule:
    engine: Engine
    schedule_id: UUID
    project_id: UUID


@pytest.fixture
def postgres(monkeypatch):
    """A disposable PostgreSQL at head with one daily schedule due at SLOT,
    for the claims only a second connection can contend for."""
    uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
    if not uri:
        missing_prerequisite(
            "set COWORK_TEST_POSTGRES_URI to a disposable database", required_by="COWORK_REQUIRE_SERVICES",
        )
    monkeypatch.setenv("DATABASE_URI", uri)
    get_app_settings.cache_clear()
    cfg = Config("alembic.ini")
    engine = create_engine(uri)
    try:
        command.upgrade(cfg, "head")
        with Session(engine) as session:
            project = Project(name=f"p-{uuid4().hex[:6]}", path="/claim")
            session.add(project)
            session.commit()
            scoped = ScopedSession(session, SYSTEM_SCOPE)
            schedule_id = ScheduleService(scoped).create_schedule(
                title="claim", prompt="p", cadence="daily", next_run_at=SLOT,
                model="default", timezone="UTC", project_id=project.id, enabled=True,
            ).id
            project_id = project.id
        yield PostgresSchedule(engine=engine, schedule_id=schedule_id, project_id=project_id)
    finally:
        engine.dispose()
        command.downgrade(cfg, "base")
        get_app_settings.cache_clear()


class LapsedRun(NamedTuple):
    run_id: UUID
    conversation_id: UUID


def _lapsed_run_and_conversation(pg: PostgresSchedule) -> LapsedRun:
    """A run whose lease ran out before it linked its conversation, and the
    conversation it was about to link."""
    from cowork.services.conversations import ConversationService

    with Session(pg.engine) as session:
        scoped = ScopedSession(session, SYSTEM_SCOPE)
        run_id = ScheduleRunService(scoped).create_run(pg.schedule_id, slot_at=SLOT, lease_seconds=-60).id
        conversation_id = ConversationService(scoped).create_conversation(
            topic="claim", project_id=pg.project_id,
        ).id
    return LapsedRun(run_id=run_id, conversation_id=conversation_id)


def _reap(pg: PostgresSchedule) -> list[ScheduleRun]:
    with Session(pg.engine) as session:
        kept = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).reap_lapsed_runs(
            now=datetime.now(timezone.utc), error="reaped",
        )
        session.commit()
        return kept


def _run_row(pg: PostgresSchedule, run_id: UUID) -> ScheduleRun:
    with Session(pg.engine) as session:
        return session.get(ScheduleRun, run_id)


@pytest.mark.postgres_service
def test_two_processes_inserting_one_slot_on_postgres_start_one_run(postgres):
    """The claim on the database hosted Cowork runs: two connections insert
    the same slot at the same moment, and exactly one of them gets the run."""
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim() -> None:
        with Session(postgres.engine) as session:
            runs = ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE))
            barrier.wait()
            try:
                runs.create_run(postgres.schedule_id, slot_at=SLOT, lease_seconds=60)
                outcomes.append("claimed")
            except SlotAlreadyClaimed:
                outcomes.append("refused")

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["claimed", "refused"]


@pytest.mark.postgres_service
def test_a_reap_on_postgres_leaves_a_run_whose_link_is_in_flight_and_keeps_its_slot_once_linked(postgres):
    """The reaper reads a lapsed run while that run is linking its
    conversation. Reading the link as not yet made would give the slot back
    while the turn starts, so the slot would run twice. The reaper skips the
    row instead, and a later poll reaps it with its link and keeps the slot."""
    run_id, conversation_id = _lapsed_run_and_conversation(postgres)
    with Session(postgres.engine) as linking:
        run = linking.exec(select(ScheduleRun).where(ScheduleRun.id == run_id).with_for_update()).one()
        run.conversation_id = conversation_id
        linking.add(run)
        linking.flush()
        reaped: list[list[ScheduleRun]] = []
        reaper = threading.Thread(target=lambda: reaped.append(_reap(postgres)))
        reaper.start()
        reaper.join(timeout=0.5)
        linking.commit()
    reaper.join(timeout=30)

    _reap(postgres)

    row = _run_row(postgres, run_id)
    assert row.status == RunStatus.failed
    assert row.conversation_id == conversation_id
    assert ensure_utc(row.slot_at) == SLOT


@pytest.mark.postgres_service
def test_a_link_on_postgres_that_waits_on_a_reap_refuses_the_run(postgres):
    """The reaper has failed the run and given its slot back, but not yet
    committed, when the run links its conversation. The link waits for the
    reap and then refuses, so the run's turn never starts beside the run
    the freed slot gets."""
    run_id, conversation_id = _lapsed_run_and_conversation(postgres)
    with Session(postgres.engine) as reaping:
        ScheduleRunService(ScopedSession(reaping, SYSTEM_SCOPE)).reap_lapsed_runs(
            now=datetime.now(timezone.utc), error="reaped",
        )
        reaping.flush()
        linked: list[bool] = []

        def link() -> None:
            with Session(postgres.engine) as session:
                linked.append(
                    ScheduleRunService(ScopedSession(session, SYSTEM_SCOPE)).set_run_conversation(
                        run_id, conversation_id,
                    )
                )

        linker = threading.Thread(target=link)
        linker.start()
        linker.join(timeout=0.5)
        reaping.commit()
    linker.join(timeout=30)

    assert linked == [False]
    row = _run_row(postgres, run_id)
    assert row.status == RunStatus.failed
    assert row.conversation_id is None
    assert row.slot_at is None
