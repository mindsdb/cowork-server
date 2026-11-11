"""The heal sweep: once per self-heal interval, one replica lists the shared
root of every org project with a recent message and records what it finds,
so a folder whose turn end wrote no row still shows in the artifacts list.

A turn end's record can be lost to the shutdown drain or a busy pool, an old
pod in a rollout writes no rows, and a self-heal request dies with its pod.
The sweep needs none of them: it finds its projects by their messages.
"""
from __future__ import annotations

import asyncio
import errno
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from cowork import scheduler, server
from cowork.db.scoped import SYSTEM_SCOPE, TenantScope, scope_for_org
from cowork.db.units import unit_session
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.schemas.responses import Role
from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_heal_sweep as sweep
from cowork.services.artifact_autopublish import SELF_HEAL_INTERVAL_S
from cowork.services.artifact_locations import keeps_locations
from cowork.services.artifact_owner_backfill import write_sentinel
from cowork.services.artifact_roots import project_artifacts_base
from cowork.services.held_artifact_roots import SENTINEL_KEY, HeldRootSentinel
from test_artifact_ownership import _engine, make_conversation, make_project, scoped, write_artifact
from test_org_list_scaling import (  # noqa: F401  (fixtures)
    ALICE,
    _add_projects,
    _headers,
    _project_id,
    _projects_root,
    client,
    org,
)
from tests._logging_assertions import assert_log_fields

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

OWNER = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(autouse=True)
def no_sweep_running():
    """This test's own sweep task, whatever an earlier test left, this
    replica's self-heal requests and sweep retries, empty, and the module's
    logger on for caplog: the migration tests' alembic env disables every
    logger that existed before it. Set by hand rather than through
    monkeypatch, so a test's monkeypatch still restores the environment
    before cleanup_tmp_projects reads the settings."""
    disabled = sweep.logger.disabled
    requests = (ap._self_heals_requested, ap._self_heal_publishes_requested)
    retries = (sweep._retries, sweep._last_pass_at)
    sweep._sweep_task = None
    sweep.logger.disabled = False
    ap._self_heals_requested, ap._self_heal_publishes_requested = set(), set()
    sweep._retries, sweep._last_pass_at = set(), None
    yield
    sweep._sweep_task = None
    sweep.logger.disabled = disabled
    ap._self_heals_requested, ap._self_heal_publishes_requested = requests
    sweep._retries, sweep._last_pass_at = retries


def _drop_rows(key: str) -> None:
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key == key, Setting.scope.is_(None))).all():
            raw.delete(row)
        raw.commit()


@pytest.fixture
def no_claim():
    """No replica has claimed a sweep yet. The row is global and the database
    is shared across modules, so a claim a test makes goes after it."""
    _drop_rows(sweep.CLAIM_KEY)
    yield
    _drop_rows(sweep.CLAIM_KEY)


@pytest.fixture
def held_sentinel():
    """The held-root backfill has finished, so the list trusts the rows. The
    sentinel is global, so it goes after the test."""
    write_sentinel(_engine(), SENTINEL_KEY, value=HeldRootSentinel().model_dump_json())
    yield
    _drop_rows(SENTINEL_KEY)


def _claimed_at() -> datetime | None:
    with Session(_engine()) as raw:
        row = raw.exec(select(Setting).where(Setting.key == sweep.CLAIM_KEY, Setting.scope.is_(None))).first()
        return None if row is None else sweep.SweepStart.model_validate_json(row.value).started_at


def _message(conversation_id: UUID, *, at: datetime | None = None) -> None:
    """One user message in the conversation, created `at`, or now."""
    with Session(_engine()) as raw:
        message = Message(conversation_id=conversation_id, role=Role.user, content={"text": "a turn"})
        if at is not None:
            message.created_at = at
        raw.add(message)
        raw.commit()


def _talked_in(project: Project, *, ago: timedelta | None = None) -> None:
    """A conversation in `project` with a message `ago`, or now."""
    at = None if ago is None else datetime.now(timezone.utc) - ago
    _message(make_conversation(project, OWNER), at=at)


def _held(project_id: UUID) -> bool:
    with Session(_engine()) as raw:
        query = select(HeldArtifactRoot).where(
            HeldArtifactRoot.project_id == project_id, HeldArtifactRoot.conversation_id.is_(None)
        )
        return raw.exec(query).first() is not None


def _sweep_events(caplog, event: str) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == sweep.logger.name and getattr(record, "event", None) == event
    ]


def _assert_pass_line(caplog, logged, event: str, **counts: int) -> None:
    """The pass logged one `event` line, and the console handler printed it
    as a WARNING with `counts`. That handler keeps only the fields in
    structured_logging.EVENT_FIELDS, so a count left out of them fails here,
    and so does a line at INFO, which no hosted deployment prints."""
    assert len(_sweep_events(caplog, event)) == 1
    assert_log_fields(logged.output(), log_level="WARNING", event=event, **counts)


def _sweep_heal(project: Project) -> ap.SelfHeal:
    """The location self-heal the sweep runs for `project`."""
    return ap.SelfHeal(
        artifacts_base=project_artifacts_base(project.path),
        scope=TenantScope(org_mode=True, org_id=str(project.org_id)),
        project_id=str(project.id),
    )


def _listings_that_fall_short(monkeypatch, *, unread: Project | None = None, broken: Project | None = None) -> None:
    """The file store, as the check lists it: `unread`'s listing could not
    read one of its folders, and `broken`'s listing fails with EIO. Every
    other project lists as it is on disk."""
    real = ap._listed_artifact_ids

    def listing(*, artifacts_base):
        if broken is not None and Path(artifacts_base) == project_artifacts_base(broken.path):
            raise OSError(errno.EIO, "Input/output error")
        listed = real(artifacts_base=artifacts_base)
        if unread is not None and Path(artifacts_base) == project_artifacts_base(unread.path):
            return listed._replace(unreadable=listed.unreadable | {"locked"})
        return listed

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing)


# ── the claim ─────────────────────────────────────────────────────────────


def test_one_replica_claims_each_interval(no_claim):
    now = datetime.now(timezone.utc)
    with scoped(None) as first:
        assert sweep.claim_sweep(first, now=now)
    # Another replica, at its next poll and later inside the interval.
    for later in (timedelta(seconds=30), timedelta(seconds=SELF_HEAL_INTERVAL_S - 1)):
        with scoped(None) as other:
            assert not sweep.claim_sweep(other, now=now + later)
    assert _claimed_at() == now

    after = now + timedelta(seconds=SELF_HEAL_INTERVAL_S)
    with scoped(None) as other:
        assert sweep.claim_sweep(other, now=after)
    assert _claimed_at() == after


def test_a_replica_that_finds_the_lock_held_claims_nothing(no_claim, monkeypatch):
    """Another replica is between its read and its write of the row."""

    @contextmanager
    def held_elsewhere(_engine, _key):
        yield False

    monkeypatch.setattr(sweep, "advisory_try_lock", held_elsewhere)
    with scoped(None) as session:
        assert not sweep.claim_sweep(session, now=datetime.now(timezone.utc))
    assert _claimed_at() is None


@pytest.mark.parametrize("earlier_sweep", [True, False], ids=["row-read-stale", "no-row-yet"])
def test_a_claim_another_replica_made_while_this_one_took_the_lock_is_seen(no_claim, monkeypatch, earlier_sweep):
    """This replica found a sweep due without the lock: the last one started
    over an interval ago, or none ever did. Before it got the lock, another
    replica claimed the interval and released it. So it reads the row again
    under the lock and claims nothing."""
    now = datetime.now(timezone.utc)
    if earlier_sweep:
        with scoped(None) as session:
            assert sweep.claim_sweep(session, now=now - timedelta(seconds=SELF_HEAL_INTERVAL_S + 60))
    theirs = now - timedelta(seconds=5)

    @contextmanager
    def claimed_meanwhile(engine, _key):
        with Session(engine) as raw:
            row = raw.exec(select(Setting).where(Setting.key == sweep.CLAIM_KEY, Setting.scope.is_(None))).first()
            row = row or Setting(key=sweep.CLAIM_KEY, value="")
            row.value = sweep.SweepStart(started_at=theirs).model_dump_json()
            raw.add(row)
            raw.commit()
        yield True

    monkeypatch.setattr(sweep, "advisory_try_lock", claimed_meanwhile)
    with scoped(None) as session:
        assert not sweep.claim_sweep(session, now=now)
    assert _claimed_at() == theirs


def test_a_value_no_sweep_wrote_is_claimed_over(no_claim):
    with Session(_engine()) as raw:
        raw.add(Setting(key=sweep.CLAIM_KEY, value="not a time"))
        raw.commit()
    now = datetime.now(timezone.utc)

    with scoped(None) as session:
        assert sweep.claim_sweep(session, now=now)
    assert _claimed_at() == now


def test_the_claim_is_committed_before_the_lock_goes(no_claim, monkeypatch):
    """The next replica to take the lock reads the row on a connection of
    its own. On PostgreSQL a claim still uncommitted when the lock goes
    reads there as the old row, so that replica would claim the same
    interval. The claim commits while it holds the lock."""
    seen: list[datetime | None] = []

    @contextmanager
    def lock(_engine, _key):
        yield True
        # The lock goes now: what the next replica to take it reads.
        seen.append(_claimed_at())

    monkeypatch.setattr(sweep, "advisory_try_lock", lock)
    now = datetime.now(timezone.utc)
    with scoped(None) as session:
        assert sweep.claim_sweep(session, now=now)
    assert seen == [now]


def test_a_first_row_another_replica_wrote_since_the_read_claims_nothing(no_claim, monkeypatch):
    """On SQLite, which has no advisory lock, two replicas can both read no
    row and write a first one. The unique index on a global settings key
    refuses the second, so that replica claims nothing and the first
    replica's row stays. The claim runs in the session run_db hands it,
    which commits when the claim returns: that commit raises unless the
    claim rolled back the refused insert."""
    theirs = datetime.now(timezone.utc) - timedelta(seconds=5)
    with Session(_engine()) as raw:
        raw.add(Setting(key=sweep.CLAIM_KEY, value=sweep.SweepStart(started_at=theirs).model_dump_json()))
        raw.commit()
    # This replica's reads, before the lock and under it, ran before that
    # write.
    monkeypatch.setattr(sweep, "_claim_row", lambda _session: None)

    with unit_session(scope=SYSTEM_SCOPE) as session:
        assert not sweep.claim_sweep(session, now=datetime.now(timezone.utc))
    assert _claimed_at() == theirs


async def test_a_replica_whose_sweep_is_still_running_claims_nothing(org_deployment, monkeypatch):
    claims: list[datetime] = []
    sweeping, release = asyncio.Event(), asyncio.Event()

    def claim(session, *, now):
        claims.append(now)
        return True

    async def held_sweep():
        sweeping.set()
        await release.wait()
        return 0

    monkeypatch.setattr(sweep, "claim_sweep", claim)
    monkeypatch.setattr(sweep, "run_heal_sweep", held_sweep)
    first = sweep.start_heal_sweep()
    # Only a task: the poll loop never waits for the claim.
    assert first is not None and claims == []
    async with asyncio.timeout(5):
        await sweeping.wait()

    assert sweep.start_heal_sweep() is None
    assert len(claims) == 1
    release.set()
    await first
    second = sweep.start_heal_sweep()
    assert second is not None
    await second
    assert len(claims) == 2


async def test_a_replica_that_loses_the_claim_never_sweeps(org_deployment, no_claim, monkeypatch):
    """Two polls inside one interval, through the real claim: the first
    claims and sweeps, and the second finds the interval claimed and sweeps
    nothing."""
    swept: list[bool] = []

    async def run_heal_sweep():
        swept.append(True)

    monkeypatch.setattr(sweep, "run_heal_sweep", run_heal_sweep)
    for _ in range(2):
        polled = sweep.start_heal_sweep()
        assert polled is not None
        await polled

    assert swept == [True]
    assert _claimed_at() is not None


async def test_the_desktop_never_sweeps(monkeypatch):
    monkeypatch.setattr(sweep, "_claim_and_sweep", lambda: pytest.fail("the desktop claimed a sweep"))

    assert sweep.start_heal_sweep() is None


async def test_a_claim_that_fails_logs_a_warning_and_sweeps_nothing(org_deployment, monkeypatch, caplog):
    swept: list[bool] = []

    def claim(session, *, now):
        raise RuntimeError("the database went away")

    async def run_heal_sweep():
        swept.append(True)

    monkeypatch.setattr(sweep, "claim_sweep", claim)
    monkeypatch.setattr(sweep, "run_heal_sweep", run_heal_sweep)
    with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
        await sweep.start_heal_sweep()

    [failed] = _sweep_events(caplog, "artifact_heal_sweep_failed")
    assert failed.reason == "claim" and failed.exc_info is not None
    assert swept == []


# ── the projects ──────────────────────────────────────────────────────────


def test_the_sweep_lists_only_org_projects_with_a_message_since_its_start(tmp_path, monkeypatch):
    org_id, other_org = str(uuid4()), str(uuid4())
    shared = tmp_path / "shared"
    monkeypatch.setattr(sweep, "resolved_shared_root", lambda: shared)
    recent, stale, quiet, desktop = (make_project(tmp_path, owner) for owner in (org_id, org_id, org_id, None))
    _talked_in(recent)
    _talked_in(stale, ago=timedelta(hours=2))
    make_conversation(quiet, OWNER)
    _talked_in(desktop)
    # A row naming a folder in another organization's partition, as agent
    # code from before the code sandbox could leave.
    planted = Project(id=uuid4(), name="planted", path=str(shared / other_org / "projects" / "theirs"), org_id=org_id)
    with Session(_engine()) as raw:
        raw.add(planted)
        raw.commit()
        raw.refresh(planted)
        raw.expunge(planted)
    _talked_in(planted)
    since = datetime.now(timezone.utc) - timedelta(minutes=30)

    with scoped(None) as session:
        heals = sweep.project_heals_since(
            session, since=since, project_ids=[recent.id, stale.id, quiet.id, desktop.id, planted.id],
        )

    assert heals == [
        ap.SelfHeal(
            artifacts_base=project_artifacts_base(recent.path),
            scope=TenantScope(org_mode=True, org_id=org_id),
            project_id=str(recent.id),
        )
    ]
    # The organization's scope with no user: the check keeps location rows
    # and publishes nothing.
    assert keeps_locations(heals[0].scope) and heals[0].scope.user_id is None


async def test_the_sweep_reaches_back_twice_the_interval_and_a_margin(tmp_path, monkeypatch):
    """A message older than one interval but inside two still brings its
    project in, so one missed sweep is covered by the next, and so does one
    a minute past two intervals, since each sweep starts up to a poll late."""
    org_id = str(uuid4())
    interval = timedelta(seconds=SELF_HEAL_INTERVAL_S)
    inside, late, outside = sorted((make_project(tmp_path, org_id) for _ in range(3)), key=lambda project: project.id)
    _talked_in(inside, ago=interval + interval / 2)
    _talked_in(late, ago=2 * interval + timedelta(minutes=1))
    _talked_in(outside, ago=timedelta(seconds=sweep.SWEEP_WINDOW_S) + timedelta(minutes=5))
    listed: list[str] = []

    async def record(heal):
        listed.append(heal.project_id)

    monkeypatch.setattr(sweep, "_record_listed_locations", record)

    counts = await sweep.run_heal_sweep(project_ids=[inside.id, late.id, outside.id])
    assert counts == sweep.SweepCounts(checked=2, failed=0)
    assert listed == [str(inside.id), str(late.id)]


async def test_a_project_whose_check_fails_never_ends_the_sweep(tmp_path, monkeypatch, caplog, owned_logger):
    org_id = str(uuid4())
    failing, healed = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda project: project.id)
    for project in (failing, healed):
        _talked_in(project)
    listed: list[str] = []

    async def record(heal):
        if heal.project_id == str(failing.id):
            raise RuntimeError("an error the check did not expect")
        listed.append(heal.project_id)

    monkeypatch.setattr(sweep, "_record_listed_locations", record)
    logged = owned_logger(sweep.logger.name, level=logging.WARNING)
    counts = await sweep.run_heal_sweep(project_ids=[failing.id, healed.id])

    assert counts == sweep.SweepCounts(checked=1, failed=1)
    assert listed == [str(healed.id)]
    [failed] = _sweep_events(caplog, "artifact_heal_sweep_failed")
    assert (failed.reason, failed.project_id) == ("project", str(failing.id)) and failed.exc_info is not None
    _assert_pass_line(caplog, logged, "artifact_heal_sweep", count=1, failed_count=1, retried_count=0)


async def test_a_check_that_falls_short_is_counted_and_kept_off_the_poll(
    org_deployment, tmp_path, monkeypatch, caplog, owned_logger
):
    """During a file store incident, one project's listing could not read a
    folder and another's failed with EIO. The check logs each and asks for
    the project's next location check; the sweep counts both as failed and
    keeps them for its own next pass, so the next poll does not start every
    such project's listing at once on the replica that swept."""
    org_id = str(uuid4())
    healthy, unread, broken = sorted((make_project(tmp_path, org_id) for _ in range(3)), key=lambda project: project.id)
    for project in (healthy, unread, broken):
        write_artifact(project_artifacts_base(project.path), "report")
        _talked_in(project)
    _listings_that_fall_short(monkeypatch, unread=unread, broken=broken)
    logged = owned_logger(sweep.logger.name, level=logging.WARNING)

    counts = await sweep.run_heal_sweep(project_ids=[healthy.id, unread.id, broken.id])

    assert counts == sweep.SweepCounts(checked=1, failed=2)
    failed = _sweep_events(caplog, "artifact_heal_sweep_failed")
    assert sorted((record.reason, record.project_id) for record in failed) == sorted(
        ("project", str(project.id)) for project in (unread, broken)
    )
    _assert_pass_line(caplog, logged, "artifact_heal_sweep", count=1, failed_count=2, retried_count=0)
    # Nothing waits for this replica's poll: the sweep's next pass has them.
    assert ap._self_heals_requested == set()
    assert ap.start_due_self_heals() == []
    assert sweep._retries == {_sweep_heal(unread), _sweep_heal(broken)}
    # A listing that read the root, though not every folder, still holds it.
    assert (_held(healthy.id), _held(unread.id), _held(broken.id)) == (True, True, False)


async def test_a_request_this_replica_held_before_the_check_stays(org_deployment, tmp_path, monkeypatch):
    """A channel turn end on this replica asked for the self-heal of each
    project before the sweep checked it. Its request, and the publish asked
    with it, stay for this replica's poll, whether the sweep's check reads
    every folder or falls short, and a check that falls short is still
    counted."""
    org_id = str(uuid4())
    healthy, unread = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda project: project.id)
    for project in (healthy, unread):
        write_artifact(project_artifacts_base(project.path), "report")
        _talked_in(project)
    _listings_that_fall_short(monkeypatch, unread=unread)
    ap.request_self_heal(_sweep_heal(healthy))
    ap.request_location_check(_sweep_heal(unread))

    counts = await sweep.run_heal_sweep(project_ids=[healthy.id, unread.id])

    assert counts == sweep.SweepCounts(checked=1, failed=1)
    assert ap._self_heals_requested == {_sweep_heal(healthy), _sweep_heal(unread)}
    assert ap._self_heal_publishes_requested == {_sweep_heal(healthy)}


async def test_a_request_another_task_makes_during_the_check_stays(org_deployment, tmp_path, monkeypatch, caplog):
    """A channel turn end on this replica asks for the project's self-heal
    while the sweep's check of it is listing. Its scope is the
    organization's with no user, so its self-heal is the sweep's own. The
    check read every folder, so it counts as checked, and the turn end's
    request, with its publish, stays for this replica's poll."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_artifacts_base(project.path), "report")
    _talked_in(project)
    listing_started, turn_end_asked = threading.Event(), threading.Event()
    real = ap._listed_artifact_ids

    def listing(*, artifacts_base):
        listing_started.set()
        assert turn_end_asked.wait(5)
        return real(artifacts_base=artifacts_base)

    async def channel_turn_end():
        while not listing_started.is_set():
            await asyncio.sleep(0.005)
        # What remote_turn.remote_turn_events asks for as a channel turn's
        # record starts.
        ap.request_turn_end_self_heal(
            artifacts_base=project_artifacts_base(project.path), scope=scope_for_org(org_id), project_id=str(project.id),
        )
        turn_end_asked.set()

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing)
    turn_end = asyncio.create_task(channel_turn_end())
    with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
        counts = await sweep.run_heal_sweep(project_ids=[project.id])
    await turn_end

    assert counts == sweep.SweepCounts(checked=1, failed=0)
    assert _sweep_events(caplog, "artifact_heal_sweep_failed") == []
    assert _held(project.id) and sweep._retries == set()
    heal = _sweep_heal(project)
    assert (ap._self_heals_requested, ap._self_heal_publishes_requested) == ({heal}, {heal})


@pytest.mark.parametrize("same_project", [False, True], ids=["another-project", "the-same-project"])
async def test_a_location_check_another_task_asks_for_during_the_check_stays_for_the_poll(
    org_deployment, tmp_path, monkeypatch, caplog, same_project
):
    """While the sweep's check of a project is listing, a location check
    this replica's poll started, in a task of its own, could not read a
    folder of another project, or of the same one, and asks for its next
    location check (request_location_check). The sweep keeps only its own
    task's requests (keep_location_checks), so that request stays for the
    poll, and the sweep's check, which read every folder, counts as checked,
    keeps no retry and logs no failure."""
    org_id = str(uuid4())
    swept, other = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda project: project.id)
    for project in (swept, other):
        write_artifact(project_artifacts_base(project.path), "report")
    _talked_in(swept)
    polled = _sweep_heal(swept if same_project else other)
    sweep_listing_started, poll_check_done = threading.Event(), threading.Event()
    real = ap._listed_artifact_ids

    def listing(*, artifacts_base):
        listed = real(artifacts_base=artifacts_base)
        if not sweep_listing_started.is_set():
            # The sweep's listing, the first, reads every folder once the
            # poll's check has asked for its next one.
            sweep_listing_started.set()
            assert poll_check_done.wait(5)
            return listed
        # The poll's listing could not read a folder.
        return listed._replace(unreadable=listed.unreadable | {"locked"})

    async def poll_started_check():
        while not sweep_listing_started.is_set():
            await asyncio.sleep(0.005)
        try:
            # What start_due_self_heals starts for a location check alone.
            await ap._record_listed_locations(polled)
        finally:
            poll_check_done.set()

    monkeypatch.setattr(ap, "_listed_artifact_ids", listing)
    poll_check = asyncio.create_task(poll_started_check())
    with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
        counts = await sweep.run_heal_sweep(project_ids=[swept.id])
    await poll_check

    assert counts == sweep.SweepCounts(checked=1, failed=0)
    assert _sweep_events(caplog, "artifact_heal_sweep_failed") == []
    assert _held(swept.id) and sweep._retries == set()
    assert (ap._self_heals_requested, ap._self_heal_publishes_requested) == ({polled}, set())


async def test_the_next_sweep_lists_a_check_that_fell_short_whatever_its_message_age(
    org_deployment, tmp_path, monkeypatch, caplog, owned_logger
):
    """The project's last message is 25 minutes old: inside this sweep's
    window, and outside the next one's. Its check fails with EIO, so its
    held row is not written. The next sweep, an interval later and with the
    file store back, still lists it and writes the row."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_artifacts_base(project.path), "report")
    _talked_in(project, ago=timedelta(minutes=25))
    real = ap._listed_artifact_ids
    _listings_that_fall_short(monkeypatch, broken=project)

    assert await sweep.run_heal_sweep(project_ids=[project.id]) == sweep.SweepCounts(checked=0, failed=1)
    assert not _held(project.id)

    monkeypatch.setattr(ap, "_listed_artifact_ids", real)
    now = datetime.now

    class AnIntervalLater(datetime):
        @classmethod
        def now(cls, tz=None):
            return now(tz) + timedelta(seconds=SELF_HEAL_INTERVAL_S)

    monkeypatch.setattr(sweep, "datetime", AnIntervalLater)
    # Only the next sweep's lines.
    caplog.clear()
    logged = owned_logger(sweep.logger.name, level=logging.WARNING)
    counts = await sweep.run_heal_sweep(project_ids=[project.id])

    # Checked as a retry: the window no longer holds the project.
    assert counts == sweep.SweepCounts(checked=1, failed=0, retried=1)
    assert _held(project.id) and sweep._retries == set()
    _assert_pass_line(caplog, logged, "artifact_heal_sweep", count=1, failed_count=0, retried_count=1)


async def test_a_replica_that_did_not_claim_lists_its_retries_an_interval_after_its_last_pass(
    org_deployment, tmp_path, monkeypatch, caplog, owned_logger
):
    """This replica's sweep fell short for one project during a file store
    incident, and other replicas claim the next sweeps. The sweep's pass
    noted when it started (_check_heals), so this replica's polls inside the
    interval list nothing, and its first poll at least an interval after
    that pass lists the project again, alone, with the file store back."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_artifacts_base(project.path), "report")
    _talked_in(project)
    real = ap._listed_artifact_ids
    _listings_that_fall_short(monkeypatch, broken=project)
    listed: list[str] = []
    check = sweep._record_listed_locations

    async def counted(heal):
        listed.append(heal.project_id)
        await check(heal)

    monkeypatch.setattr(sweep, "_record_listed_locations", counted)
    assert await sweep.run_heal_sweep(project_ids=[project.id]) == sweep.SweepCounts(checked=0, failed=1)
    assert sweep._retries == {_sweep_heal(project)}

    monkeypatch.setattr(ap, "_listed_artifact_ids", real)
    monkeypatch.setattr(sweep, "claim_sweep", lambda session, *, now: False)
    # Three polls inside the interval, 30 seconds apart in production.
    for _ in range(3):
        await sweep.start_heal_sweep()
    assert listed == [str(project.id)] and not _held(project.id)

    # The same pass, an interval ago.
    sweep._last_pass_at -= SELF_HEAL_INTERVAL_S
    # Only the retry pass's lines.
    caplog.clear()
    logged = owned_logger(sweep.logger.name, level=logging.WARNING)
    await sweep.start_heal_sweep()

    assert listed == [str(project.id)] * 2
    assert _held(project.id) and sweep._retries == set()
    _assert_pass_line(caplog, logged, "artifact_heal_sweep_retry", count=1, failed_count=0)
    assert _sweep_events(caplog, "artifact_heal_sweep") == []


async def test_a_project_query_that_fails_logs_a_warning(monkeypatch, caplog):
    def refused(session, **_):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(sweep, "project_heals_since", refused)
    monkeypatch.setattr(sweep, "_record_listed_locations", lambda heal: pytest.fail("listed with no projects"))
    with caplog.at_level(logging.WARNING, logger=sweep.logger.name):
        assert await sweep.run_heal_sweep() is None

    [failed] = _sweep_events(caplog, "artifact_heal_sweep_failed")
    assert failed.reason == "projects" and failed.exc_info is not None


# ── end to end ────────────────────────────────────────────────────────────


def _listed_slugs(client, org: str) -> set[str]:  # noqa: F811
    response = client.get("/api/v1/artifacts/", headers=_headers(org, ALICE))
    assert response.status_code == 200, response.text
    return {card["slug"] for card in response.json()}


async def test_a_folder_whose_turn_end_record_was_lost_shows_after_the_sweep(
    client, org, held_sentinel, caplog, owned_logger  # noqa: F811
):
    """A turn made the project's first folder, and its turn end's record,
    which would have written the root's held row, was lost to the shutdown
    drain. The held-root backfill had already passed the project, so the
    list hides the folder until the sweep writes the row."""
    root = _projects_root(client, org)
    _add_projects(org, root, 0, 1)
    path = root / "bulk-00000"
    project_id = _project_id(org, path)
    write_artifact(Path(path) / ".anton" / "artifacts", "report")
    with Session(_engine()) as raw:
        project = raw.get(Project, project_id)
        raw.expunge(project)
    _talked_in(project)
    assert not _held(project_id)
    assert "report" not in _listed_slugs(client, org)
    logged = owned_logger(sweep.logger.name, level=logging.WARNING)

    assert await sweep.run_heal_sweep(project_ids=[project_id]) == sweep.SweepCounts(checked=1, failed=0)

    assert _held(project_id)
    assert "report" in _listed_slugs(client, org)
    _assert_pass_line(caplog, logged, "artifact_heal_sweep", count=1, failed_count=0, retried_count=0)


# ── the poll loop and the shutdown ────────────────────────────────────────


async def test_the_scheduler_poll_starts_the_heal_sweep(monkeypatch):
    polls: list[int] = []
    starts: list[int] = []

    async def poll(fn, *, scope):
        polls.append(len(polls))
        return []

    def start_heal_sweep():
        starts.append(len(polls))
        return None

    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(scheduler, "run_db", poll)
    monkeypatch.setattr(ap, "start_due_self_heals", lambda: [])
    monkeypatch.setattr(sweep, "start_heal_sweep", start_heal_sweep)
    polling = asyncio.create_task(scheduler._scheduler_loop())
    try:
        async with asyncio.timeout(2):
            while len(starts) < 2:
                await asyncio.sleep(0)
    finally:
        polling.cancel()
        await asyncio.gather(polling, return_exceptions=True)

    assert starts[:2] == [1, 2]


async def test_a_shutdown_cancels_the_running_sweep_and_waits_for_it(org_deployment, monkeypatch):
    unwound: list[bool] = []

    async def held():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            unwound.append(True)

    class _Registry:
        def close(self):
            pass

        async def shutdown(self, *, timeout_seconds):
            return 0

    monkeypatch.setattr(sweep, "_claim_and_sweep", held)
    monkeypatch.setattr(server, "registry", _Registry())
    monkeypatch.setattr(server, "stop_scheduler", lambda: None)
    monkeypatch.setattr(server, "cancel_self_heals", lambda: [])
    running = sweep.start_heal_sweep()
    await asyncio.sleep(0)

    await server._end_remote_turns()

    assert running.cancelled()
    assert unwound == [True]
