"""The heal sweep: once per self-heal interval, one replica of the cluster
lists the shared root of every org project that had a message lately, and
records what it finds there, as a self-heal's location check does
(artifact_autopublish._record_listed_locations).

The artifacts list opens only the roots the held_artifact_roots rows name, and
three things can leave a root holding a folder that no row names. During a
rollout, old pods serve turns and write no rows. A turn end's record can be
lost to the shutdown drain or to a busy pool. And the self-heal a turn end asks
for lives only in its replica's memory
(artifact_autopublish.request_turn_end_self_heal), so it dies with the pod.
The sweep needs none of them: it finds its projects by their messages, so it
covers every project with a message in the last SWEEP_WINDOW_S, whichever pod
ran the turn. The window is twice the interval and a margin, so the next sweep
covers one that failed or never ran.

The scheduler's poll loop asks for it on every replica at each poll
(start_heal_sweep). A global settings row, CLAIM_KEY, holds when the last
sweep started. A replica claims a sweep only when that time is at least
SELF_HEAL_INTERVAL_S old, under a cross-replica advisory try-lock, so the
cluster runs one sweep per interval (claim_sweep). The sweep lists one project
at a time, so it holds one file store slot and never crowds out requests, and
it publishes nothing. A project whose check falls short stays on the sweeping
replica's retries, never on its poll's self-heals (_check_project), and that
replica lists it again, one project at a time, at its next pass an interval
later, whatever the project's message age, until a check reads every folder
(_check_heals). Org tenancy only: the desktop never sweeps.

Each pass logs one WARNING line with its counts (run_heal_sweep,
run_heal_retries), as the held-root walk logs each of its passes. Every
hosted deployment runs at LOG_LEVEL=WARNING, where an INFO line never shows,
so that line is how an operator tells a sweep that runs from one that never
does.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Collection
from datetime import datetime, timedelta, timezone
from functools import partial
from uuid import UUID

import sqlalchemy as sa
from pydantic import AwareDatetime, BaseModel, ValidationError
from sqlalchemy.exc import IntegrityError

from cowork.common.logger import log_context
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope
from cowork.db.units import run_db
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services.artifact_autopublish import (
    SELF_HEAL_INTERVAL_S,
    SelfHeal,
    _record_listed_locations,
    keep_location_checks,
)
from cowork.services.artifact_owner_backfill import advisory_try_lock
from cowork.services.artifact_roots import ProjectRow, left_out_for_another_partition, project_artifacts_base
from cowork.services.conversations import _last_message_at
from cowork.services.projects import resolved_shared_root
from cowork.services.shared_resources import advisory_lock_key

logger = logging.getLogger(__name__)

#: The global settings row that holds when the last sweep started, as a
#: SweepStart.
CLAIM_KEY = "_artifact_heal_sweep_started_at"
_LOCK_KEY = advisory_lock_key(b"cowork\0artifact_heal_sweep_v1")
#: How far back a message makes its project one the sweep lists: twice the
#: interval, so one missed sweep is still covered by the next, and a margin.
#: A sweep is claimed at the first poll at least an interval after the last
#: one started, so each starts up to a poll late (scheduler._scheduler_loop
#: polls every 30 seconds, plus the poll's own run), and replicas' clocks
#: differ a little. Five minutes covers both.
SWEEP_WINDOW_S = 2 * SELF_HEAL_INTERVAL_S + 5 * 60

# This replica's sweep, from its claim to its last project, so a second never
# starts beside it and a shutdown can cancel it (cancel_heal_sweep).
_sweep_task: asyncio.Task | None = None
# The location checks of this replica's passes that fell short, which its
# next pass runs again, whatever their projects' message age, until one
# reads and records every folder (_check_heals). In memory, as the
# self-heal requests are, so a restart drops them.
_retries: set[SelfHeal] = set()
# When this replica's last pass started, on time.monotonic(): a sweep it
# claimed, or a pass over its retries alone (_retries_due).
_last_pass_at: float | None = None


class SweepStart(BaseModel, frozen=True):
    """The value of the CLAIM_KEY row: when the last sweep started."""

    started_at: AwareDatetime


class SweepCounts(BaseModel, frozen=True):
    """What one pass did: the projects whose check read and recorded every
    folder (`checked`), those whose check fell short (`failed`), and how many
    of them it checked only because an earlier pass's check of them fell
    short (`retried`)."""

    checked: int
    failed: int
    retried: int = 0


def _claim_row(session: ScopedSession) -> Setting | None:
    query = session.select(Setting).where(Setting.key == CLAIM_KEY, Setting.scope.is_(None))
    return session.exec(query).first()


def _due(row: Setting | None, *, now: datetime) -> bool:
    """Whether a sweep is due at `now`: none has started, or the last one
    started at least SELF_HEAL_INTERVAL_S ago. A value no sweep wrote counts
    as none, so the claim overwrites it rather than stopping every sweep."""
    if row is None:
        return True
    try:
        last = SweepStart.model_validate_json(row.value).started_at
    except ValidationError:
        return True
    return now - last >= timedelta(seconds=SELF_HEAL_INTERVAL_S)


def claim_sweep(session: ScopedSession, *, now: datetime) -> bool:
    """Claim the sweep due at `now` for this replica: True once it has
    written `now` to the CLAIM_KEY row, and False when no sweep is due, or
    another replica holds the lock or claimed first.

    A read without the lock answers most polls, which find no sweep due. A
    due sweep takes the advisory try-lock and reads the row again under it,
    so it sees a claim another replica made since the first read, then writes
    `now` and commits before the lock goes. So two replicas never claim one
    interval. advisory_try_lock takes PostgreSQL's lock on a connection of
    its own and waits for nothing; SQLite has no such lock, so there it is
    always acquired, and the unique index on a global settings key still
    refuses a second replica's first row."""
    row = _claim_row(session)
    if not _due(row, now=now):
        return False
    with advisory_try_lock(session.get_bind(), _LOCK_KEY) as acquired:
        if not acquired:
            return False
        # Read again under the lock: a replica that held it since the read
        # above may have claimed.
        if row is None:
            row = _claim_row(session)
        else:
            session.refresh(row)
        if not _due(row, now=now):
            return False
        value = SweepStart(started_at=now).model_dump_json()
        if row is None:
            session.add(Setting(key=CLAIM_KEY, value=value))
        else:
            row.value = value
            session.add(row)
        try:
            session.commit()
        except IntegrityError:
            # Another replica wrote the first row since the read.
            session.rollback()
            return False
    return True


def project_heals_since(
    session: ScopedSession, *, since: datetime, project_ids: Collection[UUID] | None = None
) -> list[SelfHeal]:
    """The location self-heal of the shared root of each org project with a
    message created at or after `since`, in project id order.

    A project counts when one of its conversations' newest message is that
    recent: each conversation's last activity is the same correlated
    subquery the conversation list orders by (conversations._last_message_at),
    which PostgreSQL answers with one seek on
    `ix_messages_conversation_created`. An EXISTS on the message's time
    would let the planner turn it into a join and scan every message,
    since no index serves a filter on a message's time alone. A project
    whose folder sits in another partition of the shared root is left out,
    as the held-root backfill leaves it out
    (held_artifact_roots.run_held_root_backfill). Each heal's scope is its
    organization's with no user, so it keeps location rows
    (artifact_locations.keeps_locations) and publishes nothing.
    `project_ids` limits the query to those projects; it exists for tests,
    whose database is shared across modules. Production passes nothing."""
    active_conversation = (
        sa.select(Conversation.id)
        .where(Conversation.project_id == Project.id, _last_message_at() >= since)
        .correlate(Project)
        .exists()
    )
    query = session.select(Project).where(Project.org_id.is_not(None), active_conversation).order_by(Project.id)
    if project_ids is not None:
        query = query.where(Project.id.in_(list(project_ids)))
    # Read once for every project, not once per project.
    shared_root = resolved_shared_root()
    heals: list[SelfHeal] = []
    for project in session.exec(query).all():
        row = ProjectRow(id=UUID(str(project.id)), name=project.name, path=project.path, external=False)
        # In org tenancy an organization's partition of the shared root is
        # its id (projects.partition_of).
        if left_out_for_another_partition(row, partition=str(project.org_id), shared_root=shared_root):
            continue
        heals.append(
            SelfHeal(
                artifacts_base=project_artifacts_base(project.path),
                scope=TenantScope(org_mode=True, org_id=str(project.org_id)),
                project_id=str(project.id),
            )
        )
    return heals


async def _check_project(heal: SelfHeal) -> bool:
    """Run one project's location check
    (artifact_autopublish._record_listed_locations), and return whether it
    read and recorded every folder. That check lists the project's shared
    root, records the root's held row when it holds a folder, and checks each
    id's location row.

    A check that falls short (its listing could not read a folder or the
    root, or its listing or record failed) logs its own WARNING and asks for
    the next location check (request_location_check). It runs inside
    keep_location_checks, so that request lands in this task's own set, which
    tells this check's failure, and never reaches this replica's poll
    (start_due_self_heals): otherwise the next poll would start every such
    project at once, the burst that listing one project at a time avoids.
    The caller keeps the project for this replica's next pass instead
    (_check_heals). A turn end or a self-heal in another task that asks for
    the same project's self-heal during the check still reaches the poll,
    and is not taken for this check's failure. Logs a WARNING (`event`
    `artifact_heal_sweep_failed`, `reason` `project`) for a check that falls
    short or raises."""
    extra = {
        "event": "artifact_heal_sweep_failed", "reason": "project", **log_context(project_id=heal.project_id),
    }
    with keep_location_checks() as kept:
        try:
            await _record_listed_locations(heal)
        except Exception:
            # The check logs and keeps its own failures; this keeps one it
            # did not expect to its project.
            logger.warning(
                "artifact heal sweep could not list a project; this replica lists it again after an interval",
                exc_info=True,
                extra=extra,
            )
            return False
    if kept:
        logger.warning(
            "artifact heal sweep could not check every folder of a project; this replica lists it again after an "
            "interval",
            extra=extra,
        )
        return False
    return True


def _retries_due() -> bool:
    """Whether this replica holds checks that fell short (_retries) and its
    last pass started at least SELF_HEAL_INTERVAL_S ago, so it lists them
    again though another replica claimed the sweep."""
    if not _retries:
        return False
    return _last_pass_at is None or time.monotonic() - _last_pass_at >= SELF_HEAL_INTERVAL_S


async def _check_heals(heals: list[SelfHeal]) -> SweepCounts:
    """Run the location check of each of `heals` (_check_project), then of
    each of this replica's retries not among them, one project at a time. A
    check that reads and records every folder drops its project from the
    retries, and one that falls short keeps or adds it, so this replica's
    next pass, at least SELF_HEAL_INTERVAL_S later, lists it again whatever
    its message age (_retries_due). A folder that stays unreadable costs one
    listing of its project per interval on that replica, as a self-heal's
    does (artifact_autopublish.SELF_HEAL_INTERVAL_S)."""
    global _last_pass_at
    _last_pass_at = time.monotonic()
    retries = sorted(_retries.difference(heals), key=lambda heal: heal.project_id)
    checked = failed = 0
    for heal in [*heals, *retries]:
        if await _check_project(heal):
            checked += 1
            _retries.discard(heal)
        else:
            failed += 1
            _retries.add(heal)
    return SweepCounts(checked=checked, failed=failed, retried=len(retries))


async def run_heal_sweep(*, project_ids: Collection[UUID] | None = None) -> SweepCounts | None:
    """Sweep once: read the projects with a message in the last
    SWEEP_WINDOW_S (project_heals_since), then run each one's location check,
    and this replica's retries, one project at a time (_check_heals). A
    project whose check falls short is logged and counted, and the sweep
    goes on.

    Logs one WARNING line (`event` `artifact_heal_sweep`) with the `count` of
    projects whose check read and recorded every folder, the `failed_count`
    of those whose check fell short, and the `retried_count` of retries
    outside the window, and returns them. Returns None when the project
    query fails, which logs a WARNING; the next sweep's window covers this
    one's, and this replica's retries wait for its next pass, which lists
    them alone when due (_retries_due). `project_ids` limits the sweep's window to those
    projects; it exists for tests, whose database is shared across modules.
    Production passes nothing."""
    since = datetime.now(timezone.utc) - timedelta(seconds=SWEEP_WINDOW_S)
    try:
        heals = await run_db(
            partial(project_heals_since, since=since, project_ids=project_ids), scope=SYSTEM_SCOPE,
        )
    except Exception:
        logger.warning(
            "artifact heal sweep could not read its projects; the next sweep covers them",
            exc_info=True,
            extra={"event": "artifact_heal_sweep_failed", "reason": "projects"},
        )
        return None
    counts = await _check_heals(heals)
    logger.warning(
        "artifact heal sweep listed the projects with a recent message",
        extra={
            "event": "artifact_heal_sweep",
            "count": counts.checked,
            "failed_count": counts.failed,
            "retried_count": counts.retried,
        },
    )
    return counts


async def run_heal_retries() -> SweepCounts:
    """List this replica's retries alone (_check_heals), one project at a
    time, for a replica that holds some but did not claim the sweep. Logs one
    WARNING line (`event` `artifact_heal_sweep_retry`) with the `count` of
    projects whose check now read and recorded every folder and the
    `failed_count` of those whose check fell short again, and returns them."""
    counts = await _check_heals([])
    logger.warning(
        "artifact heal sweep listed again the projects whose check fell short",
        extra={"event": "artifact_heal_sweep_retry", "count": counts.checked, "failed_count": counts.failed},
    )
    return counts


async def _claim_and_sweep() -> None:
    """Claim this interval's sweep (claim_sweep), and run the sweep when this
    replica won the claim. When it did not, and its retries are due
    (_retries_due), list those alone, so they never wait for this replica
    to win a claim. A claim that fails logs a WARNING, and the next poll
    tries again."""
    try:
        claimed = await run_db(partial(claim_sweep, now=datetime.now(timezone.utc)), scope=SYSTEM_SCOPE)
    except Exception:
        logger.warning(
            "artifact heal sweep could not claim the interval; the next poll tries again",
            exc_info=True,
            extra={"event": "artifact_heal_sweep_failed", "reason": "claim"},
        )
        return
    if claimed:
        await run_heal_sweep()
    elif _retries_due():
        await run_heal_retries()


def start_heal_sweep() -> asyncio.Task | None:
    """Start this replica's claim of the heal sweep, and the sweep when the
    claim wins, or its retries alone when they are due, as one task of its
    own (_claim_and_sweep), unless the replica's previous one is still
    running. The scheduler's poll loop calls this once per poll on
    every replica (cowork/scheduler.py); the claim decides which replica
    sweeps. It only creates the task, so it never blocks the loop. Returns
    that task, or None on the desktop, which never sweeps, and while the
    previous one runs."""
    global _sweep_task
    if get_app_settings().tenancy_mode != "org":
        return None
    if _sweep_task is not None and not _sweep_task.done():
        return None
    _sweep_task = asyncio.create_task(_claim_and_sweep())
    return _sweep_task


def cancel_heal_sweep() -> asyncio.Task | None:
    """Cancel this replica's sweep if it is still running, for a shutdown,
    and return it for the caller to wait on, or None when none runs. The
    next sweep's window reaches back over the interval this one claimed."""
    global _sweep_task
    task, _sweep_task = _sweep_task, None
    if task is None or task.done():
        return None
    task.cancel()
    return task
