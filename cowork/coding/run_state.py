from __future__ import annotations

from cowork.coding.contracts import utc_now
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_models import TERMINAL_RUN_STATUSES, RunStatus, TaskRun

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.queued: frozenset({RunStatus.preparing, RunStatus.cancelled, RunStatus.failed}),
    RunStatus.preparing: frozenset({RunStatus.ready, RunStatus.cancelled, RunStatus.failed, RunStatus.interrupted}),
    RunStatus.ready: frozenset({
        RunStatus.running,
        RunStatus.completed,
        RunStatus.cancelled,
        RunStatus.failed,
        RunStatus.recovering,
    }),
    RunStatus.running: frozenset({
        RunStatus.awaiting_approval,
        RunStatus.completed,
        RunStatus.ready,
        RunStatus.cancelled,
        RunStatus.interrupted,
        RunStatus.failed,
        RunStatus.recovering,
    }),
    RunStatus.awaiting_approval: frozenset({
        RunStatus.running,
        RunStatus.cancelled,
        RunStatus.interrupted,
        RunStatus.failed,
        RunStatus.recovering,
    }),
    RunStatus.interrupted: frozenset({RunStatus.recovering, RunStatus.cancelled, RunStatus.failed}),
    RunStatus.recovering: frozenset({
        RunStatus.preparing,
        RunStatus.ready,
        RunStatus.running,
        RunStatus.cancelled,
        RunStatus.failed,
    }),
    RunStatus.completed: frozenset(),
    RunStatus.cancelled: frozenset(),
    RunStatus.failed: frozenset({RunStatus.recovering}),
}


class InvalidRunTransition(StateConflict):
    pass


def transition_run(run: TaskRun, target: RunStatus, *, error: str | None = None) -> TaskRun:
    """Apply the one canonical Task Run transition table."""
    if target == run.status:
        return run
    if target not in _ALLOWED_TRANSITIONS[run.status]:
        raise InvalidRunTransition(f"Task Run cannot move from {run.status.value} to {target.value}")
    now = utc_now()
    run.status = target
    run.updated_at = now
    run.last_error = error
    if target == RunStatus.running and run.started_at is None:
        run.started_at = now
    if target in TERMINAL_RUN_STATUSES:
        run.finished_at = now
        run.lease_id = None
        run.lease_expires_at = None
    return run
