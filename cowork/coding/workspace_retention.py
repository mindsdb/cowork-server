from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from cowork.coding.contracts import CodingSession, SessionStatus, WorkspaceKind, utc_now
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.runtime import RuntimeManager
from cowork.coding.store import CodingStore
from cowork.coding.workspace import WorkspaceError, WorkspaceManager

logger = logging.getLogger(__name__)

_RELEASABLE_KINDS = {WorkspaceKind.git_worktree, WorkspaceKind.local_copy}
_BUSY_RUN_STATUSES = {"queued", "preparing", "running", "awaiting_approval", "recovering"}
# Bounded so a large backlog cannot hold up startup or a single pass.
MAX_RELEASES_PER_PASS = 20


def _automation_enabled(session: CodingSession) -> bool:
    policy = session.delivery_policy
    return any((
        policy.fix_failing_checks,
        policy.mark_ready_when_passing,
        policy.merge_when_approved,
        policy.complete_source_after_merge,
        policy.archive_after_merge,
    ))


@dataclass(frozen=True)
class RetentionPolicy:
    enabled: bool = True
    keep_count: int = 10
    min_idle: timedelta = timedelta(hours=24)


@dataclass(frozen=True)
class _Item:
    key: str
    source_path: str
    workspace_path: str
    kind: WorkspaceKind


class WorkspaceRetention:
    """Release idle task workspaces to reclaim disk, and rebuild them on demand.

    A released task keeps its record and its workspace paths. Its changes are
    saved by ``WorkspaceManager.release`` and reapplied by ``restore`` at the
    same paths, so the rest of the service never sees a different workspace.
    """

    def __init__(
        self,
        *,
        store: CodingStore,
        workspaces: WorkspaceManager,
        runtimes: RuntimeManager,
        lock: threading.RLock,
        running: dict,
        maintenance: set[str],
        is_remote: Callable[[CodingSession], bool],
        control_view: Callable[[CodingSession], CodingSession],
    ) -> None:
        self.store = store
        self.workspaces = workspaces
        self.runtimes = runtimes
        self.lock = lock
        self.running = running
        self.maintenance = maintenance
        self.is_remote = is_remote
        # Run status lives in the control plane; the stored session can lag it.
        self.control_view = control_view

    def releasable(self, session: CodingSession, *, reserved: bool = False) -> bool:
        """Whether releasing this task's workspace can disturb nothing in use.

        ``reserved`` is set by the caller that already holds this task's
        maintenance reservation.
        """
        items = self._items(session)
        return bool(
            items
            and not self.is_remote(session)
            and session.workspace_released_at is None
            and not session.pinned
            and session.status not in {SessionStatus.running, SessionStatus.awaiting_approval}
            and session.run_status not in _BUSY_RUN_STATUSES
            and session.pending_approval is None
            and session.pending_question is None
            and not session.queued_instructions
            and session.id not in self.running
            and (reserved or session.id not in self.maintenance)
            and not self.runtimes.terminal_is_running(session.id)
            # Delivery automation polls its tasks' workspaces every minute and
            # would immediately rebuild a released one.
            and (session.archived or not _automation_enabled(session))
            and all(item.kind in _RELEASABLE_KINDS for item in items)
        )

    def release(self, session_id: str) -> bool:
        """Release one task's workspace. Returns whether it was released."""
        if not self._reserve(session_id):
            return False
        try:
            session = self.control_view(self.store.load_session(session_id))
            if not self.releasable(session, reserved=True):
                return False
            with self.runtimes.session_lock(session_id):
                # An idle engine process or terminal can hold the folder open,
                # which stops it being removed on Windows.
                self.runtimes.close_locked(session_id)
                # Marked first, so a failed write leaves the folders in place
                # and a crash part-way leaves a task that restores on next use.
                self._mark_released(session_id, utc_now())
                items = self._items(session)
                try:
                    for item in items:
                        if not self.workspaces.release(item.key, item.source_path, item.workspace_path, item.kind):
                            raise WorkspaceError(f"{item.workspace_path} cannot be released safely")
                except Exception as exc:
                    logger.info("Keeping the workspace for coding task %s: %s", session_id, exc)
                    try:
                        self._restore_released(items)
                    except Exception:
                        # Still marked, so the next use retries the restore.
                        logger.exception("Could not restore the workspace of coding task %s after a failed release", session_id)
                    else:
                        self._mark_released(session_id, None)
                    return False
            return True
        finally:
            self._unreserve(session_id)

    def ensure_restored(self, session_id: str, *, reserved: bool = False) -> None:
        """Rebuild a released workspace with its changes before it is used.

        ``reserved`` is set by a caller that already holds this task's
        maintenance reservation.
        """
        if self.store.load_session(session_id).workspace_released_at is None:
            return
        # Opening a task requests several views at once; the first rebuilds
        # the workspace and the others wait for it rather than fail. This is
        # the runtime lock so terminal calls, which already hold it, cannot
        # take the two locks in the opposite order.
        with self.runtimes.session_lock(session_id):
            self._restore(session_id, reserved=reserved)

    def _restore(self, session_id: str, *, reserved: bool) -> None:
        if self.store.load_session(session_id).workspace_released_at is None:
            return
        if not reserved:
            with self.lock:
                if session_id in self.maintenance:
                    raise RuntimeError("This coding task's workspace is being updated. Try again in a moment")
                self.maintenance.add(session_id)
        try:
            session = self.store.load_session(session_id)
            if session.workspace_released_at is None:
                return
            self._restore_released(self._items(session))
            self._mark_released(session_id, None)
        finally:
            if not reserved:
                self._unreserve(session_id)

    def discard(self, session: CodingSession) -> None:
        """Forget saved release state for a task that is being deleted."""
        if session.workspace_released_at is None:
            return
        for item in self._items(session):
            try:
                self.workspaces.discard_release(item.key, item.source_path)
            except (OSError, WorkspaceError, ValueError) as exc:
                logger.warning("Could not discard saved workspace state for %s: %s", item.key, exc)

    def run_policy(self, policy: RetentionPolicy, now: datetime | None = None) -> list[str]:
        """Release workspaces the policy no longer keeps. Returns released task ids."""
        now = now or utc_now()
        sessions = [self.control_view(item) for item in self.store.list_sessions()]
        archived = [item for item in sessions if item.archived and self.releasable(item)]
        candidates: list[CodingSession] = list(archived)
        if policy.enabled:
            active = sorted(
                (item for item in sessions if not item.archived and item.workspace_released_at is None),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            candidates.extend(
                item for item in active[policy.keep_count:]
                if now - item.updated_at >= policy.min_idle and self.releasable(item)
            )
        released: list[str] = []
        for session in candidates[:MAX_RELEASES_PER_PASS]:
            try:
                if self.release(session.id):
                    released.append(session.id)
            except Exception:
                logger.exception("Could not release the workspace for coding task %s", session.id)
        return released

    def _restore_released(self, items: list[_Item]) -> None:
        # Saved state, not a missing folder, marks a released item: a removal
        # that failed part-way leaves a folder that is present but incomplete.
        for item in items:
            if self.workspaces.is_released(item.key, item.kind):
                self.workspaces.restore(item.key, item.source_path, item.workspace_path, item.kind)

    def _mark_released(self, session_id: str, released_at: datetime | None) -> None:
        self.store.update_session(
            session_id,
            lambda current: setattr(current, "workspace_released_at", released_at),
            touch_updated_at=False,
        )

    def _reserve(self, session_id: str) -> bool:
        with self.lock:
            if session_id in self.running or session_id in self.maintenance:
                return False
            self.maintenance.add(session_id)
            return True

    def _unreserve(self, session_id: str) -> None:
        with self.lock:
            self.maintenance.discard(session_id)

    @staticmethod
    def _items(session: CodingSession) -> list[_Item]:
        if session.workspaces:
            return [
                _Item(
                    key=ProjectWorkspaceManager._key(session.id, workspace.folder_id),
                    source_path=workspace.source_path,
                    workspace_path=workspace.workspace_path,
                    kind=workspace.workspace_kind,
                )
                for workspace in session.workspaces
            ]
        if not session.workspace_path:
            return []
        return [_Item(session.id, session.source_path, session.workspace_path, session.workspace_kind)]
