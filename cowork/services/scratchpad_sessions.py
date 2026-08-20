"""Retention for anton's scratchpad namespace snapshots (ENG-1124).

anton persists a scratchpad's Python namespace so variables survive the pad
process being replaced — which happens on **every user turn**, because a fresh
`ChatSession` (and so a fresh `ScratchpadManager`) is built per turn. Without
that snapshot the agent rebuilds work after every reply; see anton#283.

The snapshots live under the project, keyed by conversation and then by pad:

    <project>/.anton/scratchpad-sessions/<conversation_id>/<pad>.pkl

anton owns writing them. This module owns getting rid of them, because nothing
else does: `delete_conversation` cleans DB rows and uploaded attachments but
knows nothing about these files, and only `delete_project` reclaims anything
(via `shutil.rmtree` of the whole project dir). Left alone they accumulate one
directory per conversation, forever — on desktop inside the user's own home
directory, with no storage view, no "clear data", and no uninstall cleanup.

They also matter more than plain disk: a namespace can contain the injected
`DS_*` datasource credentials (ENG-392), so an un-pruned snapshot is data at
rest outliving the conversation that was allowed to see it.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from pathlib import Path
from uuid import UUID

from cowork.common.paths import opened_subdir_nofollow

logger = logging.getLogger(__name__)

# Must match `snapshot_dir` in anton's `core/backends/local.py`.
#
# ⚠️ Cross-repo invariant: anton *sanitises* the session id into this directory name
# (`[A-Za-z0-9._-]`, leading dots stripped); we build it from a canonical UUID string
# (`_canonical_conversation_id`). Those agree because a UUID contains nothing the
# sanitiser touches — `test_uuid_needs_no_sanitising` pins exactly that. If a host ever
# supplied a non-UUID session id, anton would mangle it, we would refuse it, the two
# would disagree on the path, and the prune would silently stop matching.
_SESSIONS_DIRNAME = "scratchpad-sessions"

# anton no longer writes an unscoped bucket at all — it refuses to persist without a
# session id, precisely because `CredentialProbe` runs unscoped and parses `DS_*`
# credentials (anton#283 review). Kept as a guard anyway: an older anton in the field
# still creates it, and it is another component's live state, not an orphan.
_UNSCOPED_BUCKET = "_no-session"


def sessions_root(project_path: str | Path) -> Path:
    """The `scratchpad-sessions` directory for a project."""
    return Path(project_path) / ".anton" / _SESSIONS_DIRNAME


def _canonical_conversation_id(conversation_id: UUID | str) -> str | None:
    """The conversation id as a canonical UUID string, or None if it isn't one.

    The id reaches us from an HTTP path parameter, and it becomes a filesystem path
    segment that we then delete recursively — so it is validated at the boundary rather
    than relying on the containment check downstream (CodeQL py/path-injection, high).
    Parsing through `UUID` is the validation: it raises on anything that is not a UUID,
    and `str()` of the result is canonical lowercase hex-with-hyphens, so the value
    provably cannot traverse, cannot collide, and cannot carry a separator.
    """
    try:
        return str(UUID(str(conversation_id)))
    except (ValueError, AttributeError, TypeError):
        return None


def conversation_session_dir(
    project_path: str | Path, conversation_id: UUID | str
) -> Path | None:
    """Snapshot directory for one conversation, or None if the id isn't a UUID."""
    canonical = _canonical_conversation_id(conversation_id)
    if canonical is None:
        return None
    return sessions_root(project_path) / canonical


def _rmtree_session_child(project_path: str | Path, canonical: str) -> bool:
    """`rmtree` ``<project>/.anton/scratchpad-sessions/<canonical>`` safely.

    Resolving both legs and comparing (the previous approach) does NOT defend a
    symlinked base: `Path.resolve()` follows a `.anton` or `scratchpad-sessions`
    link the agent may have planted, so the containment check passed against an
    already-escaped path. Instead pin the sessions dir by an ``O_NOFOLLOW``
    descriptor (refusing a link at any level) and delete *canonical* relative to
    it, so the kernel resolves only that one component against the pinned inode.

    *canonical* is a validated UUID string (a direct child, no separators).
    Returns True if a directory was removed. Never raises: retention is
    best-effort and must not fail a conversation delete or block boot.
    """
    try:
        with opened_subdir_nofollow(
            Path(project_path), ".anton", _SESSIONS_DIRNAME
        ) as fd:
            try:
                st = os.lstat(canonical, dir_fd=fd)
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(st.st_mode):
                os.unlink(canonical, dir_fd=fd)  # drop a planted link, never follow it
                return False
            if not stat.S_ISDIR(st.st_mode):
                return False
            shutil.rmtree(canonical, dir_fd=fd)
            return True
    except OSError:
        logger.warning(
            "failed to remove scratchpad session dir %s in %s",
            canonical,
            project_path,
            exc_info=True,
        )
        return False


def remove_conversation_sessions(
    project_path: str | Path | None, conversation_id: UUID | str
) -> bool:
    """Drop one conversation's snapshots. Returns True if a directory was removed.

    Call this **after** the conversation's row delete has committed, the same
    ordering `delete_conversation` already uses for attachment bytes (ENG-701):
    a crash between the two must not leave a half-deleted conversation behind.
    """
    if not project_path:
        return False
    canonical = _canonical_conversation_id(conversation_id)
    if canonical is None:
        logger.warning("refusing to remove sessions for a non-UUID conversation id")
        return False
    return _rmtree_session_child(project_path, canonical)


def sweep_orphan_sessions(session) -> int:
    """Remove snapshot dirs whose conversation no longer exists. Returns the count.

    `remove_conversation_sessions` covers conversations deleted from now on; this
    covers the ones already gone — including everything deleted before this
    shipped, and any path that drops a conversation row without going through
    `ConversationService`. Cheap: one query plus a directory listing per project.

    Best-effort by contract: it runs at boot beside the other non-fatal recovery
    steps, so any failure is logged and swallowed rather than blocking startup.
    """
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.models.project import Project

    removed = 0
    try:
        # Deliberately a plain `select` on a raw session rather than the org-scoped
        # `ScopedSession.select`: this is a filesystem sweep over every project on the
        # box, so it must see all conversation ids. Scoping it would make every other
        # org's live conversations look orphaned and delete their snapshots.
        projects = session.exec(select(Project)).all()
        # Liveness is per (project, conversation), not global by conversation id. A
        # conversation that moved from project A to B keeps a directory under A; treating
        # it as live everywhere would strand that directory forever, and conversation
        # delete only ever cleans the current project. Scoping the check lets the sweep
        # reclaim the one left behind by the move.
        live_by_project: dict = {}
        for cid, pid in session.exec(
            select(Conversation.id, Conversation.project_id)
        ).all():
            live_by_project.setdefault(str(pid), set()).add(str(cid))
    except Exception:
        logger.exception(
            "scratchpad-session sweep: could not read projects/conversations"
        )
        return 0

    for project in projects:
        if not project.path:
            continue
        live = live_by_project.get(str(project.id), set())
        try:
            # Pin the sessions dir by O_NOFOLLOW descriptor and scan/delete
            # relative to it: a planted `.anton`/`scratchpad-sessions` symlink is
            # refused (raising here, caught below), and a symlinked child entry
            # is skipped rather than followed out of the tree.
            with opened_subdir_nofollow(
                Path(project.path), ".anton", _SESSIONS_DIRNAME
            ) as fd:
                for entry in os.scandir(fd):
                    if entry.name in live or entry.name == _UNSCOPED_BUCKET:
                        continue
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        continue
                    # Only conversation-shaped (UUID) names, same boundary validation
                    # as the delete path.
                    if _canonical_conversation_id(entry.name) is None:
                        continue
                    try:
                        shutil.rmtree(entry.name, dir_fd=fd)
                        removed += 1
                    except OSError:
                        logger.exception(
                            "failed to remove scratchpad session dir %s", entry.name
                        )
        except OSError:
            continue
    if removed:
        logger.info("Swept %d orphaned scratchpad session dir(s) on boot", removed)
    return removed
