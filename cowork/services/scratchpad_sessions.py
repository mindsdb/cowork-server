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
import shutil
from pathlib import Path
from uuid import UUID

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

# anton's bucket for pads created without a conversation scope. cowork-server normally
# passes one, but not always: `services/connectors/probe.py` builds a `ChatSession`
# with no conversation for connector probing. Never sweep it — it is not an orphan, it
# is another component's live state.
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


def conversation_session_dir(project_path: str | Path, conversation_id: UUID | str) -> Path | None:
    """Snapshot directory for one conversation, or None if the id isn't a UUID."""
    canonical = _canonical_conversation_id(conversation_id)
    if canonical is None:
        return None
    return sessions_root(project_path) / canonical


def _remove_contained(target: Path, root: Path) -> bool:
    """`rmtree` *target*, but only if it really sits inside *root*.

    Both legs are resolved before comparing, so a symlink or a `..` segment
    cannot redirect the delete outside the snapshot tree. Returns True if
    something was removed. Never raises — retention is best-effort and must not
    fail a conversation delete or block boot.
    """
    try:
        resolved_root = root.resolve()
        resolved = target.resolve()
    except OSError:
        return False
    if resolved == resolved_root:
        # Refuse to delete the root itself — only per-conversation children.
        return False
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        logger.warning("refusing to remove %s: outside %s", resolved, resolved_root)
        return False
    if not resolved.is_dir():
        return False
    try:
        shutil.rmtree(resolved)
        return True
    except OSError:
        logger.exception("failed to remove scratchpad session dir %s", resolved)
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
    target = conversation_session_dir(project_path, conversation_id)
    if target is None:
        logger.warning("refusing to remove sessions for a non-UUID conversation id")
        return False
    return _remove_contained(target, sessions_root(project_path))


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
        live = {str(cid) for cid in session.exec(select(Conversation.id)).all()}
    except Exception:
        logger.exception("scratchpad-session sweep: could not read projects/conversations")
        return 0

    for project in projects:
        root = sessions_root(project.path) if project.path else None
        if root is None or not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.name in live:
                continue
            if child.name == _UNSCOPED_BUCKET:
                continue
            # Rebuild the target from a validated id rather than deleting the directory
            # entry we just listed. Same boundary validation as the delete path, and it
            # keeps the allowlist ("only conversation-shaped names") in one place.
            target = conversation_session_dir(project.path, child.name)
            if target is None:
                continue
            if _remove_contained(target, root):
                removed += 1
    if removed:
        logger.info("Swept %d orphaned scratchpad session dir(s) on boot", removed)
    return removed
