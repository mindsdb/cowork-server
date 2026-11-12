"""Undo for quick edits (ENG-4227): every file an edit changed, not only the
tracked source.

A quick edit on a fullstack or component artifact usually changes files the
revision journal does not track (``app.js``, ``backend.py``). So before the
editor runs, the server reads the artifact's content files (``capture``);
after an applied edit it keeps, for each file the edit changed or created,
the bytes it had before and a hash of what the edit left (``save_undo``).
``undo`` puts every one of them back, and removes the files the edit
created, all or nothing, refusing when any of them changed since.

Stored in the artifact's journal (``.revisions/quick-edits/``): one record
per edit (``<edit id>.json``) and the bytes it restores, content-addressed
(``blobs/<sha256>``). Only the newest ``MAX_UNDO_RECORDS`` edits are kept.

Every file is reached the way the journal reaches its own: the artifact
folder and each folder below it are opened without following a link, a file
read must be a regular file within its cap, and a write goes to a new file
renamed over the name, so nothing planted at a name is written through.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import uuid
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_atomic_write,
    dir_open,
    dir_unlink,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)
from cowork.services.artifact_journal import (
    JOURNAL_DIRNAME,
    JournalFileRefused,
    iter_journal_json,
    prune_journal_folder,
    read_journal_file,
    read_journal_json,
    write_journal_file,
)
from cowork.services.artifact_lock import artifact_lock

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_UNDO_RECORDS",
    "QuickEditAlreadyUndone",
    "QuickEditFilesChanged",
    "capture",
    "new_edit_id",
    "save_undo",
    "undo",
]

#: Quick edits whose undo is kept, newest first.
MAX_UNDO_RECORDS = 10
_FOLDER = "quick-edits"
_BLOBS = "blobs"
#: A file larger than this is never captured, and a capture past the total is
#: dropped: the edit then runs without an undo (``undoable`` false).
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CAPTURE_BYTES = 32 * 1024 * 1024
_MAX_RECORD_BYTES = 256 * 1024
_MAX_FILES = 200
_EDIT_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class QuickEditFilesChanged(Exception):
    """A file the edit changed was changed again since: undo would lose it."""

    def __init__(self, paths: list[str]):
        super().__init__("These files changed after the quick edit")
        self.paths = paths


class QuickEditAlreadyUndone(Exception):
    pass


def new_edit_id() -> str:
    return uuid.uuid4().hex


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parts(rel_path: str) -> tuple[str, ...]:
    from cowork.services import artifact_files

    parts = artifact_files.relative_file_parts(rel_path)
    if JOURNAL_DIRNAME in parts:
        raise ValueError("invalid path")
    return parts


def _parent(root: PinnedDir, parts: tuple[str, ...], stack: ExitStack) -> PinnedDir:
    current = root
    for name in parts[:-1]:
        current = open_pinned_child(current, name)
        stack.callback(current.close)
    return current


@dataclass(frozen=True)
class _File:
    content: bytes
    mode: int


def _read(root: PinnedDir, rel_path: str, *, max_bytes: int = MAX_FILE_BYTES) -> _File | None:
    """The file's bytes and mode, None when nothing is at the name. Raises
    for a link, a non-regular file, or one past ``max_bytes``."""
    parts = _parts(rel_path)
    with ExitStack() as stack:
        try:
            parent = _parent(root, parts, stack)
            fd = dir_open(parent, parts[-1], os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        except FileNotFoundError:
            return None
        try:
            found = os.fstat(fd)
            if not stat.S_ISREG(found.st_mode):
                raise OSError(f"not a regular file: {rel_path}")
            if found.st_size > max_bytes:
                raise OSError(f"too large to keep for undo: {rel_path}")
            return _File(b"".join(read_bounded(fd, found.st_size)), stat.S_IMODE(found.st_mode))
        finally:
            os.close(fd)


def _write(root: PinnedDir, rel_path: str, file: _File) -> None:
    parts = _parts(rel_path)
    with ExitStack() as stack:
        dir_atomic_write(_parent(root, parts, stack), parts[-1], content=file.content, mode=file.mode)


def _remove(root: PinnedDir, rel_path: str) -> None:
    parts = _parts(rel_path)
    with ExitStack() as stack:
        try:
            dir_unlink(_parent(root, parts, stack), parts[-1])
        except FileNotFoundError:
            pass


# ── before the edit ──────────────────────────────────────────────────────


def capture(folder: Path, paths: Iterable[str]) -> dict[str, tuple[bytes, int] | None] | None:
    """What each of ``paths`` holds now (bytes and mode, or None when absent),
    or None when it is too much to keep (the edit then has no undo)."""
    captured: dict[str, tuple[bytes, int] | None] = {}
    total = 0
    try:
        with pinned_dir(folder, nofollow_base=True) as root:
            for rel_path in list(dict.fromkeys(paths))[:_MAX_FILES]:
                found = _read(root, rel_path)
                captured[rel_path] = None if found is None else (found.content, found.mode)
                total += len(found.content) if found is not None else 0
                if total > MAX_CAPTURE_BYTES:
                    return None
    except (OSError, ValueError):
        logger.warning("Could not keep the artifact's files for a quick edit's undo", exc_info=True)
        return None
    return captured


# ── after an applied edit ────────────────────────────────────────────────


def save_undo(
    folder: Path,
    *,
    edit_id: str,
    before: dict[str, tuple[bytes, int] | None],
    changed: Iterable[str],
    source_path: str | None,
    revision_id: str | None,
    previous_revision_id: str | None,
) -> list[str] | None:
    """Keep the undo of an applied edit: the files it changed or created.

    The files are ``changed`` (what anton reports) plus every captured file
    whose bytes differ now, so a file the editor wrote without naming it is
    restored too. A changed file that was not captured and did not exist
    before is one the edit created. Returns the paths kept, or None when
    nothing could be kept.
    """
    if not _EDIT_ID.match(edit_id):
        raise ValueError("invalid edit id")
    entries: list[dict] = []
    with artifact_lock(folder):
        with pinned_dir(folder, nofollow_base=True) as root:
            candidates = list(dict.fromkeys([*changed, *before]))
            for rel_path in candidates[:_MAX_FILES]:
                try:
                    now = _read(root, rel_path)
                except (OSError, ValueError):
                    return None
                was = before.get(rel_path)
                if rel_path not in before and now is None:
                    continue
                was_hash = _sha(was[0]) if was is not None else None
                now_hash = _sha(now.content) if now is not None else None
                if was_hash == now_hash:
                    continue
                if was is not None:
                    write_journal_file(folder, _FOLDER, _BLOBS, was_hash, content=was[0])
                entries.append({
                    "path": rel_path,
                    "before": was_hash,
                    "beforeMode": was[1] if was is not None else None,
                    "after": now_hash,
                })
        if not entries:
            return None
        record = {
            "version": 1,
            "id": edit_id,
            "createdAt": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "sourcePath": source_path,
            "revisionId": revision_id,
            "previousRevisionId": previous_revision_id,
            "files": entries,
            "undone": False,
        }
        write_journal_file(folder, _FOLDER, f"{edit_id}.json", content=json.dumps(record).encode("utf-8"))
        _prune(folder)
    return [entry["path"] for entry in entries]


def _records(folder: Path) -> list[dict]:
    records = [
        raw for _stem, raw in iter_journal_json(
            folder, _FOLDER, suffix=".json", accept=lambda stem: bool(_EDIT_ID.match(stem)),
            max_bytes=_MAX_RECORD_BYTES,
        )
        if isinstance(raw, dict)
    ]
    return sorted(records, key=lambda r: str(r.get("createdAt") or ""), reverse=True)


def _prune(folder: Path) -> None:
    """Keep the newest MAX_UNDO_RECORDS records and the bytes they restore."""
    try:
        kept = _records(folder)[:MAX_UNDO_RECORDS]
        prune_journal_folder(folder, _FOLDER, keep={f"{r.get('id')}.json" for r in kept})
        blobs = {
            entry.get("before") for r in kept for entry in r.get("files") or [] if isinstance(entry, dict)
        }
        prune_journal_folder(folder, _FOLDER, _BLOBS, keep={b for b in blobs if isinstance(b, str)})
    except OSError:
        # The record is saved; cleanup is best-effort.
        logger.warning("Could not prune old quick edit undo records", exc_info=True)


# ── undo ─────────────────────────────────────────────────────────────────


def _record(folder: Path, edit_id: str) -> dict:
    if not _EDIT_ID.match(edit_id or ""):
        raise FileNotFoundError("Quick edit not found")
    try:
        raw = read_journal_json(folder, _FOLDER, f"{edit_id}.json", max_bytes=_MAX_RECORD_BYTES)
    except (JournalFileRefused, ValueError):
        raw = None
    if not isinstance(raw, dict) or raw.get("id") != edit_id or not isinstance(raw.get("files"), list):
        raise FileNotFoundError("Quick edit not found")
    return raw


def undo(
    folder: Path,
    metadata: dict,
    artifact_id: str,
    edit_id: str,
    *,
    expected_revision_id: str | None,
    actor_id: str | None,
) -> dict:
    """Put back every file the quick edit changed, and remove those it
    created, all or nothing.

    ``expected_revision_id`` is the head of the tracked source the caller
    saw (``RevisionConflict`` otherwise, as the restore route). Any file the
    edit touched that changed since raises ``QuickEditFilesChanged``: undo
    would discard that later work. When the tracked source is among the
    files, its restored text is journalled as a new revision.
    """
    from cowork.services.artifact_revisions import (
        RevisionConflict,
        _append_revision,
        _current_source_locked,
        _read_manifest,
    )

    with artifact_lock(folder):
        record = _record(folder, edit_id)
        if record.get("undone"):
            raise QuickEditAlreadyUndone("This quick edit was already undone")
        source = _current_source_locked(folder, metadata, artifact_id, record.get("sourcePath") or None)
        if expected_revision_id is not None and source.revision.id != expected_revision_id:
            raise RevisionConflict(source.revision.to_json())

        plan: list[tuple[str, _File | None, _File | None]] = []  # path, restore to, current
        changed_since: list[str] = []
        with pinned_dir(folder, nofollow_base=True) as root:
            for entry in record["files"]:
                path = entry.get("path") if isinstance(entry, dict) else None
                before, after = entry.get("before"), entry.get("after")
                if not isinstance(path, str) or (before is not None and not _SHA.match(str(before))):
                    raise FileNotFoundError("Quick edit not found")
                current = _read(root, path)
                if (_sha(current.content) if current is not None else None) != after:
                    changed_since.append(path)
                    continue
                if before is None:
                    plan.append((path, None, current))
                    continue
                content = read_journal_file(folder, _FOLDER, _BLOBS, before)
                if content is None or _sha(content) != before:
                    raise FileNotFoundError("Quick edit undo data is missing")
                mode = entry.get("beforeMode")
                plan.append((path, _File(content, mode if isinstance(mode, int) else 0o644), current))
            if changed_since:
                raise QuickEditFilesChanged(changed_since)

            done: list[tuple[str, _File | None]] = []
            try:
                for path, restore_to, current in plan:
                    if restore_to is None:
                        _remove(root, path)
                    else:
                        _write(root, path, restore_to)
                    done.append((path, current))
            except BaseException:
                # Put back what this undo already changed: all or nothing.
                for path, current in reversed(done):
                    try:
                        if current is None:
                            _remove(root, path)
                        else:
                            _write(root, path, current)
                    except Exception:
                        logger.exception("Could not roll back a failed quick edit undo at %s", path)
                raise

        revision = None
        source_path = source.path
        restored_source = next((r for p, r, _c in plan if p == source_path), None)
        if restored_source is not None:
            manifest = _read_manifest(folder)
            head = manifest.head(source_path)
            revision = _append_revision(
                folder,
                manifest,
                artifact_id=artifact_id,
                rel_path=source_path,
                content=restored_source.content,
                actor_kind="manual",
                actor_id=actor_id,
                summary="Undid quick edit",
                base_revision_id=head.id if head is not None else None,
            ).to_json()

        record["undone"] = True
        write_journal_file(folder, _FOLDER, f"{edit_id}.json", content=json.dumps(record).encode("utf-8"))
        return {
            "editId": edit_id,
            "restored": [p for p, r, _c in plan if r is not None],
            "removed": [p for p, r, _c in plan if r is None],
            "revision": revision,
        }
