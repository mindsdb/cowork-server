"""Atomic artifact revisions and agent-repair handoffs.

Both records share one artifact-wide lock and journal directory so a repair can
only be compared with, accepted against, or rejected from its exact source
revision. Keeping that transaction boundary together is intentional.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cowork.services.artifact_lock import artifact_lock

EDITABLE_EXTENSIONS = frozenset({".md", ".txt", ".html", ".htm"})
MAX_SOURCE_BYTES = 2_000_000
MAX_REVISIONS = 80
JOURNAL_DIRNAME = ".revisions"
# A queued repair whose turn never reported inside this window is presumed
# dead. Without it a turn killed between minting the handoff and starting the
# agent gates that source path forever.
QUEUED_REPAIR_TTL_SECONDS = 3600


class RevisionConflict(RuntimeError):
    def __init__(self, current: dict):
        super().__init__("Artifact changed since this edit began")
        self.current = current


class RevisionValidationError(ValueError):
    pass


class RepairAlreadyPending(RevisionValidationError):
    """The path already has a repair that still gates new agent work.

    Carries the blocker so the caller can name the comment it belongs to and
    offer to discard it, rather than reporting a fact with no way to act on it.
    """

    def __init__(self, repair: dict):
        super().__init__("This artifact already has an agent repair awaiting review")
        self.repair = repair


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _journal(folder: Path) -> Path:
    return folder / JOURNAL_DIRNAME


def _manifest_path(folder: Path) -> Path:
    return _journal(folder) / "manifest.json"


def _pending_path(folder: Path) -> Path:
    return _journal(folder) / "pending-source-write.json"


def _repairs_dir(folder: Path) -> Path:
    return _journal(folder) / "repairs"


def _read_manifest(folder: Path) -> dict:
    try:
        data = json.loads(_manifest_path(folder).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "revisions": []}
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionValidationError("Artifact revision history is unreadable") from exc
    if not isinstance(data, dict) or not isinstance(data.get("revisions"), list):
        raise RevisionValidationError("Artifact revision history is invalid")
    return data


def _atomic_bytes(path: Path, content: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
        # `fsync` above durably writes the file bytes; syncing the containing
        # directory makes the rename itself durable across a host crash on
        # filesystems that support directory descriptors. Windows does not,
        # so it safely keeps the atomic-replace guarantee without this extra
        # durability step.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(folder: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(_manifest_path(folder), encoded)


def _write_pending(folder: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(_pending_path(folder), encoded, mode=0o600)


def _clear_pending(folder: Path) -> None:
    try:
        _pending_path(folder).unlink(missing_ok=True)
    except OSError:
        # The transaction is already committed or deliberately abandoned. A
        # stale marker is harmless and will be recognized on the next read.
        pass


def _write_repair(folder: Path, repair: dict) -> None:
    encoded = (json.dumps(repair, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(_repairs_dir(folder) / f"{repair['id']}.json", encoded, mode=0o600)


def _read_repair(folder: Path, repair_id: str) -> dict:
    try:
        uuid.UUID(repair_id)
        repair = json.loads(
            (_repairs_dir(folder) / f"{repair_id}.json").read_text(encoding="utf-8")
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError("Agent repair not found") from exc
    if not isinstance(repair, dict) or repair.get("id") != repair_id:
        raise FileNotFoundError("Agent repair not found")
    return repair


def _repair_records(folder: Path) -> list[dict]:
    try:
        paths = sorted(_repairs_dir(folder).glob("*.json"))
    except OSError:
        return []
    records: list[dict] = []
    for path in paths:
        try:
            repair = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(repair, dict):
            records.append(repair)
    return records


def _actionable_repairs_for_path(folder: Path, rel_path: str) -> list[dict]:
    return [
        repair
        for repair in _repair_records(folder)
        if repair.get("status") in {"queued", "ready"}
        and repair.get("path") == rel_path
    ]


def _parse_timestamp(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_superseded(manifest: dict, repair: dict) -> bool:
    """Whether the agent's revision is no longer head for the repair's path.

    Computed rather than stored, so an artifact wedged by an older build
    recovers on the first request instead of needing a migration.
    """
    revision_id = repair.get("revisionId")
    if not revision_id:
        return False
    latest = next(
        (
            entry
            for entry in reversed(manifest["revisions"])
            if entry.get("path") == repair.get("path")
        ),
        None,
    )
    return latest is not None and latest.get("id") != revision_id


def _partition_actionable_repairs(
    folder: Path, manifest: dict, rel_path: str
) -> tuple[list[dict], list[dict]]:
    """Split one path's actionable repairs into those that still gate a new
    repair and those whose turn is presumed dead.

    A superseded repair appears in neither list. It keeps its `ready` status so
    the owner can still accept or discard the agent's work, but it cannot gate
    a new turn: the content it was computed against is already history.
    """
    now = datetime.now(timezone.utc)
    blocking: list[dict] = []
    dead: list[dict] = []
    for repair in _actionable_repairs_for_path(folder, rel_path):
        if _is_superseded(manifest, repair):
            continue
        created = _parse_timestamp(repair.get("createdAt"))
        if (
            repair.get("status") == "queued"
            and created is not None
            and (now - created).total_seconds() > QUEUED_REPAIR_TTL_SECONDS
        ):
            dead.append(repair)
            continue
        blocking.append(repair)
    return blocking, dead


def has_queued_agent_repair(folder: Path, conversation_id: str) -> bool:
    """Return whether this exact agent turn has a pending repair handoff."""
    return any(
        (
            repair.get("status") == "queued"
            and repair.get("conversationId") == conversation_id
        )
        for repair in _repair_records(folder)
    )


def _finish_queued_repairs(
    folder: Path,
    repairs: list[dict],
    *,
    matching_base_id: str | None,
    revision_id: str | None,
) -> None:
    """Move repairs out of polling state after their attributed turn ends."""
    for repair in repairs:
        if repair.get("baseRevisionId") != matching_base_id:
            repair["status"] = "conflict"
        elif revision_id:
            repair["status"] = "ready"
        else:
            repair["status"] = "no_change"
        repair["revisionId"] = revision_id
        repair["updatedAt"] = _now()
        _write_repair(folder, repair)


def resolve_source(folder: Path, metadata: dict, rel_path: str | None = None) -> tuple[Path, str]:
    """Resolve an editable source without accepting an absolute/client path."""
    canonical_folder = folder.resolve(strict=False)
    candidate_rel = (rel_path or metadata.get("primary") or "").strip().replace("\\", "/")
    if not candidate_rel:
        candidates = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EDITABLE_EXTENSIONS
            and JOURNAL_DIRNAME not in p.relative_to(folder).parts
        )
        if not candidates:
            raise RevisionValidationError("Artifact has no editable source file")
        target = candidates[0]
    else:
        parts = Path(candidate_rel).parts
        if Path(candidate_rel).is_absolute() or ".." in parts or JOURNAL_DIRNAME in parts:
            raise RevisionValidationError("Invalid artifact source path")
        target = (canonical_folder / candidate_rel).resolve(strict=False)
        try:
            target.relative_to(canonical_folder)
        except ValueError as exc:
            raise RevisionValidationError("Invalid artifact source path") from exc
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError("Artifact source not found")
    if target.suffix.lower() not in EDITABLE_EXTENSIONS:
        raise RevisionValidationError("This artifact type is not source-editable")
    return target, target.resolve(strict=False).relative_to(canonical_folder).as_posix()


def _content_for_revision(folder: Path, revision: dict) -> str:
    blob = _journal(folder) / "blobs" / str(revision.get("contentHash", ""))
    try:
        return blob.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RevisionValidationError("Revision content is unavailable") from exc


def _new_revision(
    manifest: dict,
    *,
    artifact_id: str,
    rel_path: str,
    content: bytes,
    actor_kind: str,
    actor_id: str | None,
    summary: str,
    base_revision_id: str | None,
    conversation_id: str | None = None,
    comment_thread_ids: list[str] | None = None,
) -> dict:
    content_hash = _sha(content)
    revisions = manifest["revisions"]
    return {
        "id": str(uuid.uuid4()),
        "number": (int(revisions[-1].get("number", 0)) + 1) if revisions else 1,
        "artifactId": artifact_id,
        "path": rel_path,
        "createdAt": _now(),
        "actor": {"kind": actor_kind, "id": actor_id},
        "summary": (summary or "Updated artifact").strip()[:240],
        "baseRevisionId": base_revision_id,
        "contentHash": content_hash,
        "bytes": len(content),
        "conversationId": conversation_id,
        "commentThreadIds": list(dict.fromkeys(comment_thread_ids or [])),
    }


def _persist_revision(folder: Path, manifest: dict, entry: dict, content: bytes) -> dict:
    """Persist an idempotent revision entry and its content-addressed blob."""
    content_hash = str(entry.get("contentHash") or "")
    if content_hash != _sha(content):
        raise RevisionValidationError("Revision content hash is invalid")
    blob = _journal(folder) / "blobs" / content_hash
    if not blob.exists():
        _atomic_bytes(blob, content, mode=0o600)
    elif _sha(blob.read_bytes()) != content_hash:
        raise RevisionValidationError("Stored revision content is corrupt")
    revisions = manifest["revisions"]
    existing = next((item for item in revisions if item.get("id") == entry.get("id")), None)
    if existing is not None:
        return existing
    if revisions and int(entry.get("number", 0)) <= int(revisions[-1].get("number", 0)):
        entry = {**entry, "number": int(revisions[-1].get("number", 0)) + 1}
    revisions.append(entry)
    pruned = len(revisions) > MAX_REVISIONS
    if pruned:
        del revisions[:-MAX_REVISIONS]
    _write_manifest(folder, manifest)
    if pruned:
        retained_hashes = {str(item.get("contentHash")) for item in revisions}
        try:
            blobs = (_journal(folder) / "blobs").iterdir()
            for path in blobs:
                if path.name not in retained_hashes:
                    path.unlink(missing_ok=True)
        except OSError:
            # The manifest is already durable; orphan cleanup is best-effort.
            pass
    return entry


def _append_revision(
    folder: Path,
    manifest: dict,
    *,
    artifact_id: str,
    rel_path: str,
    content: bytes,
    actor_kind: str,
    actor_id: str | None,
    summary: str,
    base_revision_id: str | None,
    conversation_id: str | None = None,
    comment_thread_ids: list[str] | None = None,
) -> dict:
    entry = _new_revision(
        manifest,
        artifact_id=artifact_id,
        rel_path=rel_path,
        content=content,
        actor_kind=actor_kind,
        actor_id=actor_id,
        summary=summary,
        base_revision_id=base_revision_id,
        conversation_id=conversation_id,
        comment_thread_ids=comment_thread_ids,
    )
    return _persist_revision(folder, manifest, entry, content)


def _recover_pending_source_write(folder: Path) -> None:
    """Finish or safely abandon a source/revision transaction after a crash.

    The artifact lock must be held. A pending edit is completed only while its
    recorded base is still the manifest head and the source is either the old
    or intended content. An unrelated external write always wins.
    """
    pending_path = _pending_path(folder)
    try:
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionValidationError("Pending artifact edit is unreadable") from exc
    if not isinstance(pending, dict) or pending.get("version") != 1:
        raise RevisionValidationError("Pending artifact edit is invalid")
    entry = pending.get("revision")
    rel_path = pending.get("path")
    before_hash = pending.get("beforeContentHash")
    if not isinstance(entry, dict) or not isinstance(rel_path, str) or entry.get("path") != rel_path:
        raise RevisionValidationError("Pending artifact edit is invalid")
    parts = Path(rel_path).parts
    if Path(rel_path).is_absolute() or ".." in parts or JOURNAL_DIRNAME in parts:
        raise RevisionValidationError("Pending artifact edit path is invalid")
    target = (folder / rel_path).resolve(strict=False)
    try:
        target.relative_to(folder.resolve())
        content = (_journal(folder) / "blobs" / str(entry.get("contentHash") or "")).read_bytes()
        current = target.read_bytes()
    except (OSError, ValueError) as exc:
        raise RevisionValidationError("Pending artifact edit cannot be recovered") from exc
    if _sha(content) != entry.get("contentHash"):
        raise RevisionValidationError("Pending artifact edit content is invalid")

    manifest = _read_manifest(folder)
    revisions = manifest["revisions"]
    if any(item.get("id") == entry.get("id") for item in revisions):
        # The commit reached the manifest. A later external source edit must not
        # be overwritten merely because cleanup was interrupted.
        _clear_pending(folder)
        return

    latest = next((item for item in reversed(revisions) if item.get("path") == rel_path), None)
    latest_id = latest.get("id") if latest else None
    current_hash = _sha(current)
    if latest_id != entry.get("baseRevisionId") or current_hash not in {
        before_hash,
        entry.get("contentHash"),
    }:
        _clear_pending(folder)
        return
    if current_hash == before_hash:
        _atomic_bytes(target, content, mode=target.stat().st_mode & 0o777)
    _persist_revision(folder, manifest, entry, content)
    _clear_pending(folder)


def _commit_source_revision(
    folder: Path,
    manifest: dict,
    target: Path,
    *,
    artifact_id: str,
    rel_path: str,
    before_content: bytes,
    content: bytes,
    actor_kind: str,
    actor_id: str | None,
    summary: str,
    base_revision_id: str,
    conversation_id: str | None = None,
    comment_thread_ids: list[str] | None = None,
) -> dict:
    """Atomically couple a source replacement to its attributed revision."""
    entry = _new_revision(
        manifest,
        artifact_id=artifact_id,
        rel_path=rel_path,
        content=content,
        actor_kind=actor_kind,
        actor_id=actor_id,
        summary=summary,
        base_revision_id=base_revision_id,
        conversation_id=conversation_id,
        comment_thread_ids=comment_thread_ids,
    )
    blob = _journal(folder) / "blobs" / entry["contentHash"]
    if not blob.exists():
        _atomic_bytes(blob, content, mode=0o600)
    _write_pending(
        folder,
        {
            "version": 1,
            "path": rel_path,
            "beforeContentHash": _sha(before_content),
            "revision": entry,
        },
    )
    _atomic_bytes(target, content, mode=target.stat().st_mode & 0o777)
    _persist_revision(folder, manifest, entry, content)
    _clear_pending(folder)
    return entry


def _current_source_locked(
    folder: Path, metadata: dict, artifact_id: str, rel_path: str | None = None
) -> dict:
    """Read/capture source while the caller holds the artifact revision lock."""
    _recover_pending_source_write(folder)
    target, relative = resolve_source(folder, metadata, rel_path)
    content = target.read_bytes()
    if len(content) > MAX_SOURCE_BYTES:
        raise RevisionValidationError("Artifact source is too large to edit")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RevisionValidationError("Artifact source is not UTF-8 text") from exc
    manifest = _read_manifest(folder)
    latest = next(
        (r for r in reversed(manifest["revisions"]) if r.get("path") == relative),
        None,
    )
    if latest is None or latest.get("contentHash") != _sha(content):
        latest = _append_revision(
            folder,
            manifest,
            artifact_id=artifact_id,
            rel_path=relative,
            content=content,
            actor_kind="system",
            actor_id=None,
            summary="Captured current draft",
            base_revision_id=latest.get("id") if latest else None,
        )
    return {
        "artifactId": artifact_id,
        "path": relative,
        "content": text,
        "contentType": target.suffix.lower().lstrip("."),
        "revision": latest,
    }


def current_source(folder: Path, metadata: dict, artifact_id: str, rel_path: str | None = None) -> dict:
    """Read source plus the revision token required by a subsequent save."""
    with artifact_lock(folder):
        return _current_source_locked(folder, metadata, artifact_id, rel_path)


def current_workspace(
    folder: Path, metadata: dict, artifact_id: str, rel_path: str | None = None
) -> dict:
    """Read editable source and its history under one artifact lock.

    The viewer needs both values before its revision chrome is complete.  A
    single snapshot avoids a second HTTP request, a second identity lookup,
    and a race where an edit lands between the source and history reads.
    """
    with artifact_lock(folder):
        source = _current_source_locked(folder, metadata, artifact_id, rel_path)
        manifest = _read_manifest(folder)
        revisions = [
            revision
            for revision in manifest["revisions"]
            if revision.get("path") == source["path"]
        ]
        return {**source, "revisions": list(reversed(revisions))}


def save_source(
    folder: Path,
    metadata: dict,
    artifact_id: str,
    *,
    content: str,
    expected_revision_id: str,
    rel_path: str | None = None,
    actor_kind: str = "manual",
    actor_id: str | None = None,
    summary: str = "Edited artifact",
    conversation_id: str | None = None,
    comment_thread_ids: list[str] | None = None,
) -> dict:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise RevisionValidationError("Artifact source is too large to edit")
    with artifact_lock(folder):
        _recover_pending_source_write(folder)
        target, relative = resolve_source(folder, metadata, rel_path)
        existing = target.read_bytes()
        manifest = _read_manifest(folder)
        latest = next(
            (r for r in reversed(manifest["revisions"]) if r.get("path") == relative),
            None,
        )
        if latest is None or latest.get("contentHash") != _sha(existing):
            latest = _append_revision(
                folder,
                manifest,
                artifact_id=artifact_id,
                rel_path=relative,
                content=existing,
                actor_kind="system",
                actor_id=None,
                summary="Captured concurrent draft change",
                base_revision_id=latest.get("id") if latest else None,
            )
        if latest.get("id") != expected_revision_id:
            raise RevisionConflict(latest)
        if latest.get("contentHash") == _sha(encoded):
            return {"artifactId": artifact_id, "path": relative, "content": content, "revision": latest}

        entry = _commit_source_revision(
            folder,
            manifest,
            target,
            artifact_id=artifact_id,
            rel_path=relative,
            before_content=existing,
            content=encoded,
            actor_kind=actor_kind,
            actor_id=actor_id,
            summary=summary,
            base_revision_id=latest.get("id"),
            conversation_id=conversation_id,
            comment_thread_ids=comment_thread_ids,
        )
        return {"artifactId": artifact_id, "path": relative, "content": content, "revision": entry}


def list_revisions(folder: Path, *, rel_path: str | None = None) -> list[dict]:
    with artifact_lock(folder):
        _recover_pending_source_write(folder)
        manifest = _read_manifest(folder)
        revisions = manifest["revisions"]
        if rel_path is not None:
            revisions = [r for r in revisions if r.get("path") == rel_path]
        return list(reversed(revisions))


def _revision_with_content_locked(folder: Path, revision_id: str) -> dict:
    manifest = _read_manifest(folder)
    revision = next((r for r in manifest["revisions"] if r.get("id") == revision_id), None)
    if revision is None:
        raise FileNotFoundError("Revision not found")
    return {**revision, "content": _content_for_revision(folder, revision)}


def revision_with_content(folder: Path, revision_id: str) -> dict:
    with artifact_lock(folder):
        _recover_pending_source_write(folder)
        return _revision_with_content_locked(folder, revision_id)


def capture_agent_revision(folder: Path, *, conversation_id: str | None = None) -> dict | None:
    """Record an agent edit and finish repair handoffs attributed to this turn."""
    from cowork.services.artifact_identity import ensure_full_id

    try:
        artifact_id, metadata = ensure_full_id(folder)
        with artifact_lock(folder):
            _recover_pending_source_write(folder)
            target, relative = resolve_source(folder, metadata)
            content = target.read_bytes()
            if len(content) > MAX_SOURCE_BYTES:
                return None
            content.decode("utf-8")
            manifest = _read_manifest(folder)
            latest = next(
                (r for r in reversed(manifest["revisions"]) if r.get("path") == relative),
                None,
            )
            latest_id = latest.get("id") if latest else None
            turn_repairs = [
                repair
                for repair in _repair_records(folder)
                if repair.get("status") == "queued"
                and repair.get("conversationId") == conversation_id
            ]
            queued = [r for r in turn_repairs if r.get("path") == relative]
            # The primary can move between minting a handoff and this capture
            # (metadata["primary"] rewritten, or resolve_source's sorted
            # fallback picking a file the turn created). A handoff left on the
            # old path would never be finished by any later turn either, so it
            # ends here rather than gating that path forever.
            for repair in (r for r in turn_repairs if r.get("path") != relative):
                repair["status"] = "conflict"
                repair["updatedAt"] = _now()
                _write_repair(folder, repair)
            if latest is not None and latest.get("contentHash") == _sha(content):
                _finish_queued_repairs(
                    folder,
                    queued,
                    matching_base_id=latest_id,
                    revision_id=None,
                )
                return latest
            matching = [
                repair for repair in queued
                if repair.get("baseRevisionId") == latest_id
            ]
            entry = _append_revision(
                folder,
                manifest,
                artifact_id=artifact_id,
                rel_path=relative,
                content=content,
                actor_kind="agent",
                actor_id=None,
                summary="Agent updated artifact",
                base_revision_id=latest_id,
                conversation_id=conversation_id,
                comment_thread_ids=[
                    repair["commentThreadId"]
                    for repair in matching
                    if repair.get("commentThreadId")
                ],
            )
            _finish_queued_repairs(
                folder,
                queued,
                matching_base_id=latest_id,
                revision_id=entry["id"],
            )
            return entry
    except (OSError, UnicodeDecodeError, ValueError, RevisionValidationError, TimeoutError):
        return None


def create_agent_repair(
    folder: Path,
    metadata: dict,
    artifact_id: str,
    *,
    expected_revision_id: str,
    comment_thread_id: str,
    selector: str | None,
    thread: list[dict],
    conversation_id: str,
) -> dict:
    """Persist a structured repair handoff and return the exact agent prompt."""
    with artifact_lock(folder):
        source = _current_source_locked(folder, metadata, artifact_id)
        current = source["revision"]
        if current.get("id") != expected_revision_id:
            raise RevisionConflict(current)
        manifest = _read_manifest(folder)
        blocking, dead = _partition_actionable_repairs(folder, manifest, source["path"])
        if blocking:
            raise RepairAlreadyPending(blocking[0])
        for repair in dead:
            repair["status"] = "no_change"
            repair["updatedAt"] = _now()
            _write_repair(folder, repair)
        repair_id = str(uuid.uuid4())
        repair = {
            "id": repair_id,
            "artifactId": artifact_id,
            "path": source["path"],
            "baseRevisionId": current["id"],
            "baseContentHash": current["contentHash"],
            "commentThreadId": comment_thread_id,
            "selector": selector,
            "thread": thread,
            "conversationId": conversation_id,
            "status": "queued",
            "revisionId": None,
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        _write_repair(folder, repair)
    prompt = (
        "Address this artifact review thread. Work on the existing artifact source; "
        "do not create a replacement artifact and do not resolve the comment yourself.\n\n"
        f"Artifact id: {artifact_id}\n"
        f"Source path: {source['path']}\n"
        f"Base revision: {current['id']}\n"
        f"Repair id: {repair_id}\n"
        f"Selected element: {selector or 'General artifact feedback'}\n"
        "Complete comment thread:\n"
        f"{json.dumps(thread, ensure_ascii=False, indent=2)}\n\n"
        "Make the smallest coherent fix, preserve unrelated behavior and styling, "
        "and verify the artifact still renders."
    )
    return {"repair": repair, "prompt": prompt}


def agent_repair_detail(folder: Path, repair_id: str) -> dict:
    with artifact_lock(folder):
        _recover_pending_source_write(folder)
        repair = _read_repair(folder, repair_id)
        result = {"repair": repair}
        if repair.get("status") == "ready" and repair.get("revisionId"):
            before = _revision_with_content_locked(folder, repair["baseRevisionId"])
            after = _revision_with_content_locked(folder, repair["revisionId"])
            if before.get("path") != repair.get("path") or after.get("path") != repair.get("path"):
                raise RevisionValidationError("Agent repair revision path is inconsistent")
            result["compare"] = {"before": before, "after": after}
        return result


def active_agent_repair(folder: Path) -> dict | None:
    """Return the newest repair that still needs owner attention.

    The artifact viewer is unmounted when the user follows the generated agent
    task. Persisted discovery lets reopening the artifact resume polling or
    show the comparison instead of losing the handoff in React state.
    """
    active = [
        repair
        for repair in _repair_records(folder)
        if repair.get("status") in {"queued", "ready"}
    ]
    return max(active, key=lambda repair: str(repair.get("createdAt") or ""), default=None)


def cancel_agent_repair(folder: Path, repair_id: str, *, discard_ready: bool = False) -> dict:
    """Release a repair without accepting or rejecting its content.

    A queued repair is cancelled: its turn never produced anything. A ready one
    holds real agent work, so it is only discarded when the caller says so
    explicitly. That keeps the property the queued-only rule was protecting -
    an agent change never vanishes on its own - while giving an owner who has
    already dealt with the feedback some other way a way out.
    """
    with artifact_lock(folder):
        repair = _read_repair(folder, repair_id)
        if repair.get("status") in {"cancelled", "discarded"}:
            return repair
        if repair.get("status") == "queued":
            repair["status"] = "cancelled"
        elif discard_ready and repair.get("status") == "ready":
            repair["status"] = "discarded"
        else:
            raise RevisionValidationError("Only a queued agent repair can be cancelled")
        repair["updatedAt"] = _now()
        _write_repair(folder, repair)
        return repair


def finalize_agent_repair(
    folder: Path,
    metadata: dict,
    artifact_id: str,
    repair_id: str,
    decision: str,
    *,
    actor_id: str | None = None,
) -> dict:
    """Accept or reject a ready repair under one artifact-wide lock.

    Both the head check and a rejection restore happen while the same lock is
    held. This prevents two review tabs from accepting one suggestion while a
    concurrent request restores its pre-agent content.
    """
    if decision not in {"accepted", "rejected"}:
        raise RevisionValidationError("Invalid repair status")
    with artifact_lock(folder):
        repair = _read_repair(folder, repair_id)
        if repair.get("status") != "ready":
            raise RevisionValidationError("Agent repair is not ready for review")
        source = _current_source_locked(folder, metadata, artifact_id, repair.get("path"))
        current = source["revision"]
        if (
            decision == "rejected"
            and current.get("baseRevisionId") == repair.get("revisionId")
            and current.get("summary") == "Rejected agent suggestion"
            and repair.get("commentThreadId") in current.get("commentThreadIds", [])
        ):
            # The source/revision transaction committed but a process failure
            # interrupted the final repair-status write. Retrying the decision
            # completes that last idempotent step instead of reporting a false
            # edit conflict and stranding the repair in `ready`.
            repair["status"] = decision
            repair["updatedAt"] = _now()
            _write_repair(folder, repair)
            return repair
        if current.get("id") != repair.get("revisionId"):
            raise RevisionConflict(current)

        if decision == "rejected":
            before = _revision_with_content_locked(folder, repair["baseRevisionId"])
            if before.get("path") != repair.get("path"):
                raise RevisionValidationError("Agent repair revision path is inconsistent")
            encoded = before["content"].encode("utf-8")
            target, relative = resolve_source(folder, metadata, repair["path"])
            manifest = _read_manifest(folder)
            _commit_source_revision(
                folder,
                manifest,
                target,
                artifact_id=artifact_id,
                rel_path=relative,
                before_content=source["content"].encode("utf-8"),
                content=encoded,
                actor_kind="manual",
                actor_id=actor_id,
                summary="Rejected agent suggestion",
                base_revision_id=current["id"],
                comment_thread_ids=[repair["commentThreadId"]],
            )

        repair["status"] = decision
        repair["updatedAt"] = _now()
        _write_repair(folder, repair)
        return repair
