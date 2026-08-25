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


class RevisionConflict(RuntimeError):
    def __init__(self, current: dict):
        super().__init__("Artifact changed since this edit began")
        self.current = current


class RevisionValidationError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _journal(folder: Path) -> Path:
    return folder / JOURNAL_DIRNAME


def _manifest_path(folder: Path) -> Path:
    return _journal(folder) / "manifest.json"


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
        tmp.write_bytes(content)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(folder: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(_manifest_path(folder), encoded)


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


def _queued_repairs_for_path(folder: Path, rel_path: str) -> list[dict]:
    return [
        repair
        for repair in _repair_records(folder)
        if repair.get("status") == "queued" and repair.get("path") == rel_path
    ]


def _actionable_repairs_for_path(folder: Path, rel_path: str) -> list[dict]:
    return [
        repair
        for repair in _repair_records(folder)
        if repair.get("status") in {"queued", "ready"}
        and repair.get("path") == rel_path
    ]


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
        target = (folder / candidate_rel).resolve(strict=False)
        try:
            target.relative_to(folder.resolve())
        except ValueError as exc:
            raise RevisionValidationError("Invalid artifact source path") from exc
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError("Artifact source not found")
    if target.suffix.lower() not in EDITABLE_EXTENSIONS:
        raise RevisionValidationError("This artifact type is not source-editable")
    return target, target.relative_to(folder).as_posix()


def _content_for_revision(folder: Path, revision: dict) -> str:
    blob = _journal(folder) / "blobs" / str(revision.get("contentHash", ""))
    try:
        return blob.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RevisionValidationError("Revision content is unavailable") from exc


def _append_revision(
    folder: Path,
    manifest: dict,
    *,
    stable_id: str,
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
    blob = _journal(folder) / "blobs" / content_hash
    if not blob.exists():
        _atomic_bytes(blob, content, mode=0o600)
    revisions = manifest["revisions"]
    entry = {
        "id": str(uuid.uuid4()),
        "number": (int(revisions[-1].get("number", 0)) + 1) if revisions else 1,
        "artifactId": stable_id,
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


def _current_source_locked(
    folder: Path, metadata: dict, stable_id: str, rel_path: str | None = None
) -> dict:
    """Read/capture source while the caller holds the artifact revision lock."""
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
            stable_id=stable_id,
            rel_path=relative,
            content=content,
            actor_kind="system",
            actor_id=None,
            summary="Captured current draft",
            base_revision_id=latest.get("id") if latest else None,
        )
    return {
        "artifactId": stable_id,
        "path": relative,
        "content": text,
        "contentType": target.suffix.lower().lstrip("."),
        "revision": latest,
    }


def current_source(folder: Path, metadata: dict, stable_id: str, rel_path: str | None = None) -> dict:
    """Read source plus the revision token required by a subsequent save."""
    with artifact_lock(folder):
        return _current_source_locked(folder, metadata, stable_id, rel_path)


def save_source(
    folder: Path,
    metadata: dict,
    stable_id: str,
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
                stable_id=stable_id,
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
            return {"artifactId": stable_id, "path": relative, "content": content, "revision": latest}

        mode = target.stat().st_mode & 0o777
        _atomic_bytes(target, encoded, mode=mode)
        entry = _append_revision(
            folder,
            manifest,
            stable_id=stable_id,
            rel_path=relative,
            content=encoded,
            actor_kind=actor_kind,
            actor_id=actor_id,
            summary=summary,
            base_revision_id=latest.get("id"),
            conversation_id=conversation_id,
            comment_thread_ids=comment_thread_ids,
        )
        return {"artifactId": stable_id, "path": relative, "content": content, "revision": entry}


def list_revisions(folder: Path, *, rel_path: str | None = None) -> list[dict]:
    manifest = _read_manifest(folder)
    revisions = manifest["revisions"]
    if rel_path is not None:
        revisions = [r for r in revisions if r.get("path") == rel_path]
    return list(reversed(revisions))


def revision_with_content(folder: Path, revision_id: str) -> dict:
    manifest = _read_manifest(folder)
    revision = next((r for r in manifest["revisions"] if r.get("id") == revision_id), None)
    if revision is None:
        raise FileNotFoundError("Revision not found")
    return {**revision, "content": _content_for_revision(folder, revision)}


def snapshot_revision_head(folder: Path) -> str:
    """Ensure the primary source has a baseline and return its content hash.

    Used at the turn boundary so an agent write can be attributed even when it
    lands within the same whole-second mtime bucket as the pre-turn snapshot.
    Unsupported/big artifacts return an empty token and keep the legacy mtime
    path working.
    """
    from cowork.services.artifact_identity import ensure_stable_id

    try:
        stable_id, metadata = ensure_stable_id(folder)
        source = current_source(folder, metadata, stable_id)
        return str(source["revision"]["contentHash"])
    except (OSError, ValueError, RevisionValidationError, TimeoutError):
        return ""


def primary_source_hash(folder: Path) -> str:
    """Current primary-source hash without mutating the revision journal."""
    from cowork.services.artifact_identity import ensure_stable_id

    try:
        _stable_id, metadata = ensure_stable_id(folder)
        target, _relative = resolve_source(folder, metadata)
        content = target.read_bytes()
        if len(content) > MAX_SOURCE_BYTES:
            return ""
        content.decode("utf-8")
        return _sha(content)
    except (OSError, UnicodeDecodeError, ValueError, RevisionValidationError):
        return ""


def capture_agent_revision(folder: Path, *, conversation_id: str | None = None) -> dict | None:
    """Record an agent edit and finish repair handoffs attributed to this turn."""
    from cowork.services.artifact_identity import ensure_stable_id

    try:
        stable_id, metadata = ensure_stable_id(folder)
        with artifact_lock(folder):
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
            queued = [
                repair
                for repair in _queued_repairs_for_path(folder, relative)
                if repair.get("conversationId") == conversation_id
            ]
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
                stable_id=stable_id,
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
    stable_id: str,
    *,
    expected_revision_id: str,
    comment_thread_id: str,
    selector: str | None,
    thread: list[dict],
    conversation_id: str,
) -> dict:
    """Persist a structured repair handoff and return the exact agent prompt."""
    with artifact_lock(folder):
        source = _current_source_locked(folder, metadata, stable_id)
        current = source["revision"]
        if current.get("id") != expected_revision_id:
            raise RevisionConflict(current)
        if _actionable_repairs_for_path(folder, source["path"]):
            raise RevisionValidationError(
                "This artifact already has an agent repair awaiting review"
            )
        repair_id = str(uuid.uuid4())
        repair = {
            "id": repair_id,
            "artifactId": stable_id,
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
        f"Artifact stable id: {stable_id}\n"
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
    repair = _read_repair(folder, repair_id)
    result = {"repair": repair}
    if repair.get("status") == "ready" and repair.get("revisionId"):
        before = revision_with_content(folder, repair["baseRevisionId"])
        after = revision_with_content(folder, repair["revisionId"])
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


def cancel_agent_repair(folder: Path, repair_id: str) -> dict:
    """Cancel a repair whose agent turn never started.

    Cancellation is deliberately limited to queued repairs. Once a turn has
    produced a revision, the owner must compare and accept or reject it so an
    agent change can never disappear without an explicit decision.
    """
    with artifact_lock(folder):
        repair = _read_repair(folder, repair_id)
        if repair.get("status") == "cancelled":
            return repair
        if repair.get("status") != "queued":
            raise RevisionValidationError("Only a queued agent repair can be cancelled")
        repair["status"] = "cancelled"
        repair["updatedAt"] = _now()
        _write_repair(folder, repair)
        return repair


def finalize_agent_repair(
    folder: Path,
    metadata: dict,
    stable_id: str,
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
        source = _current_source_locked(folder, metadata, stable_id, repair.get("path"))
        current = source["revision"]
        if current.get("id") != repair.get("revisionId"):
            raise RevisionConflict(current)

        if decision == "rejected":
            before = revision_with_content(folder, repair["baseRevisionId"])
            if before.get("path") != repair.get("path"):
                raise RevisionValidationError("Agent repair revision path is inconsistent")
            encoded = before["content"].encode("utf-8")
            target, relative = resolve_source(folder, metadata, repair["path"])
            mode = target.stat().st_mode & 0o777
            _atomic_bytes(target, encoded, mode=mode)
            manifest = _read_manifest(folder)
            _append_revision(
                folder,
                manifest,
                stable_id=stable_id,
                rel_path=relative,
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
