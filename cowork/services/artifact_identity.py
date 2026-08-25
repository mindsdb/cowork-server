"""Stable artifact identity and project-scoped resolution.

Artifact folders can move between projects and published URLs can change.  The
UUID stored in ``metadata.json`` is the identity that survives those changes;
filesystem paths and the legacy eight-character id are compatibility details.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.artifacts.models import legacy_stable_id

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts

class ArtifactIdentityConflict(RuntimeError):
    """More than one scoped folder claims the same stable identity."""


def _legacy_stable_id(metadata: dict, folder: Path) -> str:
    legacy_id = str(metadata.get("id") or metadata.get("slug") or folder.name)
    created_at = str(metadata.get("createdAt") or "")
    return legacy_stable_id(legacy_id, created_at)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def ensure_stable_id(folder: Path, metadata: dict | None = None) -> tuple[str, dict]:
    """Return the folder's canonical UUID, persisting a legacy backfill.

    The fallback is deterministic, so concurrent readers choose the same value
    even before either atomic metadata write wins.
    """
    path = folder / "metadata.json"
    if metadata is None:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    raw = metadata.get("stableId")
    if raw not in (None, ""):
        try:
            return str(uuid.UUID(str(raw))), metadata
        except (ValueError, TypeError, AttributeError) as exc:
            # A present-but-invalid identity is corruption, not a legacy
            # record. Replacing it would silently detach published versions
            # and existing comment threads from the artifact.
            raise ValueError("Artifact stable identity is invalid") from exc

    stable_id = _legacy_stable_id(metadata, folder)

    # The caller may have loaded metadata before another subsystem updated it.
    # Merge into the latest durable document instead of writing the stale
    # snapshot back over unrelated metadata fields.
    try:
        latest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact metadata is unreadable") from exc
    latest_raw = latest.get("stableId")
    if latest_raw not in (None, ""):
        try:
            return str(uuid.UUID(str(latest_raw))), latest
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Artifact stable identity is invalid") from exc
    updated = dict(latest)
    updated["stableId"] = stable_id
    _atomic_json(path, updated)
    return stable_id, updated


def artifact_key(stable_id: str) -> str:
    """Canonical two-segment key accepted by the existing comments API."""
    return f"artifact/{uuid.UUID(stable_id)}"


def resolve_artifact_folder(
    sources: list["ProjectArtifacts"], stable_id: str
) -> tuple["ProjectArtifacts", Path, dict]:
    """Resolve a UUID only inside roots already authorized for the caller."""
    wanted = str(uuid.UUID(stable_id))
    matches: list[tuple["ProjectArtifacts", Path, dict]] = []
    for source in sources:
        base = Path(source.base)
        try:
            folders = sorted(base.iterdir())
        except OSError:
            continue
        for folder in folders:
            if not folder.is_dir() or not (folder / "metadata.json").is_file():
                continue
            try:
                found, metadata = ensure_stable_id(folder)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if found == wanted:
                matches.append((source, folder, metadata))

    if not matches:
        raise FileNotFoundError("Artifact not found")
    if len(matches) > 1:
        raise ArtifactIdentityConflict("Stable artifact identity is duplicated")
    return matches[0]
