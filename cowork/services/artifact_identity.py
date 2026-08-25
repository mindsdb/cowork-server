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

if TYPE_CHECKING:
    from cowork.services.artifacts import ProjectArtifacts

_LEGACY_ARTIFACT_NAMESPACE = uuid.UUID("4ba9bdf8-3f0e-4ce5-beb0-8f00a8d955e7")


class ArtifactIdentityConflict(RuntimeError):
    """More than one scoped folder claims the same stable identity."""


def _legacy_stable_id(metadata: dict, folder: Path) -> str:
    legacy_id = str(metadata.get("id") or metadata.get("slug") or folder.name)
    created_at = str(metadata.get("createdAt") or "")
    return str(uuid.uuid5(_LEGACY_ARTIFACT_NAMESPACE, f"{legacy_id}:{created_at}"))


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
    try:
        stable_id = str(uuid.UUID(str(raw))) if raw else ""
    except (ValueError, TypeError, AttributeError):
        stable_id = ""
    if stable_id:
        return stable_id, metadata

    stable_id = _legacy_stable_id(metadata, folder)
    updated = dict(metadata)
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
