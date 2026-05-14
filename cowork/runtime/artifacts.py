"""Artifact validation primitives for Cowork harnesses."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _artifact_user_files(folder: Path) -> list[Path]:
    ignored = {"metadata.json", "README.md", ".published.json"}
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.name not in ignored and ".cowork-preview" not in path.parts
    )


def _folder_mtime(folder: Path) -> float:
    latest = 0.0
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def validate_artifact_folder(folder: str | Path, artifact_root: str | Path) -> dict[str, Any] | None:
    root = Path(artifact_root).resolve()
    folder = Path(folder).resolve()
    try:
        folder.relative_to(root)
    except ValueError:
        logger.warning("Ignoring artifact outside root: %s", folder)
        return None

    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Ignoring artifact with invalid metadata JSON: %s", metadata_path)
        return None
    if not isinstance(metadata, dict):
        logger.warning("Ignoring artifact metadata that is not an object: %s", metadata_path)
        return None

    primary: Path | None = None
    primary_hint = metadata.get("primary") or metadata.get("primaryFile") or metadata.get("entrypoint")
    if isinstance(primary_hint, str) and primary_hint.strip():
        candidate = (folder / primary_hint).resolve()
        try:
            candidate.relative_to(folder)
        except ValueError:
            logger.warning("Ignoring artifact primary outside artifact folder: %s", candidate)
            return None
        if candidate.is_file():
            primary = candidate
    if primary is None:
        user_files = _artifact_user_files(folder)
        primary = user_files[0] if user_files else None
    if primary is None:
        logger.warning("Ignoring artifact without a primary file: %s", folder)
        return None

    title = str(metadata.get("name") or metadata.get("title") or folder.name).strip() or folder.name
    return {
        "title": title,
        "name": title,
        "description": str(metadata.get("description") or "").strip(),
        "type": str(metadata.get("type") or "mixed"),
        "slug": str(metadata.get("slug") or folder.name),
        "folder": str(folder),
        "path": str(primary),
        "file_path": str(primary),
        "primary": str(primary.relative_to(folder)),
        "metadata_path": str(metadata_path),
    }


def scan_artifact_root(
    artifact_root: str | Path,
    before: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    root = Path(artifact_root)
    if not root.is_dir():
        return []
    previous = before or {}
    artifacts: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        folder = metadata_path.parent
        try:
            folder_key = str(folder.resolve())
        except OSError:
            continue
        current_mtime = _folder_mtime(folder)
        if folder_key in previous and current_mtime <= previous[folder_key]:
            continue
        payload = validate_artifact_folder(folder, root)
        if payload:
            artifacts.append(payload)
    return artifacts

