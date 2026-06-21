"""Filesystem browsing for the inline path picker (browse mode).

Backs the directory navigation in the frontend's `select_path` browser: list a
directory's immediate children so the user can drill in and choose a file or
folder the agent couldn't resolve on its own.

This is a single-user desktop app bound to loopback with CORS locked to the
renderer, and the agent already has filesystem access — so listing directory
*names* is in-scope. The endpoint still hardens the obvious edges: it resolves
the real path, lists only directories the caller can read, never returns file
*contents*, and caps the entry count so a huge directory can't stall the UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

# Cap entries per listing so a directory with 100k files can't bloat the
# payload or freeze the picker. The UI tells the user when it's truncated.
_MAX_ENTRIES = 1000


def _entry(child: Path) -> dict[str, Any] | None:
    """One listing row, or None if the child can't be classified."""
    try:
        is_dir = child.is_dir()
    except OSError:
        return None
    return {"name": child.name, "path": str(child), "is_dir": is_dir}


@router.get("/list")
def list_directory(
    path: str | None = Query(None, description="Directory to list. Defaults to the user's home."),
    kind: str = Query("any", description="'file' | 'folder' | 'any' — when 'folder', files are omitted."),
    show_hidden: bool = Query(False, description="Include dot-prefixed entries."),
) -> dict[str, Any]:
    """List a directory's immediate children (folders first, then files).

    Returns ``{ path, parent, entries: [{name, path, is_dir}], truncated }``.
    ``parent`` is null at the filesystem root. Folders are always included so
    the user can navigate; files are included only when ``kind != 'folder'``.
    """
    base = Path(os.path.expanduser(path)).resolve() if path else Path.home().resolve()
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Not a directory.")

    folders: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    truncated = False
    try:
        with os.scandir(base) as it:
            for dir_entry in it:
                if not show_hidden and dir_entry.name.startswith("."):
                    continue
                entry = _entry(Path(dir_entry.path))
                if entry is None:
                    continue
                if entry["is_dir"]:
                    folders.append(entry)
                elif kind != "folder":
                    files.append(entry)
                if len(folders) + len(files) >= _MAX_ENTRIES:
                    truncated = True
                    break
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied.")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read directory: {exc}")

    folders.sort(key=lambda e: e["name"].lower())
    files.sort(key=lambda e: e["name"].lower())
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": folders + files, "truncated": truncated}
