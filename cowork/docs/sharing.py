"""Which docs are shared.

A doc is shared when its page (doc.html) has a live entry in the folder's
``.published.json``, which the publish flow writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.artifact_roots import artifacts_sources_for_scan
from cowork.services.artifacts import load_published_map

from .store import DOC_HTML, is_doc_folder


def share_record(folder: Path) -> dict[str, Any] | None:
    """The publish entry of a doc's page while it is shared, else None."""
    entry = load_published_map(folder).get(DOC_HTML)
    # Same rule as publish.published_state: unsharing keeps the entry, marked off.
    if isinstance(entry, dict) and bool(entry.get("published", True)) and entry.get("url"):
        return entry
    return None


def is_shared(folder: Path) -> bool:
    return share_record(folder) is not None


def shared_doc_folders() -> list[Path]:
    """Every shared doc in this desktop's projects."""
    found = []
    for source in artifacts_sources_for_scan(scope=LOCAL_SCOPE):
        root = Path(source.base)
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if folder.is_dir() and not folder.name.startswith(".") and is_doc_folder(folder) and is_shared(folder):
                found.append(folder)
    return found
