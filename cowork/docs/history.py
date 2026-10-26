"""A doc's version history: the versions its editing sessions left, described, and restoring one.

``store`` keeps a version each time an editing session ends (one reply of the
agent's, or a stretch of a person's typing). This reads them back as the
editor lists them, newest first and the doc as it is now on top, each with who
made it, when, and what changed since the version before (``changes``).
``restore`` brings a version back as a new write. That write is a version of
its own, and the doc it replaces is kept first, so nothing is lost and a
restore is undone by restoring the version it replaced.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .changes import describe
from .model import Doc, DocError, normalize_doc
from .service import render_page
from .store import HISTORY_KEEP, history_dir, list_history, load_doc, mutate_doc, open_session

# The menu lists this many; every kept version can be asked for.
LIST_LIMIT = 30
LIST_MAX = HISTORY_KEEP

# Kept versions never change, so what changed between two of them is worked
# out once: by doc id, older rev and newer rev.
_SUMMARIES_KEPT = 2000
_summaries: OrderedDict[tuple[str, int | None, int], str] = OrderedDict()
_summaries_lock = threading.Lock()


def read_version(folder: Path, rev: int) -> Doc:
    """Version ``rev`` of the doc, as that editing session left it."""
    if not any(entry.get("rev") == rev for entry in list_history(folder)):
        raise DocError("no_version", "that version of the doc is not kept; list its history for the ones that are.")
    return _load_version(folder, rev)


def _load_version(folder: Path, rev: int) -> Doc:
    try:
        raw = json.loads((history_dir(folder) / f"{rev:07d}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DocError("no_version", "that version of the doc could not be read.") from exc
    return normalize_doc(raw)


def list_versions(folder: Path, *, limit: int = LIST_LIMIT) -> dict[str, Any]:
    """The doc now and its newest ``limit`` kept versions, newest first: who, when, and what changed."""
    current = load_doc(folder)
    kept = [e for e in reversed(list_history(folder)) if isinstance(e.get("rev"), int) and e["rev"] < current["rev"]]
    loaded: dict[int, Doc | None] = {current["rev"]: current}

    def version(rev: int) -> Doc | None:
        """A version's doc, read only when a summary not yet worked out needs it."""
        if rev not in loaded:
            try:
                loaded[rev] = _load_version(folder, rev)
            except DocError:
                loaded[rev] = None
        return loaded[rev]

    def summary(before: dict[str, Any] | None, rev: int, *, first: bool) -> str | None:
        key = (current["id"], before["rev"] if before else None, rev)
        with _summaries_lock:
            if key in _summaries:
                _summaries.move_to_end(key)
                return _summaries[key]
        if before is None:
            # Nothing kept before it: the doc as first written, or the oldest left once older ones were let go.
            if not first:
                return None
            text = _first_draft(version(rev) or current)
        else:
            older, newer = version(before["rev"]), version(rev)
            if older is None or newer is None:
                return None
            text = describe(older, newer)
        with _summaries_lock:
            _summaries[key] = text
            while len(_summaries) > _SUMMARIES_KEPT:
                _summaries.popitem(last=False)
        return text

    versions = []
    shown = kept[-limit:]
    for n, entry in enumerate(shown, start=len(kept) - len(shown)):
        before = kept[n - 1] if n else None
        versions.append(_entry(entry, summary(before, entry["rev"], first=bool(entry.get("first")))))
    session = open_session(folder) or {}
    now = {"rev": current["rev"], "at": session.get("last") or current.get("updated_at"), "by": session.get("by"),
           **({"restored_from": session["own"]["restored_from"]} if "restored_from" in session.get("own", {}) else {})}
    return {
        "current": _entry(now, summary(kept[-1] if kept else None, current["rev"], first=bool(session.get("first")))),
        "versions": list(reversed(versions)),
    }


def _entry(entry: dict[str, Any], summary: str | None) -> dict[str, Any]:
    out = {"rev": entry["rev"], "at": entry.get("at"), "by": entry.get("by"), "summary": summary}
    if isinstance(entry.get("restored_from"), dict):
        out["restoredFrom"] = entry["restored_from"]
    return out


def _first_draft(doc: Doc) -> str:
    tabs = len(doc["tabs"])
    return "First draft" + (f" · {tabs} tabs" if tabs > 1 else "")


def restore(folder: Path, rev: int, *, by: str = "user") -> tuple[Doc, int]:
    """Bring version ``rev`` back as the doc's newest write; returns (the doc now, the rev it replaced).

    The doc as it was is kept as a version first, so restoring the rev
    returned undoes this. Every block takes the new rev, so an edit made
    against the doc before the restore (the agent mid-reply, say) is refused
    as stale rather than written over it.
    """
    entry = next((e for e in list_history(folder) if e.get("rev") == rev), None)
    old = read_version(folder, rev)
    replaced: list[int] = []

    def change(current: Doc) -> tuple[Doc, None]:
        replaced.append(current["rev"])
        restored = {**old, "id": current["id"], "rev": current["rev"] + 1, "created_at": current.get("created_at")}
        for tab in restored["tabs"]:
            for block in tab["blocks"]:
                block["v"] = restored["rev"]
        return restored, None

    doc, _ = mutate_doc(folder, change, by=by, own_version={"restored_from": {"rev": rev, "at": (entry or {}).get("at")}},
                        after=lambda d: render_page(folder, d))
    return doc, replaced[0]


__all__ = ["LIST_LIMIT", "LIST_MAX", "list_versions", "read_version", "restore"]
