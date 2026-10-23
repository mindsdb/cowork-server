"""Docs on disk: one artifact folder per doc, the doc itself in ``doc.cdoc.json``.

A doc is an ordinary anton artifact (``metadata.json`` with ``type:
"document"`` and ``primary: "doc.cdoc.json"``), so the artifacts library,
cards, provenance and deletion all work on it unchanged. What this module
adds is the doc's own write path:

- every write goes through :func:`mutate_doc`, which serialises writers of one
  doc (the agent's tool calls and the person typing in the editor) on the
  doc's lock, applies the change to a fresh read, and replaces the file
  atomically. The lock holds across processes too: hosted, two replicas
  serve one shared tree, and each would otherwise write its own rev N+1 over
  the other's;
- every write belongs to an editing session: one reply of the agent's, or a
  stretch of a person's typing. When the next session starts, the doc as the
  last one left it is kept as a version in ``.revisions/doc-history/``
  (housekeeping the content walk already skips), so the doc before any agent
  reply can be brought back, and the versions read as who did what, when.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from anton.core.artifacts import ArtifactStore

from cowork.services.artifact_lock import artifact_lock

from .model import Doc, DocError, normalize_doc
from .render import doc_html

logger = logging.getLogger(__name__)

DOC_FILENAME = "doc.cdoc.json"
DOC_SUFFIX = ".cdoc.json"
# The doc rendered as a standalone page: what "Open in browser" shows and what
# sharing publishes (through the same publish flow, and access, as any HTML
# artifact). Rewritten after every write, so it never lags the doc.
DOC_HTML = "doc.html"
HISTORY_DIR = (".revisions", "doc-history")
HISTORY_KEEP = 100
# And only as many of the newest as fit in this: at up to MAX_DOC_BYTES each,
# a hundred versions could hold 800 MB of one doc.
HISTORY_MAX_BYTES = 50 * 1024 * 1024
# A person's session ends when they pause this long, or once it has run this
# long; the agent's ends with its reply. Hours of typing are then a handful of
# versions, not one, and not one every few seconds.
SESSION_PAUSE_S = 5 * 60
SESSION_MAX_S = 30 * 60
MAX_DOC_BYTES = 8 * 1024 * 1024
_MAX_DOC_MB = MAX_DOC_BYTES // (1024 * 1024)

_locks: dict[str, _DocLock] = {}
_locks_guard = threading.Lock()

T = TypeVar("T")


class NewDoc(NamedTuple):
    folder: Path
    slug: str
    artifact_id: str
    doc: Doc


class _DocLock:
    """One doc's write lock: re-entrant in this process, and held across processes.

    Threads here queue on an ``RLock``. The outermost hold also takes the
    artifact's lock file (``artifact_lock``), which another process on the
    same tree (a second replica) takes too. The lock file is not re-entrant,
    so a write made while the lock is already held (``cloud.apply_edit`` holds
    it around its check and the write it guards) re-enters only the ``RLock``.
    """

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._local = threading.RLock()
        self._depth = 0  # changed only by the thread holding _local
        self._file = ExitStack()

    def __enter__(self) -> None:
        self._local.acquire()
        try:
            if self._depth == 0:
                self._lock_file()
        except BaseException:
            self._local.release()
            raise
        self._depth += 1

    def _lock_file(self) -> None:
        try:
            # Made here, without its parents: artifact_lock makes them, and a
            # write racing the doc's deletion would bring its folder back.
            (self._folder / ".revisions").mkdir(exist_ok=True)
            self._file.enter_context(artifact_lock(self._folder))
        except FileNotFoundError as exc:
            raise DocError("no_doc", "this doc no longer exists.") from exc
        except TimeoutError as exc:
            raise DocError("busy", "this doc is being saved elsewhere; try again in a moment.") from exc

    def __exit__(self, *exc_info: object) -> None:
        self._depth -= 1
        try:
            if self._depth == 0:
                self._file.close()
        finally:
            self._local.release()


def _lock_for(path: Path) -> _DocLock:
    key = str(path.resolve())
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = _DocLock(Path(key).parent)
        return lock


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_doc_primary(name: str | None) -> bool:
    return bool(name) and str(name).lower().endswith(DOC_SUFFIX)


def _read_meta(folder: Path) -> dict[str, Any]:
    """The artifact's ``metadata.json``; {} when it is missing or unreadable."""
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def doc_file(folder: Path) -> Path:
    """The doc file in an artifact folder: the recorded primary, else the default name."""
    primary = _read_meta(folder).get("primary")
    if is_doc_primary(primary):
        candidate = (folder / primary).resolve()
        if candidate.parent == folder.resolve():
            return candidate
    return folder / DOC_FILENAME


def is_doc_folder(folder: Path) -> bool:
    return doc_file(folder).is_file()


def _read_json(path: Path, *, missing: DocError) -> Any:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise missing from exc
    if len(raw) > MAX_DOC_BYTES:
        raise DocError("too_big", f"this doc is larger than {_MAX_DOC_MB} MB and cannot be opened.")
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise DocError("bad_doc", f"{path.name} is not valid JSON ({exc}).") from exc


def load_doc(folder: Path) -> Doc:
    path = doc_file(folder)
    return normalize_doc(_read_json(path, missing=DocError("no_doc", f"this artifact is not a doc (no {DOC_FILENAME}).")))


def atomic_write(path: Path, payload: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".doc-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _encode(doc: Doc) -> bytes:
    payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DOC_BYTES:
        raise DocError("too_big", f"the doc would exceed {_MAX_DOC_MB} MB; move bulky data to its own tab "
                       "or a dataset artifact.")
    return payload


# ---------------------------------------------------------------- history


def history_dir(folder: Path) -> Path:
    return folder.joinpath(*HISTORY_DIR)


def _read_index(folder: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads((history_dir(folder) / "index.json").read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def open_session(folder: Path) -> dict[str, Any] | None:
    """The session that made the doc as it is now: who, which agent reply, when it started and last wrote."""
    try:
        data = json.loads((history_dir(folder) / "open.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_session(folder: Path, session: dict[str, Any]) -> None:
    hist = history_dir(folder)
    hist.mkdir(parents=True, exist_ok=True)
    atomic_write(hist / "open.json", json.dumps(session).encode("utf-8"))


def _stamp(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _seconds_since(stamp: Any, now: float) -> float:
    try:
        return now - datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")


def _new_session(by: str, turn: str | None, stamp: str, own: dict[str, Any] | None = None) -> dict[str, Any]:
    session: dict[str, Any] = {"by": by, "turn": turn, "started": stamp, "last": stamp}
    if own is not None:
        session["own"] = own
    return session


def _session_ends(session: dict[str, Any] | None, *, by: str, turn: str | None, now: float) -> bool:
    """Whether a write by ``by`` (in agent reply ``turn``) starts a session rather than continuing ``session``."""
    if session is None or session.get("by") != by or "own" in session:
        return True
    if by == "agent":
        return session.get("turn") != turn
    return (_seconds_since(session.get("last"), now) >= SESSION_PAUSE_S
            or _seconds_since(session.get("started"), now) >= SESSION_MAX_S)


def _keep_version(folder: Path, doc: Doc, session: dict[str, Any] | None) -> None:
    """Keep ``doc`` as the version ``session`` left behind."""
    index = _read_index(folder)
    if index and index[-1].get("rev") == doc["rev"]:
        return
    hist = history_dir(folder)
    hist.mkdir(parents=True, exist_ok=True)
    payload = _encode(doc)
    atomic_write(hist / f"{doc['rev']:07d}.json", payload)
    made = session or {}  # a doc written before sessions were recorded: when, but not who
    entry: dict[str, Any] = {"rev": doc["rev"], "at": made.get("last") or doc.get("updated_at") or now_iso(),
                             "started": made.get("started"), "by": made.get("by"),
                             "title": doc.get("title", ""), "bytes": len(payload)}
    if made.get("first"):
        entry["first"] = True  # the doc as first written, not what's left once older versions are let go
    entry.update(made.get("own") or {})
    index.append(entry)
    dropped = len(index) - _versions_kept(hist, index)
    for old in index[:dropped]:
        try:
            (hist / f"{int(old['rev']):07d}.json").unlink()
        except (OSError, KeyError, ValueError):
            pass
    atomic_write(hist / "index.json", json.dumps(index[dropped:]).encode("utf-8"))


def _record_write(folder: Path, current: Doc, *, by: str, turn: str | None,
                  own: dict[str, Any] | None) -> None:
    """Continue the editing session for a write that replaces ``current``, or start one.

    Starting one first keeps ``current`` as the version the previous session
    left. A write that is a version of its own (``own``) always starts one,
    and the next write after it does too.
    """
    now = time.time()
    session = open_session(folder)
    if own is not None or _session_ends(session, by=by, turn=turn, now=now):
        if current["rev"] > 0:
            _keep_version(folder, current, session)
        session = _new_session(by, turn, _stamp(now), own)
    else:
        session["last"] = _stamp(now)
    _write_session(folder, session)


def _versions_kept(hist: Path, index: list[dict[str, Any]]) -> int:
    """How many of the newest versions stay: HISTORY_KEEP at most, within HISTORY_MAX_BYTES, and never none."""
    total = 0
    for kept, entry in enumerate(reversed(index)):
        total += _version_bytes(hist, entry)
        if kept == HISTORY_KEEP or (kept and total > HISTORY_MAX_BYTES):
            return kept
    return len(index)


def _version_bytes(hist: Path, entry: dict[str, Any]) -> int:
    size = entry.get("bytes")
    if isinstance(size, int):
        return size
    try:  # a version kept before the index recorded sizes
        return (hist / f"{int(entry['rev']):07d}.json").stat().st_size
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def list_history(folder: Path) -> list[dict[str, Any]]:
    """Kept versions, newest first: each the doc as one editing session left it, and who, and when."""
    return list(reversed(_read_index(folder)))


# ---------------------------------------------------------------- writes


def write_html(folder: Path, doc: Doc) -> None:
    """Refresh doc.html from the doc. Best-effort: the doc file is the truth."""
    try:
        atomic_write(folder / DOC_HTML, doc_html(doc).encode("utf-8"))
    except Exception:  # noqa: BLE001 - a page that failed to render must not fail the edit
        logger.warning("could not render %s", folder / DOC_HTML, exc_info=True)


def doc_lock(folder: Path) -> _DocLock:
    """The lock every write to this doc holds, in this process and across processes.

    Re-entrant: a write may be made while holding it. Raises DocError("busy")
    when another process has held it for longer than ``artifact_lock`` waits.
    """
    return _lock_for(doc_file(folder))


def with_doc(folder: Path, fn: Callable[[Doc], T]) -> T:
    """``fn`` of the doc as it is now, under the doc's lock, so no write lands in between."""
    with doc_lock(folder):
        return fn(load_doc(folder))


def mutate_doc(
    folder: Path,
    change: Callable[[Doc], tuple[Doc, T]],
    *,
    by: str,
    turn: str | None = None,
    own_version: dict[str, Any] | None = None,
    after: Callable[[Doc], None] | None = None,
) -> tuple[Doc, T]:
    """Read, change, write — under the doc's lock. ``change`` returns (new doc, result).

    Each write continues its editing session or starts the next, which keeps
    the doc as the last session left it (``_record_write``); ``turn`` names
    the agent reply a write is part of. ``own_version``: this write is a
    version of its own (a restore), kept with these fields. ``after`` runs on
    the new doc once it is written, still under the lock: the page rendered
    from it can't be overtaken by an older one.
    """
    path = doc_file(folder)
    with _lock_for(path):
        current = load_doc(folder)
        new_doc, result = change(current)
        if new_doc is current:
            return current, result
        new_doc["updated_at"] = now_iso()
        normalize_doc(new_doc)  # refuses a change that would write a doc that no longer loads
        payload = _encode(new_doc)
        try:
            _record_write(folder, current, by=by, turn=turn, own=own_version)
        except OSError:
            # History is a safety net; never fail the edit over it.
            logger.warning("could not keep a version of %s", folder, exc_info=True)
        atomic_write(path, payload)
        if after is not None:
            after(new_doc)
        return new_doc, result


def create_doc_folder(artifacts_dir: Path, *, title: str, description: str, doc: Doc,
                      by: str = "agent", turn: str | None = None) -> NewDoc:
    """Register a new document artifact and write its doc (the doc's id becomes the artifact's).

    Writing it opens the doc's first editing session, ``by`` and in ``turn``.
    """
    store = ArtifactStore(artifacts_dir)
    artifact = store.create(name=title, description=description, type="document", primary=DOC_FILENAME)
    folder = store.folder_for(artifact.slug)
    # No lock: nothing else knows this folder until this returns.
    stamp = now_iso()
    written = {**doc, "id": artifact.id, "created_at": doc.get("created_at") or stamp, "updated_at": stamp}
    atomic_write(folder / DOC_FILENAME, _encode(written))
    try:
        _write_session(folder, {**_new_session(by, turn, stamp), "first": True})
    except OSError:
        logger.warning("could not record who wrote %s", folder, exc_info=True)
    return NewDoc(folder, artifact.slug, artifact.id, written)


def _doc_folders(artifacts_dir: Path) -> list[Path]:
    base = Path(artifacts_dir)
    if not base.is_dir():
        return []
    return sorted(f for f in base.iterdir() if f.is_dir() and not f.name.startswith(".") and is_doc_folder(f))


def find_doc_folder(artifacts_dir: Path, ref: str) -> Path:
    """A doc folder by artifact id (dashed or not), slug, or exact title."""
    ref = (ref or "").strip()
    if not ref:
        raise DocError("no_doc", "name the doc: its id, slug or title (list_docs shows them).")
    if not Path(artifacts_dir).is_dir():
        raise DocError("no_doc", f"no doc {ref!r}: this project has no artifacts yet.")
    hexref = ref.replace("-", "").lower()
    by_title = []
    for folder in _doc_folders(artifacts_dir):
        if folder.name == ref:
            return folder
        meta = _read_meta(folder)
        if str(meta.get("id", "")).replace("-", "").lower() == hexref:
            return folder
        if str(meta.get("name", "")).strip().lower() == ref.lower():
            by_title.append(folder)
    if len(by_title) == 1:
        return by_title[0]
    if len(by_title) > 1:
        raise DocError("ambiguous", f"{len(by_title)} docs are titled {ref!r}; pass the id from list_docs.")
    raise DocError("no_doc", f"no doc {ref!r} in this project; list_docs shows the docs that exist.")


def list_doc_folders(artifacts_dir: Path) -> list[dict[str, Any]]:
    """The project's docs, newest first: id, slug, title, rev and tabs.

    Read lightly (no validation of every block), so a project full of long
    docs lists quickly; a doc that won't open says so when it is opened.
    """
    out = []
    for folder in _doc_folders(artifacts_dir):
        try:
            raw = _read_json(doc_file(folder), missing=DocError("no_doc", ""))
        except DocError:
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("tabs"), list):
            continue
        tabs = [t for t in raw["tabs"] if isinstance(t, dict)]
        out.append({
            "id": _read_meta(folder).get("id") or raw.get("id", ""),
            "slug": folder.name,
            "title": str(raw.get("title") or "Untitled doc"),
            "rev": raw.get("rev", 0),
            "updated_at": raw.get("updated_at", ""),
            "tabs": [{"id": t.get("id", ""), "title": t.get("title", ""), "blocks": len(t.get("blocks") or [])}
                     for t in tabs],
        })
    out.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return out
