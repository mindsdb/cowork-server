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

The agent's code can write in an artifact's folder, so every read and write
here goes through the folder opened without following a link: a link (or a
second name) planted where the doc keeps a file is refused or replaced, never
followed, and a write never brings back a folder that was deleted. The
history goes through the artifact journal (``cowork.services.artifact_journal``),
which works the same way.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from anton.core.artifacts import ArtifactStore

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_atomic_write,
    dir_lstat,
    dir_mkdir,
    dir_open,
    dir_rmtree,
    dir_unlink,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)
from cowork.services.artifact_journal import (
    read_journal_file,
    read_journal_json,
    remove_journal_file,
    write_journal_file,
)
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
# The history's folder in the artifact's journal (.revisions).
_HISTORY = HISTORY_DIR[1]
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


@contextmanager
def doc_subfolder(folder: Path, *names: str, create: bool = False) -> Iterator[PinnedDir]:
    """The doc's folder, or the folder ``names`` below it, every level opened without
    following a link. With ``create`` the folders below it are made when missing; the
    doc's folder never is, so a write racing the doc's deletion can't bring it back.

    Raises DocError: no_doc when a folder is gone, bad_doc when a link (or anything but
    a folder) stands at one of the names.
    """
    with ExitStack() as opened:
        try:
            current = opened.enter_context(pinned_dir(folder, nofollow_base=True))
            for name in names:
                if create:
                    try:
                        dir_mkdir(current, name)
                    except FileExistsError:
                        pass
                current = open_pinned_child(current, name)
                opened.callback(current.close)
        except FileNotFoundError as exc:
            raise DocError("no_doc", "this doc no longer exists.") from exc
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise DocError("bad_doc", "a link stands where this doc keeps a folder; it is not followed.") from exc
            raise
        yield current


def read_child(d: PinnedDir, name: str, *, max_bytes: int) -> bytes:
    """The bytes of the file ``name`` in ``d``, not following a link.

    Only a regular file with one name is read (a hard link could be any file the
    agent's code can read), and never past ``max_bytes``. Raises FileNotFoundError
    when there is none, DocError("bad_doc") for anything else at the name, and
    DocError("too_big") past the cap.
    """
    try:
        fd = dir_open(d, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise DocError("bad_doc", f"{name} in this doc's folder is a link; it is not followed.") from exc
        raise
    try:
        found = os.fstat(fd)
        if not stat.S_ISREG(found.st_mode) or found.st_nlink != 1:
            raise DocError("bad_doc", f"{name} in this doc's folder is not a file this doc wrote.")
        if found.st_size > max_bytes:
            raise DocError("too_big", f"{name} is larger than {max_bytes // (1024 * 1024) or 1} MB and cannot be read.")
        return b"".join(read_bounded(fd, found.st_size))
    finally:
        os.close(fd)


def write_child(folder: Path, name: str, payload: bytes, *, below: tuple[str, ...] = ()) -> None:
    """Replace the file ``name`` in the doc's folder (or the folder ``below`` it, made when
    missing) in one durable step. A link or second name at ``name`` is replaced, never
    written through."""
    with doc_subfolder(folder, *below, create=True) as d:
        try:
            dir_atomic_write(d, name, content=payload)
        except FileNotFoundError as exc:  # the folder went while this wrote
            raise DocError("no_doc", "this doc no longer exists.") from exc


def child_exists(folder: Path, name: str, *, below: tuple[str, ...] = ()) -> bool:
    """Whether a regular file is at ``name`` (not followed) in the doc's folder or ``below`` it."""
    try:
        with doc_subfolder(folder, *below) as d:
            return stat.S_ISREG(dir_lstat(d, name).st_mode)
    except (DocError, OSError):
        return False


def remove_child(folder: Path, name: str) -> None:
    """Remove the name ``name`` from the doc's folder; a link goes, never what it points to."""
    try:
        with doc_subfolder(folder) as d:
            dir_unlink(d, name)
    except FileNotFoundError:
        pass


def _read_meta(folder: Path) -> dict[str, Any]:
    """The artifact's ``metadata.json``; {} when it is missing or unreadable."""
    try:
        with doc_subfolder(folder) as d:
            meta = json.loads(read_child(d, "metadata.json", max_bytes=1024 * 1024))
    except (OSError, ValueError, DocError):
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
    return child_exists(folder, doc_file(folder).name)


def _read_json(folder: Path, *, missing: DocError) -> Any:
    """The doc file in ``folder``, parsed; ``missing`` when there is none."""
    name = doc_file(folder).name
    try:
        with doc_subfolder(folder) as d:
            raw = read_child(d, name, max_bytes=MAX_DOC_BYTES)
    except FileNotFoundError as exc:
        raise missing from exc
    except DocError as exc:
        if exc.code == "no_doc":
            raise missing from exc
        if exc.code == "too_big":
            raise DocError("too_big", f"this doc is larger than {_MAX_DOC_MB} MB and cannot be opened.") from exc
        raise
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise DocError("bad_doc", f"{name} is not valid JSON ({exc}).") from exc


def load_doc(folder: Path) -> Doc:
    return normalize_doc(_read_json(folder, missing=DocError("no_doc", f"this artifact is not a doc (no {DOC_FILENAME}).")))


def _encode(doc: Doc) -> bytes:
    payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DOC_BYTES:
        raise DocError("too_big", f"the doc would exceed {_MAX_DOC_MB} MB; move bulky data to its own tab "
                       "or a dataset artifact.")
    return payload


# ---------------------------------------------------------------- history


def version_name(rev: int) -> str:
    """A kept version's file in the history folder."""
    return f"{rev:07d}.json"


def read_version_bytes(folder: Path, rev: int) -> bytes | None:
    """Kept version ``rev`` as stored, or None when there is none. Raises OSError
    (JournalFileRefused) for anything at its name the history never writes."""
    return read_journal_file(folder, _HISTORY, version_name(rev), max_bytes=MAX_DOC_BYTES)


def _read_index(folder: Path) -> list[dict[str, Any]]:
    try:
        data = read_journal_json(folder, _HISTORY, "index.json")
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def open_session(folder: Path) -> dict[str, Any] | None:
    """The session that made the doc as it is now: who, which agent reply, when it started and last wrote."""
    try:
        data = read_journal_json(folder, _HISTORY, "open.json")
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_session(folder: Path, session: dict[str, Any]) -> None:
    write_journal_file(folder, _HISTORY, "open.json", content=json.dumps(session).encode("utf-8"))


def _stamp(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _seconds_since(stamp: Any, now: float) -> float:
    try:
        return now - datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")


def _new_session(by: str, turn: str | None, stamp: str) -> dict[str, Any]:
    return {"by": by, "turn": turn, "started": stamp, "last": stamp}


def _session_ends(session: dict[str, Any] | None, *, by: str, turn: str | None, now: float) -> bool:
    """Whether a write by ``by`` (in agent reply ``turn``) starts a session rather than continuing ``session``."""
    if session is None or session.get("by") != by:
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
    payload = _encode(doc)
    write_journal_file(folder, _HISTORY, version_name(doc["rev"]), content=payload)
    made = session or {}  # a doc written before sessions were recorded: when, but not who
    entry: dict[str, Any] = {"rev": doc["rev"], "at": made.get("last") or doc.get("updated_at") or now_iso(),
                             "started": made.get("started"), "by": made.get("by"),
                             "title": doc.get("title", ""), "bytes": len(payload)}
    if made.get("first"):
        entry["first"] = True  # the doc as first written, not what's left once older versions are let go
    index.append(entry)
    dropped = len(index) - _versions_kept(folder, index)
    for old in index[:dropped]:
        try:
            remove_journal_file(folder, _HISTORY, version_name(int(old["rev"])))
        except (OSError, KeyError, ValueError, TypeError):
            pass
    write_journal_file(folder, _HISTORY, "index.json", content=json.dumps(index[dropped:]).encode("utf-8"))


def _record_write(folder: Path, current: Doc, *, by: str, turn: str | None) -> None:
    """Continue the editing session for a write that replaces ``current``, or start one.

    Starting one first keeps ``current`` as the version the previous session left.
    """
    now = time.time()
    session = open_session(folder)
    if _session_ends(session, by=by, turn=turn, now=now):
        if current["rev"] > 0:
            _keep_version(folder, current, session)
        session = _new_session(by, turn, _stamp(now))
    else:
        session["last"] = _stamp(now)
    _write_session(folder, session)


def _versions_kept(folder: Path, index: list[dict[str, Any]]) -> int:
    """How many of the newest versions stay: HISTORY_KEEP at most, within HISTORY_MAX_BYTES, and never none."""
    total = 0
    for kept, entry in enumerate(reversed(index)):
        total += _version_bytes(folder, entry)
        if kept == HISTORY_KEEP or (kept and total > HISTORY_MAX_BYTES):
            return kept
    return len(index)


def _version_bytes(folder: Path, entry: dict[str, Any]) -> int:
    size = entry.get("bytes")
    if isinstance(size, int):
        return size
    try:  # a version kept before the index recorded sizes
        return len(read_version_bytes(folder, int(entry["rev"])) or b"")
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def list_history(folder: Path) -> list[dict[str, Any]]:
    """Kept versions, newest first: each the doc as one editing session left it, and who, and when."""
    return list(reversed(_read_index(folder)))


# ---------------------------------------------------------------- writes


def write_html(folder: Path, doc: Doc) -> None:
    """Refresh doc.html from the doc. Best-effort: the doc file is the truth."""
    try:
        write_child(folder, DOC_HTML, doc_html(doc).encode("utf-8"))
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
    after: Callable[[Doc], None] | None = None,
) -> tuple[Doc, T]:
    """Read, change, write — under the doc's lock. ``change`` returns (new doc, result).

    Each write continues its editing session or starts the next, which keeps
    the doc as the last session left it (``_record_write``); ``turn`` names
    the agent reply a write is part of. ``after`` runs on the new doc once it
    is written, still under the lock: the page rendered from it can't be
    overtaken by an older one.
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
            _record_write(folder, current, by=by, turn=turn)
        except OSError:
            # History is a safety net; never fail the edit over it.
            logger.warning("could not keep a version of %s", folder, exc_info=True)
        write_child(folder, path.name, payload)
        if after is not None:
            after(new_doc)
        return new_doc, result


def create_doc_folder(artifacts_dir: Path, *, title: str, description: str, doc: Doc,
                      by: str = "agent", turn: str | None = None) -> NewDoc:
    """Register a new document artifact and write its doc (the doc's id becomes the artifact's).

    Writing it opens the doc's first editing session, ``by`` and in ``turn``.
    """
    _encode(doc)  # a doc too big to write is refused before any folder is made
    store = ArtifactStore(artifacts_dir)
    artifact = store.create(name=title, description=description, type="document", primary=DOC_FILENAME)
    folder = store.folder_for(artifact.slug)
    # No lock: nothing else knows this folder until this returns.
    stamp = now_iso()
    written = {**doc, "id": artifact.id, "created_at": doc.get("created_at") or stamp, "updated_at": stamp}
    try:
        write_child(folder, DOC_FILENAME, _encode(written))
    except BaseException:
        # A folder with no doc in it would sit in the library as a broken artifact.
        _remove_folder(Path(artifacts_dir), artifact.slug)
        raise
    try:
        _write_session(folder, {**_new_session(by, turn, stamp), "first": True})
    except OSError:
        logger.warning("could not record who wrote %s", folder, exc_info=True)
    return NewDoc(folder, artifact.slug, artifact.id, written)


def _remove_folder(artifacts_dir: Path, slug: str) -> None:
    """Remove the artifact folder ``slug`` this process just made, through the artifacts
    folder's descriptor: a link at the name goes, never what it points to."""
    try:
        with pinned_dir(artifacts_dir) as d:
            dir_rmtree(d, slug)
    except OSError:
        logger.warning("could not remove the unfinished doc %s", artifacts_dir / slug, exc_info=True)


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
            raw = _read_json(folder, missing=DocError("no_doc", ""))
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
