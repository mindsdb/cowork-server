"""A shared doc's live copy in MindsHub, so its owner can edit the shared link.

A doc lives on this machine; sharing publishes it as a static page. While it is
shared, this module keeps MindsHub inference holding the doc as it is now (each
write is mirrored, a beat after it lands) and listens for the owner's edits from
the shared page. Each edit is applied through the same write path as the editor
and the agent (``service.apply``, ``by="user"``, with the browser's rev as the
stale guard), announced to any open pane, mirrored, and only then reported back,
so the page reads the new copy before it hears the edit landed. This machine
stays the only writer: there is nothing to merge, and an edit made while the app
is closed waits in MindsHub until it opens.

MindsHub sends each edit to every computer the owner is signed in on, and again
on every reconnect until one reports it. So a computer that doesn't hold the doc
stays silent (the one that does will answer), and an edit that landed but whose
report was lost is reported again rather than applied twice.

One-person desktops only (``page_editor.link_editing_available``).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from cowork.db.scoped import LOCAL_SCOPE
from cowork.services import share_hooks
from cowork.services.comments_proxy import resolve_inference_endpoint

from . import page_editor, republish
from .debounce import Debouncer
from .model import DocError
from .service import announce, apply, refresh_page
from .page_editor import link_editing_available
from .sharing import share_record, shared_doc_folders
from .store import atomic_write, doc_lock, load_doc, with_doc

logger = logging.getLogger(__name__)

MIRROR_DELAY_S = 0.4
# The edits from the shared page that landed here, by id, beside the doc's history.
APPLIED_EDITS = (".revisions", "shared-page-edits.json")
APPLIED_EDITS_KEEP = 200
# The stream sends a keepalive every ~20s; three missed means it is gone.
STREAM_READ_TIMEOUT_S = 75.0
BACKOFF_MAX_S = 30.0
# A MindsHub without shared-doc editing answers 404: stop asking for a while
# instead of failing (and logging) on every keystroke in a shared doc.
UNSUPPORTED_PAUSE_S = 600.0

_mirrors = Debouncer(MIRROR_DELAY_S, what="mirroring a shared doc")
_lock = threading.Lock()
_listener: _EditListener | None = None
_folders: dict[str, Path] = {}  # SharedPage.artifact_id -> doc folder
_unsupported_until = 0.0


# --- the page a doc is shared as: MindsHub addresses its live copy by it ---

# A shared page's URL ends /view/<user dir>/<report id>; both are plain tokens.
_PAGE_URL = re.compile(r"/view/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)(?:[/?#]|$)")


class SharedPage(NamedTuple):
    """Where a doc is shared: MindsHub addresses its live copy by ``user_dir/report_id``."""

    user_dir: str
    report_id: str
    artifact_key: str

    @property
    def artifact_id(self) -> str:
        return f"{self.user_dir}/{self.report_id}"


def page_from_record(entry: dict[str, Any] | None) -> SharedPage | None:
    """The page a publish entry names, or None for an address that isn't a plain shared page."""
    match = _PAGE_URL.search(str(entry.get("url") or "")) if entry else None
    if match is None:
        return None
    return SharedPage(match.group(1), match.group(2),
                      str(entry.get("artifact_key") or entry.get("lambda_artifact_key") or ""))


def shared_page(folder: Path) -> SharedPage | None:
    return page_from_record(share_record(folder))


def _supported() -> bool:
    return time.monotonic() >= _unsupported_until


def _note_status(status: int) -> None:
    global _unsupported_until
    if status == 404:
        _unsupported_until = time.monotonic() + UNSUPPORTED_PAUSE_S


def _endpoint() -> tuple[str, str]:
    """(inference base incl. /v1, bearer) — the credential comments and publish use."""
    base, key = resolve_inference_endpoint()
    return (base or "").rstrip("/"), key or ""


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _doc_url(base: str, page: SharedPage, suffix: str = "") -> str:
    return f"{base}/artifact-docs/{page.user_dir}/{page.report_id}{suffix}"


# --- mirroring this machine's copy ---


def mirror_now(folder: Path) -> bool:
    """Send the doc as it is now to MindsHub. True when MindsHub has it."""
    _mirrors.cancel(str(folder))
    page = shared_page(folder)
    if page is None or not _supported():
        return False
    base, key = _endpoint()
    if not base or not key:
        return False
    doc = load_doc(folder)
    body = {"rev": doc["rev"], "doc": doc, **({"artifactKey": page.artifact_key} if page.artifact_key else {})}
    try:
        res = httpx.put(_doc_url(base, page, "/state"), json=body, headers=_headers(key), timeout=20.0)
    except httpx.HTTPError:
        logger.warning("could not mirror shared doc %s", page.artifact_id, exc_info=True)
        return False
    if res.status_code >= 300:
        _note_status(res.status_code)
        logger.warning("mirroring shared doc %s refused: %s", page.artifact_id, res.status_code)
        return False
    with _lock:
        _folders[page.artifact_id] = folder
    return True


def schedule_mirror(folder: Path, *, delay: float | None = None) -> bool:
    """Mirror ``folder`` after ``delay`` (MIRROR_DELAY_S) unless another write comes first."""
    if not link_editing_available() or shared_page(folder) is None:
        return False
    _mirrors.schedule(str(folder), lambda: mirror_now(folder), delay=delay)
    ensure_listening()
    return True


# --- sharing starting and stopping (the publish service's share hooks) ---


def before_share(folder: Path) -> None:
    """The doc's page is about to be uploaded: render it fresh, with the owner's editor.

    What goes out is the doc as it is now, even if nothing changed since the
    last render, or the editor build arrived after it.
    """
    with_doc(folder, lambda doc: page_editor.render_page(folder, doc, shared=True))


def after_share(folder: Path) -> None:
    """The doc was just shared (or its access changed): give MindsHub the live copy, and listen."""
    if not link_editing_available():
        return
    mirror_now(folder)
    ensure_listening()


def after_unshare(record: dict[str, Any], folder: Path | None) -> None:
    """Sharing stopped, or the doc is being deleted (no ``folder``): drop the live
    copy and any edits still waiting, then take the editor off the page."""
    page = page_from_record(record)
    if page is not None and link_editing_available():
        _drop_live_copy(page)
    if folder is not None:
        refresh_page(folder)  # after the drop: a doc that no longer loads must not keep its copy live


def _drop_live_copy(page: SharedPage) -> None:
    with _lock:
        _folders.pop(page.artifact_id, None)
    base, key = _endpoint()
    if not base or not key:
        return
    try:
        httpx.delete(_doc_url(base, page), headers=_headers(key), timeout=20.0)
    except httpx.HTTPError:
        logger.warning("could not drop the live copy of %s", page.artifact_id, exc_info=True)


def install() -> None:
    """Have the publish service keep shared docs in step (at startup, before anything is shared)."""
    share_hooks.register(share_hooks.DocShareHooks(before_share, after_share, after_unshare))


# --- finding the doc an edit is for ---


def folder_for(artifact_id: str) -> Path | None:
    with _lock:
        hit = _folders.get(artifact_id)
    if hit is not None and getattr(shared_page(hit), "artifact_id", None) == artifact_id:
        return hit
    found = {page.artifact_id: folder for folder in shared_doc_folders() if (page := shared_page(folder))}
    with _lock:
        _folders.update(found)
        return _folders.get(artifact_id)


# --- the owner's edits from the shared page ---


def _rejected(code: str, message: str) -> dict[str, Any]:
    return {"status": "rejected", "error": {"code": code, "message": message[:500]}}


def _applied_edits(folder: Path) -> dict[str, Any]:
    try:
        seen = json.loads(folder.joinpath(*APPLIED_EDITS).read_bytes())
    except (OSError, ValueError):
        return {}
    return seen if isinstance(seen, dict) else {}


def _remember_applied(folder: Path, edit_id: str, outcome: dict[str, Any]) -> None:
    seen = _applied_edits(folder)
    seen.pop(edit_id, None)
    seen[edit_id] = outcome
    path = folder.joinpath(*APPLIED_EDITS)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(dict(list(seen.items())[-APPLIED_EDITS_KEEP:])).encode("utf-8"))
    except OSError:
        logger.warning("could not record edit %s as applied", edit_id, exc_info=True)


def apply_edit(event: dict[str, Any]) -> dict[str, Any] | None:
    """Apply one edit from the shared page; returns the outcome to report, or None when the doc isn't here."""
    folder = folder_for(str(event.get("artifact_id") or ""))
    if folder is None:
        return None
    edit_id = str(event.get("id") or "")
    base_rev = event.get("base_rev")
    with doc_lock(folder):  # the check and the write it guards, with no other write in between
        outcome = _applied_edits(folder).get(edit_id)
        if outcome is None:
            try:
                doc, result = apply(
                    folder, event.get("ops") or [], tab=event.get("tab"), by="user",
                    if_rev=int(base_rev) if base_rev is not None else None,
                )
            except (DocError, TypeError, ValueError) as err:
                return _rejected(getattr(err, "code", "bad_op"), str(err))
            outcome = {"status": "applied", "rev": doc["rev"]}
            _remember_applied(folder, edit_id, outcome)
            announce(doc["id"], doc, result, by="user", client_id=event.get("client_id"))
            republish.schedule(folder, scope=LOCAL_SCOPE)
    mirror_now(folder)  # before reporting: the page rereads the copy when it hears back
    return outcome


def report(artifact_id: str, edit_id: str, outcome: dict[str, Any]) -> None:
    base, key = _endpoint()
    if not base or not key or "/" not in artifact_id:
        return
    user_dir, report_id = artifact_id.split("/", 1)
    try:
        httpx.post(
            f"{base}/artifact-docs/{user_dir}/{report_id}/edits/{edit_id}/result",
            json=outcome, headers=_headers(key), timeout=20.0,
        )
    except httpx.HTTPError:
        logger.warning("could not report edit %s on %s", edit_id, artifact_id, exc_info=True)


def sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Decode an SSE stream's ``data:`` frames; comments and blanks are skipped."""
    data: list[str] = []

    def decoded() -> Iterator[dict[str, Any]]:
        try:
            yield json.loads("\n".join(data))
        except ValueError:
            pass

    for line in lines:
        if line.startswith(":"):
            continue
        if not line:
            if data:
                yield from decoded()
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield from decoded()


class _EditListener(threading.Thread):
    """Holds the owner's edit stream open, reconnecting with backoff."""

    def __init__(self) -> None:
        super().__init__(name="shared-doc-edits", daemon=True)
        self.stopping = threading.Event()

    def run(self) -> None:
        backoff = 1.0
        while not self.stopping.is_set():
            base, key = _endpoint()
            if not base or not key or not _supported():
                self.stopping.wait(BACKOFF_MAX_S)
                continue
            try:
                timeout = httpx.Timeout(connect=15.0, read=STREAM_READ_TIMEOUT_S, write=15.0, pool=15.0)
                with httpx.stream("GET", f"{base}/artifact-docs/stream", headers=_headers(key), timeout=timeout) as res:
                    if res.status_code != 200:
                        _note_status(res.status_code)
                        raise httpx.HTTPStatusError("stream refused", request=res.request, response=res)
                    backoff = 1.0
                    for event in sse_events(res.iter_lines()):
                        if self.stopping.is_set():
                            return
                        if event.get("type") == "doc.edit" and event.get("id"):
                            handle_edit_event(event)
            except Exception:  # noqa: BLE001 - reconnect; edits wait in MindsHub meanwhile
                logger.info("shared-doc edit stream dropped; reconnecting", exc_info=True)
            self.stopping.wait(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_S)


def handle_edit_event(event: dict[str, Any]) -> None:
    """Apply one ``doc.edit`` off the stream and report its outcome; one bad edit never stops the stream."""
    try:
        outcome = apply_edit(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not apply edit %s", event.get("id"), exc_info=True)
        outcome = _rejected("error", str(exc))
    if outcome is not None:
        report(str(event.get("artifact_id") or ""), str(event["id"]), outcome)


def ensure_listening() -> None:
    global _listener
    if not link_editing_available():
        return
    with _lock:
        if _listener is not None and _listener.is_alive():
            return
        _listener = _EditListener()
        _listener.start()


def stop() -> None:
    """Shutdown: stop listening and drop mirrors still waiting (``sync.start`` catches up next time)."""
    global _listener
    with _lock:
        listener, _listener = _listener, None
    _mirrors.cancel_all()
    if listener is not None:
        listener.stopping.set()


def pending_mirrors() -> list[str]:
    return _mirrors.pending()
