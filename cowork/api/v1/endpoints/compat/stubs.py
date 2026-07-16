"""Compat stub endpoints.  # SHIM:client-compat

These exist solely so the Cowork renderer doesn't 404 on endpoints that
haven't been migrated to cowork-server yet. Each returns a safe empty
response. Replace with real implementations as they're ported over.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session

from cowork.db.session import get_session

logger = logging.getLogger(__name__)

# ── Integrations ─────────────────────────────────────────────────────

integrations_router = APIRouter()


@integrations_router.get("")
def list_integrations():
    return []


@integrations_router.post("/{service}/oauth/start")
def oauth_start(service: str, body: dict[str, Any] | None = None):
    return {"url": None, "error": "OAuth not yet available in cowork-server"}


# ── Attachments ──────────────────────────────────────────────────────
# TODO(migration): Per MIGRATION.md, attachments should be removed.
# The client should upload via POST /v1/files/ and reference files as
# input_file content blocks in the Responses request input field.
# These endpoints exist as a compat bridge for the current client.

attachments_router = APIRouter()
_SessionDep = Annotated[Session, Depends(get_session)]


def _attachment_purpose(project_name: str, session_id: str) -> str:
    # `project_name` is still part of the client-facing route but is
    # deliberately ignored: purposes are keyed by conversation id only, so a
    # project rename can't strand attachments (ENG-338).
    from cowork.services.files import attachment_purpose
    return attachment_purpose(session_id)


def _to_attachment(file) -> dict:
    """Legacy FileAttachment shape the rail's Task Uploads rows render
    (name / mime / size + ISO timestamps). Returning the OpenAI-style
    FileResponse here (filename / bytes / epoch-seconds) left the rows
    showing raw file ids and a 1970s age (ENG-264)."""
    created = file.created_at.isoformat() if file.created_at else None
    return {
        "id": str(file.id),
        "name": file.filename,
        "mime": file.content_type,
        "size": file.size,
        "created_at": created,
        # File rows are immutable once uploaded — created is updated.
        "updated_at": created,
        "purpose": file.purpose,
    }


@attachments_router.get("/{project_name}/{session_id}")
def list_attachments(
    project_name: str,
    session_id: str,
    session: _SessionDep,
    ids: list[str] | None = Query(default=None),
):
    from cowork.services.files import FileService
    rows = FileService(session).list_file_rows(purpose=_attachment_purpose(project_name, session_id))
    if ids:
        wanted = {str(i) for i in ids}
        rows = [r for r in rows if str(r.id) in wanted]
    return [_to_attachment(r) for r in rows]


@attachments_router.post("/{project_name}/{session_id}/upload")
async def upload_attachment(
    project_name: str,
    session_id: str,
    session: _SessionDep,
    files: list[UploadFile] = File(...),
):
    from cowork.services.files import FileService
    if ":" in session_id:
        # Real clients mint UUIDs (or the legacy timestamp format) — neither
        # contains a colon. Rejecting colon-bearing ids keeps every
        # "attachment:{session_id}" tag unambiguously parseable by the
        # legacy-format rekey in cowork.db.migrations (ENG-338).
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id must not contain ':'",
        )
    svc = FileService(session)
    purpose = _attachment_purpose(project_name, session_id)
    results = []
    for f in files:
        try:
            created = await svc.create_file(upload=f, purpose=purpose)
            results.append(_to_attachment(svc.get_file_row(UUID(created.id))))
        except Exception as exc:
            # Previously any failure here surfaced as an opaque 500 with no
            # server log (e.g. an over-long `purpose` failing model
            # validation — see the files.purpose width fix). Log the real
            # cause and return an actionable error instead of a bare crash.
            logger.exception(
                "Attachment upload failed (project=%s session=%s file=%s)",
                project_name, session_id, getattr(f, "filename", "?"),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store attachment '{getattr(f, 'filename', '')}'.",
            ) from exc
    return results


@attachments_router.delete("/{attachment_id}")
def delete_attachment(attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    FileService(session).delete_file(attachment_id)
    return {"ok": True}


@attachments_router.delete("/{project_name}/{session_id}/{attachment_id}")
def delete_attachment_scoped(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    FileService(session).delete_file(attachment_id)
    return {"ok": True}


@attachments_router.get("/{project_name}/{session_id}/{attachment_id}/raw")
def attachment_raw(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    try:
        content_type, filename, path = FileService(session).get_file_content(attachment_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    # Inline, not attachment: the rail's row click opens this URL in a
    # browser tab expecting the file to RENDER (image/pdf/text). The
    # default attachment disposition silently downloads instead, which
    # reads as "open does nothing" (ENG-264).
    return FileResponse(
        path,
        media_type=content_type,
        filename=filename,
        content_disposition_type="inline",
    )


def _unique_project_target(project_dir: Path, filename: str) -> Path:
    safe_name = os.path.basename(filename or "upload").strip() or "upload"
    target = project_dir / safe_name
    if not target.exists():
        return target
    stem = target.stem or "upload"
    suffix = target.suffix
    for i in range(2, 10_000):
        candidate = project_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail="Could not choose a unique project filename")


@attachments_router.post("/{project_name}/{session_id}/{attachment_id}/move-to-project")
def move_attachment_to_project(project_name: str, session_id: str, attachment_id: UUID, session: _SessionDep):
    from cowork.services.files import FileService
    from cowork.services.projects import ProjectService

    try:
        project = ProjectService(session).get_project_by_name(project_name)
        content_type, filename, source = FileService(session).get_file_content(attachment_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    project_dir = Path(project.path)
    project_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_project_target(project_dir, filename)
    shutil.copy2(source, target)
    FileService(session).delete_file(attachment_id)
    return {
        "ok": True,
        "project_path": target.name,
        "absolute_path": str(target),
        "content_type": content_type or mimetypes.guess_type(str(target))[0] or "application/octet-stream",
    }


# ── Scratchpad ───────────────────────────────────────────────────────

scratchpad_router = APIRouter()


@scratchpad_router.post("/cancel")
def cancel_scratchpad():
    return {"ok": True}


# ── Browse (Browser Control M1, read-only) ───────────────────────────
#
# The server is the command authority; the Electron main process owns the
# CDP socket and executes. This router is the HTTP contract between them:
#   GET  /status                 — availability + control_state
#   GET  /resume?session_id=     — reconnect/resume snapshot
#   POST /commands/next          — long-poll for the next queued command
#   POST /commands/{id}/result   — poster returns a command's result
#   POST /bridge/hello           — poller announces a (re)connect
#   POST /bridge/state           — poller pushes a bridge-state change
#   POST /control/stop           — set the stopped pre-dispatch gate (<1s)
#   POST /control/takeover       — mark taken_over
#
# Every field here is content-free: host-only domain, action type, timing,
# typed codes only.

from pydantic import BaseModel

from cowork.schemas.browser import (
    BridgeCommandResult,
    BridgeState,
    BrowserActionType,
    ControlState,
    ResumeState,
    host_only,
)

browse_router = APIRouter()


def _get_control_service(session: Session):
    from cowork.services.browser.control import BrowserControlService

    return BrowserControlService(session)


def _bridge_state_payload(sess) -> dict:
    """The availability + state snapshot the bridge endpoints return.

    Carries `session_id` so the Electron poller can learn it from
    `/bridge/hello` (its only handshake entry point) and use it for the
    session-keyed endpoints (`/commands/next`, `/resume`, `/bridge/state`).
    """
    return {
        "session_id": str(sess.id),
        "available": sess.available,
        "control_state": sess.control_state,
        "bridge_state": sess.bridge_state,
        "requires_reapproval": sess.requires_reapproval,
    }


@browse_router.get("/status")
def browse_status(session: _SessionDep, conversation_id: str | None = Query(default=None)):
    """Availability + control_state.

    Without a `conversation_id` this reports the legacy shape
    (`{"available": False}`) so existing clients keep working. With one, it
    reports the session's live availability and control_state.
    """
    if not conversation_id:
        return {"available": False}
    control = _get_control_service(session)
    sess = control.get_by_conversation(conversation_id)
    if sess is None:
        return {"available": False, "control_state": ControlState.active.value}
    return {
        "available": sess.available,
        "control_state": sess.control_state,
        "bridge_state": sess.bridge_state,
        "domain": sess.active_domain,
        "requires_reapproval": sess.requires_reapproval,
    }


@browse_router.get("/resume")
def browse_resume(session: _SessionDep, session_id: str = Query(...)):
    """Reconnect/resume snapshot for a session (content-free)."""
    from cowork.services.browser.actions import BrowserActionStore

    control = _get_control_service(session)
    sess = control.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")
    store = BrowserActionStore(session)
    last = store.last_observed(sess.id)
    return ResumeState(
        session_id=str(sess.id),
        available=sess.available,
        control_state=ControlState(sess.control_state),
        bridge_state=BridgeState(sess.bridge_state),
        domain=sess.active_domain,
        requires_reapproval=sess.requires_reapproval,
        last_result_code=(last.result_code if last else None),
        last_action_type=(last.action_type if last else None),
        action_count=store.action_count(sess.id),
    ).model_dump()


class _CommandsNextRequest(BaseModel):
    session_id: str
    wait_s: float = 25.0


@browse_router.post("/commands/next")
async def browse_commands_next(req: _CommandsNextRequest, session: _SessionDep):
    """Long-poll for the next queued command for a session.

    The Electron poller calls this after `bridge/hello`. Returns the command
    to execute, or `{"command": None}` when the wait elapses so the poller
    can re-poll.

    The control gate is enforced HERE too, not just pre-dispatch: if the
    session is `stopped` / `taken_over`, we never hand the poller a command.
    Instead we drain any queued/awaiting commands to a terminal result (so
    their producers stop waiting) and return `{"command": None, "blocked":
    ...}`. This closes the race where a command was enqueued just before a
    Stop landed and would otherwise still be pulled and executed.
    """
    from cowork.services.browser.bridge import bridge_command_service
    from cowork.schemas.browser import ResultCode

    control = _get_control_service(session)
    sess = control.get_session(req.session_id)
    if sess is not None and sess.control_state != ControlState.active.value:
        await bridge_command_service.drain_session(
            req.session_id,
            ResultCode.error,
            detail=f"session {sess.control_state}",
        )
        return {"command": None, "blocked": sess.control_state}

    cmd = await bridge_command_service.next(req.session_id, wait_s=req.wait_s)
    if cmd is None:
        return {"command": None}

    # Re-check the gate AFTER the wakeup: a Stop can land while we were
    # awaiting `next()` (its drain sees an empty queue), then a producer
    # that had already passed its own pre-dispatch check enqueues and wakes
    # this await. Without this re-read, that command would be handed to the
    # extension after the Stop. Expire the ORM cache first — the Stop was
    # committed by a different request/DB session.
    session.expire_all()
    sess = control.get_session(req.session_id)
    if sess is not None and sess.control_state != ControlState.active.value:
        # Resolve the pulled command terminally so its producer stops
        # waiting, then drain anything else that raced in.
        await bridge_command_service.fail(
            cmd.command_id,
            ResultCode.error,
            detail=f"session {sess.control_state}",
        )
        await bridge_command_service.drain_session(
            req.session_id,
            ResultCode.error,
            detail=f"session {sess.control_state}",
        )
        return {"command": None, "blocked": sess.control_state}

    return {"command": cmd.model_dump()}


@browse_router.post("/commands/{command_id}/result")
async def browse_commands_result(command_id: str, result: BridgeCommandResult):
    """The poster returns a command's result; resolves the awaiting future."""
    from cowork.services.browser.bridge import bridge_command_service

    # Path is authoritative for the command id.
    payload = result.model_copy(update={"command_id": command_id})
    resolved = await bridge_command_service.resolve(command_id, payload)
    return {"resolved": resolved, "command_id": command_id}


class _BridgeHelloRequest(BaseModel):
    session_id: str | None = None
    conversation_id: str | None = None
    domain: str | None = None
    target_changed: bool = False


@browse_router.post("/bridge/hello")
def browse_bridge_hello(req: _BridgeHelloRequest, session: _SessionDep):
    """The poller announces a (re)connect.

    UPSERT semantics: when a `conversation_id` (and, on first attach, an
    approved `domain`) is supplied, this creates-or-updates the
    `BrowserSession` and the approved-tab grant so a session exists before
    any command is dispatched — closing the gap where nothing created a
    session/grant in production. Without a `conversation_id`, it falls back
    to the legacy `session_id` lookup (404 if unknown).

    A `target_changed` hello (Chrome restarted, target ids changed) marks
    the session `lost` and requires re-approval while preserving history; a
    stopped session stays stopped.

    Hello NEVER clears a pending `requires_reapproval`: the poller re-hellos
    automatically (with a domain) after a Chrome restart, and letting that
    call `approve()` would silently self-approve server-side without any
    user action. Only the explicit, user-driven `/browse/control/approve`
    endpoint clears the flag / refreshes grants once it is set.
    """
    control = _get_control_service(session)

    # Upsert path: a conversation-scoped hello creates/updates the session
    # (and grants the approved host) rather than 404-ing on first connect.
    if req.conversation_id:
        from cowork.services.browser.approval import BrowserApprovalService

        approval = BrowserApprovalService(session)
        existing = control.get_by_conversation(req.conversation_id)
        if existing is not None and existing.requires_reapproval:
            # Pending re-approval: no grant refresh, no flag clear — the
            # poller just learns the session state (requires_reapproval=true).
            sess = existing
        elif req.domain:
            sess = approval.approve(req.conversation_id, req.domain)
        else:
            sess = approval.get_or_create_session(req.conversation_id)
        if sess is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="unknown conversation",
            )
        sess = control.on_bridge_state(
            sess.id, BridgeState.connected, target_changed=req.target_changed
        )
        return _bridge_state_payload(sess)

    if not req.session_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id or conversation_id is required",
        )
    sess = control.on_bridge_state(
        req.session_id, BridgeState.connected, target_changed=req.target_changed
    )
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")
    return _bridge_state_payload(sess)


class _ApproveRequest(BaseModel):
    conversation_id: str
    domain: str


@browse_router.post("/control/approve")
def browse_control_approve(req: _ApproveRequest, session: _SessionDep):
    """Approve a tab: upsert the conversation's session + grant its host.

    This is the production entry point a tab approval in the desktop UI
    calls. It makes the subsequent agent-tool `send()` session lookup +
    permission check succeed for the approved host-only domain.
    """
    from cowork.services.browser.approval import BrowserApprovalService

    sess = BrowserApprovalService(session).approve(
        req.conversation_id, req.domain
    )
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="unknown conversation"
        )
    return {
        "session_id": str(sess.id),
        "active_domain": sess.active_domain,
        "control_state": sess.control_state,
        "available": sess.available,
    }


class _BridgeStateRequest(BaseModel):
    session_id: str
    bridge_state: BridgeState
    target_changed: bool = False


@browse_router.post("/bridge/state")
def browse_bridge_state(req: _BridgeStateRequest, session: _SessionDep):
    """The poller pushes a bridge-state change (connected/lost/…)."""
    control = _get_control_service(session)
    sess = control.on_bridge_state(
        req.session_id, req.bridge_state, target_changed=req.target_changed
    )
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")
    return _bridge_state_payload(sess)


class _ControlRequest(BaseModel):
    conversation_id: str


class _StopRequest(_ControlRequest):
    # Client-generated stop token (UUID string). The renderer sends it with
    # the user's stop; the Electron poller re-sends the SAME id as its
    # acknowledgement, which must be idempotent (see browse_control_stop).
    stop_id: str | None = None


async def _drain_gated_session(sess) -> None:
    """Drain any queued/in-flight commands for a gated session.

    Called right after `stop`/`takeover` so an already-in-flight command's
    awaiting producer resolves to a terminal (non-ok) result immediately
    instead of hanging until timeout, and any queued command is dropped so a
    poller can't pull it after the gate landed.
    """
    if sess is None:
        return
    from cowork.services.browser.bridge import bridge_command_service
    from cowork.schemas.browser import ResultCode

    await bridge_command_service.drain_session(
        str(sess.id), ResultCode.error, detail=f"session {sess.control_state}"
    )


@browse_router.post("/control/stop")
async def browse_control_stop(req: _StopRequest, session: _SessionDep):
    """Set the stopped pre-dispatch gate synchronously (<1s, persisted).

    Returns 200 even when no session exists yet so the Stop button never
    errors; the gate is set the moment a session is created too, via the
    persisted control_state. Any queued/in-flight command is drained so it
    never resolves as a success after the Stop.

    The server is the single source of truth for this gate: it is cleared
    ONLY by a fresh user turn (`resume_on_new_turn`, from POST /responses).
    No re-approval is required after a Stop — Electron's local
    `stopRequested` latch merely closes the hand-out→execute race and
    self-clears.

    Idempotent by `stop_id`: the poller acknowledges the gate by re-POSTing
    the renderer's `stop_id`. An already-applied `stop_id` is a pure ack —
    it neither changes control_state (the session may have been resumed by
    a fresh user turn in between; the ack must NOT re-stop it) nor drains.
    Absent/new stop_ids keep the full stop behavior (legacy callers too).
    """
    control = _get_control_service(session)
    sess, applied = control.apply_stop(req.conversation_id, stop_id=req.stop_id)
    if applied:
        await _drain_gated_session(sess)
    if sess is None:
        return {"stopped": True, "control_state": ControlState.stopped.value, "session": None}
    return {"stopped": applied, "control_state": sess.control_state, "session": str(sess.id)}


@browse_router.post("/control/takeover")
async def browse_control_takeover(req: _ControlRequest, session: _SessionDep):
    """Mark the conversation's browser session taken_over."""
    control = _get_control_service(session)
    sess = control.takeover_by_conversation(req.conversation_id)
    await _drain_gated_session(sess)
    if sess is None:
        return {"taken_over": True, "control_state": ControlState.taken_over.value, "session": None}
    return {"taken_over": True, "control_state": sess.control_state, "session": str(sess.id)}

