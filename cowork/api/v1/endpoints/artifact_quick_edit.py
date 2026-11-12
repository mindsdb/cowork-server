"""POST /api/v1/artifacts/workspace/{project_ref}/{artifact_id}/quick-edits (ENG-4227).

The artifact viewer's chat bar sends a change here. Jev routes it (anton's
``route_request``): a change that needs only the artifact's own files runs
anton's artifact editor right here, with no chat turn, checks the open
preview, and saves ONE new draft revision; anything else is handed back to
the app (``done`` with status ``agent``), which sends the message to the
artifact's chat as a normal turn.

The response is ``text/event-stream``:

* ``route``         ``{"route": "edit"|"agent", "confidence": float, "ms": int, "reason": str}``;
  with no decisions model for the caller's provider, at once
  ``{"route": "agent", "confidence": 1, "ms": 0, "reason": "no_decisions_model"}``
  and neither Jev nor the editor runs
* ``progress``      ``{"stage": str}``
* ``check_request`` ``{"check_id": "acheck:<hex>", "artifact_slug": str, "timeout_s": int}``,
  answered with the existing ``POST /api/v1/artifacts/checks/{check_id}``
  (same broker as a turn's checks, authorized against this request's owner)
* ``done``          ``{"status", "summary", "revisionId", "previousRevisionId", "changed", "check",
  "editId", "undoable", "ms"}``
* ``error``         ``{"message": str}``

``POST .../quick-edits/{edit_id}/undo`` puts back every file an applied
edit changed and removes those it created, all or nothing
(``services/artifact_quick_edits.py``); the tracked source's restore is
journalled as a revision.

Everything that can refuse the request does so before the stream starts, as
an ordinary HTTP error: 403 (permissions, owner), 501 (quick edits off or the
installed anton has no quick editor), 409 (a quick edit is already running on
the artifact, an agent repair still gates it, or ``expectedRevisionId`` is
stale, in the PUT draft route's shape), 404/422 as the neighbouring routes.

The editor works on the artifact folder on the event loop, as anton's editor
does inside an in-process turn; anton's workspace writes all-or-nothing and
never follows a link. This route's own file store work (the revision journal,
the file list the router reads) goes through ``run_efs``. A client that goes
away cancels the edit, and anton's editor restores every file it touched.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cowork.api.v1.endpoints.artifact_workspace import ArtifactIdDep, ScopeDep, _not_found_detail, _owner_workspace
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope
from cowork.db.units import run_efs, run_to_completion
from cowork.harnesses.anton_harness.preview_checker import ask_preview
from cowork.services import artifact_quick_edits
from cowork.services import quick_edit as quick_edit_service
from cowork.services.artifact_revisions import (
    RepairAlreadyPending,
    RevisionConflict,
    RevisionValidationError,
    begin_quick_edit,
    record_quick_edit,
)
from cowork.services.product_permissions import require_product_permission
from cowork.services.quick_edit import AntonQuickEdit, QuickEditBackend
from cowork.streaming.preview_checks import PreviewCheckBroker, preview_checks
from cowork.streaming.sse import sse_frame
from cowork.streaming.turn_index import TurnOwner

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
}

#: The whole edit, router to final check. Past it the edit is cancelled, which
#: restores its files, and the app hears "failed".
EDIT_TIMEOUT_S = 240.0
#: How often the stream looks for a client that went away while nothing is
#: being sent, and how often it sends a comment to keep proxies from closing it.
_DISCONNECT_POLL_S = 1.0
_KEEPALIVE_S = 15.0
#: The router reads the artifact's file names; more than this tells it nothing
#: more about whether a change is a quick one.
_MAX_ROUTER_FILES = 200
_MAX_FINDINGS = 20
_MAX_FINDING_TEXT = 1000
_MAX_STAGE = 40
#: What the app is told when the edit could not run; the reason is logged.
_ERROR_MESSAGE = "Couldn't run the quick edit. Try again, or ask the agent."
_TIMEOUT_SUMMARY = "The edit took too long, so nothing was changed."

#: The route event's reason when no decisions model serves the caller.
NO_DECISIONS_REASON = "no_decisions_model"

QUICK_EDIT_STATUSES = frozenset({"applied", "escalate", "failed", "noop"})


class _HistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class _QuickEditBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Earlier messages of the bar's thread, so "make it bigger" has a referent.
    history: list[_HistoryEntry] = Field(default_factory=list, max_length=20)
    expectedRevisionId: str | None = Field(default=None, min_length=1, max_length=80)
    conversationId: UUID | None = None
    forceRoute: Literal["edit", "agent"] | None = None


def _lock_key(scope: TenantScope, artifact_id: str) -> str:
    return f"{scope.org_id or ''}:{artifact_id}"


def _broker_key(artifact_id: str) -> str:
    """The key this edit's preview checks are filed under. Never a
    conversation id, so no turn's end closes them."""
    return f"quick-edit:{artifact_id}"


def _artifact_kind(metadata: dict) -> str:
    kind = metadata.get("type") if isinstance(metadata, dict) else None
    return kind if isinstance(kind, str) and kind else "html-app"


def _router_input(anton: AntonQuickEdit, folder: Path, kind: str) -> tuple[list[str], dict]:
    """What the router reads from the folder, in one file store hop: the
    artifact's content files as anton's editor sees them, and the optional
    context (name, description, the shape of its data) when this anton's
    router takes it. An empty file list makes the router send the message to
    the agent."""
    files: list[str] = []
    if anton.workspace_cls is not None:
        try:
            entries = anton.workspace_cls(folder, kind=kind).map()
            files = [str(getattr(entry, "path", entry)) for entry in entries[:_MAX_ROUTER_FILES]]
        except Exception:
            logger.warning("Could not list the artifact's files for the quick edit router", exc_info=True)
    context: dict = {}
    if anton.describe_artifact is not None:
        try:
            described = anton.describe_artifact(folder)
        except Exception:
            described = None
        if isinstance(described, dict):
            context = {
                key: str(described.get(key) or "")
                for key in ("name", "description", "data")
                if key in anton.route_extras
            }
    return files, context


def _capture_for_undo(anton: AntonQuickEdit, folder: Path, kind: str, source_path: str) -> dict | None:
    """Every file the editor may change, as it is now: the content files
    anton's workspace lists, the tracked source, and anton's presentation
    program (the editor snapshots it too)."""
    paths: list[str] = [source_path]
    if anton.workspace_cls is not None:
        try:
            paths += [str(getattr(e, "path", e)) for e in anton.workspace_cls(folder, kind=kind).map()]
        except Exception:
            logger.warning("Could not list the artifact's files for the quick edit's undo", exc_info=True)
            return None
    try:
        from anton.core.artifacts.internal_files import PRESENTATION_PROGRAM_FILENAME

        paths.append(PRESENTATION_PROGRAM_FILENAME)
    except ImportError:
        pass
    return artifact_quick_edits.capture(folder, paths)


def _finish_applied(
    folder: Path,
    metadata: dict,
    artifact_id: str,
    *,
    source_path: str,
    base_revision_id: str,
    summary: str,
    conversation_id: str | None,
    before: dict | None,
    changed: list[str],
) -> tuple[dict | None, str | None]:
    """An applied edit's file store work, in one hop: the tracked source's
    revision, then the undo of every file the edit changed."""
    revision = record_quick_edit(
        folder, metadata, artifact_id, rel_path=source_path, base_revision_id=base_revision_id,
        summary=summary, conversation_id=conversation_id,
    )
    if before is None:
        return revision, None
    edit_id = artifact_quick_edits.new_edit_id()
    try:
        kept = artifact_quick_edits.save_undo(
            folder, edit_id=edit_id, before=before, changed=changed, source_path=source_path,
            revision_id=revision["id"] if revision else None,
            previous_revision_id=base_revision_id if revision else None,
        )
    except (OSError, ValueError):
        logger.warning("Could not keep the undo of a quick edit", exc_info=True)
        kept = None
    return revision, edit_id if kept else None


def _check_json(check: Any) -> dict | None:
    """The final check as the app shows it: never the screenshot."""
    if check is None:
        return None
    findings = []
    for finding in list(getattr(check, "findings", None) or [])[:_MAX_FINDINGS]:
        findings.append({
            "kind": str(getattr(finding, "kind", "") or ""),
            "message": str(getattr(finding, "message", "") or "")[:_MAX_FINDING_TEXT],
            "source": (str(getattr(finding, "source", "") or "")[:_MAX_FINDING_TEXT] or None),
        })
    return {
        "status": str(getattr(check, "status", "unverified")),
        "channel": str(getattr(check, "channel", "") or ""),
        "note": str(getattr(check, "note", "") or "")[:_MAX_FINDING_TEXT],
        "findings": findings,
    }


def _done(
    *,
    status_: str,
    summary: str = "",
    revision_id: str | None = None,
    previous_revision_id: str | None = None,
    changed: list[str] | None = None,
    check: dict | None = None,
    edit_id: str | None = None,
    started: float,
) -> str:
    return sse_frame("done", {
        "status": status_,
        "summary": summary,
        "revisionId": revision_id,
        "previousRevisionId": previous_revision_id,
        "changed": list(changed or []),
        "check": check,
        # The id POST .../quick-edits/{editId}/undo takes; set when this
        # edit applied and its undo was kept.
        "editId": edit_id,
        "undoable": edit_id is not None,
        "ms": _ms(started),
    })


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


@dataclass
class QuickEditRun:
    """One quick edit, from the route to the end of its stream."""

    anton: AntonQuickEdit
    scope: TenantScope
    folder: Path
    metadata: dict
    artifact_id: str
    body: _QuickEditBody
    # From begin_quick_edit: the tracked source and the head the edit follows.
    source_path: str
    base_revision_id: str
    started: float
    lock_key: str
    broker: PreviewCheckBroker = preview_checks
    # Tests replace it; None reads quick_edit.resolve_backend at call time.
    resolve_backend: Callable[[TenantScope], Any] | None = None
    # None reads EDIT_TIMEOUT_S at call time.
    edit_timeout_s: float | None = None
    _released: bool = field(default=False, init=False)

    @property
    def kind(self) -> str:
        return _artifact_kind(self.metadata)

    def release(self) -> None:
        if not self._released:
            self._released = True
            quick_edit_service.quick_edit_locks.release(self.lock_key)

    async def _route(self, backend: QuickEditBackend, decisions) -> tuple[str, float, str]:
        body = self.body
        if body.forceRoute is not None:
            return body.forceRoute, 1.0, "forced"
        if not backend.route_base_url or not backend.route_api_key:
            # anton names a decisions model, but this caller has no endpoint
            # to ask it on (Jev opted out, or no key): the chat, as for no
            # decisions model at all.
            return "agent", 1.0, NO_DECISIONS_REASON
        files, context = await run_efs(partial(_router_input, self.anton, self.folder, self.kind), scope=self.scope)
        extra: dict = {**context}
        model = backend.route_model or decisions.model
        if model:
            extra["model"] = model
        if "history" in self.anton.route_extras and body.history:
            extra["history"] = [entry.model_dump() for entry in body.history]
        decision = await self.anton.route_request(
            body.message,
            files=files,
            kind=self.kind,
            base_url=backend.route_base_url,
            api_key=backend.route_api_key,
            **extra,
        )
        route = getattr(decision, "route", "agent")
        if route not in ("edit", "agent"):
            route = "agent"
        try:
            confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return route, confidence, str(getattr(decision, "reason", "") or "")

    def _checker(self, emit: Callable[[str], None]):
        """The editor's checker chain, ending in the viewer's preview, asked
        through the preview check broker on THIS response's stream."""
        owner = TurnOwner(org_id=self.scope.org_id, user_id=self.scope.user_id)

        async def publish(check_id: str, slug: str, timeout: int) -> bool:
            emit(sse_frame("check_request", {"check_id": check_id, "artifact_slug": slug, "timeout_s": timeout}))
            return True

        async def preview_checker(artifact_slug: str, timeout_s: float) -> dict | None:
            return await ask_preview(
                broker=self.broker,
                key=_broker_key(self.artifact_id),
                artifact_slug=artifact_slug,
                timeout_s=timeout_s,
                publish=publish,
                owner=owner,
            )

        checkers = self.anton.checkers
        slug = self.folder.name
        if checkers is not None and hasattr(checkers, "select_checker"):
            return checkers.select_checker(self.kind, slug=slug, preview_checker=preview_checker)
        if checkers is not None and hasattr(checkers, "PreviewChecker"):
            return checkers.PreviewChecker(preview_checker, slug=slug)
        return None

    async def _capture(self) -> dict | None:
        """The artifact's content files as they are before the edit, for its
        undo; None when they are too many or too large to keep."""
        return await run_efs(
            partial(_capture_for_undo, self.anton, self.folder, self.kind, self.source_path), scope=self.scope,
        )

    async def _save(self, result: Any, before: dict | None) -> tuple[dict | None, str | None]:
        """The applied edit's revision and its undo. Run to completion even
        when the client goes away now: the files are already written."""
        conversation_id = str(self.body.conversationId) if self.body.conversationId else None
        return await run_to_completion(run_efs(
            partial(
                _finish_applied,
                self.folder,
                self.metadata,
                self.artifact_id,
                source_path=self.source_path,
                base_revision_id=self.base_revision_id,
                summary=str(getattr(result, "summary", "") or ""),
                conversation_id=conversation_id,
                before=before,
                changed=[str(path) for path in (getattr(result, "changed", None) or [])],
            ),
            scope=self.scope,
            deadline=False,
        ))

    async def execute(self, emit: Callable[[str], None]) -> None:
        """Route, edit, check, save; each step's frame goes to ``emit``."""
        backend: QuickEditBackend | None = None
        try:
            if self.body.forceRoute == "agent":
                emit(sse_frame("route", {"route": "agent", "confidence": 1.0, "ms": 0, "reason": "forced"}))
                emit(_done(status_="agent", started=self.started))
                return
            user, decisions = await quick_edit_service.decisions_for_scope(self.anton, self.scope)
            if not decisions.available:
                # No decisions model serves this person's provider (their own
                # OpenAI or Anthropic keys, say): neither Jev nor the editor
                # runs, not even when the app forces the edit, and the app
                # sends the message to the chat.
                emit(sse_frame("route", {"route": "agent", "confidence": 1.0, "ms": 0, "reason": NO_DECISIONS_REASON}))
                emit(_done(status_="agent", started=self.started))
                return
            backend = await (self.resolve_backend or quick_edit_service.resolve_backend)(self.scope, user, decisions)
            route_started = time.monotonic()
            route, confidence, reason = await self._route(backend, decisions)
            emit(sse_frame("route", {"route": route, "confidence": confidence, "ms": _ms(route_started), "reason": reason}))
            if route == "agent":
                emit(_done(status_="agent", started=self.started))
                return

            before = await self._capture()

            def on_progress(event: dict) -> None:
                stage = event.get("stage") if isinstance(event, dict) else None
                if isinstance(stage, str) and stage:
                    emit(sse_frame("progress", {"stage": stage[:_MAX_STAGE]}))

            timeout_s = self.edit_timeout_s or EDIT_TIMEOUT_S
            try:
                result = await asyncio.wait_for(
                    self.anton.run_quick_edit(
                        self.folder,
                        self.body.message,
                        kind=self.kind,
                        llm=backend.llm,
                        checker=self._checker(emit),
                        history=[entry.model_dump() for entry in self.body.history],
                        on_progress=on_progress,
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                # wait_for cancelled the edit, and a cancelled edit restores.
                logger.warning("Quick edit on %s timed out after %ss", self.artifact_id, timeout_s)
                emit(_done(status_="failed", summary=_TIMEOUT_SUMMARY, started=self.started))
                return

            status_ = getattr(result, "status", "failed")
            if status_ not in QUICK_EDIT_STATUSES:
                status_ = "failed"
            changed = [str(path) for path in (getattr(result, "changed", None) or [])]
            summary = str(getattr(result, "summary", "") or "")
            revision = edit_id = None
            if status_ == "applied":
                revision, edit_id = await self._save(result, before)
                if self.body.conversationId is not None:
                    await quick_edit_service.record_conversation_note(
                        self.scope,
                        conversation_id=self.body.conversationId,
                        message=self.body.message,
                        text=quick_edit_service.conversation_note_text(
                            summary=summary, changed=changed, revision=revision,
                        ),
                    )
            emit(_done(
                status_=status_,
                summary=summary,
                revision_id=revision["id"] if revision else None,
                previous_revision_id=self.base_revision_id if revision else None,
                changed=changed,
                check=_check_json(getattr(result, "check", None)),
                edit_id=edit_id,
                started=self.started,
            ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Quick edit on %s failed", self.artifact_id)
            emit(sse_frame("error", {"message": _ERROR_MESSAGE}))
        finally:
            self.broker.close_conversation(_broker_key(self.artifact_id))
            if backend is not None:
                await asyncio.shield(backend.close())


async def quick_edit_events(run: QuickEditRun, request: Request) -> AsyncIterator[str]:
    """The response body: the edit's frames as they come.

    The edit runs in a task of its own, so a client that goes away (seen
    here, or as this generator being closed) cancels it, and the lock is
    released only once it has unwound, after anton restored the files.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    worker = asyncio.ensure_future(_drive(run, queue))
    last_sent = time.monotonic()
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=_DISCONNECT_POLL_S)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    logger.info("Quick edit client for %s went away; cancelling the edit", run.artifact_id)
                    return
                if time.monotonic() - last_sent >= _KEEPALIVE_S:
                    last_sent = time.monotonic()
                    yield ": keep-alive\n\n"
                continue
            if frame is None:
                return
            last_sent = time.monotonic()
            yield frame
    finally:
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        run.release()


async def _drive(run: QuickEditRun, queue: asyncio.Queue) -> None:
    try:
        await run.execute(queue.put_nowait)
    finally:
        queue.put_nowait(None)


@router.post("/workspace/{project_ref}/{artifact_id}/quick-edits")
async def quick_edit_artifact(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    body: _QuickEditBody,
    request: Request,
    scope: ScopeDep,
):
    """Edit the artifact's draft from the viewer's chat bar; see the module."""
    # The agent repair route's permissions: a quick edit is agent work on the
    # artifact's source.
    await require_product_permission(scope, "artifact.manage")
    await require_product_permission(scope, "product.execute")
    anton = quick_edit_service.anton_quick_edit() if get_app_settings().quick_edit_enabled else None
    if anton is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Quick edits are not available on this server.",
        )
    started = time.monotonic()
    _source, folder, metadata, _capabilities = await _owner_workspace(scope, project_ref, artifact_id)

    lock_key = _lock_key(scope, artifact_id)
    if not quick_edit_service.quick_edit_locks.acquire(lock_key):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "A quick edit is already running on this artifact", "reason": "quick_edit_running"},
        )
    try:
        begun = await run_efs(
            partial(
                begin_quick_edit, folder, metadata, artifact_id, expected_revision_id=body.expectedRevisionId,
            ),
            scope=scope,
        )
    except BaseException as exc:
        quick_edit_service.quick_edit_locks.release(lock_key)
        if isinstance(exc, RevisionConflict):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"message": str(exc), "currentRevision": exc.current},
            ) from exc
        if isinstance(exc, RepairAlreadyPending):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": str(exc),
                    "reason": "repair_pending",
                    "repairId": exc.repair.get("id"),
                    "commentThreadId": exc.repair.get("commentThreadId"),
                },
            ) from exc
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_not_found_detail(exc)) from exc
        if isinstance(exc, (RevisionValidationError, TimeoutError)):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise

    run = QuickEditRun(
        anton=anton,
        scope=scope,
        folder=folder,
        metadata=metadata,
        artifact_id=artifact_id,
        body=body,
        source_path=begun["path"],
        base_revision_id=begun["revision"]["id"],
        started=started,
        lock_key=lock_key,
    )
    try:
        return StreamingResponse(
            quick_edit_events(run, request), media_type="text/event-stream", headers=_SSE_HEADERS,
        )
    except BaseException:
        run.release()
        raise


class _UndoBody(BaseModel):
    # The tracked source's head as the viewer has it (the done event's
    # revisionId, or the head it already held when the edit left the source
    # alone). Null skips that check; changed files are refused either way.
    expectedRevisionId: str | None = Field(default=None, min_length=1, max_length=80)


@router.post("/workspace/{project_ref}/{artifact_id}/quick-edits/{edit_id}/undo")
async def undo_quick_edit(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    edit_id: str,
    body: _UndoBody,
    scope: ScopeDep,
):
    """Undo one quick edit: every file it changed goes back, every file it
    created goes, all or nothing.

    409: a quick edit is running on the artifact, a file the edit changed was
    changed again since (``reason: files_changed`` with ``paths``), it was
    already undone, or ``expectedRevisionId`` is stale (the draft PUT's
    shape). 404: no such edit, or its undo is no longer kept (only the last
    ten are).
    """
    await require_product_permission(scope, "artifact.manage")
    _source, folder, metadata, _capabilities = await _owner_workspace(scope, project_ref, artifact_id)
    lock_key = _lock_key(scope, artifact_id)
    if not quick_edit_service.quick_edit_locks.acquire(lock_key):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "A quick edit is already running on this artifact", "reason": "quick_edit_running"},
        )
    try:
        actor_id = str(scope.user_id) if scope and scope.user_id else None
        return await run_efs(
            partial(
                artifact_quick_edits.undo, folder, metadata, artifact_id, edit_id,
                expected_revision_id=body.expectedRevisionId, actor_id=actor_id,
            ),
            scope=scope,
        )
    except artifact_quick_edits.QuickEditFilesChanged as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "reason": "files_changed", "paths": exc.paths},
        ) from exc
    except artifact_quick_edits.QuickEditAlreadyUndone as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail={"message": str(exc), "reason": "already_undone"},
        ) from exc
    except RevisionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "currentRevision": exc.current},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_not_found_detail(exc)) from exc
    except (RevisionValidationError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        quick_edit_service.quick_edit_locks.release(lock_key)
