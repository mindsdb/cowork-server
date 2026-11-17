"""Docs API: create, read, edit, watch and export structured doc artifacts.

A doc is addressed like every other artifact workspace route: the project
(``projectId`` or ``local``) and the artifact's UUID. Reading needs review
access; every write needs ``artifact.manage`` and ownership, exactly as a
source edit through the artifact workspace does.

Writes from the editor and from the agent's tools share one path
(``cowork.docs.service``), so the person typing and the agent writing are
serialised on the same lock and announced on the same hub; ``/events`` streams
those announcements to every open editor. ``/history`` lists the versions the
doc's editing sessions left and restores one, for its owner alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from cowork.api.v1.artifact_scope import review_artifact_for_request, scoped_project_id_for_request
from cowork.api.v1.content_disposition import attachment_disposition
from cowork.api.v1.endpoints.artifact_workspace import ArtifactIdDep
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession, TenantScope, get_tenant_scope
from cowork.db.units import FilesystemBusy, run_db, run_efs
from cowork.docs.export import CONVERTED_FORMATS, MEDIA_TYPES, export_bytes, export_filename, offered_formats
from cowork.services.artifact_export import ExportRefusal, ExportRefused, export_slot, org_export_refused
from cowork.services.artifact_identity import opened_artifact_folder
from cowork.services.render_pdf_client import RenderPdfFailed
from cowork.docs.history import LIST_LIMIT, LIST_MAX, list_versions, read_version, restore
from cowork.docs.hub import conv_topic, doc_topic, hub
from cowork.docs.images import MAX_IMAGE_BYTES, store_image
from cowork.docs.model import DocError
from cowork.docs.ops import MAX_OPS
from cowork.docs.service import apply, create_doc
from cowork.docs.store import doc_file, is_doc_folder, load_doc
from cowork.docs.sync import after_write
from cowork.models.conversation import Conversation
from cowork.services.artifact_ownership import record_artifact_owner
from cowork.services.artifact_permissions import artifact_capabilities, current_capabilities, require_artifact_owner
from cowork.services.artifact_roots import plan_project_roots, readable_sources
from cowork.services.artifacts import card_for_folder, origin_conversation_id
from cowork.services.conversations import ConversationService
from cowork.services.product_permissions import require_product_permission

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

_KEEPALIVE_S = 15.0
_POLL_S = 2.0

_STATUS_FOR_CODE = {
    "stale": status.HTTP_409_CONFLICT,
    "no_doc": status.HTTP_404_NOT_FOUND,
    "no_version": status.HTTP_404_NOT_FOUND,
    "too_big": status.HTTP_413_CONTENT_TOO_LARGE,
    # Another replica held the doc's lock past the wait: the edit is fine, resend it.
    "busy": status.HTTP_503_SERVICE_UNAVAILABLE,
    # A doc file that won't load is the server's problem, not the request's.
    "bad_doc": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def _doc_error(err: DocError) -> HTTPException:
    code = _STATUS_FOR_CODE.get(err.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
    return HTTPException(status_code=code, detail=err.to_dict())


async def _resolve(scope: TenantScope, project_ref: str, artifact_id: str, *, write: bool):
    """(source, folder, metadata, is_own) for a doc the caller may read, or may write."""
    source, folder, metadata, is_own = await review_artifact_for_request(scope, project_ref, artifact_id)
    if not await run_efs(partial(is_doc_folder, folder), scope=scope):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This artifact is not a doc")
    if write:
        if not is_own:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the doc's owner can edit it")
        await run_db(lambda unit: require_artifact_owner(unit, source, folder.name), scope=scope)
    return source, folder, metadata, is_own


def _card(scope: TenantScope, source, folder: Path) -> dict[str, Any] | None:
    """The doc's artifact card: its publish state, access and paths, as the library shows them (owner only)."""
    try:
        card = card_for_folder(folder, scope=scope, project_id=source.project_id,
                               project_name=source.project_name, artifacts_base=folder.parent)
        return card.to_json() if card is not None else None
    except Exception:  # noqa: BLE001 - the doc is there either way; the card is what the library adds
        logger.warning("no card for doc %s", folder, exc_info=True)
        return None


# ---------------------------------------------------------------- create


class _CreateBody(BaseModel):
    projectId: UUID
    title: str = Field(min_length=1, max_length=300)
    markdown: str | None = Field(default=None, max_length=2_000_000)


def _project_plan(session: ScopedSession, *, project_ref: str):
    """The plan of one request-selected project's roots, as a database unit."""
    project_id = scoped_project_id_for_request(session, project_ref)
    try:
        return plan_project_roots(session, project_id)
    except ValueError as exc:
        # The project can disappear between catalog recovery and root discovery.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project") from exc


@router.post("/")
async def create_blank_doc(body: _CreateBody, scope: TenantScope = Depends(get_tenant_scope)):
    """A doc the person starts themselves (the agent's own path is create_doc)."""
    await require_product_permission(scope, "artifact.manage")
    plan = await run_db(partial(_project_plan, project_ref=str(body.projectId)), scope=scope)
    sources = await run_efs(partial(readable_sources, plan), scope=scope)
    # A project in a folder the person chose (external) is as good a home as any: the
    # agent writes its artifacts there too.
    source = next((s for s in sources if not getattr(s, "external", False)), sources[0] if sources else None)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown project")
    try:
        made = await run_efs(
            lambda: create_doc(Path(source.base), title=body.title, markdown=body.markdown or "", by="user"),
            scope=scope,
        )
    except DocError as err:
        raise _doc_error(err) from err
    # Its creator owns it; in org mode, unrecorded, every edit after this would be refused.
    await run_db(
        lambda unit: record_artifact_owner(unit, source.project_id, made.slug, scope.user_id, action="create"),
        scope=scope,
    )
    card = await run_efs(partial(_card, scope, source, made.folder), scope=scope)
    return {"artifactId": made.artifact_id, "slug": made.slug, "doc": made.doc, "card": card}


# ---------------------------------------------------------------- conversation events


def _owns_conversation(session: ScopedSession, *, conversation_id: UUID) -> bool:
    """Whether the caller owns the conversation: ``owned_ids_query`` applies the
    org and, in org mode, the author."""
    owned = ConversationService(session).owned_ids_query().where(Conversation.id == conversation_id)
    return session.exec(owned).first() is not None


def _conversation_stored(session: ScopedSession, *, conversation_id: UUID) -> bool:
    """Whether any organization stores the conversation. Run under SYSTEM_SCOPE,
    which looks past the org, so another org's conversation is not mistaken
    for one not stored yet."""
    return session.get(Conversation, conversation_id) is not None


async def _may_follow_conversation(scope: TenantScope, conversation_id: UUID) -> bool:
    """The caller's own conversation, or one not stored yet.

    The composer allocates a new chat's id before the server stores it, and
    this stream opens as that first turn starts, so an id with no row yet may
    be followed: nothing is announced on it before the owner's turn creates
    it. One that exists must be the caller's, since its events carry doc
    titles and cards.
    """
    if await run_db(partial(_owns_conversation, conversation_id=conversation_id), scope=scope):
        return True
    return not await run_db(partial(_conversation_stored, conversation_id=conversation_id), scope=SYSTEM_SCOPE)


@router.get("/events")
async def conversation_doc_events(
    request: Request,
    conversation_id: UUID = Query(...),
    scope: TenantScope = Depends(get_tenant_scope),
):
    """Docs the agent creates or changes in one conversation, as they happen.

    Only in-process turns announce here (desktop and local). The chat still
    receives each doc's card at the end of the turn either way.
    """
    if not await _may_follow_conversation(scope, conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    topic = conv_topic(str(conversation_id))
    return _sse(request, topic, initial=None, replay=True)


# ---------------------------------------------------------------- one doc


@router.get("/{project_ref}/{artifact_id}")
async def get_doc(project_ref: str, artifact_id: ArtifactIdDep, scope: TenantScope = Depends(get_tenant_scope)):
    source, folder, metadata, is_own = await _resolve(scope, project_ref, artifact_id, write=False)
    try:
        doc = await run_efs(partial(load_doc, folder), scope=scope)
    except DocError as err:
        raise _doc_error(err) from err
    capabilities = await current_capabilities(
        scope, await run_db(lambda unit: artifact_capabilities(unit, source, folder.name), scope=scope))
    # A doc's export is a download, a reader's as much as its owner's: what this deployment converts.
    capabilities = {**capabilities, "exportFormats": offered_formats(org_mode=bool(scope.org_mode))}
    # The card carries the share's access (its password, the invited emails): the owner's alone.
    card = await run_efs(partial(_card, scope, source, folder), scope=scope) if is_own else None
    return {
        "artifactId": artifact_id,
        "slug": folder.name,
        "doc": doc,
        "card": card,
        "capabilities": capabilities,
        "originConversationId": origin_conversation_id(metadata),
    }


class _OpsBody(BaseModel):
    ops: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_OPS)
    tab: str | None = Field(default=None, max_length=100)
    ifRev: int | None = None
    clientId: str | None = Field(default=None, max_length=100)


@router.post("/{project_ref}/{artifact_id}/ops")
async def edit_doc(
    project_ref: str, artifact_id: ArtifactIdDep, body: _OpsBody, scope: TenantScope = Depends(get_tenant_scope)
):
    await require_product_permission(scope, "artifact.manage")
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=True)
    try:
        doc, result = await run_efs(
            lambda: apply(folder, body.ops, tab=body.tab, by="user", if_rev=body.ifRev), scope=scope
        )
    except DocError as err:
        raise _doc_error(err) from err
    after_write(folder, doc, result, by="user", scope=scope, client_id=body.clientId)
    return {"doc": doc, **result}


async def _body_within(request: Request, limit: int) -> bytes:
    """The request body, refused as too large once it passes ``limit`` bytes."""
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise _doc_error(DocError("too_big", f"an image is at most {limit // (1024 * 1024)} MB."))
    return bytes(data)


@router.post("/{project_ref}/{artifact_id}/images")
async def add_doc_image(
    project_ref: str, artifact_id: ArtifactIdDep, request: Request, scope: TenantScope = Depends(get_tenant_scope)
):
    """Keep an image the person pasted or dropped beside the doc; answers the src that shows it.

    The editor then puts it in the doc with an ordinary edit.
    """
    await require_product_permission(scope, "artifact.manage")
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=True)
    data = await _body_within(request, MAX_IMAGE_BYTES)
    try:
        src = await run_efs(partial(store_image, folder, data), scope=scope)
    except DocError as err:
        raise _doc_error(err) from err
    return {"src": src}


@router.get("/{project_ref}/{artifact_id}/events")
async def doc_events(
    project_ref: str, artifact_id: ArtifactIdDep, request: Request, scope: TenantScope = Depends(get_tenant_scope)
):
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=False)
    return _sse(request, doc_topic(artifact_id), initial=folder, scope=scope)


# How a refused or failed Word or PDF export answers, as the artifact export does.
_EXPORT_REFUSAL_STATUS = {
    ExportRefusal.NOT_OFFERED: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ExportRefusal.TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    ExportRefusal.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ExportRefusal.BUSY: status.HTTP_429_TOO_MANY_REQUESTS,
}
_PDF_FAILURE_DETAIL = {
    413: "This doc can't be exported as a PDF.",
    422: "This doc can't be exported as a PDF.",
    502: "Couldn't export this doc as a PDF.",
    503: "PDF export is busy; try again.",
    504: "PDF export took too long.",
}


def _converted(source, folder_name: str, doc, fmt: str, org_mode: bool) -> bytes:
    """A Word or PDF export, with the doc's folder pinned for the images beside it."""
    with opened_artifact_folder(source, folder_name) as pinned:
        return export_bytes(doc, fmt, folder=pinned, org_mode=org_mode)


@router.get("/{project_ref}/{artifact_id}/export")
async def export_doc(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: TenantScope = Depends(get_tenant_scope),
    format: Literal["md", "html", "pdf", "docx"] = Query(default="md"),
):
    """The doc as a file to download. Markdown and HTML always; Word and PDF as
    the deployment offers them (``offered_formats``): one conversion at a time,
    a busy server answering 429 at once, as the artifact export does."""
    org_mode = bool(scope.org_mode)
    if format not in offered_formats(org_mode=org_mode):
        if org_export_refused(org_mode=org_mode):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Word and PDF export aren't available on this deployment. Export to HTML instead.",
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="PDF export isn't set up here. Export to Word or HTML instead.",
        )
    source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=False)
    try:
        doc = await run_efs(partial(load_doc, folder), scope=scope)
        if format in CONVERTED_FORMATS:
            async with export_slot(scope=scope, export_format=format):
                payload = await run_efs(partial(_converted, source, folder.name, doc, format, org_mode), scope=scope)
        else:
            payload = await run_in_threadpool(export_bytes, doc, format)
    except DocError as err:
        raise _doc_error(err) from err
    except ExportRefused as exc:
        raise HTTPException(status_code=_EXPORT_REFUSAL_STATUS[exc.refusal], detail=str(exc)) from exc
    except RenderPdfFailed as exc:
        # The reason never quotes the render service, so it is safe to log.
        logger.info("PDF export of a doc failed: %s", exc)
        raise HTTPException(status_code=exc.status, detail=_PDF_FAILURE_DETAIL[exc.status]) from exc
    except Exception as exc:
        logger.warning("doc export to %s failed", format, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not export the doc as {format.upper()}") from exc
    return Response(
        content=payload,
        media_type=MEDIA_TYPES[format],
        headers={"Content-Disposition": attachment_disposition(export_filename(doc, format)),
                 "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------- history
#
# The owner's alone, like writing: a version can hold what was later taken out.


@router.get("/{project_ref}/{artifact_id}/history")
async def doc_history(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: TenantScope = Depends(get_tenant_scope),
    limit: int = Query(default=LIST_LIMIT, ge=1, le=LIST_MAX),
):
    """The doc now and its kept versions, newest first: who made each, when, and what changed."""
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=True)
    try:
        return await run_efs(lambda: list_versions(folder, limit=limit), scope=scope)
    except DocError as err:
        raise _doc_error(err) from err


@router.get("/{project_ref}/{artifact_id}/history/{rev}")
async def doc_version(
    project_ref: str, artifact_id: ArtifactIdDep, rev: int, scope: TenantScope = Depends(get_tenant_scope)
):
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=True)
    try:
        return {"doc": await run_efs(partial(read_version, folder, rev), scope=scope)}
    except DocError as err:
        raise _doc_error(err) from err


class _RestoreBody(BaseModel):
    clientId: str | None = Field(default=None, max_length=100)


@router.post("/{project_ref}/{artifact_id}/history/{rev}/restore")
async def restore_version(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    rev: int,
    scope: TenantScope = Depends(get_tenant_scope),
    body: _RestoreBody | None = None,
):
    """Bring version ``rev`` back. ``replacedRev`` is the version restoring undoes this."""
    await require_product_permission(scope, "artifact.manage")
    _source, folder, _metadata, _is_own = await _resolve(scope, project_ref, artifact_id, write=True)
    try:
        doc, replaced = await run_efs(partial(restore, folder, rev), scope=scope)
    except DocError as err:
        raise _doc_error(err) from err
    after_write(folder, doc, None, by="user", scope=scope, client_id=body.clientId if body else None)
    return {"doc": doc, "replacedRev": replaced}


# ---------------------------------------------------------------- SSE


def _frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _mtime(folder: Path) -> int:
    try:
        return doc_file(folder).stat().st_mtime_ns
    except OSError:
        return 0


def _latest_write(doc: dict[str, Any]) -> tuple[dict[str, list[str]], str]:
    """For a write the hub never announced: the blocks it touched, and who wrote them."""
    touched: dict[str, list[str]] = {}
    authors: dict[str, int] = {}
    for tab in doc["tabs"]:
        ids = [b["id"] for b in tab["blocks"] if b.get("v") == doc["rev"]]
        if ids:
            touched[tab["id"]] = ids
        for b in tab["blocks"]:
            if b.get("v") == doc["rev"] and b.get("by"):
                authors[b["by"]] = authors.get(b["by"], 0) + 1
    by = max(authors, key=authors.get) if authors else "unknown"
    return touched, by


def _sse(
    request: Request,
    topic: str,
    *,
    initial: Path | None,
    scope: TenantScope | None = None,
    replay: bool = False,
) -> StreamingResponse:
    """Hub events for ``topic``; for a doc stream, also catch writes the hub never saw.

    The hub is this process's own. A write made by another process (the other
    replica serving the same shared tree, or a hand edit of the file) changes
    the file without an announcement here, so a doc stream also watches the
    file's mtime and sends the doc when it moved without an event for that rev.
    A doc stream (``initial``) needs ``scope``: the file reads run on the file
    store's threads.
    """

    async def poll_mtime(last: int) -> int:
        # A busy file store skips one poll rather than ending the stream.
        try:
            return await run_efs(partial(_mtime, initial), scope=scope)
        except FilesystemBusy:
            return last

    async def events():
        queue = hub.subscribe(topic, replay=replay)
        last_rev = -1
        last_mtime = 0
        try:
            if initial is not None:
                last_mtime = await poll_mtime(0)
                try:
                    doc = await run_efs(partial(load_doc, initial), scope=scope)
                    last_rev = doc["rev"]
                    yield _frame("doc.ready", {"rev": doc["rev"]})
                except (DocError, FilesystemBusy):
                    pass
            else:
                yield _frame("ready", {})
            idle = 0.0
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_POLL_S)
                except asyncio.TimeoutError:
                    idle += _POLL_S
                    if initial is not None:
                        mtime = await poll_mtime(last_mtime)
                        if mtime != last_mtime:
                            last_mtime = mtime
                            try:
                                doc = await run_efs(partial(load_doc, initial), scope=scope)
                            except (DocError, FilesystemBusy):
                                continue
                            if doc["rev"] != last_rev:
                                last_rev = doc["rev"]
                                touched, by = _latest_write(doc)
                                yield _frame("doc.updated", {"type": "doc.updated", "rev": doc["rev"],
                                                             "by": by, "doc": doc, "touched": touched, "removed": {}})
                                idle = 0.0
                                continue
                    if idle >= _KEEPALIVE_S:
                        idle = 0.0
                        yield ": keepalive\n\n"
                    continue
                idle = 0.0
                if initial is not None:
                    last_rev = max(last_rev, int(event.get("rev") or 0))
                    last_mtime = await poll_mtime(last_mtime)
                yield _frame(event.get("type", "message"), event)
        finally:
            hub.unsubscribe(topic, queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
