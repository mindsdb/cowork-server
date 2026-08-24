from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Callable
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from starlette.background import BackgroundTask

from cowork.api.v1.endpoints.guards import require_local, require_local_tenancy
from cowork.coding.contracts import (
    ApprovalRequest,
    BranchRequest,
    CommitRequest,
    EventPage,
    RenameSessionRequest,
    SessionCreateRequest,
    SessionPage,
    SessionUpdateRequest,
    TerminalInputRequest,
    TerminalResizeRequest,
    TerminalStartRequest,
    TerminalStatus,
    TurnRequest,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.codex_config import LOCAL_PROXY_TOKEN
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    PlaybookConfigureRequest,
    PlaybookItemsRequest,
    ProjectCreateRequest,
    ProjectPage,
    ProjectUpdateRequest,
    PublishRequest,
    PullRequestActionRequest,
    SourceContextRequest,
)
from cowork.coding.redaction import redact_text
from cowork.coding.service import CodingService, get_coding_service
from cowork.coding.workspace import WorkspaceError
from cowork.common.settings.user_settings import Provider, provider_api_key_str
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.services.settings import SettingService

router = APIRouter(dependencies=[Depends(require_local), Depends(require_local_tenancy)])
logger = logging.getLogger(__name__)
SessionDep = Annotated[Session, Depends(get_session)]
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]


def _service() -> CodingService:
    return get_coding_service()


def _settings(session: Session, scope: TenantScope):
    return SettingService(session, scope).load()


def _credentials(settings) -> EngineCredentials:
    return EngineCredentials(
        minds_url=settings.minds_url,
        minds_api_key=provider_api_key_str(settings, Provider.MINDS_CLOUD),
    )


def _integration_service(scope: ScopeDep):
    integrations = DeveloperIntegrationService(scope)
    try:
        yield integrations
    finally:
        integrations.close()


IntegrationsDep = Annotated[DeveloperIntegrationService, Depends(_integration_service)]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, (ValueError, WorkspaceError, RuntimeError)):
        return HTTPException(status_code=409, detail=redact_text(str(exc))[:4_000])
    return HTTPException(status_code=500, detail="Coding operation failed")


def _call[Result](operation: Callable[..., Result], *args, **kwargs) -> Result:
    """Translate the coding boundary's typed failures without endpoint boilerplate."""
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        if not isinstance(exc, (KeyError, ValueError, WorkspaceError, RuntimeError)):
            logger.exception(
                "Unexpected coding operation failure in %s",
                getattr(operation, "__qualname__", type(operation).__name__),
            )
        raise _http_error(exc) from exc


_INFERENCE_PATHS = {"models", "responses", "responses/compact"}


def _inference_url(minds_url: str, path: str, query: str = "") -> str:
    base = minds_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    url = f"{base}/{path}"
    return f"{url}?{query}" if query else url


def _inference_headers(request: Request, api_key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    for name in ("content-type",):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


def _require_inference_client(request: Request) -> None:
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(credential, LOCAL_PROXY_TOKEN):
        raise HTTPException(status_code=401, detail="invalid inference client")


def _inference_body(body: bytes) -> bytes:
    """Remove Codex-only transport metadata rejected by MindsHub Inference."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or "client_metadata" not in payload:
        return body
    payload.pop("client_metadata")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


@router.get("/engines")
def engines():
    return _service().capabilities()


@router.get("/models")
def models(session: SessionDep, scope: ScopeDep, engine_id: str = Query(default="codex", alias="engineId")):
    settings = _settings(session, scope)
    return {"items": _call(_service().discover_models, engine_id, _credentials(settings))}


@router.post("/runtime/prepare-shutdown")
def prepare_shutdown():
    """Persist resumable task state before the desktop stops the process tree."""
    return {"interrupted": _service().prepare_shutdown()}


@router.api_route("/inference/{path:path}", methods=["GET", "POST"])
async def inference_proxy(path: str, request: Request, session: SessionDep, scope: ScopeDep):
    if path not in _INFERENCE_PATHS:
        raise HTTPException(status_code=404, detail="inference route not found")
    _require_inference_client(request)

    credentials = _credentials(_settings(session, scope))
    if not credentials.minds_api_key:
        raise HTTPException(status_code=409, detail="MindsHub is not connected")

    body = _inference_body(await request.body())
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                _inference_url(credentials.minds_url, path, request.url.query),
                headers=_inference_headers(request, credentials.minds_api_key),
                content=body,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail="MindsHub inference is unavailable") from exc

    async def close_upstream() -> None:
        await upstream.aclose()
        await client.aclose()

    response_headers = {
        name: value
        for name in ("content-type", "retry-after", "x-mindshub-dropped-params", "x-request-id")
        if (value := upstream.headers.get(name))
    }
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )


@router.get("/workspace/inspect")
def inspect_workspace(path: str):
    return _service().inspect_workspace(path)


@router.get("/projects", response_model=ProjectPage)
def list_code_projects():
    return _service().projects.list()


@router.post("/projects")
def create_code_project(body: ProjectCreateRequest):
    return _call(_service().projects.create, body)


@router.get("/projects/{project_id}")
def get_code_project(project_id: str):
    return _call(_service().projects.get, project_id)


@router.patch("/projects/{project_id}")
def update_code_project(project_id: str, body: ProjectUpdateRequest):
    return _call(_service().projects.update, project_id, body)


@router.delete("/projects/{project_id}", status_code=204)
def delete_code_project(project_id: str):
    _call(_service().delete_project, project_id)


@router.get("/projects/{project_id}/folders")
def inspect_code_project_folders(project_id: str):
    return {"items": _call(_service().projects.inspect_folders, project_id)}


@router.post("/projects/{project_id}/playbook")
def configure_code_project_playbook(project_id: str, body: PlaybookConfigureRequest):
    return _call(_service().playbooks.configure, project_id, body.repository, body.branch)


@router.get("/projects/{project_id}/playbook")
def code_project_playbook(project_id: str):
    return _call(_service().playbooks.status, project_id)


@router.delete("/projects/{project_id}/playbook", status_code=204)
def remove_code_project_playbook(project_id: str):
    _call(_service().playbooks.remove, project_id)


@router.post("/projects/{project_id}/playbook/refresh")
def refresh_code_project_playbook(project_id: str):
    return _call(_service().playbooks.refresh, project_id)


@router.post("/projects/{project_id}/playbook/apply")
def apply_code_project_playbook(project_id: str):
    return _call(_service().playbooks.apply_update, project_id)


@router.post("/projects/{project_id}/playbook/items")
def update_code_project_playbook_items(project_id: str, body: PlaybookItemsRequest):
    return _call(_service().playbooks.set_enabled, project_id, body.enabled_paths)


@router.get("/projects/{project_id}/integrations")
def code_project_integrations(project_id: str, integrations: IntegrationsDep):
    project = _call(_service().projects.get, project_id)
    return {"items": _call(integrations.statuses, project)}


@router.post("/projects/{project_id}/source-context")
def read_code_project_source(project_id: str, body: SourceContextRequest, integrations: IntegrationsDep):
    project = _call(_service().projects.get, project_id)
    return _call(integrations.read, project, body)


@router.get("/sessions", response_model=SessionPage)
def list_sessions(include_archived: bool = Query(default=False, alias="includeArchived")):
    return _service().list_sessions(include_archived)


@router.post("/sessions")
def create_session(body: SessionCreateRequest, session: SessionDep, scope: ScopeDep):
    settings = _settings(session, scope)
    return _call(
        _service().create_session,
        body,
        _credentials(settings),
        default_engine=settings.coding_agent_engine,
        default_model=settings.coding_agent_model,
    )


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    return _call(_service().get_session, session_id)


@router.get("/sessions/{session_id}/workspace/files")
def workspace_files(
    session_id: str,
    query: str = Query(default="", max_length=512),
    limit: int = Query(default=40, ge=1, le=100),
):
    return {"items": _call(_service().workspace_files, session_id, query, limit)}


@router.get("/sessions/{session_id}/extensions")
def extensions(session_id: str, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().extension_inventory,
        session_id,
        _credentials(_settings(session, scope)),
    )


@router.get("/sessions/{session_id}/platform")
def platform_status(session_id: str, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().platform_status,
        session_id,
        _credentials(_settings(session, scope)),
    )


@router.post("/sessions/{session_id}/windows-sandbox/setup")
def setup_windows_sandbox(session_id: str, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().setup_windows_sandbox,
        session_id,
        _credentials(_settings(session, scope)),
    )


@router.patch("/sessions/{session_id}")
def update_session(session_id: str, body: SessionUpdateRequest):
    return _call(_service().update_session_config, session_id, body)


@router.post("/sessions/{session_id}/rename")
def rename_session(session_id: str, body: RenameSessionRequest):
    return _call(_service().rename_session, session_id, body.title)


@router.post("/sessions/{session_id}/archive")
def archive_session(session_id: str):
    return _call(_service().set_archived, session_id, True)


@router.post("/sessions/{session_id}/unarchive")
def unarchive_session(session_id: str):
    return _call(_service().set_archived, session_id, False)


@router.post("/sessions/{session_id}/fork")
def fork_session(session_id: str, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().fork_session,
        session_id,
        _credentials(_settings(session, scope)),
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    _call(_service().delete_session, session_id)


@router.get("/sessions/{session_id}/events", response_model=EventPage)
def events(session_id: str, after: int = Query(default=0, ge=0)):
    return _call(_service().events, session_id, after)


@router.get("/sessions/{session_id}/stream")
async def stream_events(request: Request, session_id: str, after: int = Query(default=0, ge=0)):
    service = _service()
    _call(service.get_session, session_id)

    async def generate():
        cursor = after
        yield "retry: 1000\n\n"
        while not await request.is_disconnected():
            page = await asyncio.to_thread(service.wait_for_events, session_id, cursor, 15.0)
            if not page.items:
                yield ": keep-alive\n\n"
                continue
            for event in page.items:
                cursor = event.seq
                payload = event.model_dump(mode="json")
                yield f"id: {event.seq}\nevent: coding-event\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/sessions/{session_id}/turns")
def submit_turn(session_id: str, body: TurnRequest, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().submit_turn,
        session_id,
        body.prompt,
        _credentials(_settings(session, scope)),
        body.attachments,
    )


@router.post("/sessions/{session_id}/steer")
def steer(session_id: str, body: TurnRequest):
    return _call(_service().steer, session_id, body.prompt, body.attachments)


@router.post("/sessions/{session_id}/queue")
def queue_turn(session_id: str, body: TurnRequest):
    return _call(_service().queue_turn, session_id, body.prompt, body.attachments)


@router.delete("/sessions/{session_id}/queue/{instruction_id}")
def remove_queued_turn(session_id: str, instruction_id: str):
    return _call(_service().remove_queued_turn, session_id, instruction_id)


@router.post("/sessions/{session_id}/queue/{instruction_id}/steer")
def steer_queued_turn(session_id: str, instruction_id: str):
    return _call(_service().steer_queued_turn, session_id, instruction_id)


@router.post("/sessions/{session_id}/queue/run")
def run_next_queued(session_id: str, session: SessionDep, scope: ScopeDep):
    return _call(
        _service().run_next_queued,
        session_id,
        _credentials(_settings(session, scope)),
    )


@router.post("/sessions/{session_id}/cancel")
def cancel(session_id: str):
    return _call(_service().cancel, session_id)


@router.get("/sessions/{session_id}/terminal")
def terminal(session_id: str, after: int = Query(default=0, ge=0)):
    return _call(_service().terminal, session_id, after)


@router.post("/sessions/{session_id}/terminal/start")
def start_terminal(
    session_id: str,
    body: TerminalStartRequest,
    session: SessionDep,
    scope: ScopeDep,
):
    return _call(
        _service().start_terminal,
        session_id,
        _credentials(_settings(session, scope)),
        body.cols,
        body.rows,
    )


@router.post("/sessions/{session_id}/terminal/input")
def terminal_input(session_id: str, body: TerminalInputRequest):
    return _call(_service().write_terminal, session_id, body.data_base64)


@router.post("/sessions/{session_id}/terminal/resize")
def terminal_resize(session_id: str, body: TerminalResizeRequest):
    return _call(_service().resize_terminal, session_id, body.cols, body.rows)


@router.post("/sessions/{session_id}/terminal/stop")
def terminal_stop(session_id: str):
    return _call(_service().stop_terminal, session_id)


@router.get("/sessions/{session_id}/terminal/stream")
async def stream_terminal(request: Request, session_id: str, after: int = Query(default=0, ge=0)):
    service = _service()
    _call(service.get_session, session_id)

    async def generate():
        cursor = after
        yield "retry: 1000\n\n"
        while not await request.is_disconnected():
            page = await asyncio.to_thread(service.wait_for_terminal, session_id, cursor, 15.0)
            for chunk in page.items:
                cursor = chunk.seq
                payload = chunk.model_dump(mode="json")
                yield f"id: {chunk.seq}\nevent: terminal-output\ndata: {json.dumps(payload)}\n\n"
            if page.status != TerminalStatus.running:
                payload = page.model_dump(mode="json", exclude={"items"})
                # Keep terminal-state frames compatible with the TerminalPage
                # contract without replaying the buffered output a second time.
                payload["items"] = []
                yield f"event: terminal-state\ndata: {json.dumps(payload)}\n\n"
                return
            if not page.items:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/approvals/{approval_id}")
def approval(session_id: str, approval_id: str, body: ApprovalRequest):
    return _call(_service().resolve_approval, session_id, approval_id, body.decision)


@router.get("/sessions/{session_id}/git")
def git_state(session_id: str):
    return _call(_service().git_state, session_id)


@router.get("/sessions/{session_id}/git/all")
def git_states(session_id: str):
    return {"items": _call(_service().git_states, session_id)}


@router.get("/sessions/{session_id}/diff")
def diff(session_id: str):
    return {"files": _call(_service().diff, session_id)}


@router.post("/sessions/{session_id}/branch")
def create_branch(session_id: str, body: BranchRequest):
    return _call(_service().create_branch, session_id, body.name)


@router.post("/sessions/{session_id}/commit")
def commit(session_id: str, body: CommitRequest):
    return _call(_service().commit, session_id, body.message)


@router.post("/sessions/{session_id}/apply")
def apply_to_source(session_id: str):
    return _call(_service().apply_to_source, session_id)


@router.post("/sessions/{session_id}/validate")
def validate_project(session_id: str):
    return {"items": _call(_service().validate_project, session_id)}


@router.get("/sessions/{session_id}/delivery")
def delivery_plan(session_id: str, integrations: IntegrationsDep):
    return _call(_service().delivery_plan, session_id, integrations)


@router.post("/sessions/{session_id}/draft-pull-requests")
def create_draft_pull_requests(session_id: str, body: DraftPullRequestRequest, integrations: IntegrationsDep):
    return {
        "items": _call(
            _service().create_draft_pull_requests,
            session_id,
            body,
            integrations,
        )
    }


@router.post("/sessions/{session_id}/pull-request-action")
def pull_request_action(session_id: str, body: PullRequestActionRequest, integrations: IntegrationsDep):
    return _call(_service().pull_request_action, session_id, body, integrations)


@router.post("/sessions/{session_id}/publish")
def publish_task_update(session_id: str, body: PublishRequest, integrations: IntegrationsDep):
    coding = _service()
    task = _call(coding.get_session, session_id)
    if not task.project_id:
        raise HTTPException(status_code=409, detail="This task is not linked to a Code Project")
    project = _call(coding.projects.get, task.project_id)
    delivery = _call(integrations.publish, project, body)
    coding.store.update_session(
        task.id,
        lambda current: current.deliveries.append(delivery),
    )
    return delivery
