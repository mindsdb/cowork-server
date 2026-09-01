from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from cowork.api.v1.endpoints.guards import require_local, require_local_tenancy
from cowork.coding.connector_capabilities import (
    ConnectorCapability,
    ConnectorCapabilityIssueRequest,
)
from cowork.coding.contracts import (
    ApprovalRequest,
    BranchRequest,
    CommitRequest,
    DeliveryAutomationClaim,
    DeliveryAutomationClaimRequest,
    DeliveryAutomationPolicy,
    EventPage,
    RenameSessionRequest,
    SessionCreateRequest,
    SessionPage,
    SessionRecoverRequest,
    SessionUpdateRequest,
    TerminalCreateRequest,
    TerminalInputRequest,
    TerminalPage,
    TerminalRenameRequest,
    TerminalResizeRequest,
    TerminalShellInventory,
    TerminalStartRequest,
    TerminalStatus,
    TerminalTabPage,
    TerminalTabState,
    TurnRequest,
)
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_models import TaskResourceScope
from cowork.coding.delivery_automation import DeliveryAutomationService
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.engines.codex_config import LOCAL_PROXY_TOKEN
from cowork.coding.inference_proxy import (
    inference_body as _inference_body,  # noqa: F401 - retained for endpoint-test compatibility.
)
from cowork.coding.inference_proxy import (
    inference_headers as _inference_headers,  # noqa: F401 - retained for endpoint-test compatibility.
)
from cowork.coding.inference_proxy import (
    inference_url as _inference_url,  # noqa: F401 - retained for endpoint-test compatibility.
)
from cowork.coding.inference_proxy import (
    proxy_inference,
)
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    DraftPullRequestRequest,
    PlaybookConfigureRequest,
    PlaybookItemsRequest,
    ProjectCreateRequest,
    ProjectFolder,
    ProjectPage,
    ProjectActionRunRequest,
    ProjectActionRunResponse,
    ProjectActionPage,
    ReviewFileActionRequest,
    ProjectUpdateRequest,
    PublishRequest,
    PullRequestActionRequest,
    SourceActionRequest,
    SourceContextRequest,
    WorkItemSearchRequest,
)
from cowork.coding.redaction import redact_text
from cowork.coding.runtime_protocol import (
    ComputerUpdateRequest,
    RegistrationTokenResponse,
)
from cowork.coding.service import CodingService, get_coding_service
from cowork.coding.shells import shell_inventory
from cowork.coding.skill_models import (
    SkillLibraryDocument,
    SkillLibraryPage,
    SkillLibrarySource,
    SkillSourceAssignmentsRequest,
    SkillSourceCreateRequest,
    SkillSourceItemsRequest,
)
from cowork.coding.workspace import WorkspaceError
from cowork.coding.workspace_models import (
    WorkspaceEntryPage,
    WorkspaceFileContent,
    WorkspaceResourcePage,
    WorkspaceSearchPage,
)
from cowork.common.settings.user_settings import Provider, provider_api_key_str
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.services.settings import SettingService
from cowork.services.skills import CodeSkillService

router = APIRouter(dependencies=[Depends(require_local), Depends(require_local_tenancy)])
logger = logging.getLogger(__name__)


@router.get("/terminal-shells", response_model=TerminalShellInventory)
def terminal_shells():
    return shell_inventory()


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
    if isinstance(exc, (StateConflict, WorkspaceError, RuntimeError)):
        return HTTPException(status_code=409, detail=redact_text(str(exc))[:4_000])
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=redact_text(str(exc))[:4_000])
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


def _require_inference_client(request: Request) -> None:
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(credential, LOCAL_PROXY_TOKEN):
        raise HTTPException(status_code=401, detail="invalid inference client")


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


@router.get("/computers")
def coding_computers():
    return _service().control.list_computers()


@router.patch("/computers/{computer_id}")
def update_coding_computer(computer_id: str, body: ComputerUpdateRequest):
    return _call(_service().control.rename_computer, computer_id, body.name)


@router.delete("/computers/{computer_id}", status_code=204)
def revoke_coding_computer(computer_id: str):
    _call(_service().control.revoke_computer, computer_id)


@router.post("/runtime/registration-token", response_model=RegistrationTokenResponse)
def runtime_registration_token():
    return RegistrationTokenResponse(registration_token=_service().control.issue_registration_token())


@router.post("/runs/{run_id}/connector-capabilities", response_model=ConnectorCapability)
def issue_connector_capability(run_id: str, body: ConnectorCapabilityIssueRequest):
    grant, token = _call(
        _service().control.issue_connector_grant,
        run_id,
        body.provider,
        body.connection_name,
        list(body.actions),
        body.resource_constraints,
        timedelta(seconds=body.expires_in_seconds),
    )
    return ConnectorCapability(
        id=grant.id,
        provider=grant.provider,
        token=token,
        actions=grant.actions,
        resource_constraints=grant.resource_constraints,
        expires_at=grant.expires_at,
    )


@router.api_route("/inference/{path:path}", methods=["GET", "POST"])
async def inference_proxy(path: str, request: Request, session: SessionDep, scope: ScopeDep):
    _require_inference_client(request)
    credentials = _credentials(_settings(session, scope))
    return await proxy_inference(request, path, credentials)


@router.get("/workspace/inspect")
def inspect_workspace(path: str):
    return _call(_service().inspect_workspace, path)


@router.get("/skills/library", response_model=SkillLibraryPage)
def code_skill_library(
    scope: ScopeDep,
    project_id: str | None = Query(default=None, alias="projectId"),
):
    return _call(_service().skill_library.catalog, CodeSkillService(scope), project_id)


@router.get("/skills/library/content", response_model=SkillLibraryDocument)
def code_skill_document(
    scope: ScopeDep,
    item_id: str = Query(alias="itemId", min_length=1, max_length=2_000),
    path: str | None = Query(default=None, max_length=2_000),
):
    return _call(_service().skill_library.document, CodeSkillService(scope), item_id, path)


@router.post("/skills/sources", response_model=SkillLibrarySource)
def add_code_skill_source(body: SkillSourceCreateRequest):
    return _call(_service().skill_library.add, body.repository, body.branch, body.name)


@router.post("/skills/sources/{source_id}/refresh", response_model=SkillLibrarySource)
def refresh_code_skill_source(source_id: str):
    return _call(_service().skill_library.refresh, source_id)


@router.post("/skills/sources/{source_id}/apply", response_model=SkillLibrarySource)
def apply_code_skill_source(source_id: str):
    return _call(_service().skill_library.apply_update, source_id)


@router.delete("/skills/sources/{source_id}", status_code=204)
def remove_code_skill_source(source_id: str):
    _call(_service().skill_library.remove, source_id)


@router.put("/projects/{project_id}/skills/{source_id}", response_model=SkillLibraryPage)
def update_code_project_skill_source(
    project_id: str,
    source_id: str,
    body: SkillSourceItemsRequest,
):
    return _call(
        _service().skill_library.set_project_items,
        project_id,
        source_id,
        body.enabled_paths,
    )


@router.put("/skills/sources/{source_id}/projects", response_model=SkillLibraryPage)
def update_code_skill_source_projects(
    source_id: str,
    body: SkillSourceAssignmentsRequest,
):
    return _call(
        _service().skill_library.set_project_assignments,
        source_id,
        body.assignments,
    )


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


@router.post("/project-resources/inspect")
def inspect_local_project_resource(body: ProjectFolder):
    return _call(_service().projects.resolve_local_resource, body)


@router.get("/projects/{project_id}/resources")
def code_project_resources(project_id: str):
    project = _call(_service().projects.get, project_id)
    availability = _service().control.resource_availability(project)
    states = {item.resource_id: item for item in availability.items}
    return {
        "items": [
            {"resource": resource, "availability": states[resource.id]}
            for resource in project.resources
        ]
    }


@router.get("/projects/{project_id}/computers")
def eligible_code_project_computers(
    project_id: str,
    resource_ids: list[str] | None = Query(default=None, alias="resourceId"),
    engine_id: str | None = Query(default=None, alias="engineId"),
):
    project = _call(_service().projects.get, project_id)
    scope = TaskResourceScope(
        all_project_resources=resource_ids is None,
        resource_ids=resource_ids or [],
    )
    return {"items": _call(_service().control.eligible_computers, project, scope, engine_id)}


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


@router.post("/projects/{project_id}/work-items/search")
def search_code_project_work(project_id: str, body: WorkItemSearchRequest, integrations: IntegrationsDep):
    project = _call(_service().projects.get, project_id)
    return _call(integrations.search, project, body)


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
        code_skills=CodeSkillService(scope),
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


@router.get(
    "/sessions/{session_id}/workspace/resources",
    response_model=WorkspaceResourcePage,
)
def workspace_resources(session_id: str):
    return _call(_service().workspace_resources, session_id)


@router.get(
    "/sessions/{session_id}/workspace/entries",
    response_model=WorkspaceEntryPage,
)
def workspace_entries(
    session_id: str,
    resource_id: str = Query(alias="resourceId", min_length=1, max_length=128),
    path: str = Query(default="", max_length=32_768),
):
    return _call(_service().workspace_entries, session_id, resource_id, path)


@router.get(
    "/sessions/{session_id}/workspace/file",
    response_model=WorkspaceFileContent,
)
def workspace_file(
    session_id: str,
    resource_id: str = Query(alias="resourceId", min_length=1, max_length=128),
    path: str = Query(min_length=1, max_length=32_768),
    line_start: int | None = Query(default=None, alias="lineStart", ge=1),
    line_end: int | None = Query(default=None, alias="lineEnd", ge=1),
):
    return _call(
        _service().workspace_file,
        session_id,
        resource_id,
        path,
        line_start,
        line_end,
    )


@router.get(
    "/sessions/{session_id}/workspace/search",
    response_model=WorkspaceSearchPage,
)
def workspace_search(
    session_id: str,
    query: str = Query(min_length=1, max_length=512),
    resource_id: str | None = Query(default=None, alias="resourceId", max_length=128),
    limit: int = Query(default=60, ge=1, le=100),
):
    return _call(_service().workspace_search, session_id, query, resource_id, limit)


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


@router.put("/sessions/{session_id}/delivery-policy")
def update_delivery_policy(session_id: str, body: DeliveryAutomationPolicy):
    return _call(DeliveryAutomationService(_service().store).update_policy, session_id, body)


@router.post("/sessions/{session_id}/delivery-automation/claim", response_model=DeliveryAutomationClaim)
def claim_delivery_automation(session_id: str, body: DeliveryAutomationClaimRequest):
    return _call(DeliveryAutomationService(_service().store).claim_fix, session_id, body)


@router.post("/sessions/{session_id}/rename")
def rename_session(session_id: str, body: RenameSessionRequest):
    return _call(_service().rename_session, session_id, body.title)


@router.post("/sessions/{session_id}/archive")
def archive_session(session_id: str):
    return _call(_service().set_archived, session_id, True)


@router.post("/sessions/{session_id}/unarchive")
def unarchive_session(session_id: str):
    return _call(_service().set_archived, session_id, False)


@router.post("/sessions/{session_id}/pin")
def pin_session(session_id: str):
    return _call(_service().set_pinned, session_id, True)


@router.post("/sessions/{session_id}/unpin")
def unpin_session(session_id: str):
    return _call(_service().set_pinned, session_id, False)


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


@router.post("/sessions/{session_id}/recover")
def recover(session_id: str, body: SessionRecoverRequest):
    return _call(
        _service().recover,
        session_id,
        body.computer_id,
        allow_recreate=body.allow_recreate,
    )


@router.get("/sessions/{session_id}/recovery-options")
def recovery_options(session_id: str):
    return _call(_service().recovery_plan, session_id)


@router.get("/sessions/{session_id}/terminals", response_model=TerminalTabPage)
def terminals(session_id: str):
    return _call(_service().terminals, session_id)


@router.post("/sessions/{session_id}/project-actions/run", response_model=ProjectActionRunResponse)
def run_project_action(
    session_id: str,
    body: ProjectActionRunRequest,
    session: SessionDep,
    scope: ScopeDep,
):
    return _call(
        _service().run_project_action,
        session_id,
        body,
        _credentials(_settings(session, scope)),
    )


@router.get("/sessions/{session_id}/project-actions", response_model=ProjectActionPage)
def project_actions(session_id: str):
    return _call(_service().project_action_page, session_id)


@router.post("/sessions/{session_id}/terminals", response_model=TerminalTabState)
def create_terminal(session_id: str, body: TerminalCreateRequest):
    return _call(_service().create_terminal_tab, session_id, body.label)


@router.patch("/sessions/{session_id}/terminals/{terminal_id}", response_model=TerminalTabState)
def rename_terminal(session_id: str, terminal_id: str, body: TerminalRenameRequest):
    return _call(_service().rename_terminal_tab, session_id, terminal_id, body.label)


@router.delete("/sessions/{session_id}/terminals/{terminal_id}", status_code=204)
def delete_terminal(session_id: str, terminal_id: str):
    _call(_service().delete_terminal_tab, session_id, terminal_id)


@router.get("/sessions/{session_id}/terminals/{terminal_id}")
def terminal_tab(session_id: str, terminal_id: str, after: int = Query(default=0, ge=0)):
    return _call(_service().terminal_tab, session_id, terminal_id, after)


@router.post("/sessions/{session_id}/terminals/{terminal_id}/start")
def start_terminal_tab(
    session_id: str,
    terminal_id: str,
    body: TerminalStartRequest,
    session: SessionDep,
    scope: ScopeDep,
):
    return _call(
        _service().start_terminal_tab,
        session_id,
        terminal_id,
        _credentials(_settings(session, scope)),
        body.cols,
        body.rows,
        body.shell,
    )


@router.post("/sessions/{session_id}/terminals/{terminal_id}/input")
def terminal_tab_input(session_id: str, terminal_id: str, body: TerminalInputRequest):
    return _call(_service().write_terminal_tab, session_id, terminal_id, body.data_base64)


@router.post("/sessions/{session_id}/terminals/{terminal_id}/resize")
def terminal_tab_resize(session_id: str, terminal_id: str, body: TerminalResizeRequest):
    return _call(_service().resize_terminal_tab, session_id, terminal_id, body.cols, body.rows)


@router.post("/sessions/{session_id}/terminals/{terminal_id}/stop")
def terminal_tab_stop(session_id: str, terminal_id: str):
    return _call(_service().stop_terminal_tab, session_id, terminal_id)


@router.get("/sessions/{session_id}/terminals/{terminal_id}/stream")
async def stream_terminal_tab(
    request: Request,
    session_id: str,
    terminal_id: str,
    after: int = Query(default=0, ge=0),
):
    service = _service()
    _call(service.terminal_tab, session_id, terminal_id)
    return _terminal_stream_response(
        request,
        after,
        lambda cursor, timeout: service.wait_for_terminal_tab(
            session_id,
            terminal_id,
            cursor,
            timeout,
        ),
    )


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
        body.shell,
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
    return _terminal_stream_response(
        request,
        after,
        lambda cursor, timeout: service.wait_for_terminal(session_id, cursor, timeout),
    )


def _terminal_stream_response(
    request: Request,
    after: int,
    wait_for_page: Callable[[int, float], TerminalPage],
) -> StreamingResponse:

    async def generate():
        cursor = after
        yield "retry: 1000\n\n"
        while not await request.is_disconnected():
            page = await asyncio.to_thread(wait_for_page, cursor, 15.0)
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


@router.post("/sessions/{session_id}/review/file")
def review_file(session_id: str, body: ReviewFileActionRequest):
    return {
        "files": _call(
            _service().review_file_action,
            session_id,
            body.folder_id,
            body.path,
            body.action,
        )
    }


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
    return _call(_service().publish_task_update, session_id, body, integrations)


@router.post("/sessions/{session_id}/source-action")
def source_action(session_id: str, body: SourceActionRequest, integrations: IntegrationsDep):
    return _call(_service().complete_task_source, session_id, body, integrations)
