from __future__ import annotations

import asyncio
import hmac
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session

from cowork.api.v1.endpoints.guards import require_local_tenancy
from cowork.coding.connector_capabilities import ConnectorInvocationRequest
from cowork.coding.control_models import RUNTIME_PROTOCOL_VERSION, RuntimeEvent
from cowork.coding.control_service import RuntimeAuthenticationError, StaleRuntimeEvent
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.inference_proxy import proxy_inference
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import SourceContextRequest, WorkItemSearchRequest
from cowork.coding.runtime_protocol import (
    ComputerHeartbeatRequest,
    ComputerRegistrationRequest,
    ComputerRegistrationResponse,
    RuntimeCommandAckRequest,
    RuntimeCommandPage,
    RuntimeFenceRequest,
    RuntimeLease,
    RuntimeLeaseRequest,
)
from cowork.coding.service import get_coding_service
from cowork.common.settings.user_settings import Provider, provider_api_key_str
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session as get_db_session
from cowork.services.settings import SettingService

# Remote computers are supported by the desktop control plane. Hosted/org
# activation requires the tenant-bound service resolver and SQL store; until
# that boundary is wired, fail closed rather than sharing desktop-global state.
router = APIRouter(dependencies=[Depends(require_local_tenancy)])


def _control():
    return get_coding_service().control


def _runtime_token(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Runtime authentication required")
    return token


def _authenticate(request: Request, computer_id: str):
    try:
        return _control().authenticate_runtime(computer_id, _runtime_token(request))
    except RuntimeAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _require_protocol(version: str) -> None:
    if not hmac.compare_digest(version, RUNTIME_PROTOCOL_VERSION):
        raise HTTPException(
            status_code=426,
            detail=f"Runtime protocol {RUNTIME_PROTOCOL_VERSION} is required",
        )


@router.post("/register", response_model=ComputerRegistrationResponse)
def register_runtime(body: ComputerRegistrationRequest):
    _require_protocol(body.protocol_version)
    try:
        computer, runtime_token = _control().register_runtime(
            body.registration_token,
            body.name,
            body.capabilities,
        )
    except RuntimeAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=426, detail=str(exc)) from exc
    return ComputerRegistrationResponse(computer=computer, runtime_token=runtime_token)


@router.post("/computers/{computer_id}/heartbeat")
def heartbeat_runtime(
    computer_id: str,
    body: ComputerHeartbeatRequest,
    request: Request,
):
    _require_protocol(body.protocol_version)
    _authenticate(request, computer_id)
    return _control().heartbeat(computer_id, body.active_run_count)


@router.post("/computers/{computer_id}/lease", response_model=RuntimeLease | None)
async def acquire_runtime_lease(
    computer_id: str,
    body: RuntimeLeaseRequest,
    request: Request,
):
    _require_protocol(body.protocol_version)
    _authenticate(request, computer_id)
    deadline = time.monotonic() + body.wait_seconds
    while True:
        lease = await asyncio.to_thread(get_coding_service().remote.acquire_lease, computer_id)
        if lease is not None:
            return lease
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.25)


@router.post("/runs/{run_id}/events")
def accept_runtime_event(
    run_id: str,
    event: RuntimeEvent,
    request: Request,
):
    if event.run_id != run_id:
        raise HTTPException(status_code=409, detail="Runtime event targets another Task Run")
    _authenticate(request, event.computer_id)
    try:
        return get_coding_service().accept_runtime_event(event)
    except StaleRuntimeEvent as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/commands/claim", response_model=RuntimeCommandPage)
def claim_runtime_commands(
    run_id: str,
    body: RuntimeFenceRequest,
    computer_id: str,
    request: Request,
):
    _require_protocol(body.protocol_version)
    _authenticate(request, computer_id)
    try:
        return RuntimeCommandPage(items=_control().claim_commands(
            run_id,
            computer_id,
            body.lease_id,
            body.epoch,
        ))
    except StaleRuntimeEvent as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/commands/ack")
def acknowledge_runtime_command(
    run_id: str,
    body: RuntimeCommandAckRequest,
    computer_id: str,
    request: Request,
):
    _require_protocol(body.protocol_version)
    _authenticate(request, computer_id)
    try:
        return _control().acknowledge_command(
            run_id,
            body.command_id,
            computer_id,
            body.lease_id,
            body.epoch,
            body.result,
            body.error,
        )
    except StaleRuntimeEvent as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc


@router.post("/computers/{computer_id}/connector-capabilities/{grant_id}/invoke")
def invoke_connector_capability(
    computer_id: str,
    grant_id: str,
    body: ConnectorInvocationRequest,
    request: Request,
    scope: TenantScope = Depends(get_tenant_scope),
):
    _require_protocol(body.protocol_version)
    service = get_coding_service()
    try:
        try:
            grant_record = service.control.store.get_grant(grant_id)
        except KeyError as exc:
            raise RuntimeAuthenticationError("Connector capability authentication failed") from exc
        service.control.authenticate_run_token(
            grant_record.run_id,
            computer_id,
            _runtime_token(request),
        )
        grant = service.control.authorize_connector(
            grant_id,
            body.grant_token,
            body.action,
            {
                name: str(body.payload[name])
                for name in ("repository", "target_url", "url")
                if name in body.payload
            },
            computer_id,
        )
        run = service.control.store.get_run(grant.run_id)
        if run.computer_id != computer_id:
            raise RuntimeAuthenticationError("Connector capability belongs to another computer")
        task = service.control.store.get_task(run.task_id)
        if not task.project_id or task.execution_project is None:
            raise ValueError("Connector capabilities require a Code Project")
        if not any(
            connection.provider == grant.provider and connection.name == grant.connection_name
            for connection in task.execution_project.connections
        ):
            raise RuntimeAuthenticationError("Connector capability is outside the task execution snapshot")
        project = service.projects.get(task.project_id)
        integrations = DeveloperIntegrationService(scope)
        try:
            if body.action == "read_source":
                payload = SourceContextRequest.model_validate({
                    **body.payload,
                    "provider": grant.provider,
                    "connection_name": grant.connection_name,
                })
                return integrations.read(project, payload)
            if body.action == "search_work":
                payload = WorkItemSearchRequest.model_validate({
                    **body.payload,
                    "provider": grant.provider,
                    "connection_name": grant.connection_name,
                })
                return integrations.search(project, payload)
            if grant.provider != "github":
                raise ValueError("Pull request status requires GitHub")
            target_url = str(body.payload.get("target_url") or "")
            if not target_url:
                raise ValueError("A pull request URL is required")
            return integrations.pull_request_status(project, target_url, grant.connection_name)
        finally:
            integrations.close()
    except RuntimeAuthenticationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc).strip("'")) from exc


@router.api_route("/computers/{computer_id}/runs/{run_id}/inference/{path:path}", methods=["GET", "POST"])
async def runtime_inference_proxy(
    computer_id: str,
    run_id: str,
    path: str,
    request: Request,
    scope: TenantScope = Depends(get_tenant_scope),
    session: Session = Depends(get_db_session),
):
    try:
        _control().authenticate_run_token(run_id, computer_id, _runtime_token(request))
    except RuntimeAuthenticationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    settings = SettingService(session, scope).load()
    credentials = EngineCredentials(
        minds_url=settings.minds_url,
        minds_api_key=provider_api_key_str(settings, Provider.MINDS_CLOUD),
    )
    return await proxy_inference(request, path, credentials)
