"""Routes the web app uses to add a member's custom MCP servers and switch their tools.

Org mode only (409 otherwise), for the signed-in member. Every route forwards
the member's own Authorization, to auth or, for a listing, to the inference
gateway in-cluster, and keeps nothing: auth runs the sign-in and holds the
tokens, and no route hands a token to this server. Every route but DELETE
answers 404, and ``/config`` ``available: false``, while ``custom_mcp_enabled``
is off; a member can always remove a stored server. A refusal from auth or the
gateway answers with its status and ``{"error": <code>, "detail": <message>}``,
auth's own shape, with the code unchanged. The desktop app's routes are
``custom_mcp.py`` and stay loopback only.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.common.settings.app_settings import OAuthSettings, TurnQueueSettings
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.schemas.custom_mcp import (
    CloudConnectRequest,
    CloudFinishRequest,
    CloudReconnectRequest,
    CloudSwitchesRequest,
)
from cowork.services.connectors.custom_mcp import is_connection_id
from cowork.services.connectors.oauth import auth_proxy

_log = logging.getLogger("cowork.connectors.custom_mcp_cloud")

ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]

#: Over the gateway's 30 second member-tools deadline, so its own answer arrives first.
_GATEWAY_TIMEOUT_SECONDS = 32.0
_MEMBER_TOOLS_PATH = "/v1/datasources/custom-mcp/member-tools"
_MEMBER_TOOLS_PROTOCOL_VERSION = 1
#: auth's own code for a body it could not read.
_INVALID_REQUEST = "invalid_request"


async def _require_org(scope: ScopeDep) -> None:
    """Refuse the whole surface outside org mode, before any body is read."""
    if not scope.org_mode:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="custom_mcp_unavailable")


def _require_enabled() -> None:
    """404 while the custom MCP setting is off."""
    if not TurnQueueSettings().custom_mcp_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")


def _require_name(name: str) -> str:
    """The path's connection name, 404 unless it is a custom MCP connection id; no other text reaches auth's path."""
    if not is_connection_id(name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    return name


router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode)), Depends(_require_org)])
NameDep = Annotated[str, Depends(_require_name)]
Enabled = Depends(_require_enabled)


class _Refused(Exception):
    """auth or the gateway refused, with its own code; or the gateway could not be reached, with ours."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message

    def response(self) -> JSONResponse:
        """The refusal as the web app reads it: the status and ``{"error", "detail"}``."""
        return JSONResponse({"error": self.code, "detail": self.message}, status_code=self.status_code)


def _answers_refusals(route: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap a route so a ``_Refused`` it raises becomes that refusal's answer."""

    @functools.wraps(route)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await route(*args, **kwargs)
        except _Refused as refused:
            return refused.response()

    return wrapper


async def _auth(method: str, path: str, request: Request, *, params: dict | None = None,
                json_body: dict | None = None) -> dict[str, Any]:
    """Relay one call to auth's custom MCP routes on the member's credential.

    Returns auth's answer. Raises ``_Refused`` when auth refused with a code,
    or with a 400 that carries none (``invalid_request``, a fixed message);
    auth's own ``HTTPException`` otherwise (unreachable, unauthenticated).
    """
    try:
        return await auth_proxy.proxy_custom_mcp(
            method, path, request, OAuthSettings(), params=params, json_body=json_body,
        )
    except HTTPException as exc:
        if isinstance(exc.detail, dict) and isinstance(exc.detail.get("code"), str):
            raise _Refused(exc.status_code, exc.detail["code"], str(exc.detail.get("message", ""))) from exc
        # A codeless 400 holds auth's raw body, field errors included; the web shows a fixed message instead.
        if exc.status_code == status.HTTP_400_BAD_REQUEST:
            raise _Refused(exc.status_code, _INVALID_REQUEST, "The request was refused.") from exc
        raise


async def _gateway_member_tools(name: str, request: Request) -> dict[str, Any]:
    """List the member's server ``name`` through the gateway, which stores the snapshot in auth.

    Sends the member's own Authorization and no service key. Returns the
    snapshot ``{tools_revision, tools}``. Raises ``_Refused`` for an
    unconfigured address, no answer, a refusal (with the gateway's code) or
    an answer that is not a snapshot.
    """
    base = TurnQueueSettings().datasource_gateway_base_url
    if not base:
        raise _Refused(503, "gateway_disabled", "The tool listing service is not configured.")
    authorization = request.headers.get("authorization")
    headers = {"authorization": authorization} if authorization else {}
    body = {"protocol_version": _MEMBER_TOOLS_PROTOCOL_VERSION, "name": name}
    try:
        async with httpx.AsyncClient(timeout=_GATEWAY_TIMEOUT_SECONDS, follow_redirects=False) as client:
            resp = await client.post(f"{base.rstrip('/')}{_MEMBER_TOOLS_PATH}", json=body, headers=headers)
    except httpx.HTTPError as exc:
        _log.warning("custom MCP listing got no answer from the gateway: %s", type(exc).__name__)
        raise _Refused(502, "gateway_unreachable", "The tool listing service did not answer.") from exc
    try:
        answer = resp.json()
    except (ValueError, RecursionError):
        answer = None
    if resp.status_code >= 400:
        code = answer.get("code") if isinstance(answer, dict) else None
        message = answer.get("detail") if isinstance(answer, dict) else None
        code = code if isinstance(code, str) and code else "gateway_error"
        _log.info("custom MCP listing refused by the gateway: status %s, %s", resp.status_code, code)
        raise _Refused(resp.status_code, code, message if isinstance(message, str) else "The listing failed.")
    if (
        not isinstance(answer, dict)
        or not _is_revision(answer.get("tools_revision"))
        or not isinstance(answer.get("tools"), list)
    ):
        raise _Refused(502, "gateway_unreachable", "The tool listing service answered something unreadable.")
    return answer


def _is_revision(value: object) -> bool:
    """Whether ``value`` is a snapshot revision auth could have written."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tools_on(snapshot: dict[str, Any]) -> int:
    """How many tools of a snapshot are switched on."""
    return sum(1 for tool in snapshot["tools"] if isinstance(tool, dict) and tool.get("allowed") is True)


async def _server_details(name: str, request: Request) -> dict[str, Any] | None:
    """A just signed-in server's details, after its first listing while it has no snapshot.

    Reads the member's servers from auth for the id, label, URL and
    ``tools_revision``, and lists through the gateway only at revision 0, so a
    second poll and a reconnect never list again. A failed listing, or a
    failed read of the member's servers, leaves the server connected with no
    tools and ``tools_unavailable`` true; the failed read answers only the
    name. Returns ``{name, id, label, server_url, tools_on, tools_unavailable}``,
    or None when the server is not in the member's list.
    """
    try:
        items = (await _auth("GET", "/connections", request)).get("items")
    except (_Refused, HTTPException) as exc:
        # auth already connected the server; a failure here must not read as a failed sign-in.
        _log.warning("a custom MCP sign-in finished but the member's servers could not be read: status %s",
                     exc.status_code)
        return {"name": name, "tools_on": 0, "tools_unavailable": True}
    row = next(
        (item for item in items if isinstance(item, dict) and item.get("name") == name), None,
    ) if isinstance(items, list) else None
    if row is None or not _is_revision(row.get("tools_revision")):
        _log.warning("a custom MCP sign-in finished but its server is not in the member's list")
        return None
    tools = {"tools_on": row.get("tools_on", 0), "tools_unavailable": False}
    if row["tools_revision"] == 0:
        try:
            tools = {"tools_on": _tools_on(await _gateway_member_tools(name, request)), "tools_unavailable": False}
        except _Refused:
            tools = {"tools_on": 0, "tools_unavailable": True}
    return {"name": name, "id": row.get("id"), "label": row.get("label"), "server_url": row.get("server_url"),
            **tools}


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Whether web custom MCP servers are available here, and the redirect URI a member's own client registers.

    Unavailable while this server's setting is off or auth's answers 404.
    """
    if not TurnQueueSettings().custom_mcp_enabled:
        return {"available": False, "redirect_uri": None}
    try:
        config = await auth_proxy.proxy_custom_mcp("GET", "/config", request, OAuthSettings())
    except HTTPException as exc:
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            return {"available": False, "redirect_uri": None}
        raise
    return {"available": True, "redirect_uri": config.get("redirect_uri")}


@router.post("/connect", dependencies=[Enabled])
@_answers_refusals
async def connect(body: CloudConnectRequest, request: Request) -> dict[str, Any]:
    """Start adding a server through auth; a server with no login connects at once and gets its first listing.

    Answers auth's start answer; a ``connected`` one also carries the
    server's details (``_server_details``).
    """
    answer = await _auth("POST", "/start", request, json_body=body.model_dump())
    name = answer.get("name")
    if answer.get("status") != "connected" or not is_connection_id(name):
        return answer
    details = await _server_details(name, request)
    return {**answer, **details} if details is not None else answer


@router.get("/status", dependencies=[Enabled])
@_answers_refusals
async def sign_in_status(request: Request, state: str = Query(min_length=1, max_length=512)) -> dict[str, Any]:
    """The outcome of a sign-in; on ``success`` the server's details, after its first listing when it has none.

    Any other outcome, ``refused`` included, passes through unchanged and lists nothing.
    """
    outcome = await auth_proxy.proxy_status(state, request, OAuthSettings())
    name = outcome.get("name")
    if outcome.get("status") != "success" or not is_connection_id(name):
        return outcome
    details = await _server_details(name, request)
    return {"status": "success", **details} if details is not None else outcome


@router.get("/connections/{name}/tools", dependencies=[Enabled])
@_answers_refusals
async def get_tools(name: NameDep, request: Request) -> dict[str, Any]:
    """List the server live and store the snapshot; when that fails, the stored snapshot marked stale.

    Answers ``{tools_revision, tools, stale}``, with the failure's ``code``
    when stale. With no stored snapshot the listing's refusal answers.
    """
    try:
        snapshot = await _gateway_member_tools(name, request)
    except _Refused as failure:
        try:
            stored = await _auth("GET", f"/connections/{name}/tools", request)
        except (_Refused, HTTPException):
            raise failure from None
        revision = stored.get("tools_revision")
        if not _is_revision(revision) or revision == 0:
            raise failure from None
        return {**stored, "stale": True, "code": failure.code}
    return {**snapshot, "stale": False}


@router.put("/connections/{name}/tools", dependencies=[Enabled])
@_answers_refusals
async def put_tools(name: NameDep, body: CloudSwitchesRequest, request: Request) -> dict[str, Any]:
    """Save which stored tools are switched on; talks to auth only, so it works when the server is down."""
    return await _auth("PUT", f"/connections/{name}/tools", request, json_body=body.model_dump())


@router.post("/connections/{name}/reconnect", dependencies=[Enabled])
@_answers_refusals
async def reconnect(name: NameDep, body: CloudReconnectRequest, request: Request) -> dict[str, Any]:
    """Sign in to the server again with its stored client; answers as auth's start does, with no listing."""
    return await _auth("POST", f"/connections/{name}/reconnect", request, json_body=body.model_dump())


@router.post("/finish", dependencies=[Enabled])
@_answers_refusals
async def finish(body: CloudFinishRequest, request: Request) -> dict[str, Any]:
    """Finish a sign-in in this member's session; auth's refusal code passes through unchanged."""
    return await _auth("POST", "/finish", request, json_body=body.model_dump())


@router.delete("/connections/{name}", status_code=status.HTTP_204_NO_CONTENT)
@_answers_refusals
async def delete_connection(name: NameDep, request: Request) -> Response:
    """Remove the member's server and its credentials; open while the setting is off."""
    await _auth("DELETE", f"/connections/{name}", request)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
