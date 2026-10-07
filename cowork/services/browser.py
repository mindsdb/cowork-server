"""The user's MindsHub browser instance (ENG-3298).

MindsHub provisions a per-user browser (`br-<hash>`) through the same
`/instance` lambda as every other hosted agent. This module is the server's
half of it: provisioning and status as the caller, a viewer embed URL for the
renderer, and the block that tells anton where the browser is for one turn.

Every call is made AS the caller, with their MindsHub credential
(`hub_credential`). Nothing here holds a credential of its own, and anton
reaches the instance with its own turn credential, not anything from here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

AGENT = "browser"
#: One browser profile per user, shared across conversations so a login made
#: in one is there in the next. Mirrors anton's BrowserConfig default.
PROFILE = "main"
_TIMEOUT_S = 15.0
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# What a provisioned instance's endpoint must look like: the hosted agents'
# Cloudflare-fronted host, nothing else. Guards the URL anton is handed.
_ENDPOINT = re.compile(r"^https://br-[a-z0-9-]+\.(?:[a-z0-9-]+\.)*(?:4nton\.ai|mindshub\.ai)$")


class BrowserServiceError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def instance_api_base() -> str:
    """Where `/instance` is served for this deployment.

    The provisioning lambdas sit behind the API Gateway custom domain: the
    legacy `4nton.ai` in prod, and each non-prod MindsHub api host (dev,
    staging, a PR env's own `api-pr-…`) through the Cloudflare worker's SAM
    dispatch. Same split as the publish API (`publish_url_for_endpoint`),
    derived from the turn host so a PR env provisions in its own stack.
    """
    from cowork.common.settings.app_settings import default_turn_minds_api_host

    host = default_turn_minds_api_host().rstrip("/")
    return "https://4nton.ai" if host == "https://api.mindshub.ai" else host


def valid_endpoint(url: str) -> bool:
    return bool(url) and _ENDPOINT.match(url.rstrip("/")) is not None


async def _call(method: str, url: str, bearer: str, *, json: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    if not bearer:
        raise BrowserServiceError(401, "Sign in to MindsHub to use the browser.")

    async def _send() -> httpx.Response:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S)) as client:
            return await client.request(
                method, url, json=json, params=params, headers={"Authorization": f"Bearer {bearer}"},
            )

    try:
        response = await asyncio.wait_for(_send(), _TIMEOUT_S)
    except Exception as exc:
        logger.warning("browser %s %s failed: %s", method, url, type(exc).__name__)
        raise BrowserServiceError(502, "MindsHub did not answer. Try again in a moment.") from exc
    if response.status_code in (401, 403):
        raise BrowserServiceError(response.status_code, _message(response) or "MindsHub refused this request.")
    if response.status_code >= 400:
        logger.warning("browser %s %s returned HTTP %s", method, url, response.status_code)
        raise BrowserServiceError(502, _message(response) or f"MindsHub returned HTTP {response.status_code}.")
    try:
        return response.json()
    except ValueError as exc:
        raise BrowserServiceError(502, "MindsHub returned an unreadable answer.") from exc


def _message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body.get("detail") or "")
    return ""


def _view(raw: dict) -> dict:
    """The lambda's status/provision answer, narrowed to what the renderer needs."""
    endpoint = str(raw.get("endpoint") or "")
    return {
        "provisioned": bool(raw.get("provisioned", raw.get("status") not in (None, "none", "not_found"))),
        "status": str(raw.get("status") or "none"),
        "endpoint": endpoint if valid_endpoint(endpoint) else "",
    }


async def fetch_status(bearer: str) -> dict:
    raw = await _call("GET", f"{instance_api_base()}/instance", bearer, params={"agent": AGENT})
    return _view(raw if isinstance(raw, dict) else {})


async def provision(bearer: str) -> dict:
    """Create (or wake) the caller's browser instance. Safe to call again."""
    raw = await _call("POST", f"{instance_api_base()}/instance", bearer, json={"agent": AGENT})
    view = _view(raw if isinstance(raw, dict) else {})
    view["provisioned"] = True
    return view


async def embed(bearer: str, endpoint: str, session_id: str = PROFILE) -> dict:
    """A fresh viewer URL for one browser session, to put in an iframe.

    The worker mints it (`POST /_embed`, mindshub_services ENG-3295): a token
    good for that session's viewer and event stream only, for an hour.
    """
    if not valid_endpoint(endpoint):
        raise BrowserServiceError(409, "The browser isn't set up yet.")
    if not _SESSION_ID.match(session_id or ""):
        raise BrowserServiceError(400, "session_id must be 1-64 letters, digits, '-' or '_'.")
    raw = await _call("POST", f"{endpoint.rstrip('/')}/_embed", bearer, json={"session_id": session_id})
    if not isinstance(raw, dict) or not str(raw.get("view_url") or "").startswith(endpoint.rstrip("/") + "/"):
        raise BrowserServiceError(502, "The browser returned an unexpected viewer URL.")
    return {"session_id": session_id, "view_url": raw["view_url"], "expires_at": int(raw.get("expires_at") or 0)}


def turn_block(user_settings: Any) -> Optional[dict]:
    """Where the browser is, for one anton turn; None when it isn't on.

    The same shape anton's ``BrowserConfig.from_dict`` reads, both in-process
    (desktop) and in a cloud pod's ``TurnRequestV1.browser``.
    """
    if not getattr(user_settings, "browser_enabled", False):
        return None
    endpoint = str(getattr(user_settings, "browser_url", "") or "").rstrip("/")
    if not valid_endpoint(endpoint):
        return None
    return {"base_url": endpoint, "profile": PROFILE}


def anton_browser_config(user_settings: Any):
    """`turn_block` as anton's BrowserConfig, or None when it's off or anton
    predates the browser tool (cowork-server and anton deploy independently)."""
    block = turn_block(user_settings)
    if block is None:
        return None
    try:
        from anton.core.browser.config import BrowserConfig
    except ImportError:
        return None
    return BrowserConfig.from_dict(block)
