"""The user's MindsHub browser instance (ENG-3298).

MindsHub provisions a per-user browser (`br-<hash>`) through the same
`/instance` lambda as every other hosted agent. This module is the server's
half of it: provisioning and status as the caller, a viewer embed URL for the
renderer, and the block that tells anton where the browser is for one turn.

Every call is made AS the caller, with their MindsHub credential
(`hub_credential`). Nothing here holds a credential of its own, and anton
reaches the instance with its own turn credential, not anything from here.

That is the `mindshub` mode, the default. In `standalone` mode an admin points
cowork at a browser service they run themselves (the same Node service with a
static bearer token, `browser_service_url` + `browser_service_api_key`): there
is nothing to provision, every call carries that key instead of the caller's
MindsHub credential, and anton is handed the key in its turn block. The service
is shared by the org, so each user gets their own profile on it
(`standalone_profile`).
"""

from __future__ import annotations

import asyncio
import hashlib
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

MINDSHUB = "mindshub"
STANDALONE = "standalone"
# Plain http only for a service on this machine, for local testing; anything
# that crosses a network carries the key, so it must be https.
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_NOT_CONFIGURED = "Set the browser service URL and API key in Settings to use the browser."


#: Status for "MindsHub failed us". Not 502: the cluster's ingress swaps a 5xx
#: body for its own page, and the reason is what the user needs to see.
UPSTREAM_FAILED = 424


class BrowserServiceError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _KeyRejected(BrowserServiceError):
    """The standalone service refused its API key: a config error to show,
    unlike an unreachable service, which status reports as a state."""


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


def browser_mode(user_settings: Any) -> str:
    """`standalone` only when set so; anything else is the MindsHub default."""
    return STANDALONE if getattr(user_settings, "browser_mode", MINDSHUB) == STANDALONE else MINDSHUB


def normalize_service_url(url: str) -> str:
    """A standalone service origin without its trailing slash, or "" if it
    isn't one: https to any host, http only to loopback, and nothing past the
    origin (no path, query or fragment) so `/_embed` lands where we expect."""
    raw = (url or "").strip()
    try:
        parsed = httpx.URL(raw)
    except Exception:
        return ""
    if parsed.scheme == "https":
        ok_scheme = True
    else:
        ok_scheme = parsed.scheme == "http" and parsed.host in _LOOPBACK
    if not ok_scheme or not parsed.host or parsed.userinfo:
        return ""
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment or "?" in raw or "#" in raw:
        return ""
    return raw.rstrip("/")


def standalone_service(user_settings: Any) -> Optional[tuple[str, str]]:
    """(url, api key) of the standalone browser service, or None when the mode
    isn't standalone or the config is incomplete or invalid."""
    if browser_mode(user_settings) != STANDALONE:
        return None
    url = normalize_service_url(str(getattr(user_settings, "browser_service_url", "") or ""))
    secret = getattr(user_settings, "browser_service_api_key", None)
    key = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret or "")
    if not url or not key.strip():
        return None
    return url, key.strip()


def standalone_profile(user_id: Optional[str]) -> str:
    """This user's profile on a shared standalone service.

    Deterministic, so a login made in one turn is there in the next, and a
    hash, so the service never sees the user id itself. No user id (desktop,
    local scope) means one person, who gets `PROFILE` like on MindsHub.
    """
    if not user_id:
        return PROFILE
    return "u-" + hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:24]


async def _call(
    method: str, url: str, bearer: str, *,
    json: Optional[dict] = None, params: Optional[dict] = None, standalone: bool = False,
) -> Any:
    # Messages name what failed: MindsHub, or the admin's own service. Never
    # the bearer, which in standalone mode is the service's shared key.
    if not bearer:
        raise BrowserServiceError(401, _NOT_CONFIGURED if standalone else "Sign in to MindsHub to use the browser.")

    async def _send() -> httpx.Response:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S)) as client:
            return await client.request(
                method, url, json=json, params=params, headers={"Authorization": f"Bearer {bearer}"},
            )

    try:
        response = await asyncio.wait_for(_send(), _TIMEOUT_S)
    except Exception as exc:
        logger.warning("browser %s %s failed: %r", method, url, exc)
        what = (
            f"The browser service at {_host(url)} did not answer" if standalone
            else f"MindsHub did not answer at {_host(url)}"
        )
        raise BrowserServiceError(UPSTREAM_FAILED, f"{what} ({type(exc).__name__}). Try again in a moment.") from exc
    if response.status_code in (401, 403):
        if standalone:
            # Not passed through as a 401: the caller's own sign-in is fine,
            # and a renderer treating it as an expired session would be wrong.
            raise _KeyRejected(UPSTREAM_FAILED, f"The browser service at {_host(url)} rejected the API key.")
        raise BrowserServiceError(response.status_code, _message(response) or "MindsHub refused this request.")
    where = f"The browser service at {_host(url)}" if standalone else _host(url)
    if response.status_code >= 400:
        logger.warning("browser %s %s returned HTTP %s: %s", method, url, response.status_code, response.text[:300])
        reason = _message(response) or "no reason given"
        raise BrowserServiceError(UPSTREAM_FAILED, f"{where} returned HTTP {response.status_code}: {reason}")
    try:
        return response.json()
    except ValueError as exc:
        raise BrowserServiceError(UPSTREAM_FAILED, f"{where} returned an unreadable answer.") from exc


def _host(url: str) -> str:
    return httpx.URL(url).host


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


async def service_status(user_settings: Any) -> dict:
    """The standalone service's health, in the same shape as `fetch_status`.

    Always "provisioned": there is nothing to create. A service that doesn't
    answer is a state the renderer shows ("unreachable"), not an error; a
    rejected key is an error, since only fixing the config helps.
    """
    service = standalone_service(user_settings)
    if service is None:
        return {"provisioned": False, "status": "not_configured", "endpoint": ""}
    url, key = service
    try:
        raw = await _call("GET", f"{url}/healthz", key, standalone=True)
    except _KeyRejected:
        raise
    except BrowserServiceError:
        return {"provisioned": True, "status": "unreachable", "endpoint": url}
    ok = isinstance(raw, dict) and bool(raw.get("ok"))
    return {"provisioned": True, "status": "running" if ok else "unreachable", "endpoint": url}


async def embed(
    bearer: str, endpoint: str, session_id: str = PROFILE, *, standalone: bool = False,
) -> dict:
    """A fresh viewer URL for one browser session, to put in an iframe.

    The worker mints it (`POST /_embed`, mindshub_services ENG-3295): a token
    good for that session's viewer and event stream only, for an hour. A
    standalone service serves the same route itself, with its API key as the
    bearer; its endpoint was checked by `standalone_service`, not
    `valid_endpoint`, which stays MindsHub-only.
    """
    if standalone and not endpoint:
        raise BrowserServiceError(409, _NOT_CONFIGURED)
    if not standalone and not valid_endpoint(endpoint):
        raise BrowserServiceError(409, "The browser isn't set up yet.")
    if not _SESSION_ID.match(session_id or ""):
        raise BrowserServiceError(400, "session_id must be 1-64 letters, digits, '-' or '_'.")
    raw = await _call(
        "POST", f"{endpoint.rstrip('/')}/_embed", bearer, json={"session_id": session_id}, standalone=standalone,
    )
    if not isinstance(raw, dict) or not str(raw.get("view_url") or "").startswith(endpoint.rstrip("/") + "/"):
        raise BrowserServiceError(UPSTREAM_FAILED, "The browser returned an unexpected viewer URL.")
    return {"session_id": session_id, "view_url": raw["view_url"], "expires_at": int(raw.get("expires_at") or 0)}


async def embed_for(user_settings: Any, bearer: str, user_id: Optional[str], session_id: Optional[str] = None) -> dict:
    """`embed` for the configured mode: the caller's MindsHub instance with
    their credential, or the standalone service with its key. The session
    defaults to the user's profile, the one anton drives; on a standalone
    service it can't be anything else."""
    if browser_mode(user_settings) == STANDALONE:
        service = standalone_service(user_settings)
        if service is None:
            raise BrowserServiceError(409, _NOT_CONFIGURED)
        url, key = service
        profile = standalone_profile(user_id)
        # The service is shared and its key opens every session on it, so a
        # caller only ever gets a viewer for their own profile, not one named
        # in the request.
        if session_id and session_id != profile:
            raise BrowserServiceError(403, "That browser session belongs to someone else.")
        return await embed(key, url, profile, standalone=True)
    return await embed(bearer, str(getattr(user_settings, "browser_url", "") or ""), session_id or PROFILE)


def turn_block(user_settings: Any, user_id: Optional[str] = None) -> Optional[dict]:
    """Where the browser is, for one anton turn; None when it isn't on.

    The same shape anton's ``BrowserConfig.from_dict`` reads, both in-process
    (desktop) and in a cloud pod's ``TurnRequestV1.browser``. In standalone
    mode it carries the service's key as ``token`` (anton has no other way to
    authenticate there) and the user's own profile; an anton that predates
    ``token`` ignores it. Holds a secret then: never log the block.
    """
    if not getattr(user_settings, "browser_enabled", False):
        return None
    if browser_mode(user_settings) == STANDALONE:
        service = standalone_service(user_settings)
        if service is None:
            return None
        url, key = service
        return {"base_url": url, "profile": standalone_profile(user_id), "token": key}
    endpoint = str(getattr(user_settings, "browser_url", "") or "").rstrip("/")
    if not valid_endpoint(endpoint):
        return None
    return {"base_url": endpoint, "profile": PROFILE}


def anton_browser_config(user_settings: Any, user_id: Optional[str] = None):
    """`turn_block` as anton's BrowserConfig, or None when it's off or anton
    predates the browser tool (cowork-server and anton deploy independently)."""
    block = turn_block(user_settings, user_id)
    if block is None:
        return None
    try:
        from anton.core.browser.config import BrowserConfig
    except ImportError:
        return None
    return BrowserConfig.from_dict(block)
