"""The agent's browser: the browser router on the web, and the desktop app's
own browser on the desktop.

Two hosts serve anton's browser tool the same API, and this module is the
server's half of both. Which one a deployment has follows its tenancy: org
tenancy is the web app, anything else the desktop app (``on_web``).

On the web the host is the browser router (``COWORK_BROWSER_ROUTER_URL``,
``router_url``), served on the cowork host under ``/browser``. It starts one
browser per person and routes each caller to their own by the credential they
present, so nothing here holds a credential of its own:

- Settings and the chat call it as the caller, with their MindsHub credential
  (``mindshub_credential``): the browser's state (``router_status``), a warm-up
  that carries the person's time zone and language (``warm``,
  ``RouterPrefs``), a viewer link for one session (``embed``), the person's
  approval of a step the browser held (``approve``), their Skip of one
  (``decline``) and signing out of every
  site (``clear_logins``). A viewer link opens once; the app mints a new one
  whenever it shows the viewer, and nothing here stores one.
- A web turn gets a router block with no credential (``turn_block``), which
  anton reaches with the turn's own key. Before the producer queues the turn,
  it scopes that key with the router (``scope_turn``): to the task's session
  and the project's saved tabs, which also starts the person's browser, or to
  no session at all for a turn with no browser, so no code holding the key
  can scope it later. The router takes a scope only beside ``scope_key``,
  which only this server and the router hold. A key the router hasn't scoped
  reaches nothing.
- Whether the person may use the browser at all is their MindsHub entitlement
  (``browser_entitlement``), which the router enforces on every call and
  ``/browse/status`` reports, an organization admin's switch included.

Sites see the browser's traffic come from MindsDB's own addresses.

On the desktop the browser runs inside the Cowork app. The app registers its
loopback address and per-launch key with this server
(``PUT /runtime-credential/browser``, held in memory by
``cowork.common.settings.runtime_credential``), and an in-process turn's anton
is handed a config for it (``desktop_browser_config``). Nothing here calls it,
and the router setting is ignored there. A desktop signed in to MindsHub asks
MindsHub only whether its organization turned the browser off
(``desktop_organization_off``); one with the person's own model key asks
nothing.

Each task drives a session of its own (``task_session``), and a project's saved
tabs live under a key of their own (``project_key``). Both are hashes, so
neither host sees a person, conversation or project id.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx
from urllib.parse import urlsplit
from anton.core.browser.config import BrowserConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.runtime_credential import get_browser_host, get_minds_credential
from cowork.services.browser_contract import (
    NOT_ENTITLED,
    ORGANIZATION_OFF,
    SCOPE_KEY_HEADER,
    SESSION_ID,
    BrowserDisabledReason,
    BrowserEntitlement,
)

if TYPE_CHECKING:
    from cowork.common.settings.user_settings import UserSettings

logger = logging.getLogger(__name__)

#: The longest any call here waits on the router. A slower answer, the
#: router's own browser_starting at its 105 s deadline included, reaches the
#: app as 424 browser_unreachable.
_TIMEOUT_S = 15.0

#: Status for "the browser router failed us". Not 502: Cloudflare, in front of
#: the cowork host, swaps a 502 or 504 body for its own page, and the reason is
#: what the person needs to see.
UPSTREAM_FAILED = 424

#: What a warm-up sends for a time zone and a language: an IANA zone and a
#: BCP 47 tag. Anything else is left out rather than refused, since it only
#: shapes how the browser presents.
_TIMEZONE = re.compile(r"[A-Za-z_+\-/0-9]{1,64}")
_LOCALE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,3}")

_NO_BROWSER = "The browser isn't available here."


# ── Errors ──────────────────────────────────────────────────────────────────


class BrowserErrorDetail(BaseModel):
    """The ``detail`` of a ``/browse`` error: a code the app can branch on, and
    a sentence it can show. A refusal from the router keeps the router's code
    (``not_entitled``, ``daily_limit``)."""

    code: str
    message: str


class BrowserServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.detail = BrowserErrorDetail(code=code, message=message)


class _RouterRefusal(BaseModel):
    """The router's error body."""

    model_config = ConfigDict(extra="ignore")

    error: str = ""
    message: str = ""


# ── What the routes and the router exchange ─────────────────────────────────

#: A browser's state. ``none``, ``starting`` and ``running`` are the router's
#: answer for the caller's own browser; ``unavailable`` means this deployment
#: offers no browser to the caller (no router on the web, or no browser the
#: desktop app registered), ``not_entitled`` that the caller's account may
#: not use it, and ``organization_off`` that an admin of the caller's
#: organization turned it off.
BrowseState = Literal["none", "starting", "running", "unavailable", "not_entitled", "organization_off"]


class BrowseStatus(BaseModel):
    """``GET /browse/status``. ``enabled`` is the person's own switch.
    ``available`` is whether this deployment offers the browser to them,
    whatever the switch says, so a person who turned it off can turn it back
    on."""

    enabled: bool
    available: bool
    status: BrowseState


class RouterStatus(BaseModel):
    """The router's ``_status`` and ``_warm`` answer."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["none", "starting", "running"]


class RouterPrefs(BaseModel):
    """The person's browser preferences, as the router's ``_warm`` takes them:
    the page clock and language follow the person's own. A key a warm-up
    leaves out keeps what the router holds."""

    model_config = ConfigDict(extra="ignore")

    timezone: str | None = None
    locale: str | None = None


class BrowserEmbed(BaseModel):
    """A viewer link for one session, as the router mints it for the caller.
    The link opens the viewer once, by ``ticket_expires_at``; the viewer it
    opens closes at ``lease_expires_at``, and the app mints a new link to go
    on. Both are Unix seconds."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    view_url: str
    ticket_expires_at: int
    lease_expires_at: int


class ScopeRequest(BaseModel):
    """The router's ``_scope`` body: the one session and project a turn's key
    may reach, or ``session_id`` null for a turn with no browser, which binds
    the key to nothing. ``project`` is left out for a turn with no project."""

    session_id: str | None = Field(pattern=SESSION_ID.pattern)
    project: str | None = Field(default=None, pattern=SESSION_ID.pattern)

    def body(self) -> dict:
        """The JSON the router takes: ``session_id`` always, null included."""
        return self.model_dump(exclude={"project"} if self.project is None else set())


class ApproveRequest(BaseModel):
    """``POST /browse/approve``: the person approves one step the browser held
    in one of their sessions."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=SESSION_ID.pattern)
    pending_id: str = Field(pattern=SESSION_ID.pattern)


class BrowserApproval(BaseModel):
    """The router's answer to an approval: the held step may now run once,
    until ``expires_at`` (Unix seconds)."""

    model_config = ConfigDict(extra="ignore")

    pending_id: str
    approved: Literal[True]
    expires_at: int


class DeclineRequest(ApproveRequest):
    """``POST /browse/decline``: the person skips one step the browser held
    in one of their sessions."""


class BrowserDecline(BaseModel):
    """The router's answer to a decline: the held step, and any approval
    recorded for it, are gone. Also the answer for an id it no longer holds."""

    model_config = ConfigDict(extra="ignore")

    declined: Literal[True]


class LoginsCleared(BaseModel):
    """The router's answer to signing out of every site."""

    model_config = ConfigDict(extra="ignore")

    cleared: Literal[True]


# ── Where the browser is ───────────────────────────────────────────────────


def on_web() -> bool:
    """Org tenancy: the web app, whose browser is the router. Everywhere else
    the browser is the desktop app's own."""
    return get_app_settings().tenancy_mode == "org"


def router_url() -> str:
    """The browser router's base URL, or "" when this deployment has none.
    Only on the web: the desktop app has its own browser. Validated and
    stripped of its trailing slash by the app setting."""
    return get_app_settings().browser_router_url if on_web() else ""


def _require_router() -> str:
    router = router_url()
    if not router:
        raise BrowserServiceError(409, "unavailable", _NO_BROWSER)
    return router


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_session(user_id: str | None, conversation_id: str) -> str:
    """The browser session for one task: tabs of its own in the person's
    browser, so two tasks never share tabs. Hashed, so the host never sees the
    ids; the person is in it so the name means nothing outside their browser."""
    return "t-" + _digest(f"{user_id or ''}:{conversation_id}")[:24]


def project_key(user_id: str | None, project_id: Any) -> str:
    """The opaque key a project's saved tabs live under, or "" without a project."""
    if project_id is None or str(project_id) == "":
        return ""
    return "p-" + _digest(f"{user_id or ''}:{project_id}")[:24]


# ── Calls to the router ────────────────────────────────────────────────────


async def _call(
    method: str, url: str, bearer: str, *, json: dict | None = None, headers: dict[str, str] | None = None,
) -> Any:
    """One router call as ``bearer``, with any extra ``headers``. Never logs or
    echoes the bearer or a header's value.

    A 401, 403 or 429 passes on with the router's code and sentence: they are
    its answer about the caller (a stale sign-in, ``not_entitled``,
    ``daily_limit``). Any other failure is UPSTREAM_FAILED with the router's
    code, since Cloudflare would swap a 502 or 504 body for its own page.
    """
    if not bearer:
        raise BrowserServiceError(401, "unauthorized", "Sign in to MindsHub to use the browser.")

    async def _send() -> httpx.Response:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S)) as client:
            return await client.request(
                method, url, json=json, headers={**(headers or {}), "Authorization": f"Bearer {bearer}"},
            )

    try:
        response = await asyncio.wait_for(_send(), _TIMEOUT_S)
    except Exception as exc:
        logger.warning(
            "browser router call failed",
            extra={"event": "browser_router_unreachable", "method": method, "route": httpx.URL(url).path,
                   "error_type": type(exc).__name__},
        )
        raise BrowserServiceError(
            UPSTREAM_FAILED, "browser_unreachable", "The browser didn't answer. Try again in a moment.",
        ) from exc
    if response.status_code >= 400:
        try:
            refusal = _RouterRefusal.model_validate_json(response.content)
        except ValidationError:
            refusal = _RouterRefusal()
        status = response.status_code if response.status_code in (401, 403, 429) else UPSTREAM_FAILED
        if status == UPSTREAM_FAILED:
            logger.warning(
                "browser router refused a call",
                extra={"event": "browser_router_refused", "method": method, "route": httpx.URL(url).path,
                       "status_code": response.status_code, "reason": refusal.error},
            )
        raise BrowserServiceError(
            status,
            refusal.error or "browser_error",
            refusal.message or f"The browser answered HTTP {response.status_code}.",
        )
    try:
        return response.json()
    except ValueError as exc:
        raise BrowserServiceError(UPSTREAM_FAILED, "bad_response", "The browser returned an unreadable answer.") from exc


_Answer = TypeVar("_Answer", bound=BaseModel)


def _parsed(model: type[_Answer], raw: Any) -> _Answer:
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise BrowserServiceError(UPSTREAM_FAILED, "bad_response", "The browser returned an unexpected answer.") from exc


async def router_status(bearer: str) -> RouterStatus:
    """The caller's own browser's state, from the router."""
    return _parsed(RouterStatus, await _call("GET", f"{_require_router()}/_status", bearer))


def router_prefs(*, timezone: str | None = None, locale: str | None = None) -> RouterPrefs:
    """The person's browser preferences for a warm-up: the time zone and
    language the app sent, when they look like an IANA zone and a BCP 47 tag."""
    return RouterPrefs(
        timezone=timezone if timezone and _TIMEZONE.fullmatch(timezone) else None,
        locale=locale if locale and _LOCALE.fullmatch(locale) else None,
    )


async def warm(bearer: str, prefs: RouterPrefs) -> RouterStatus:
    """Start or claim the caller's browser without waiting for it, and update
    the preferences the router keeps for them. Safe to repeat."""
    body = {"prefs": prefs.model_dump(exclude_none=True)}
    return _parsed(RouterStatus, await _call("POST", f"{_require_router()}/_warm", bearer, json=body))


class OpenedSession(BaseModel):
    """``POST /browse/open``'s answer: the task's session, now at the address."""

    session_id: str


def person_url(raw: str) -> str | None:
    """An address the person typed, as the browser takes it: http or https with
    a host, ``https://`` added when they typed none. None for anything else."""
    text = (raw or "").strip()
    if not text or len(text) > 4096 or any(c.isspace() for c in text):
        return None
    if "://" not in text:
        # "javascript:…" or "mailto:…" is a scheme, not a host; "localhost:3000" is a port.
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:(?!\d)", text):
            return None
        text = f"https://{text}"
    parsed = urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    return text


async def person_open(bearer: str, *, user_id: str | None, conversation_id: str, url: str | None) -> OpenedSession:
    """The person opens ``url`` in a task's browser themselves, from the pane:
    the task's session (``task_session``) opens at it, or, open already, goes
    there. No ``url`` opens it on a blank tab and leaves an open one as it is.
    As the caller, so it is their browser and their allowance; no turn and no
    agent are involved. Raises BrowserServiceError."""
    router = _require_router()
    session_id = task_session(user_id, conversation_id)
    opened = await _call("POST", f"{router}/sessions", bearer, json={"session_id": session_id, **({"url": url} if url else {})})
    if url and isinstance(opened, dict) and opened.get("created") is False:
        await _call("POST", f"{router}/sessions/{session_id}/action", bearer, json={"action": "navigate", "url": url})
    return OpenedSession(session_id=session_id)


async def embed(bearer: str, session_id: str) -> BrowserEmbed:
    """A viewer link for one of the caller's sessions, minted by the router
    with their own credential. The router routes by that credential, so a
    session id can only name the caller's own. A link the router answers
    outside that session's viewer page is refused."""
    router = _require_router()
    answer = _parsed(BrowserEmbed, await _call("POST", f"{router}/_embed", bearer, json={"session_id": session_id}))
    page = answer.view_url.split("?", 1)[0]
    if answer.session_id != session_id or page != f"{router}/sessions/{session_id}/view":
        raise BrowserServiceError(UPSTREAM_FAILED, "bad_response", "The browser returned an unexpected viewer link.")
    return answer


async def approve(bearer: str, request: ApproveRequest) -> BrowserApproval:
    """Record the caller's approval of a step the browser held in one of their
    sessions, so anton's one resend of it runs. The router takes it only with
    the person's own credential; it never runs the step itself."""
    router = _require_router()
    answer = _parsed(BrowserApproval, await _call(
        "POST", f"{router}/sessions/{request.session_id}/approve", bearer,
        json={"pending_id": request.pending_id},
    ))
    if answer.pending_id != request.pending_id:
        raise BrowserServiceError(UPSTREAM_FAILED, "bad_response", "The browser approved a different step.")
    return answer


async def decline(bearer: str, request: DeclineRequest) -> BrowserDecline:
    """Record the caller's Skip of a step the browser held: the router drops
    the step and any approval already recorded for it, so an approval that
    landed after this server stopped waiting on it can't let a resend run."""
    router = _require_router()
    return _parsed(BrowserDecline, await _call(
        "POST", f"{router}/sessions/{request.session_id}/decline", bearer,
        json={"pending_id": request.pending_id},
    ))


async def clear_logins(bearer: str) -> LoginsCleared:
    """Sign the caller out of every site: the router empties their running
    browser's cookies and deletes their saved sign-ins."""
    return _parsed(LoginsCleared, await _call("DELETE", f"{_require_router()}/logins", bearer))


async def scope_turn(*, turn_key: str, router: str, config: BrowserConfig | None) -> None:
    """Bind a web turn's key with the router before the producer queues the
    turn: to the turn's session and project, which also starts the person's
    browser, or, with no ``config``, to no session, so the key reaches nothing
    and no code holding it can scope it later. The router refuses an unscoped
    key everything. It takes the scope only with ``scope_key`` beside the key.
    Raises BrowserServiceError."""
    body = ScopeRequest(
        session_id=config.profile if config is not None else None,
        project=(config.project or None) if config is not None else None,
    )
    scope_key = _scope_key()
    headers = {SCOPE_KEY_HEADER: scope_key} if scope_key else None
    await _call("POST", f"{router}/_scope", turn_key, json=body.body(), headers=headers)


def _scope_key() -> str:
    """The router's scope key, read from its file on every call: the kubelet
    fills the mounted Secret when it first appears and rewrites it on rotation,
    and an environment variable would keep whatever the pod started with. ""
    when there is no file or it is empty; the router then refuses the scope."""
    path = get_app_settings().browser_scope_key_file
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


async def browser_entitlement(*, bearer: str, org_id: str, user_id: str) -> BrowserEntitlement:
    """Whether the caller may use the browser on the web: the ``browser``
    block of their MindsHub entitlements, through the same cached read as
    their permissions (cowork.services.mindshub_teams.fetch_mindshub_permissions).

    Raises MindsHubUnreachable when MindsHub couldn't be asked or answered
    without the block, so an outage is never reported as a refusal."""
    from cowork.services.mindshub_client import MindsHubUnreachable
    from cowork.services.mindshub_teams import fetch_mindshub_permissions

    if not bearer:
        raise BrowserServiceError(401, "unauthorized", "Sign in to MindsHub to use the browser.")
    permissions = await fetch_mindshub_permissions(bearer_token=bearer, org_id=org_id, user_id=user_id)
    if permissions.signed_out:
        raise BrowserServiceError(401, "unauthorized", "Sign in to MindsHub again to use the browser.")
    if not permissions.reachable or permissions.browser is None:
        raise MindsHubUnreachable("Couldn't reach MindsHub to confirm your browser access. Try again in a moment.")
    return permissions.browser


#: The last answer a MindsHub-signed-in desktop read: whether its
#: organization turned the browser off. Kept through an outage, since the
#: desktop browser runs on the person's own machine and connection.
_desktop_organization_off = False


async def desktop_organization_off() -> bool:
    """Whether the organization a signed-in desktop belongs to turned the
    browser off, read from MindsHub with the credential the app handed over,
    before each turn and on each status read. A desktop with no MindsHub
    credential (the person's own model key) has no organization and asks
    nothing. ``not_permitted`` doesn't apply here: ``browser.use`` governs
    the web only. When MindsHub can't answer, the last answer stands."""
    global _desktop_organization_off
    from cowork.services.mindshub_teams import fetch_mindshub_permissions

    credential = get_minds_credential()
    if not credential:
        _desktop_organization_off = False
        return False
    try:
        permissions = await fetch_mindshub_permissions(bearer_token=credential, org_id="", user_id="")
    except Exception as exc:
        logger.warning(
            "could not read the desktop's organization browser switch; the last answer stands",
            extra={"event": "desktop_browser_switch_unread", "error_type": type(exc).__name__},
        )
        return _desktop_organization_off
    if not permissions.reachable or permissions.browser is None or permissions.signed_out:
        # MindsHub couldn't say, or refused the credential itself: neither is
        # the switch's answer.
        logger.warning(
            "could not read the desktop's organization browser switch; the last answer stands",
            extra={"event": "desktop_browser_switch_unread", "signed_out": permissions.signed_out},
        )
        return _desktop_organization_off
    _desktop_organization_off = permissions.browser.disabled_reason == BrowserDisabledReason.ORGANIZATION_OFF
    return _desktop_organization_off


async def browse_status(user_settings: UserSettings, *, bearer: str, org_id: str, user_id: str) -> BrowseStatus:
    """``GET /browse/status`` for the caller.

    The desktop answers from the browser the app registered, and a signed-in
    desktop also from its organization's switch (``desktop_organization_off``).
    On the web without a router there is no browser, and nothing is asked. An
    account whose organization turned the browser off is ``organization_off``,
    and one whose entitlement is otherwise off ``not_entitled``, both read from
    MindsHub before the router is asked. Otherwise the router says how the
    caller's browser is.
    """
    enabled = user_settings.browser_enabled
    if not on_web():
        if get_browser_host() is None:
            return BrowseStatus(enabled=enabled, available=False, status="unavailable")
        if await desktop_organization_off():
            return BrowseStatus(enabled=enabled, available=False, status=ORGANIZATION_OFF)
        return BrowseStatus(enabled=enabled, available=True, status="running")
    if not router_url():
        return BrowseStatus(enabled=enabled, available=False, status="unavailable")
    entitlement = await browser_entitlement(bearer=bearer, org_id=org_id, user_id=user_id)
    if entitlement.disabled_reason == BrowserDisabledReason.ORGANIZATION_OFF:
        return BrowseStatus(enabled=enabled, available=False, status=ORGANIZATION_OFF)
    if not entitlement.enabled:
        return BrowseStatus(enabled=enabled, available=False, status=NOT_ENTITLED)
    return BrowseStatus(enabled=enabled, available=True, status=(await router_status(bearer)).status)


# ── A turn's browser ───────────────────────────────────────────────────────


def turn_block(
    user_settings: UserSettings, *, user_id: str | None, conversation_id: str, project_id: Any,
) -> BrowserConfig | None:
    """A web turn's browser: the router, the task's own session, and the
    project's saved-tabs key. No credential: anton reaches the router with the
    turn's own key, once ``scope_turn`` has scoped it. The producer puts
    ``to_dict()`` in the job, which anton reads back with ``from_dict``.

    None when this deployment has no router, when the person turned the
    browser off, and for a turn with no person behind it: an organization's
    channel turn, which comes from a chat anyone in it can write to, while the
    browser holds one person's sign-ins.
    """
    router = router_url()
    if not router or not user_id or not user_settings.browser_enabled:
        return None
    return BrowserConfig(
        base_url=router,
        profile=task_session(user_id, conversation_id),
        project=project_key(user_id, project_id),
    )


def desktop_browser_config(*, conversation_id: str, project_id: Any) -> BrowserConfig | None:
    """An in-process turn's browser: the one the desktop app registered, with
    the task's own session. None when the app registered none. anton's own
    check refuses a malformed address or key; that is logged without either
    and the turn runs without a browser."""
    host = get_browser_host()
    if host is None:
        return None
    try:
        return BrowserConfig(
            base_url=host.url,
            profile=task_session(None, conversation_id),
            project=project_key(None, project_id),
            token=host.token,
        )
    except ValidationError:
        logger.warning(
            "the desktop app's browser was refused by the agent's browser config",
            extra={"event": "desktop_browser_refused"},
        )
        return None
