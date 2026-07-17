"""`BridgeClient` — the single interface WS3's agent tool consumes.

Each of `inspect / navigate / scroll / wait`:
  1. is_blocked gate (stopped / taken_over → permission_denied verdict);
  2. permission check against the session's grant (cross-domain policy);
  3. `BridgeCommandService.execute(...)` under a bounded timeout;
  4. persists the action (pending → in_flight → observed/failed) with a
     content-free digest, and returns a typed `BrowserToolVerdict`.

Single-in-flight per session: a second command while one is outstanding
returns immediately (`permission_denied` verdict with a busy detail) rather
than racing the bridge. The verdict carries the WS4-internal `result_code`;
the agent tool maps it to a canonical `BrowserErrorKind`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from uuid import UUID

from sqlmodel import Session

from cowork.schemas.browser import (
    ACTION_TYPE_TO_CLASS,
    LLM_ACTION_TO_TYPE,
    BridgeCommandResult,
    BrowserActionType,
    BrowserToolVerdict,
    ControlState,
    ResultCode,
    coerce_enum,
    coerce_uuid,
    host_only,
    registrable_host,
)
from cowork.services.browser import BROWSER_CONNECT_FLOW_STEPS
from cowork.services.browser.actions import BrowserActionStore
from cowork.services.browser.bridge import BridgeCommandService, bridge_command_service
from cowork.services.browser.control import BrowserControlService
from cowork.services.browser.permissions import BrowserPermissionService

logger = logging.getLogger(__name__)

# No-session verdict detail. Truthful and actionable: the ONLY way a browser
# session comes to exist is the in-app connect flow — there is no extension.
# Without this the LLM invents a nonexistent "Chrome extension" setup flow
# when no tab is connected (observed incident).
NO_SESSION_DETAIL = (
    "No browser tab is connected. Ask the user to connect one in the "
    f"desktop app: {BROWSER_CONNECT_FLOW_STEPS}. There is no browser "
    "extension to install."
)

# open_url denial detail — the user-anchor guard failed. Deterministic and
# server-side: the model cannot bypass it by claiming the user asked.
OPEN_URL_NOT_ANCHORED_DETAIL = (
    "open_url is only allowed for sites the user explicitly asked for. "
    "Ask the user to name the exact site or URL."
)


def url_user_anchored(host_or_url: str, user_texts: list[str]) -> bool:
    """True iff the target's registrable host appears in a USER message.

    The deterministic server-side guard behind `open_url` (M1
    \"user-directed URL = implicit grant\"): a cross-domain open is allowed
    ONLY when the user's own words name the site. The check is a
    case-insensitive substring match of the target's PSL registrable host
    (`registrable_host`, so `www.bbc.co.uk` and `https://bbc.co.uk/news`
    both reduce to `bbc.co.uk`) against the raw text of the conversation's
    USER messages — assistant/tool text must never satisfy it (the caller
    passes user-authored texts only). An empty host or no user text never
    matches.
    """
    host = registrable_host(host_or_url)
    if not host:
        return False
    needle = host.lower()
    for text in user_texts:
        if isinstance(text, str) and needle in text.lower():
            return True
    return False


# One asyncio.Lock per server session id enforces single-in-flight.
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class BridgeClient:
    """Brokers a single read-only browser action for a session."""

    def __init__(
        self,
        db_session: Session,
        *,
        broker: BridgeCommandService | None = None,
    ) -> None:
        self._db = db_session
        self._broker = broker or bridge_command_service
        self._control = BrowserControlService(db_session)
        self._permissions = BrowserPermissionService(db_session)
        self._actions = BrowserActionStore(db_session)

    # ── verb-agnostic entrypoint (the send_browser_command surface) ─
    async def send(
        self,
        conversation_id: UUID | str,
        action: str,
        *,
        href: str | None = None,
        direction: str | None = None,
        user_texts: list[str] | None = None,
    ) -> BrowserToolVerdict:
        """Dispatch a browser command for a conversation by LLM verb.

        `action` is the LLM-facing verb (`inspect | follow_link | scroll |
        wait | open_url`). Resolves the conversation's `BrowserSession`,
        translates the verb to a stored `action_type`, and runs it. An
        unknown verb or a conversation with no browser session returns a
        typed verdict (never raises) so the agent tool can map it to a
        canonical error kind.

        `open_url` (user-directed URL = implicit grant) additionally needs
        `user_texts` — the raw texts of the conversation's USER messages —
        for the deterministic user-anchor guard (`url_user_anchored`): the
        target's registrable host must appear in the user's own words, else
        the call is denied and NO grant is created. On success the session
        is retargeted (new host granted, prior grants revoked,
        `active_domain` moved) BEFORE dispatch, so the normal `_run`
        permission check passes against the new grant.
        """
        action_type = LLM_ACTION_TO_TYPE.get(action)
        if action_type is None:
            return BrowserToolVerdict(
                result_code=ResultCode.error,
                action_type=BrowserActionType.inspect,
                detail=f"unsupported action '{action}'",
            )

        sess = self._control.get_by_conversation(conversation_id)
        if sess is None:
            return BrowserToolVerdict(
                result_code=ResultCode.error,
                action_type=action_type,
                detail=NO_SESSION_DETAIL,
            )

        if action_type == BrowserActionType.open_url:
            return await self._open_url(sess, href, user_texts or [])

        # A navigate (follow_link) targets the href's host — the permission
        # check must run against IT, not fall back to the session's approved
        # active_domain. Without this, `follow_link` with a cross-site href
        # would ride the same-site grant. The grant match itself is
        # registrable-host equality (host_matches_grant, inside the
        # permission service): active_domain is Electron's
        # PSL-registrable-or-exact host, so a same-site SUBDOMAIN href
        # stays allowed, matching exactly what Electron itself accepts.
        domain = (
            host_only(href)
            if action_type == BrowserActionType.navigate and href
            else None
        )
        return await self._run(
            sess.id, action_type, domain=domain, href=href, direction=direction
        )

    async def _open_url(self, sess, href: str | None, user_texts: list[str]):
        """User-directed `open_url`: anchor guard → retarget → dispatch.

        Order matters:
          1. USER-ANCHOR GUARD (deterministic, server-side): the target's
             registrable host must appear in a USER message. Failure →
             `permission_denied`, nothing dispatched, NO grant created.
          2. Control gates: a stopped/taken-over or reapproval-pending
             session refuses via `_run`'s normal gates WITHOUT retargeting —
             a blocked session must not accumulate new grants.
          3. Retarget: grant the new host, revoke prior grants for other
             hosts, move `active_domain` (single-active-domain invariant).
          4. Dispatch through `_run`, whose permission check now passes
             against the fresh grant and which persists the action row with
             action_type `open_url`.
        """
        target_host = host_only(href) if href else ""
        if not href or not target_host:
            return BrowserToolVerdict(
                result_code=ResultCode.error,
                action_type=BrowserActionType.open_url,
                detail="open_url requires a full http(s) `url`",
            )
        if not url_user_anchored(href, user_texts):
            return BrowserToolVerdict(
                result_code=ResultCode.permission_denied,
                action_type=BrowserActionType.open_url,
                domain=target_host,
                detail=OPEN_URL_NOT_ANCHORED_DETAIL,
            )

        # Control/reapproval gates BEFORE any grant mutation: let `_run`
        # produce its normal gate verdicts without creating a grant.
        if not self._control.is_blocked(sess.id) and not sess.requires_reapproval:
            from cowork.services.browser.approval import BrowserApprovalService

            # The grant domain is the PSL-registrable-or-exact host — the
            # SAME function Electron uses for its grants — so the permission
            # check's registrable-host equality covers subdomains exactly
            # like a tab approval would (`www.bbc.co.uk` → grant
            # `bbc.co.uk`).
            BrowserApprovalService(self._db).retarget_domain(
                sess.id, registrable_host(target_host)
            )

        return await self._run(
            sess.id,
            BrowserActionType.open_url,
            domain=target_host,
            href=href,
        )

    # ── public verbs (the send_browser_command surface) ───────────
    async def inspect(
        self, session_id: UUID | str, *, domain: str | None = None
    ) -> BrowserToolVerdict:
        return await self._run(session_id, BrowserActionType.inspect, domain=domain)

    async def navigate(
        self, session_id: UUID | str, *, href: str, domain: str | None = None
    ) -> BrowserToolVerdict:
        # The target's bare host is what the grant is checked against
        # (registrable-host equality with Electron's PSL-derived grant,
        # inside the permission service); only the host is ever used
        # server-side.
        target_domain = domain or host_only(href)
        return await self._run(
            session_id, BrowserActionType.navigate, domain=target_domain, href=href
        )

    async def scroll(
        self,
        session_id: UUID | str,
        *,
        direction: str | None = None,
        domain: str | None = None,
    ) -> BrowserToolVerdict:
        return await self._run(
            session_id, BrowserActionType.scroll, domain=domain, direction=direction
        )

    async def wait(
        self, session_id: UUID | str, *, domain: str | None = None
    ) -> BrowserToolVerdict:
        return await self._run(session_id, BrowserActionType.wait, domain=domain)

    # ── core ──────────────────────────────────────────────────────
    async def _run(
        self,
        session_id: UUID | str,
        action_type: BrowserActionType,
        *,
        domain: str | None = None,
        href: str | None = None,
        direction: str | None = None,
    ) -> BrowserToolVerdict:
        sid = coerce_uuid(session_id)
        sid_str = str(sid)
        action_class = ACTION_TYPE_TO_CLASS[action_type]

        sess = self._control.get_session(sid)
        if sess is None:
            return BrowserToolVerdict(
                result_code=ResultCode.error,
                action_type=action_type,
                detail=NO_SESSION_DETAIL,
            )

        # 1. Control gate (pre-dispatch): stopped / taken_over never dispatch.
        if self._control.is_blocked(sid):
            # `stopped` / `taken_over` are CONTROL terminal states, not error
            # kinds. Carry the control_state separately so the agent tool /
            # UI renders a distinct stopped / taken-over terminal state rather
            # than collapsing it into `permission_denied`.
            return BrowserToolVerdict(
                result_code=ResultCode.permission_denied,
                action_type=action_type,
                domain=sess.active_domain,
                detail=f"session {sess.control_state}",
                control_state=coerce_enum(ControlState, sess.control_state),
            )

        # 1b. Re-approval gate: after a Chrome restart / target change the
        # old tab grants remain rows in the DB, but the approved TAB is gone.
        # Nothing may dispatch until a fresh approval clears the flag —
        # otherwise a same-domain inspect/follow_link would ride a stale
        # grant. `unapproved_tab` maps to the canonical `permission_denied`.
        if sess.requires_reapproval:
            return BrowserToolVerdict(
                result_code=ResultCode.unapproved_tab,
                action_type=action_type,
                domain=sess.active_domain,
                detail="tab changed; re-approval required",
            )

        # Resolve the effective target host: an explicit domain, else the
        # session's approved active domain. `target_host` is the bare
        # hostname (host_only) — it is what gets persisted/traced; grant
        # matching against it is registrable-host equality inside the
        # permission service.
        target_host = host_only(domain) if domain else (sess.active_domain or "")

        # 2. Permission check (cross-domain policy, registrable-host match).
        verdict = self._permissions.check(sid, target_host, action_class)
        if not verdict.granted:
            return BrowserToolVerdict(
                result_code=ResultCode.unapproved_tab,
                action_type=action_type,
                domain=target_host or sess.active_domain,
                detail=verdict.reason,
            )

        lock = _session_locks[sid_str]
        if lock.locked():
            return BrowserToolVerdict(
                result_code=ResultCode.permission_denied,
                action_type=action_type,
                domain=target_host,
                detail="a browser action is already in flight for this session",
            )

        async with lock:
            command_id = self._broker.new_command_id()
            idem = f"{sid_str}:{action_type.value}:{target_host}:{href or ''}:{direction or ''}"
            self._actions.append_pending(
                session_id=sid,
                command_id=command_id,
                idempotency_key=idem,
                action_type=action_type,
                domain=target_host or None,
            )
            self._actions.mark_in_flight(command_id)

            started = time.monotonic()
            try:
                result: BridgeCommandResult = await self._broker.execute(
                    session_id=sid_str,
                    action_type=action_type,
                    conversation_id=str(sess.conversation_id),
                    domain=target_host or None,
                    href=href,
                    direction=direction,
                    command_id=command_id,
                )
            except asyncio.CancelledError:
                # A cancelled producer (e.g. `/responses/cancel` teardown)
                # must not strand the row `in_flight` forever — the broker
                # discards the command, and we mark the row terminal-failed
                # (never observed=ok) before letting cancellation propagate.
                self._actions.mark_failed(
                    command_id,
                    result_code=ResultCode.error,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                raise
            duration_ms = int((time.monotonic() - started) * 1000)

            # ── no-false-success guard (BEFORE persistence) ──────────
            # An `ok` with NO observable result is NOT a real success. We
            # downgrade the EFFECTIVE result_code to `error` here — before
            # the row is persisted and before the tool traces its span — so
            # the DB never records a false `observed`/`ok`. The store then
            # persists it as `failed` with no digest (record_observed's
            # non-ok path), and WS3 maps `error` to the canonical kind.
            # `result.observed` being None/empty is the "nothing observed"
            # signal; a blob carrying content (text/links/etc.) is a genuine
            # observation even when it distils to an empty content-free
            # digest, so it stays `ok`.
            effective_code = result.result_code
            effective_detail = result.detail
            if effective_code == ResultCode.ok and not result.observed:
                effective_code = ResultCode.error
                effective_detail = (
                    "action completed with no observable result; "
                    "not recorded as success"
                )

            self._actions.record_observed(
                command_id,
                result_code=effective_code,
                transient=result.observed,
                duration_ms=duration_ms,
            )

            return BrowserToolVerdict(
                result_code=effective_code,
                action_type=action_type,
                observed=result.observed if effective_code == ResultCode.ok else None,
                citations=self._citations(result)
                if effective_code == ResultCode.ok
                else [],
                domain=target_host or sess.active_domain,
                action_id=command_id,
                detail=effective_detail,
            )

    @staticmethod
    def _citations(result: BridgeCommandResult) -> list[dict]:
        if result.result_code != ResultCode.ok or not result.observed:
            return []
        cites = result.observed.get("citations")
        return cites if isinstance(cites, list) else []


async def send_browser_command(
    conversation_id: UUID | str,
    action: str,
    *,
    href: str | None = None,
    direction: str | None = None,
    user_texts: list[str] | None = None,
) -> BrowserToolVerdict:
    """Module-level `send_browser_command` surface for the agent tool.

    Opens (and always closes) its own short-lived DB session so a single
    browser action is fully self-contained. This is the seam WS3's browser
    tool reaches through its thin `_get_bridge_client()` indirection.
    `user_texts` (the conversation's USER-message texts) is required for
    `open_url`'s user-anchor guard and ignored by every other verb.
    """
    from cowork.db.session import get_open_session

    db = get_open_session()
    try:
        return await BridgeClient(db).send(
            conversation_id,
            action,
            href=href,
            direction=direction,
            user_texts=user_texts,
        )
    finally:
        db.close()
