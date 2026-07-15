"""`BrowserApprovalService` — the server-side session + grant upsert path.

Closes the gap where nothing created a `BrowserSession` / `BrowserTabGrant`
in production: without this, `BridgeClient.send` always returned "no browser
session for conversation" and the permission check always denied.

When the user approves a tab in the desktop UI (WS1/WS2), the Electron bridge
tells the server the approved conversation + host-only domain. This service
creates-or-updates:

- the `BrowserSession` for the conversation (set `active_domain`, mark
  connected/available unless a control gate forbids it); and
- a `BrowserTabGrant` for the approved host (a single `navigate`-class grant,
  which the permission check accepts for both `read` and `navigate`).

Everything is content-free: the domain is reduced to a bare host before it
is ever persisted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.browser import BrowserSession, BrowserTabGrant
from cowork.models.conversation import Conversation
from cowork.schemas.browser import (
    BridgeState,
    BrowserActionClass,
    ControlState,
    PermissionDecision,
    coerce_uuid,
    host_only,
)


class BrowserApprovalService:
    """Creates/updates a browser session + its single approved-tab grant."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── session upsert ────────────────────────────────────────────
    def get_or_create_session(
        self,
        conversation_id: UUID | str,
        *,
        project_id: UUID | str | None = None,
        active_domain: str | None = None,
    ) -> BrowserSession | None:
        """Return the conversation's browser session, creating it if needed.

        `project_id` is resolved from the conversation when not supplied. If
        the conversation does not exist, returns `None` (nothing to attach
        to) rather than raising.
        """
        cid = coerce_uuid(conversation_id)
        sess = self._session.exec(
            select(BrowserSession).where(BrowserSession.conversation_id == cid)
        ).first()
        host = host_only(active_domain) if active_domain else None

        if sess is None:
            pid = coerce_uuid(project_id) if project_id else None
            if pid is None:
                conv = self._session.get(Conversation, cid)
                if conv is None:
                    return None
                pid = conv.project_id
            sess = BrowserSession(
                conversation_id=cid,
                project_id=pid,
                active_domain=host,
            )
        elif host:
            sess.active_domain = host
        return self._save(sess)

    # ── grant upsert ──────────────────────────────────────────────
    def grant_domain(
        self,
        session_id: UUID | str,
        domain: str,
        *,
        action_class: BrowserActionClass | str = BrowserActionClass.navigate,
    ) -> BrowserTabGrant | None:
        """Create-or-refresh a granted tab for a host on a session.

        A `navigate`-class grant satisfies both `read` and `navigate`
        permission checks on the same host (M1: one approved tab / one
        active domain).
        """
        sid = coerce_uuid(session_id)
        host = host_only(domain)
        if not host:
            return None
        ac = (
            action_class.value
            if isinstance(action_class, BrowserActionClass)
            else str(action_class)
        )
        grant = self._session.exec(
            select(BrowserTabGrant).where(
                BrowserTabGrant.session_id == sid,
                BrowserTabGrant.domain == host,
                BrowserTabGrant.action_class == ac,
            )
        ).first()
        now = datetime.now(timezone.utc)
        if grant is None:
            grant = BrowserTabGrant(
                session_id=sid,
                domain=host,
                action_class=ac,
                decision=PermissionDecision.granted.value,
                granted_at=now,
            )
        else:
            # Re-approving refreshes a previously revoked/expired grant.
            grant.decision = PermissionDecision.granted.value
            grant.granted_at = now
            grant.expires_at = None
        self._session.add(grant)
        self._session.commit()
        self._session.refresh(grant)
        return grant

    def approve(
        self,
        conversation_id: UUID | str,
        domain: str,
        *,
        project_id: UUID | str | None = None,
    ) -> BrowserSession | None:
        """Upsert the session for a conversation and grant the approved host.

        The single production entry point a tab approval calls: it makes the
        subsequent `send()` session lookup + permission check succeed. Marks
        the session connected/available unless a control gate forbids it.
        Returns the session, or `None` if the conversation does not exist.
        """
        host = host_only(domain)
        sess = self.get_or_create_session(
            conversation_id, project_id=project_id, active_domain=host
        )
        if sess is None:
            return None
        self.grant_domain(sess.id, host)
        # A fresh approval clears any prior "needs re-approval" flag and (when
        # the gate is active) brings the session online.
        sess.requires_reapproval = False
        sess.bridge_state = BridgeState.connected.value
        if sess.control_state == ControlState.active.value:
            sess.available = True
        return self._save(sess)

    # ── helpers ───────────────────────────────────────────────────
    def _save(self, sess: BrowserSession) -> BrowserSession:
        self._session.add(sess)
        self._session.commit()
        self._session.refresh(sess)
        return sess
