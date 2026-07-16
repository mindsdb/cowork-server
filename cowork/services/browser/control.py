"""`BrowserControlService` — the per-session control gate + reconnect logic.

Owns `control_state` on `BrowserSession`:

- `stop()` sets `stopped` synchronously (the pre-dispatch gate) and persists
  it — it survives reconnect and is NEVER auto-cleared. Only a fresh user
  turn clears it, via `resume_on_new_turn()` (called from the API layer).

Shared Stop lifecycle (single definition across server + Electron):
the SERVER is the single source of truth for the Stop gate. Stop gates the
session server-side, and a FRESH USER TURN resumes it via
`resume_on_new_turn()` — nothing else clears it. Electron keeps a local
`stopRequested` latch ONLY to close the hand-out→execute race (a command
already handed to the wire just before the Stop landed); that latch
self-clears and is NOT a "requires re-approval" gate. Re-approval is
required only after take-over / lost (`requires_reapproval`), never after a
plain Stop.

Stop ack tokens: the renderer sends a client-generated `stop_id` with the
user's stop, and the Electron poller re-sends the SAME `stop_id` when it
acknowledges the gate by POSTing /browse/control/stop itself. A `stop_id`
the session has already applied is a PURE acknowledgement — it must not
change `control_state` (the session may legitimately be `active` again
after `resume_on_new_turn`) and must not drain. Only a NEW `stop_id` (or a
legacy call without one) applies the stop. Tokens live in-memory only (see
`_last_stop_ids`).
- `takeover()` sets `taken_over` and flips `available=False` so the poller
  pauses issuing browser actions.
- `is_blocked()` reports whether the gate currently forbids dispatch.
- `on_bridge_state()` mirrors the Electron-main bridge state; a Chrome
  restart (target ids changed) marks the session `lost` + requires
  re-approval while preserving history; a `stopped` session stays stopped.
- `reconnect()` restores availability after a clean reconnect but refuses to
  clear a `stopped` gate.
"""
from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.browser import BrowserSession
from cowork.schemas.browser import BridgeState, ControlState, coerce_enum, coerce_uuid

# Last applied stop token per conversation (str(conversation_id) → stop_id).
# In-memory ONLY, like the broker's command queue: the server is a single
# process, so no DB column is needed. A restart forgets the tokens — worst
# case the poller's ack after a restart is treated as a new stop and
# re-stops once (the next user turn resumes it); it can never miss a stop.
_last_stop_ids: dict[str, str] = {}


class BrowserControlService:
    """Reads/writes the control + bridge state of a browser session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── lookup ────────────────────────────────────────────────────
    def get_session(self, session_id: UUID | str) -> BrowserSession | None:
        return self._session.get(BrowserSession, coerce_uuid(session_id))

    def get_by_conversation(self, conversation_id: UUID | str) -> BrowserSession | None:
        cid = coerce_uuid(conversation_id)
        return self._session.exec(
            select(BrowserSession).where(BrowserSession.conversation_id == cid)
        ).first()

    # ── control gate ──────────────────────────────────────────────
    def is_blocked(self, session_id: UUID | str) -> bool:
        """True when the control gate forbids dispatching a new action."""
        sess = self.get_session(session_id)
        if sess is None:
            return False
        return sess.control_state in (
            ControlState.stopped.value,
            ControlState.taken_over.value,
        )

    def _set_gate(
        self, sess: BrowserSession | None, state: ControlState
    ) -> BrowserSession | None:
        """Set the control gate on `sess` and pause availability, persisting.

        `None` in / `None` out so the callers stay a one-liner regardless of
        whether the session (or its conversation) exists.
        """
        if sess is None:
            return None
        sess.control_state = state.value
        sess.available = False
        return self._save(sess)

    def stop(self, session_id: UUID | str) -> BrowserSession | None:
        """Set the `stopped` pre-dispatch gate (synchronous, persisted)."""
        return self._set_gate(self.get_session(session_id), ControlState.stopped)

    def stop_by_conversation(self, conversation_id: UUID | str) -> BrowserSession | None:
        """Mark the conversation's browser session stopped.

        Used by the `/responses/cancel` hook so cancelling a turn also gates
        the browser for that conversation. When no browser session exists
        yet, one is CREATED so the gate persists — otherwise a Stop arriving
        before the first `/bridge/hello` would be lost and the later hello
        would mint a fresh `active` session. A nonexistent conversation
        stays a safe no-op.
        """
        return self._set_gate(
            self._get_or_create_by_conversation(conversation_id),
            ControlState.stopped,
        )

    def apply_stop(
        self, conversation_id: UUID | str, *, stop_id: str | None = None
    ) -> tuple[BrowserSession | None, bool]:
        """Apply a stop idempotently by its client-generated `stop_id`.

        Returns `(session, applied)`. When `stop_id` matches the LAST stop
        already applied for this conversation, the call is a PURE
        acknowledgement (the Electron poller re-sends the renderer's
        `stop_id` to confirm the gate): `applied` is False and
        `control_state` is left untouched — the session may legitimately be
        `active` again after `resume_on_new_turn`, and the ack must not
        re-stop it. A NEW `stop_id`, or none at all (legacy/curl callers),
        applies the stop exactly like `stop_by_conversation` and records
        the token. Tokens are in-memory per-process (`_last_stop_ids`); a
        restart forgets them, worst case one redundant re-stop.
        """
        key = str(coerce_uuid(conversation_id))
        if stop_id is not None and _last_stop_ids.get(key) == stop_id:
            return self.get_by_conversation(conversation_id), False
        sess = self._set_gate(
            self._get_or_create_by_conversation(conversation_id),
            ControlState.stopped,
        )
        if stop_id is not None and sess is not None:
            _last_stop_ids[key] = stop_id
        return sess, True

    def resume_on_new_turn(self, conversation_id: UUID | str) -> BrowserSession | None:
        """Clear a `stopped` gate when a FRESH USER TURN starts.

        This is the ONLY resume path in the shared server/Electron Stop
        lifecycle: the server owns the gate, and a new user turn is the
        explicit signal to proceed again, so it resets `stopped` → `active`
        (the API layer calls this from POST /responses). Stop gates the
        turn it cancelled — it survives reconnect and is never
        auto-cleared. No re-approval is needed after a plain Stop
        (Electron's local `stopRequested` latch only closes the
        hand-out→execute race and self-clears). A `taken_over` gate is NOT
        cleared here: the user is actively driving the browser, and only an
        explicit re-approval ends a takeover. Availability is restored only
        when the bridge is connected and no re-approval is pending.
        """
        sess = self.get_by_conversation(conversation_id)
        if sess is None or sess.control_state != ControlState.stopped.value:
            return sess
        sess.control_state = ControlState.active.value
        sess.available = (
            sess.bridge_state == BridgeState.connected.value
            and not sess.requires_reapproval
        )
        return self._save(sess)

    def takeover(self, session_id: UUID | str) -> BrowserSession | None:
        """Mark `taken_over` and pause the bridge from issuing actions."""
        return self._set_gate(self.get_session(session_id), ControlState.taken_over)

    def takeover_by_conversation(
        self, conversation_id: UUID | str
    ) -> BrowserSession | None:
        """Mark the conversation's browser session taken_over, creating the
        session first when none exists (same persistence rationale as
        `stop_by_conversation`)."""
        return self._set_gate(
            self._get_or_create_by_conversation(conversation_id),
            ControlState.taken_over,
        )

    def _get_or_create_by_conversation(
        self, conversation_id: UUID | str
    ) -> BrowserSession | None:
        """The conversation's session, creating one if the conversation
        exists but has no browser session yet. `None` when the conversation
        itself doesn't exist (gate call becomes a no-op)."""
        sess = self.get_by_conversation(conversation_id)
        if sess is not None:
            return sess
        # Local import: approval.py is the session-upsert owner and does not
        # import control.py, so this stays cycle-free.
        from cowork.services.browser.approval import BrowserApprovalService

        return BrowserApprovalService(self._session).get_or_create_session(
            conversation_id
        )

    # ── bridge state mirroring ────────────────────────────────────
    def on_bridge_state(
        self,
        session_id: UUID | str,
        bridge_state: BridgeState | str,
        *,
        target_changed: bool = False,
    ) -> BrowserSession | None:
        """Mirror a bridge-state push from the Electron main process.

        - `connected` → available (unless the gate is stopped/taken_over).
        - `lost` / `disconnected` → not available.
        - A Chrome restart (`target_changed`) marks `lost` and requires
          re-approval, preserving history; a stopped session stays stopped.
        """
        sess = self.get_session(session_id)
        if sess is None:
            return None
        bs = coerce_enum(BridgeState, bridge_state)

        if target_changed:
            sess.bridge_state = BridgeState.lost.value
            sess.available = False
            sess.requires_reapproval = True
            return self._save(sess)

        sess.bridge_state = bs.value
        if bs == BridgeState.connected:
            # Never auto-clear a stopped/taken-over gate; only make the
            # session available when the gate permits.
            if sess.control_state == ControlState.active.value:
                sess.available = True
        else:
            # Any non-connected state (lost / disconnected / awaiting_approval)
            # cannot execute commands.
            sess.available = False
        return self._save(sess)

    def reconnect(self, session_id: UUID | str) -> BrowserSession | None:
        """Restore availability after a clean reconnect.

        Refuses to clear a `stopped` gate — a stopped session stays stopped
        even across a reconnect.
        """
        sess = self.get_session(session_id)
        if sess is None:
            return None
        sess.bridge_state = BridgeState.connected.value
        # A stopped / taken-over gate is never cleared by a reconnect: the
        # session only becomes available again while the gate is `active`.
        sess.available = sess.control_state == ControlState.active.value
        return self._save(sess)

    # ── helpers ───────────────────────────────────────────────────
    def _save(self, sess: BrowserSession) -> BrowserSession:
        self._session.add(sess)
        self._session.commit()
        self._session.refresh(sess)
        return sess
