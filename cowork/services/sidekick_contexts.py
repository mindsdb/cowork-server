"""Load and save a conversation's Chimera sidekick (sidekick_contexts).

anton hands its sidekick out as a JSON-shaped dict (``ChatSession.
sidekick_state``) and takes the same dict back on the next session
(``ChatSessionConfig.sidekick_state``). This maps that dict onto the row and
back. The row has no org column, so every method first reaches the
conversation through ``ConversationService._owned``: a conversation the
caller cannot see has no sidekick to read or write.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from cowork.common.history_scrub import scrub_credentials, scrub_message_dict
from cowork.db.scoped import ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.sidekick_context import SIDEKICK_IDLE, SIDEKICK_OPEN, SidekickContext

logger = logging.getLogger(__name__)

# anton's SidekickState keys that have a column of their own. Anything else in
# the dict is dropped on save: there is nowhere to keep it.
_STATE_KEYS = ("sidekick_id", "model", "messages", "summary", "status", "last_call_at", "signals")
_STATUSES = frozenset({SIDEKICK_IDLE, SIDEKICK_OPEN})


def _parse_instant(value) -> datetime | None:
    """An ISO-8601 instant as an aware UTC datetime, or None when unreadable."""
    if isinstance(value, datetime):
        instant = value
    elif isinstance(value, str) and value:
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    # SQLite keeps no offset, so a stored instant comes back naive; it was UTC.
    return instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant.astimezone(UTC)


def state_from_row(row: SidekickContext) -> dict:
    """The row as anton's SidekickState dict, as stored (see scrubbed_state)."""
    instant = _parse_instant(row.last_call_at)
    return {
        "sidekick_id": row.sidekick_id,
        "model": row.model,
        "messages": list(row.messages or []),
        "summary": row.summary,
        "status": row.status,
        "last_call_at": instant.isoformat() if instant is not None else None,
        "signals": row.signals,
    }


def scrubbed_state(state: dict) -> dict:
    """``state`` with its messages and summary scrubbed for replay, as a copy.

    They go back to a model, so a credential vaulted since they were written is
    masked, as replayed history is (cowork.common.history_scrub). Run it after
    the turn registers its vault's secrets, as the harness does for history.
    """
    summary = state.get("summary")
    return {
        **state,
        "messages": [
            scrub_message_dict(dict(message)) if isinstance(message, dict) else message
            for message in state.get("messages") or []
        ],
        "summary": scrub_credentials(summary) if isinstance(summary, str) and summary else summary,
    }


class SidekickContextService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _conversation(self, conversation_id: UUID) -> Conversation | None:
        from cowork.services.conversations import ConversationService

        return ConversationService(self.session)._owned(conversation_id)

    def _row(self, conversation_id: UUID) -> SidekickContext | None:
        return self.session.exec(
            self.session.select(SidekickContext).where(
                SidekickContext.conversation_id == conversation_id
            )
        ).first()

    def load(self, conversation_id: UUID) -> dict | None:
        """The conversation's sidekick state for anton, or None when it has none."""
        if self._conversation(conversation_id) is None:
            return None
        row = self._row(conversation_id)
        return state_from_row(row) if row is not None else None

    def is_open(self, conversation: Conversation) -> bool:
        """Whether the sidekick last handed its work back to the main agent.

        Takes the conversation the caller has already reached, so the request's
        first unit pays one lookup for it.
        """
        row = self._row(conversation.id)
        return row is not None and row.status == SIDEKICK_OPEN

    def save(self, conversation_id: UUID, state: dict) -> bool:
        """Upsert the conversation's row from anton's state; commits.

        False, with nothing written, when the conversation is gone or out of
        the caller's sight, or when ``state`` is not a dict carrying a message
        list: a turn's cleanup calls this, after the answer is settled, and a
        bad state must cost the sidekick its memory, never the turn.
        """
        if not isinstance(state, dict) or not isinstance(state.get("messages", []), list):
            logger.warning("sidekick state for conversation %s is malformed; not saved", conversation_id)
            return False
        if self._conversation(conversation_id) is None:
            return False
        row = self._row(conversation_id) or SidekickContext(conversation_id=conversation_id)
        values = {key: state.get(key) for key in _STATE_KEYS}
        status = values["status"] if values["status"] in _STATUSES else SIDEKICK_IDLE
        row.sidekick_id = str(values["sidekick_id"]) if values["sidekick_id"] is not None else None
        row.model = str(values["model"]) if values["model"] is not None else None
        row.messages = list(values["messages"] or [])
        row.summary = values["summary"] if isinstance(values["summary"], str) else None
        row.status = status
        row.last_call_at = _parse_instant(values["last_call_at"])
        row.signals = values["signals"]
        row.updated_at = datetime.now(UTC)
        self.session.add(row)
        self.session.commit()
        return True

    def stage_delete(self, conversation_id: UUID) -> None:
        """Delete the conversation's row inside the caller's transaction.

        For the conversation-delete cascade and a history rewind: the caller has
        already reached the conversation, and commits.
        """
        row = self._row(conversation_id)
        if row is not None:
            self.session.delete(row)
