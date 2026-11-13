"""Who may run a turn through a channel bot, and the requests of those not yet allowed.

A sender is allowed per external chat (channel type, chat id, sender id), so
one approval covers every thread of the chat. Messages from anyone else start
nothing; the runtime records a pending request for the owner instead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.db.scoped import ScopedSession
from cowork.models.channel import (
    SENDER_ALLOWED,
    SENDER_BLOCKED,
    SENDER_DISPLAY_NAME_MAX,
    SENDER_PENDING,
    ChannelPairingCode,
    ChannelSender,
)

log = logging.getLogger(__name__)

PENDING_PER_CHANNEL_MAX = 50
PENDING_PER_SENDER_MAX = 5
PENDING_TTL = timedelta(days=7)
# Telegram posts a linked channel's messages under this one service account,
# so allowing it would admit every post of every linked channel.
_NEVER_ALLOWED = {("telegram", "777000")}


def never_allowed(channel_type: str, sender_id: str) -> bool:
    """Whether a platform service account that stands for many people sent this."""
    return (channel_type, sender_id) in _NEVER_ALLOWED


class ChannelSenderService:
    """Reads and records the senders of channel chats, in one scoped session."""

    def __init__(self, session: ScopedSession) -> None:
        """Bind the service to a scoped session; every read and write is org scoped by it."""
        self.session = session

    def find(self, channel_type: str, group_id: str, sender_id: str) -> ChannelSender | None:
        """The sender's row in that chat, or None when they never wrote there."""
        return self.session.exec(
            self.session.select(ChannelSender).where(
                ChannelSender.channel_type == channel_type,
                ChannelSender.external_group_id == group_id,
                ChannelSender.external_sender_id == sender_id,
            )
        ).first()

    def status_of(self, channel_type: str, group_id: str, sender_id: str) -> str | None:
        """The sender's status in this chat, or None when they never wrote there.

        A platform service account that stands for many people reads as blocked.
        """
        if never_allowed(channel_type, sender_id):
            return SENDER_BLOCKED
        row = self.find(channel_type, group_id, sender_id)
        return row.status if row is not None else None

    def list(self, *, status: str | None = None, channel_type: str | None = None) -> list[ChannelSender]:
        """Senders the owner can act on, newest first; expired pending requests are left out."""
        stmt = self.session.select(ChannelSender)
        if status is not None:
            stmt = stmt.where(ChannelSender.status == status)
        if channel_type is not None:
            stmt = stmt.where(ChannelSender.channel_type == channel_type)
        now = datetime.now(timezone.utc)
        rows = [
            row for row in self.session.exec(stmt).all()
            if not never_allowed(row.channel_type, row.external_sender_id)
            and not (row.status == SENDER_PENDING and _expired(row, now))
        ]
        return sorted(rows, key=lambda row: _aware(row.created_at) or now, reverse=True)

    def set_status(self, sender_row_id: UUID, status: str) -> ChannelSender | None:
        """Approve or block one sender in their chat; None when no such row is in scope. Commits."""
        row = self.session.get(ChannelSender, sender_row_id)
        if row is None:
            return None
        if status == SENDER_ALLOWED and never_allowed(row.channel_type, row.external_sender_id):
            raise ValueError("this platform account stands for many people and cannot be allowed")
        row.status = status
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def delete(self, sender_row_id: UUID) -> bool:
        """Forget one sender; their next message is a new pending request. Commits."""
        row = self.session.get(ChannelSender, sender_row_id)
        if row is None:
            return False
        self.session.delete(row)
        self.session.commit()
        return True

    def drop_channel(self, channel_type: str) -> None:
        """Stage the removal of every sender and pairing code of a channel type.

        Called when the channel's bot is disconnected, so a different bot of
        the same type inherits no approvals. The caller commits.
        """
        for model in (ChannelSender, ChannelPairingCode):
            for row in self.session.exec(self.session.select(model).where(model.channel_type == channel_type)).all():
                self.session.delete(row)

    def record_request(
        self,
        *,
        channel_type: str,
        group_id: str,
        sender_id: str,
        sender_name: str | None,
        is_group: bool,
    ) -> None:
        """Record or refresh a pending request from a sender the owner has not allowed.

        A refresh updates only the last-seen time and name, never the status,
        so it cannot demote a sender allowed meanwhile or undo a block. A new
        request past the per-channel or per-sender cap is dropped and logged.
        Side effects: prunes pending requests older than PENDING_TTL; commits.
        """
        if never_allowed(channel_type, sender_id):
            return
        now = datetime.now(timezone.utc)
        name = sender_name[:SENDER_DISPLAY_NAME_MAX] if sender_name else None
        row = self.find(channel_type, group_id, sender_id)
        if row is not None:
            self._refresh(row, name, now)
            return
        self._prune_expired(channel_type, now)
        if self._over_cap(channel_type, sender_id, now):
            log.warning(
                "channel %s: pending sender requests are at their cap; a new request was dropped",
                channel_type,
            )
            return
        self.session.add(
            ChannelSender(
                channel_type=channel_type,
                external_group_id=group_id,
                external_sender_id=sender_id,
                is_group=is_group,
                status=SENDER_PENDING,
                display_name=name,
                last_seen_at=now,
                created_at=now,
            )
        )
        try:
            self.session.commit()
        except IntegrityError:
            # Another thread of the same chat wrote the first row meanwhile.
            self.session.rollback()
            row = self.find(channel_type, group_id, sender_id)
            if row is not None:
                self._refresh(row, name, now)

    def _refresh(self, row: ChannelSender, name: str | None, now: datetime) -> None:
        """Touch a known sender's last-seen time and name; the status is left alone."""
        row.last_seen_at = now
        if name:
            row.display_name = name
        self.session.add(row)
        self.session.commit()

    def _pending(self, channel_type: str, now: datetime) -> list[ChannelSender]:
        """Unexpired pending requests on this channel type."""
        rows = self.session.exec(
            self.session.select(ChannelSender).where(
                ChannelSender.channel_type == channel_type,
                ChannelSender.status == SENDER_PENDING,
            )
        ).all()
        return [row for row in rows if not _expired(row, now)]

    def _over_cap(self, channel_type: str, sender_id: str, now: datetime) -> bool:
        """Whether one more pending request would pass the channel or sender cap."""
        pending = self._pending(channel_type, now)
        if len(pending) >= PENDING_PER_CHANNEL_MAX:
            return True
        return sum(1 for row in pending if row.external_sender_id == sender_id) >= PENDING_PER_SENDER_MAX

    def _prune_expired(self, channel_type: str, now: datetime) -> None:
        """Delete pending requests created more than PENDING_TTL ago."""
        rows = self.session.exec(
            self.session.select(ChannelSender).where(
                ChannelSender.channel_type == channel_type,
                ChannelSender.status == SENDER_PENDING,
            )
        ).all()
        stale = [row for row in rows if _expired(row, now)]
        for row in stale:
            self.session.delete(row)
        if stale:
            self.session.commit()


def _aware(value: datetime | None) -> datetime | None:
    """A stored time as UTC-aware; SQLite returns it naive."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _expired(row: ChannelSender, now: datetime) -> bool:
    """Whether a pending request is older than PENDING_TTL, by its creation time."""
    created = _aware(row.created_at)
    return created is not None and now - created > PENDING_TTL
