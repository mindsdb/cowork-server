"""Channel event log — inbound/outbound audit + inbound de-duplication.

"""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.models.channel import ChannelEvent


class ChannelEventService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def is_duplicate_inbound(self, channel_type: str, dedupe_key: str | None) -> bool:
        if not dedupe_key:
            return False
        row = self.session.exec(
            select(ChannelEvent).where(
                ChannelEvent.channel_type == channel_type,
                ChannelEvent.dedupe_key == dedupe_key,
                ChannelEvent.direction == "inbound",
            )
        ).first()
        return row is not None

    def record_inbound(
        self,
        channel_type: str,
        *,
        dedupe_key: str | None,
        external_message_id: str | None = None,
        status: str = "received",
    ) -> UUID | None:
        event = ChannelEvent(
            channel_type=channel_type,
            direction="inbound",
            status=status,
            dedupe_key=dedupe_key,
            external_message_id=external_message_id,
        )
        self.session.add(event)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            return None
        self.session.refresh(event)
        return event.id

    def set_status(self, event_id: UUID, status: str, *, error: str | None = None) -> None:
        event = self.session.get(ChannelEvent, event_id)
        if event is None:
            return
        event.status = status
        if error is not None:
            event.error = error
        self.session.add(event)
        self.session.commit()
