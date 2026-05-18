from __future__ import annotations

from sqlmodel import Session, select

from cowork.models.pin import Pin


class PinService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_pins(self) -> list[Pin]:
        return list(self.session.exec(select(Pin).order_by(Pin.created_at.desc())).all())

    def pin_item(self, item_type: str, item_id: str, title: str | None = None) -> Pin:
        existing = self.session.exec(
            select(Pin).where(Pin.item_type == item_type, Pin.item_id == item_id)
        ).first()
        if existing:
            if title is not None:
                existing.title = title
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            return existing
        pin = Pin(item_type=item_type, item_id=item_id, title=title or item_id)
        self.session.add(pin)
        self.session.commit()
        self.session.refresh(pin)
        return pin

    def unpin_item(self, item_type: str, item_id: str) -> bool:
        pin = self.session.exec(
            select(Pin).where(Pin.item_type == item_type, Pin.item_id == item_id)
        ).first()
        if not pin:
            return False
        self.session.delete(pin)
        self.session.commit()
        return True
