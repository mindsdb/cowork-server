from __future__ import annotations

from cowork.db.scoped import ScopedSession
from cowork.models.pin import Pin


class PinService:
    """Pins are personal: the org filter comes from the scoped session, the
    user filter is applied here (user_id has no automatic scoping)."""

    def __init__(self, session: ScopedSession) -> None:
        self.session = session

    def _own_pins(self):
        stmt = self.session.select(Pin)
        if self.session.scope.org_mode:
            stmt = stmt.where(Pin.user_id == self.session.scope.user_id)
        return stmt

    def list_pins(self) -> list[Pin]:
        return list(
            self.session.exec(self._own_pins().order_by(Pin.created_at.desc())).all()
        )

    def _find(self, item_type: str, item_id: str) -> Pin | None:
        return self.session.exec(
            self._own_pins().where(Pin.item_type == item_type, Pin.item_id == item_id)
        ).first()

    def pin_item(self, item_type: str, item_id: str, title: str | None = None) -> Pin:
        existing = self._find(item_type, item_id)
        if existing:
            if title is not None:
                existing.title = title
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            return existing
        pin = Pin(
            item_type=item_type,
            item_id=item_id,
            title=title or item_id,
            user_id=self.session.scope.user_id,
        )
        self.session.add(pin)
        self.session.commit()
        self.session.refresh(pin)
        return pin

    def unpin_item(self, item_type: str, item_id: str) -> bool:
        pin = self._find(item_type, item_id)
        if not pin:
            return False
        self.session.delete(pin)
        self.session.commit()
        return True
