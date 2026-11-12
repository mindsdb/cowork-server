"""The phones each person registered for push: database units.

Every function here is a unit for ``cowork.db.units.run_db`` and is run under
SYSTEM_SCOPE, with the organization and person as explicit filters. A device
token is one app install on one phone, not an organization's row: when
someone else signs in on that phone, or the same person switches
organization, the next registration has to find and move the row wherever it
is, and an org-scoped session would not see it there. Every read and delete a
person's request reaches is still bounded by their own org_id and user_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from cowork.db.scoped import ScopedSession
from cowork.models.push_device import PushDevice

Platform = Literal["ios", "android"]
Environment = Literal["production", "sandbox"]


@dataclass(frozen=True)
class Device:
    """A registered device as the senders need it, detached from any session."""

    platform: str
    token: str
    environment: str | None


def register_device(
    session: ScopedSession, *, org_id: str, user_id: str, token: str, platform: Platform,
    app_version: str | None, environment: Environment | None,
) -> None:
    """Record ``token`` as this person's, in this organization. A token
    already on file, whoever it was registered to, moves to them."""
    now = datetime.now(timezone.utc)
    row = session.exec(session.select(PushDevice).where(PushDevice.token == token)).first()
    if row is None:
        row = PushDevice(org_id=org_id, user_id=user_id, platform=platform, token=token)
    row.org_id = org_id
    row.user_id = user_id
    row.platform = platform
    row.app_version = app_version
    row.environment = environment
    row.last_seen_at = now
    session.add(row)


def unregister_device(session: ScopedSession, *, org_id: str, user_id: str, token: str) -> bool:
    """Delete ``token`` when it is this person's in this organization.
    Returns whether a row went; someone else's token is left alone."""
    row = session.exec(
        session.select(PushDevice)
        .where(PushDevice.token == token)
        .where(PushDevice.org_id == org_id)
        .where(PushDevice.user_id == user_id)
    ).first()
    if row is None:
        return False
    session.delete(row)
    return True


def devices_for(session: ScopedSession, *, org_id: str, user_id: str) -> list[Device]:
    rows = session.exec(
        session.select(PushDevice)
        .where(PushDevice.org_id == org_id)
        .where(PushDevice.user_id == user_id)
    ).all()
    return [Device(platform=row.platform, token=row.token, environment=row.environment) for row in rows]


def forget_token(session: ScopedSession, *, token: str) -> None:
    """Delete a token APNs or FCM said no longer reaches an app."""
    row = session.exec(session.select(PushDevice).where(PushDevice.token == token)).first()
    if row is not None:
        session.delete(row)


def set_environment(session: ScopedSession, *, token: str, environment: Environment) -> None:
    """Remember which APNs environment a token turned out to belong to."""
    row = session.exec(session.select(PushDevice).where(PushDevice.token == token)).first()
    if row is not None and row.environment != environment:
        row.environment = environment
        session.add(row)
