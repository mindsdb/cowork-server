from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field

from .base import BaseSQLModel


class PushDevice(BaseSQLModel, table=True):
    """One phone the mobile app registered for push (cowork.services.push).

    A token names one app install on one device, so it is unique across the
    whole table: when a different person signs in on that phone, their
    registration moves the row to them. Org mode only; the desktop never
    writes it.
    """

    __tablename__ = "push_devices"

    org_id: str = Field(max_length=36, description="Organization the person registered under")
    user_id: str = Field(max_length=36, description="Person the device delivers to")
    platform: str = Field(max_length=16, description="ios | android")
    token: str = Field(max_length=512, description="APNs device token or FCM registration token")
    environment: str | None = Field(
        default=None, max_length=16,
        description="APNs environment: production | sandbox. NULL means production first.",
    )
    app_version: str | None = Field(default=None, max_length=64)
    last_seen_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the app last registered this token",
    )

    __table_args__ = (
        UniqueConstraint("token", name="uq_push_devices_token"),
        Index("ix_push_devices_org_id_user_id", "org_id", "user_id"),
    )
