"""Live apps' server-written records (cowork.services.live_apps).

None of them carries an ``org_id``: a launch, a share and a copy each name two
people's spaces or none, so the service filters on the partition columns
itself and the scoped session's org filter never applies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel

# The longest address SMTP carries (RFC 5321's 256-octet path less its angle
# brackets, RFC 3696's erratum 1690), and the longest anton's LiveAppLauncher
# takes.
EMAIL_MAX_LENGTH = 254


#: Why a launch stopped: the slot's report (idle, max_time, app_exited,
#: terminated), anton's claim of the slot (replaced, relaunched, or
#: launch_failed when the launch that claimed it didn't finish), a slot
#: found no longer running it with no report (lost), or a move of its
#: conversation to another project, which anton no longer finds it under
#: (moved).
StopReason = Literal[
    "idle", "max_time", "app_exited", "terminated", "replaced", "relaunched", "launch_failed", "lost", "moved"
]

#: The stop reasons a person is shown: a relaunch keeps the app, so it is
#: never shown as a stop.
ShownStopReason = Literal[
    "idle", "max_time", "app_exited", "terminated", "replaced", "launch_failed", "lost", "moved"
]


class LiveAppLaunchRecord(BaseSQLModel, table=True):
    """One launch of an app into a slot: running until it stops.

    anton's claim of a slot stops its running record before the launch, and
    the launch, once the slot accepts it, starts a new one (CoworkLiveAppHost),
    so a refused launch leaves nobody able to open the slot. The access route
    reads the slot's running record, never the request, for the owner; the
    stopped page, the card and the turn read the rest.
    """

    __tablename__ = "live_app_launches"
    __table_args__ = (
        # At most one running record per slot.
        sa.Index(
            "uq_live_app_launches_running_slot",
            "slot",
            unique=True,
            sqlite_where=sa.text("state = 'running'"),
            postgresql_where=sa.text("state = 'running'"),
        ),
        sa.Index("ix_live_app_launches_launcher", "launcher_partition_id", "launcher_user_id"),
    )

    slot: int
    launcher_user_id: str = Field(max_length=36)
    launcher_partition_id: str = Field(max_length=36)
    launcher_email: str = Field(max_length=EMAIL_MAX_LENGTH)
    project_id: UUID
    slug: str = Field(max_length=255)
    artifact_dir: str = Field(max_length=4096)
    address: str = Field(max_length=512)
    boot_id: str = Field(max_length=64)
    started_at: datetime = Field(sa_type=sa.DateTime(timezone=True))  # type: ignore
    state: Literal["running", "stopped"] = Field(default="running", sa_type=sa.String(16))  # type: ignore
    stopped_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True))  # type: ignore
    stop_reason: StopReason | None = Field(default=None, sa_type=sa.String(16))  # type: ignore
    # The idle or maximum hours the slot reported with an idle or max_time stop.
    limit_hours: float | None = None
    # Whether the launcher's turn has been told of the stop (turn_live_apps).
    stop_noticed: bool = False


class LiveAppShare(BaseSQLModel, table=True):
    """One person's app, shared with one colleague's email address, or with
    everyone on the deployment when ``recipient_email`` is None.

    ``recipient_email`` is stored lowercased; ``owner_email`` is the owner's
    verified address when they shared, the one name of another member the
    recipient is shown.
    """

    __tablename__ = "live_app_shares"
    __table_args__ = (
        sa.UniqueConstraint(
            "owner_partition_id",
            "project_id",
            "slug",
            "recipient_email",
            name="uq_live_app_shares_app_recipient",
        ),
        # A unique constraint lets NULLs repeat, so one share with everyone
        # per app has an index of its own.
        sa.Index(
            "uq_live_app_shares_app_everyone",
            "owner_partition_id",
            "project_id",
            "slug",
            unique=True,
            sqlite_where=sa.text("recipient_email IS NULL"),
            postgresql_where=sa.text("recipient_email IS NULL"),
        ),
        sa.Index("ix_live_app_shares_recipient_email", "recipient_email"),
    )

    owner_partition_id: str = Field(max_length=36)
    owner_user_id: str = Field(max_length=36)
    owner_email: str = Field(max_length=EMAIL_MAX_LENGTH)
    project_id: UUID
    slug: str = Field(max_length=255)
    recipient_email: str | None = Field(default=None, max_length=EMAIL_MAX_LENGTH)


class LiveAppCopy(BaseSQLModel, table=True):
    """An app a person duplicated from a share into a project of their own.

    The card's "received from" line reads it, never the copy's
    ``metadata.json``, which the person's agent can rewrite.
    """

    __tablename__ = "live_app_copies"
    __table_args__ = (
        sa.UniqueConstraint("partition_id", "project_id", "slug", name="uq_live_app_copies_artifact"),
    )

    partition_id: str = Field(max_length=36)
    project_id: UUID
    slug: str = Field(max_length=255)
    owner_email: str = Field(max_length=EMAIL_MAX_LENGTH)
    share_id: UUID
