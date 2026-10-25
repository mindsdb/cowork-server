"""Request and response bodies of the live-app share routes (cowork.api.v1.endpoints.artifacts)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, StrictBool

from cowork.models.live_app import ShownStopReason
from cowork.schemas.base import CamelRequest, CamelResponse


class LiveAppShareRequest(CamelRequest):
    """Share one of the caller's apps with one address, or with everyone."""

    project_id: UUID
    slug: str = Field(min_length=1, max_length=255)
    # Room for any address a client sends, so the route answers one past
    # EMAIL_MAX_LENGTH with its own sentence rather than a 422.
    email: str | None = Field(default=None, max_length=320)
    # JSON true only: "yes" or 1 is no choice of everyone.
    everyone: StrictBool = False


class LiveAppShareResponse(CamelResponse):
    share_id: UUID
    slug: str
    # None for a share with everyone.
    email: str | None
    everyone: bool
    shared_at: datetime


class LiveAppCopyRef(CamelResponse):
    """Where the caller's copy of a shared app lives."""

    project_id: UUID
    slug: str


class SharedLiveAppResponse(CamelResponse):
    """One app someone let the caller duplicate, as Cowork lists it."""

    share_id: UUID
    slug: str
    name: str
    description: str
    owner_email: str
    everyone: bool
    shared_at: datetime
    # The caller's copy from this share while its folder is there: one copy
    # per person per share, so the list offers it instead of Duplicate.
    your_copy: LiveAppCopyRef | None = None


class LiveAppDuplicateRequest(CamelRequest):
    share_id: UUID
    # The caller's own project the copy lands in.
    project_id: UUID
    # The caller confirmed the disclaimer: the app was built by its owner, runs
    # with the caller's connections, and is theirs to review first. JSON true
    # only.
    confirmed: StrictBool = False


class LiveAppStateResponse(CamelResponse):
    """The latest launch of one of the caller's apps, for its card's state line."""

    project_id: UUID
    slug: str
    slot: int
    launch_id: UUID
    state: Literal["running", "stopped"]
    address: str
    started_at: datetime
    stopped_at: datetime | None
    stop_reason: ShownStopReason | None
    stop_reason_text: str | None


class LiveAppPreviewRequest(CamelRequest):
    launch_id: UUID


class LiveAppPreviewToken(CamelRequest):
    token: str = Field(pattern=r"^[0-9a-f]{64}$")


class LiveAppPreviewResponse(CamelResponse):
    id: UUID
    intent: str


class LiveAppPreviewGrant(CamelResponse):
    grant: str
