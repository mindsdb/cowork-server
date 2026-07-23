from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from cowork.schemas.base import CamelRequest, CamelResponse


class ApprovalKind(str, Enum):
    action = "action"
    auth = "auth"


class ApprovalStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    edited = "edited"
    skipped = "skipped"
    expired = "expired"


# Versioned action descriptors — discriminated union on `kind`, version: 1.
# Later phases add kinds (auth is here; schedule, rule linkage next); never
# bolt fields onto an existing v1 schema without bumping `version`.
class ActionDescriptorV1(BaseModel):
    """An executable action: the harness runs tool+args verbatim on approve."""

    version: Literal[1] = 1
    kind: Literal["action"] = "action"
    tool: str = Field(description="Tool to execute on approve (e.g. browser_click)")
    args: dict[str, Any] = Field(default_factory=dict, description="Exact tool arguments approved")
    summary: str = Field(default="", description="One-line human description of the action")


class AuthDescriptorV1(BaseModel):
    """An auth wall the agent can't cross — the card hands the tab to the human."""

    version: Literal[1] = 1
    kind: Literal["auth"] = "auth"
    app_name: str = Field(description="Display name of the app needing sign-in (e.g. Gmail)")
    tab_id: str | None = Field(default=None, description="Browser tab to focus for the user")
    reason: str = Field(default="", description="Why the agent hit the wall")


ApprovalDescriptor = Annotated[ActionDescriptorV1 | AuthDescriptorV1, Field(discriminator="kind")]

_DESCRIPTOR_ADAPTER = TypeAdapter(ApprovalDescriptor)


def parse_descriptor(data: dict[str, Any]) -> ActionDescriptorV1 | AuthDescriptorV1:
    """Validate a stored descriptor dict back into the versioned union."""
    return _DESCRIPTOR_ADAPTER.validate_python(data)


class ApprovalCreateRequest(CamelRequest):
    conversation_id: UUID
    kind: ApprovalKind
    descriptor: ApprovalDescriptor
    draft: str = ""
    ttl_seconds: int = 259200


class ApprovalResolveRequest(CamelRequest):
    resolution: Literal["approved", "edited", "skipped"]
    edited_draft: str | None = None


class ApprovalResponse(CamelResponse):
    id: UUID
    conversation_id: UUID
    kind: str
    status: str
    action_descriptor: dict[str, Any]
    draft: str
    receipt: dict[str, Any] | None
    ttl_seconds: int
    expires_at: datetime
    resolved_at: datetime | None
    created_at: datetime
