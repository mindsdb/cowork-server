from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from cowork.coding.control_models import RUNTIME_PROTOCOL_VERSION

ConnectorAction = Literal["read_source", "search_work", "pull_request_status"]


class ConnectorCapabilityIssueRequest(BaseModel):
    provider: Literal["github", "linear"]
    connection_name: str = Field(min_length=1, max_length=512)
    actions: list[ConnectorAction] = Field(min_length=1, max_length=8)
    resource_constraints: dict[str, str] = Field(default_factory=dict, max_length=16)
    expires_in_seconds: int = Field(default=900, ge=60, le=3_600)


class ConnectorCapability(BaseModel):
    id: str
    provider: Literal["github", "linear"]
    token: str
    actions: list[str]
    resource_constraints: dict[str, str]
    expires_at: datetime


class ConnectorInvocationRequest(BaseModel):
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    grant_token: str = Field(min_length=32, max_length=512)
    action: ConnectorAction
    payload: dict[str, object] = Field(default_factory=dict)
