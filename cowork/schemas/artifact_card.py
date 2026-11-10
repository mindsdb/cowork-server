"""An artifact card: what the artifacts list, an inline chat card and a live
app copy send for one artifact folder. cowork.services.artifacts
._prepare_artifact_card builds it.

Fields are declared in the order the card's keys go out, and `to_json` sends
only the fields that were set, so a key a card leaves out (the access
password and emails in org mode, `draftUrl` without a primary, `capabilities`
on an inline card) stays out. A route or the stream must send `to_json()`,
never the model: FastAPI would send every unset field with its default.

Each default is the value a consumer reading the card takes for a missing key.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cowork.models.artifact_owner_choice import AccessMode


class ArtifactCard(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, validate_assignment=True)

    id: str
    slug: str
    title: str
    description: str
    type: str
    kind: str
    ext: str
    updated: str
    mtime: int
    live: bool
    bg: str
    file_count: int
    folder: str
    path: str
    primary: str | None
    project_id: str | None
    project_name: str
    origin_conversation_id: str
    published_url: str = ""
    modified: bool = False
    # The access fields, in ArtifactAccessView's order and types.
    access_emails: list[str] | None = Field(default_factory=list)
    org_allowed: bool | None = False
    owner_only: bool | None = False
    allowed_team_ids: list[str] | None = Field(default_factory=list)
    allowed_user_ids: list[str] | None = Field(default_factory=list)
    access_mode: AccessMode = AccessMode.PUBLIC
    access_protected: bool = False
    access_password: str = ""
    artifact_key: str = ""
    serve_url: str
    draft_url: str = ""
    capabilities: dict | None = None
    received_from: dict | None = None

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_unset=True)
