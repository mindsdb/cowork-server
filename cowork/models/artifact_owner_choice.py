"""The sharing an artifact's owner last chose, kept where an agent can't write.

An artifact's .published.json entry says its mode and whether it is published,
but an agent turn can rewrite that file: the pod mounts the artifact folder
writable. This row says the same two things, written only when the owner saves
the artifact's sharing themselves, so a re-publish that isn't the owner's is
checked against it and never goes wider (cowork.services.artifact_owner_choice,
read by cowork.services.publish.publish_artifact).

One row per artifact per organization, keyed by the artifact's authorization
key: the key its MindsHub rule is stored under and its publish carries. The
org and owner are UUID columns. MindsHub mode only.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel

# The column's bound: an authorization key is "artifact/<uuid>", and the
# identity alias that issues it stores it in as many characters
# (cowork.models.artifact_identity.ArtifactIdentity.canonical_artifact_id).
ARTIFACT_KEY_MAX_CHARS = 64


class AccessMode(StrEnum):
    """Who a published artifact's link opens for, narrowest first: the people
    its restricted rule names, anyone with the password, or anyone."""

    RESTRICTED = "restricted"
    PASSWORD = "password"
    PUBLIC = "public"


class ArtifactOwnerChoice(BaseSQLModel, table=True):
    __tablename__ = "artifact_owner_choices"
    __table_args__ = (
        sa.UniqueConstraint("org_id", "artifact_key", name="uq_artifact_owner_choice_org_key"),
    )

    org_id: UUID = Field(index=True, description="Owning organization")
    artifact_key: str = Field(
        max_length=ARTIFACT_KEY_MAX_CHARS, description="The artifact's authorization key"
    )
    # An AccessMode value.
    access_mode: str = Field(max_length=16, description="The mode the owner chose")
    published: bool = Field(description="False when the owner's artifact is unpublished")
    owner_user_id: UUID = Field(description="User id of the owner who chose it")
