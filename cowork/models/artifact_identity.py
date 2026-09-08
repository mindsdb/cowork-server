"""Server-owned aliases for agent-writable local artifact identities."""

from sqlalchemy import UniqueConstraint
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class ArtifactIdentity(BaseSQLModel, table=True):
    __tablename__ = "artifact_identities"
    __table_args__ = (
        UniqueConstraint(
            "org_id", "local_artifact_id", name="uq_artifact_identity_org_local"
        ),
    )

    org_id: str = Field(index=True, max_length=36)
    owner_keycloak_id: str = Field(max_length=36)
    local_artifact_id: str = Field(max_length=32)
    # NULL while an allocation is pending. The row's random primary key is
    # auth's idempotency key, persisted BEFORE contacting auth. Neither field
    # is read from the artifact's mutable metadata or publication sidecar.
    canonical_artifact_id: str | None = Field(default=None, max_length=64)

    # Intentionally no cascading artifact/conversation FK: deleting files must
    # never release a name for a different owner, or detach its comment history.
