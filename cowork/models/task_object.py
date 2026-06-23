from uuid import UUID

from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class TaskObject(BaseSQLModel, table=True):
    """Index of the objects a task (conversation) owns — its created
    artifacts and attached files — so a task can be moved to another
    project together with everything it produced.

    The authoritative source for an artifact's owner is its on-disk
    `metadata.json` provenance (written by the shared ArtifactStore for
    every harness), and for a file it's `files.purpose`. This table is a
    fast, durable index over those: populated when an artifact is claimed
    via `create_artifact`, and reconciled from provenance at move time so
    artifacts created by any harness (or before this table existed) are
    still attributable. `project_id` is denormalized so a move can both
    look up a task's objects and keep their project pointer correct.
    """

    __tablename__ = "task_objects"

    conversation_id: UUID = Field(
        foreign_key="conversations.id",
        index=True,
        description="The task (conversation) that owns this object.",
    )
    project_id: UUID = Field(
        foreign_key="projects.id",
        index=True,
        description="Project the object currently lives in (kept in sync on move).",
    )
    kind: str = Field(max_length=16, description="'artifact' or 'file'.")
    # For an artifact this is its slug (folder name under the project's
    # `.anton/artifacts/`); for a file it is the File row's UUID (as text).
    ref: str = Field(max_length=255, index=True, description="Artifact slug or file id.")
