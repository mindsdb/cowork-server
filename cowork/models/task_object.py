from uuid import UUID

from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class TaskObject(BaseSQLModel, table=True):
    """Index of the objects a task (conversation) owns — its created
    artifacts and attached files — so a task can be moved to another
    project together with everything it produced.

    This table indexes what a task CREATED so the task can be moved together
    with its work. It is not an authorization record: in organization mode the
    owner of an artifact is its `shared_resource_attributions` row (ENG-2961),
    and legacy per-conversation roots derive it from the conversation. Rows are
    written at the end of the turn that created the artifact; on the desktop
    they are also reconciled from `metadata.json` provenance at move time.
    `project_id` is denormalized so a move can both look up a task's objects
    and keep their project pointer correct.
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
