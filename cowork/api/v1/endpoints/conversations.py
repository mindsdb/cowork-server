from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.models.project import Project
from cowork.schemas.conversations import (
    ConversationCreateRequest,
    ConversationListItem,
    ConversationMoveRequest,
    ConversationUpdateRequest,
)
from cowork.services.conversations import ConversationService
from cowork.services.task_objects import TaskObjectService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _serialize_conversation(c):
    return ConversationListItem.serialize({
        "id": c.id,
        "title": c.topic,
        "preview": c.topic,
        "updated_at": c.modified_at or c.created_at,
        "created_at": c.created_at,
        "project": c.project.name if c.project else None,
        "project_path": c.project.path if c.project else None,
        "project_id": c.project_id,
    })


@router.get("/")
def list_conversations(
    session: SessionDep,
    project_id: UUID | None = None,
    project: str | None = None,
    limit: int = 50,
):
    all_projects = project == "all"
    resolved_project_id = project_id
    if not all_projects and resolved_project_id is None and project:
        from cowork.services.projects import ProjectService
        proj = ProjectService(session).get_project_by_name_or_none(project)
        if proj is not None:
            resolved_project_id = proj.id
    convs = ConversationService(session).list_conversations(
        project_id=resolved_project_id, limit=limit, all_projects=all_projects,
    )
    return {"conversations": [_serialize_conversation(c) for c in convs]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreateRequest, session: SessionDep):
    svc = ConversationService(session)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    conversation = svc.create_conversation(
        topic=body.topic or body.title or "Untitled task", project_id=project_id
    )
    return _serialize_conversation(conversation)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: UUID, session: SessionDep):
    try:
        return _serialize_conversation(ConversationService(session).get_conversation(conversation_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: UUID, body: ConversationUpdateRequest, session: SessionDep):
    svc = ConversationService(session)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    try:
        conversation = svc.update_conversation(
            conversation_id, topic=body.topic or body.title, project_id=project_id
        )
        return _serialize_conversation(conversation)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{conversation_id}/move")
def move_conversation(conversation_id: UUID, body: ConversationMoveRequest, session: SessionDep):
    """Move a task to another project. With `move_objects` (default), the
    artifacts the task created are relocated into the destination project
    and its attachment files are re-tagged; otherwise only the task's
    project pointer changes. The destination project must already exist
    (the client creates a new one first, then moves to its id)."""
    svc = ConversationService(session)
    try:
        conversation = svc.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    dest = None
    if body.project_id is not None:
        dest = session.get(Project, body.project_id)
    elif body.project:
        dest = svc.project_by_name(body.project)
    if dest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Destination project not found")

    source = conversation.project
    if body.move_objects and source is not None and dest.id != source.id:
        TaskObjectService(session).relocate_to_project(conversation, source, dest)

    conversation = svc.update_conversation(conversation_id, project_id=dest.id)
    return _serialize_conversation(conversation)


@router.get("/{conversation_id}/items")
def get_messages(conversation_id: UUID, session: SessionDep):
    try:
        return ConversationService(session).get_messages(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: UUID, session: SessionDep):
    found = ConversationService(session).delete_conversation(conversation_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"ok": True}


@router.delete("/{conversation_id}/turns/{turn_index}")
def delete_conversation_turn(conversation_id: UUID, turn_index: int, session: SessionDep):
    """Delete a turn (user+assistant exchange) and everything after it.

    turn_index is the 0-based index counting only assistant messages.
    """
    svc = ConversationService(session)
    try:
        deleted = svc.delete_turn(conversation_id, turn_index)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"ok": True, "deleted": deleted}
