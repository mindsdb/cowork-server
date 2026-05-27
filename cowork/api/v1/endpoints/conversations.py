from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.conversations import ConversationCreateRequest, ConversationListItem, ConversationUpdateRequest
from cowork.services.conversations import ConversationService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_conversations(
    session: SessionDep,
    project_id: UUID | None = None,
    project: str | None = None,
    limit: int = 50,
):
    all_projects = project == "all"
    convs = ConversationService(session).list_conversations(
        project_id=project_id, limit=limit, all_projects=all_projects,
    )
    return {"conversations": [
        ConversationListItem.serialize({
            "id": c.id,
            "title": c.topic,
            "preview": c.topic,
            "updated_at": c.modified_at or c.created_at,
            "created_at": c.created_at,
            "project": None,
            "project_id": c.project_id,
        })
        for c in convs
    ]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreateRequest, session: SessionDep):
    return ConversationService(session).create_conversation(
        topic=body.topic, project_id=body.project_id
    )


@router.get("/{conversation_id}")
def get_conversation(conversation_id: UUID, session: SessionDep):
    try:
        return ConversationService(session).get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: UUID, body: ConversationUpdateRequest, session: SessionDep):
    try:
        return ConversationService(session).update_conversation(
            conversation_id, topic=body.topic, project_id=body.project_id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


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
