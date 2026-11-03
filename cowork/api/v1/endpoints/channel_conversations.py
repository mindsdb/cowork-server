"""Org admins reading channel conversations: the ones the channel runtime
created for a route, which have no owner and so no `/conversations` access.

Read and delete only. Posting, renaming and moving stay on the owner-only
`/conversations` routes, which never admit these rows. Org mode only: outside
it the admin check admits everyone, so every route here answers 404.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cowork.api.v1.permissions import AuthenticatedOrgAdmin, require
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSessionDep
from cowork.models.conversation import Conversation
from cowork.schemas.conversations import ChannelConversationItem
from cowork.services.conversations import ConversationService, InvalidPaginationParams


def _org_mode_only() -> None:
    """404 outside org mode, before the admin check that would admit anyone there."""
    if get_app_settings().tenancy_mode != "org":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


router = APIRouter(dependencies=[Depends(_org_mode_only), Depends(require(AuthenticatedOrgAdmin))])


def _serialize(conversation: Conversation, updated_at: datetime | None = None) -> dict:
    """One list item; `updated_at` is the last-activity time, as the conversation list uses it."""
    return ChannelConversationItem.serialize({
        "id": conversation.id,
        "title": conversation.topic,
        "channel_type": conversation.channel_type,
        "created_at": conversation.created_at,
        "updated_at": updated_at or conversation.created_at,
    })


@router.get("/")
def list_channel_conversations(scoped: ScopedSessionDep, limit: int = Query(50, ge=1, le=200)):
    """The org's channel conversations, most recently active first."""
    rows = ConversationService(scoped).list_channel_conversations(limit=limit)
    return {"conversations": [_serialize(c, updated_at=activity) for c, activity in rows]}


@router.get("/{conversation_id}")
def get_channel_conversation(conversation_id: UUID, scoped: ScopedSessionDep):
    """One channel conversation, or 404 if it is not one in the caller's org."""
    svc = ConversationService(scoped)
    conversation = svc.channel_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return _serialize(conversation, updated_at=svc.last_message_at(conversation_id))


@router.get("/{conversation_id}/items")
def get_channel_conversation_items(
    conversation_id: UUID,
    scoped: ScopedSessionDep,
    limit: int | None = None,
    before: str | None = None,
):
    """History in the same two modes as `/conversations/{id}/items`: a bare
    list with neither `limit` nor `before`, else the paginated envelope."""
    svc = ConversationService(scoped)
    try:
        if limit is None and before is None:
            return svc.channel_messages(conversation_id)
        page_kwargs = {"before": before}
        if limit is not None:
            page_kwargs["limit"] = limit
        page = svc.channel_messages_page(conversation_id, **page_kwargs)
    except InvalidPaginationParams as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return page.model_dump(by_alias=True)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_channel_conversation(conversation_id: UUID, scoped: ScopedSessionDep) -> None:
    """Delete a channel conversation; its route stays and starts a new one on the next message."""
    if not ConversationService(scoped).delete_channel_conversation(conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
