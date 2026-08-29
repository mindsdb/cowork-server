"""Artifact capabilities derived from the scoped project and owning task."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, status

from cowork.models.conversation import Conversation


def artifact_owner_id(session, source):
    """Resolve the creating user from the conversation-scoped artifact root."""
    scope = getattr(session, "scope", None)
    if not scope or not scope.org_mode:
        return getattr(scope, "user_id", None)
    try:
        conversation_id = UUID(Path(source.base).parent.parent.name)
        conversation = session.get(Conversation, conversation_id)
        if conversation is not None and str(conversation.project_id) == str(source.project_id):
            return conversation.created_by
    except (OSError, ValueError, TypeError):
        pass
    return None


def artifact_capabilities(session, source) -> dict:
    """Return the permissions the API will enforce for this artifact.

    Desktop is a single-user boundary. In organization mode, artifact bytes live
    below the creating conversation directory; that conversation's ``created_by``
    is the owner. Project visibility grants review, never source mutation.
    """
    scope = getattr(session, "scope", None)
    if not scope or not scope.org_mode:
        return {
            "role": "owner",
            "canPreview": True,
            "canComment": True,
            "canEdit": True,
            "canAddressWithAgent": True,
            "canResolveComments": True,
        }

    owner_id = artifact_owner_id(session, source)
    is_owner = bool(owner_id and scope.user_id and str(owner_id) == str(scope.user_id))
    return {
        "role": "owner" if is_owner else "reviewer",
        "canPreview": True,
        "canComment": True,
        "canEdit": is_owner,
        "canAddressWithAgent": is_owner,
        "canResolveComments": is_owner,
    }


def require_artifact_owner(session, source) -> dict:
    capabilities = artifact_capabilities(session, source)
    if not capabilities["canEdit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the artifact owner can change this draft",
        )
    return capabilities
