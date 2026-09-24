"""Artifact capabilities derived from each artifact's recorded owner."""
from __future__ import annotations

from fastapi import HTTPException, status

from cowork.principal import Principal, can_manage_org

_OWNER_ONLY = "Only the artifact owner can change this draft"
_OWNER_UNKNOWN = "Artifact owner is unknown"


def artifact_capabilities(session, source, slug: str, *, resolution=None) -> dict:
    """Return the permissions the API will enforce for this artifact.

    Desktop is a single-user boundary. In organization mode the owner is a
    property of each ARTIFACT, not of its root: a project-level root is shared
    by every member of the project (ENG-2056), so the owner is the artifact's
    recorded attribution (legacy per-conversation roots: the conversation's
    creator). Project visibility grants review, never source mutation. An
    artifact nobody is recorded as owning says so with ``ownerUnknown``.
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
    if resolution is None:
        from cowork.services.artifact_ownership import resolve_artifact_owner

        resolution = resolve_artifact_owner(session, source, slug)
    owner_id = resolution.owner_user_id
    is_owner = bool(owner_id and scope.user_id and str(owner_id) == str(scope.user_id))
    capabilities = {
        "role": "owner" if is_owner else "reviewer",
        "canPreview": True,
        "canComment": True,
        "canEdit": is_owner,
        "canAddressWithAgent": is_owner,
        "canResolveComments": is_owner,
    }
    if resolution.unknown:
        capabilities["ownerUnknown"] = True
    return capabilities


def require_artifact_owner(session, source, slug: str, *, resolution=None) -> dict:
    capabilities = artifact_capabilities(session, source, slug, resolution=resolution)
    if not capabilities["canEdit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_OWNER_UNKNOWN if capabilities.get("ownerUnknown") else _OWNER_ONLY,
        )
    return capabilities


def may_delete_ownerless_artifact(
    session, source, slug: str, principal, *, resolution=None
) -> bool:
    """D7: an org admin may delete an artifact whose owner is unknown.

    The only thing `unknown` grants anyone. The principal must be the request's
    own (same user and org as the scope), and `can_manage_org` alone is not
    enough: without a principal there is no admin exception at all.

    ``resolution`` lets a caller that also checks ownership resolve once.
    """
    scope = getattr(session, "scope", None)
    if not scope or not scope.org_mode or not isinstance(principal, Principal):
        return False
    if principal.user_id != scope.user_id or principal.org_id != scope.org_id:
        return False
    if not can_manage_org(principal):
        return False
    if resolution is None:
        from cowork.services.artifact_ownership import resolve_artifact_owner

        resolution = resolve_artifact_owner(session, source, slug)
    return resolution.unknown
