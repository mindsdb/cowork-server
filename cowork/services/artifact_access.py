"""Provision the auth rule that lets an organization review a private draft."""
from __future__ import annotations

from uuid import UUID

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.services.artifact_identity import artifact_key


class ArtifactAccessUnavailable(RuntimeError):
    pass


def draft_review_access_key(artifact_id: str) -> str:
    """Internal auth rule for draft collaborators, separate from sharing.

    Comments remain keyed by ``artifact/<uuid>``. This second key exists only
    inside the access service so enabling same-org draft review can never
    broaden an email-only or owner-only published link.
    """
    return f"artifact-draft/{UUID(artifact_id)}"


async def provision_draft_review_access(
    artifact_id: str,
    scope: TenantScope,
    *,
    owner_user_id: str | None = None,
    settings: TurnQueueSettings | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Upsert owner + same-org review access and return the canonical key.

    Only org mode has a server-established user and organization identity.
    Desktop keeps using the rule created by a restricted publish until a
    signed desktop ownership claim is available; it must never fabricate ids.
    """
    key = artifact_key(artifact_id)
    access_key = draft_review_access_key(artifact_id)
    if not scope.org_mode or not scope.user_id or not scope.org_id:
        raise ArtifactAccessUnavailable("Draft collaboration requires a signed-in organization")
    if not owner_user_id:
        raise ArtifactAccessUnavailable("Artifact ownership could not be established")
    settings = settings or TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise ArtifactAccessUnavailable("Draft collaboration authorization is not configured")

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.post(
            f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access/",
            json={
                "artifact_id": access_key,
                "owner_keycloak_id": str(owner_user_id),
                "organization_id": str(scope.org_id),
                "allowed_emails": [],
                "org_allowed": True,
                "access_version": 1,
            },
            headers={"X-Internal-Auth": settings.auth_internal_secret},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ArtifactAccessUnavailable("Could not authorize draft collaboration") from exc
    finally:
        if owns_client:
            await client.aclose()
    return key


async def revoke_draft_review_access(
    artifact_id: str,
    *,
    settings: TurnQueueSettings | None = None,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Remove the private-draft rule before deleting its artifact.

    An unconfigured deployment cannot have provisioned this rule through
    Cowork, so there is nothing to revoke. A configured but unavailable auth
    service fails closed: the artifact remains available to its owner instead
    of leaving a comment capability orphaned after its files are gone.
    """
    access_key = draft_review_access_key(artifact_id)
    settings = settings or TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        return False

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.post(
            f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access/delete/",
            json={"artifact_id": access_key},
            headers={"X-Internal-Auth": settings.auth_internal_secret},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ArtifactAccessUnavailable("Could not revoke draft collaboration") from exc
    finally:
        if owns_client:
            await client.aclose()
    return True
