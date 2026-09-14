"""Translate local metadata IDs through an immutable, tenant-scoped SQL alias.

Agents may choose metadata IDs; they may never choose global comment identities.
Only auth issues new global IDs. A historical ID is adopted only after auth
confirms its durable binding already belongs to this owner and organization.
"""

from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.artifact_identity import ArtifactIdentity
from cowork.models.conversation import Conversation
from cowork.services.artifact_access import ArtifactAccessUnavailable
from cowork.services.artifact_identity import artifact_key


@contextmanager
def _session(scope: TenantScope):
    if not scope.org_mode or not scope.org_id or not scope.user_id:
        raise ArtifactAccessUnavailable(
            "Artifact authorization requires a signed-in organization"
        )
    # Publish runs in a worker thread; never share a request's SQL session.
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        yield ScopedSession(raw, scope)


def _row(session, local_id, *, lock=False):
    query = session.select(ArtifactIdentity).where(
        ArtifactIdentity.local_artifact_id == UUID(local_id).hex
    )
    if lock:
        query = query.with_for_update()
    return session.exec(query).first()


def _check_owner(row, owner_user_id):
    if row is None or not owner_user_id or row.owner_keycloak_id != str(owner_user_id):
        raise ArtifactAccessUnavailable(
            "This artifact identity belongs to another owner"
        )


def existing_authorization_key(
    local_id: str, scope: TenantScope, *, owner_user_id: str | None = None
) -> str | None:
    """Read an existing alias. Never contact auth or allocate during a read."""
    with _session(scope) as session:
        row = _row(session, local_id)
        if row is None:
            return None
        if owner_user_id is not None:
            _check_owner(row, owner_user_id)
        if not row.canonical_artifact_id:
            raise ArtifactAccessUnavailable("Artifact authorization is still pending")
        return row.canonical_artifact_id


def _allocate_or_adopt(local_id, owner_user_id, scope, request_id):
    settings = TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise ArtifactAccessUnavailable("Artifact authorization is not configured")
    identity = {
        "owner_keycloak_id": str(owner_user_id),
        "organization_id": str(scope.org_id),
    }
    base = f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access"
    try:
        with httpx.Client(
            timeout=5.0, headers={"X-Internal-Auth": settings.auth_internal_secret}
        ) as client:
            response = client.post(
                f"{base}/claim/",
                json={
                    **identity,
                    "artifact_id": artifact_key(local_id),
                    "create_if_missing": False,
                },
            )
            if response.is_success:
                if response.json().get("ok") is not True:
                    raise ValueError("Invalid ownership confirmation")
                return artifact_key(local_id)
            if (
                response.status_code != 409
                or response.json().get("code") != "artifact_ownership_unclaimed"
            ):
                response.raise_for_status()
            response = client.post(
                f"{base}/allocate/",
                json={
                    **identity,
                    "allocation_request_id": str(request_id),
                },
            )
            response.raise_for_status()
            key = response.json()["artifact_id"]
            namespace, _, identifier = key.partition("/")
            if namespace != "artifact":
                raise ValueError("Invalid allocated artifact namespace")
            return artifact_key(str(UUID(identifier)))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ArtifactAccessUnavailable(
            "Could not establish artifact authorization"
        ) from exc


def ensure_authorization_key(
    local_id: str, scope: TenantScope, *, owner_user_id: str
) -> str:
    """Allocate after the caller verified the artifact's Conversation owner."""
    if not owner_user_id or str(owner_user_id) != str(scope.user_id):
        raise ArtifactAccessUnavailable(
            "Only the artifact owner can authorize this draft"
        )
    local_id = UUID(local_id).hex
    with _session(scope) as session:
        row = _row(session, local_id)
        if row is None:
            row = ArtifactIdentity(
                org_id=str(scope.org_id),
                owner_keycloak_id=str(owner_user_id),
                local_artifact_id=local_id,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                # Concurrent first use must reuse the winner's nonce and owner.
                session.rollback()
                row = _row(session, local_id)
                if row is None:
                    raise
        _check_owner(row, owner_user_id)
        if row.canonical_artifact_id:
            return row.canonical_artifact_id
        request_id = row.id

    canonical = _allocate_or_adopt(local_id, owner_user_id, scope, request_id)
    with _session(scope) as session:
        row = _row(session, local_id, lock=True)
        _check_owner(row, owner_user_id)
        if row.canonical_artifact_id and row.canonical_artifact_id != canonical:
            raise ArtifactAccessUnavailable(
                "Artifact authorization identity changed unexpectedly"
            )
        row.canonical_artifact_id = canonical
        session.add(row)
        session.commit()
    return canonical


def publish_authorization_key(
    local_id: str, artifacts_base: Path, scope: TenantScope
) -> str:
    """Verify the server conversation root before minting publish identity."""
    from cowork.services.artifact_roots import artifacts_sources_for_project

    try:
        conversation_id = UUID(Path(artifacts_base).parent.parent.name)
    except (TypeError, ValueError) as exc:
        raise ArtifactAccessUnavailable(
            "Artifact ownership could not be established"
        ) from exc
    with _session(scope) as session:
        conversation = session.get(Conversation, conversation_id)
        if (
            conversation is None
            or not conversation.created_by
            or str(conversation.created_by) != str(scope.user_id)
        ):
            raise ArtifactAccessUnavailable(
                "Only the artifact owner can publish this draft"
            )
        sources = artifacts_sources_for_project(session, conversation.project_id)
        if not any(Path(source.base) == Path(artifacts_base) for source in sources):
            raise ArtifactAccessUnavailable(
                "Artifact root does not belong to its owner"
            )
        owner_user_id = conversation.created_by
    return ensure_authorization_key(local_id, scope, owner_user_id=owner_user_id)
