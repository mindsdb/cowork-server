"""Provision the auth rule that lets an organization review a private draft."""
from __future__ import annotations

from uuid import UUID

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.services.artifact_identity import artifact_key


class ArtifactAccessUnavailable(RuntimeError):
    pass


class ArtifactAudienceRejected(ValueError):
    """Auth refused a restricted rule's teams or people: a team outside the
    artifact's organization, or a person who isn't a member of it. A
    ValueError, so the access route answers 400 with the reason."""


def _audience_rejection(response: httpx.Response) -> str:
    """Auth's 400 for an unknown team or a non-member, as one sentence naming
    them; "" for any other answer."""
    if response.status_code != 400:
        return ""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    if body.get("code") == "not_a_member":
        ids = ", ".join(str(i) for i in body.get("allowed_user_ids") or [])
        return f"Only people in your organization can be added by name; share with anyone else by email: {ids}"
    if body.get("code") == "unknown_team":
        ids = ", ".join(str(i) for i in body.get("allowed_team_ids") or [])
        return f"Each team must be a team of your organization: {ids}"
    return ""


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
    """Upsert owner + same-org review access for an already-issued global id.

    The caller resolves agent metadata through the trusted SQL identity alias
    before entering this helper; auth requires a matching durable binding.

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
    scope: TenantScope,
    *,
    settings: TurnQueueSettings | None = None,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Remove the private-draft rule after verifying the scoped caller owns it.

    The caller must enforce artifact ownership before entering this helper.
    Auth also checks this owner and organization against its durable binding;
    knowing an artifact id alone must never authorize deleting its policy.

    An unconfigured deployment cannot have provisioned this rule through
    Cowork, so there is nothing to revoke. A configured but unavailable auth
    service fails closed: the artifact remains available to its owner instead
    of leaving a comment capability orphaned after its files are gone.
    """
    access_key = draft_review_access_key(artifact_id)
    if not scope.org_mode or not scope.user_id or not scope.org_id:
        raise ArtifactAccessUnavailable("Draft collaboration requires a signed-in organization")
    settings = settings or TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        return False

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.post(
            f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access/delete/",
            json={
                "artifact_id": access_key,
                "owner_keycloak_id": str(scope.user_id),
                "organization_id": str(scope.org_id),
            },
            headers={"X-Internal-Auth": settings.auth_internal_secret},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ArtifactAccessUnavailable("Could not revoke draft collaboration") from exc
    finally:
        if owns_client:
            await client.aclose()
    return True


def sync_artifact_team_access(
    *,
    artifact_id: str,
    scope: TenantScope,
    owner_user_id: str,
    workspace_id: str | None,
    allowed_emails: list[str],
    org_allowed: bool,
    allowed_team_ids: list[str],
    access_version: int,
    allowed_user_ids: list[str] | None = None,
    notify_user_ids: list[str] | None = None,
    notify_emails: list[str] | None = None,
    artifact_title: str = "",
    artifact_url: str = "",
    settings: TurnQueueSettings | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Write a restricted artifact's teams and people to auth (ENG-3787).

    The publish lambda upserts the same rule from the anton publisher's
    access block, which carries emails and the org flag but no teams or
    people by id, so this upserts the whole rule again, with the same
    version, under auth's ArtifactAccess names: ``org_allowed``,
    ``allowed_team_ids``, ``allowed_user_ids`` (Keycloak ids of people in the
    org) and ``allowed_emails`` (people outside it). Auth checks each team
    belongs to the org and each person is a member of it, and refuses the
    whole rule otherwise: ArtifactAudienceRejected, naming them. Called after
    a successful publish, so it lands second. Raises ArtifactAccessUnavailable
    when it can't write, so a restricted publish never reports a team
    audience auth doesn't hold.

    ``notify_user_ids`` and ``notify_emails`` are the people a person's own
    save chose to tell: with either non-empty the write carries auth's
    ``notify_recipients: true`` and both lists, and auth emails them, naming
    the artifact by ``artifact_title`` and linking ``artifact_url``. The
    lambda's write, a re-sync and a retry never carry any of it.
    """
    if not scope.org_mode or not scope.org_id or not owner_user_id:
        raise ArtifactAccessUnavailable("Team sharing requires a signed-in organization")
    settings = settings or TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise ArtifactAccessUnavailable("Team sharing authorization is not configured")
    owns_client = client is None
    client = client or httpx.Client(timeout=5.0)
    try:
        response = client.post(
            f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access/",
            json={
                "artifact_id": artifact_id,
                "owner_keycloak_id": str(owner_user_id),
                "organization_id": str(scope.org_id),
                "workspace_id": workspace_id or None,
                "allowed_emails": allowed_emails,
                "org_allowed": org_allowed,
                "allowed_team_ids": allowed_team_ids,
                "allowed_user_ids": list(allowed_user_ids or []),
                "access_version": access_version,
                **(
                    {
                        "notify_recipients": True,
                        "notify_user_ids": list(notify_user_ids or []),
                        "notify_emails": list(notify_emails or []),
                        "artifact_title": artifact_title,
                        "artifact_url": artifact_url,
                    }
                    if notify_user_ids or notify_emails
                    else {}
                ),
            },
            headers={"X-Internal-Auth": settings.auth_internal_secret},
        )
        rejected = _audience_rejection(response)
        if rejected:
            raise ArtifactAudienceRejected(rejected)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ArtifactAccessUnavailable(
            "Could not share the artifact with the selected teams and people"
        ) from exc
    finally:
        if owns_client:
            client.close()
