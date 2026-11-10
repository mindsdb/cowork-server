"""MindsHub's artifact access rules: the one that lets an organization review
a private draft, and the sharing rule of a published artifact."""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.services.artifact_identity import artifact_key

if TYPE_CHECKING:
    from cowork.services.artifact_audience import ShareNotice


class ArtifactAccessUnavailable(RuntimeError):
    pass


class ArtifactAccessRefused(RuntimeError):
    """MindsHub refused an artifact rule's read or write with a 409
    or a 403: the artifact's recorded owner or organization doesn't match
    the caller's, or the owner's role no longer allows artifact changes.
    Carries its detail. Asking again gets the same answer, so the access
    route answers 409, never the 503 that reads as "try again"."""


class ArtifactAudienceRejected(ValueError):
    """MindsHub refused a restricted rule's write with a 400: a team
    outside the artifact's organization, a person who isn't a member of it,
    or a field it won't take. A ValueError, so the access route answers 400
    with the reason; a retry of the same rule can never succeed, so it is
    never the retryable 503."""


# What MindsHub's not_a_member refusal, and the access route's own check of
# the people a save adds, tell the person.
NOT_A_MEMBER_REFUSAL = (
    "Only people in your organization can be added by name; share with anyone else by email"
)
# What MindsHub's unknown_team refusal, and the access route's own check of a
# team an admin adds, tell the person.
FOREIGN_TEAM_REFUSAL = "Each team must be a team of your organization"


class ArtifactAudience(BaseModel):
    """The org flag, teams and people of an artifact's rule, under MindsHub's
    artifact rule names (cowork.services.sharing_contract.AudienceField), as
    MindsHub echoes them after a write."""

    org_allowed: bool
    allowed_team_ids: list[str]
    allowed_user_ids: list[str]


class ArtifactRule(ArtifactAudience):
    """An artifact's rule as MindsHub stores it (read_artifact_access):
    the one record of who a restricted artifact is shared with. Its
    .published.json entry keeps the mode and the password, never this.
    ``allowed_emails`` are people outside the organization, by address."""

    allowed_emails: list[str]

    @classmethod
    def nobody(cls) -> ArtifactRule:
        """No one but the owner: what an artifact MindsHub holds no rule for
        is shared with."""
        return cls(org_allowed=False, allowed_team_ids=[], allowed_user_ids=[], allowed_emails=[])

    @property
    def names_nobody(self) -> bool:
        """Shared with no one but the owner: restricted to them alone."""
        return not (
            self.org_allowed or self.allowed_team_ids or self.allowed_user_ids or self.allowed_emails
        )


class _ArtifactRuleKey(BaseModel):
    """Which rule a call to MindsHub's internal artifact-access routes is
    about, and whose: the artifact's recorded owner and organization."""

    artifact_id: str
    owner_keycloak_id: str
    organization_id: str


class _ArtifactRuleWrite(_ArtifactRuleKey):
    """A restricted rule's teams and people, written beside what the lambda
    wrote (sync_artifact_team_access). Only the fields a call sets go on the
    wire: MindsHub keeps a field the caller leaves out."""

    workspace_id: str | None
    access_version: int
    allowed_team_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None
    # A person's save with "Notify people" chosen: MindsHub emails them.
    notify_recipients: bool | None = None
    notify_user_ids: list[str] | None = None
    notify_emails: list[str] | None = None
    artifact_title: str | None = None
    artifact_url: str | None = None


def _json_body(response: httpx.Response) -> dict | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _refusal_detail(response: httpx.Response, *, refused: str) -> str:
    """MindsHub's 409 or 403 as one sentence saying it ``refused`` the call
    ("the change", "the read"); "" for any other answer."""
    if response.status_code not in (403, 409):
        return ""
    body = _json_body(response) or {}
    detail = body.get("detail") if isinstance(body.get("detail"), str) else body.get("code")
    return f"MindsHub refused {refused}: {detail}" if detail else f"MindsHub refused {refused}."


def _field_messages(why: object) -> list[str]:
    """The messages of one field's validation error, however MindsHub nests
    them: a string, a list of them, or, for an item of a list field, a dict
    of the item's index to its own error."""
    if isinstance(why, dict):
        return [message for nested in why.values() for message in _field_messages(nested)]
    if isinstance(why, list):
        return [message for nested in why for message in _field_messages(nested)]
    return [str(why)]


def _rejection_reason(response: httpx.Response) -> str:
    """MindsHub's 400 as one sentence naming what it refused; "" for any
    other answer."""
    if response.status_code != 400:
        return ""
    body = _json_body(response)
    if body is None:
        return "The sharing change was refused."
    if body.get("code") == "not_a_member":
        ids = ", ".join(str(i) for i in body.get("allowed_user_ids") or [])
        return f"{NOT_A_MEMBER_REFUSAL}: {ids}"
    if body.get("code") == "unknown_team":
        ids = ", ".join(str(i) for i in body.get("allowed_team_ids") or [])
        return f"{FOREIGN_TEAM_REFUSAL}: {ids}"
    if isinstance(body.get("detail"), str) and body["detail"]:
        return f"The sharing change was refused: {body['detail']}"
    # A field validation error: {"<field>": ["<why>", ...], ...}, or for an
    # item of a list field {"<field>": {"<index>": ["<why>"]}}.
    fields = "; ".join(
        f"{field}: {' '.join(dict.fromkeys(_field_messages(why)))}" for field, why in body.items()
    )
    return f"The sharing change was refused: {fields}" if fields else "The sharing change was refused."


def _post_internal(
    route: str,
    body: _ArtifactRuleKey,
    *,
    settings: TurnQueueSettings,
    client: httpx.Client | None,
    failure: str,
    refused: str,
) -> httpx.Response:
    """POST ``body``, the fields it sets, to MindsHub's internal
    artifact-access ``route`` with the shared secret, and return its answer.
    A 409 or 403 is ArtifactAccessRefused, saying MindsHub ``refused`` the
    call; no answer at all is ArtifactAccessUnavailable (``failure``). Every
    other answer is the caller's to read."""
    owns_client = client is None
    client = client or httpx.Client(timeout=5.0)
    try:
        response = client.post(
            f"{settings.auth_internal_base_url.rstrip('/')}/v1/internal/artifact-access/{route}",
            json=body.model_dump(mode="json", exclude_unset=True),
            headers={"X-Internal-Auth": settings.auth_internal_secret},
        )
    except httpx.HTTPError as exc:
        raise ArtifactAccessUnavailable(failure) from exc
    finally:
        if owns_client:
            client.close()
    refusal = _refusal_detail(response, refused=refused)
    if refusal:
        raise ArtifactAccessRefused(refusal)
    return response


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


def _internal_settings(scope: TenantScope, settings: TurnQueueSettings | None) -> TurnQueueSettings:
    """MindsHub's internal endpoint and secret, for a call about an artifact
    of ``scope``'s organization."""
    if not scope.org_mode or not scope.org_id:
        raise ArtifactAccessUnavailable("Sharing with the organization requires a signed-in organization")
    settings = settings or TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise ArtifactAccessUnavailable("Sharing with the organization is not configured")
    return settings


def read_artifact_access(
    *,
    artifact_id: str,
    scope: TenantScope,
    owner_user_id: str,
    settings: TurnQueueSettings | None = None,
    client: httpx.Client | None = None,
) -> ArtifactRule | None:
    """The artifact's rule as MindsHub stores it, the one record of who a
    restricted artifact is shared with: ``org_allowed``,
    ``allowed_team_ids``, ``allowed_user_ids`` and ``allowed_emails``. None
    when it holds none (its 404 ``no_rule``): in MindsHub mode only a
    restricted publish leaves a rule, and a public or password one, or an
    unpublish, deletes it.

    Asked as the delete is, for the artifact's recorded owner and
    organization. A 409 or 403 is ArtifactAccessRefused. Any other failure,
    including an answer without all four fields, is
    ArtifactAccessUnavailable: a caller that can't read the rule never
    guesses it.
    """
    if not owner_user_id:
        raise ArtifactAccessUnavailable("Artifact ownership could not be established")
    settings = _internal_settings(scope, settings)
    failure = "Could not read who the artifact is shared with"
    response = _post_internal(
        "read/",
        _ArtifactRuleKey(
            artifact_id=artifact_id,
            owner_keycloak_id=str(owner_user_id),
            organization_id=str(scope.org_id),
        ),
        settings=settings,
        client=client,
        failure=failure,
        refused="the read",
    )
    if response.status_code == 404 and (_json_body(response) or {}).get("code") == "no_rule":
        return None
    if not response.is_success:
        raise ArtifactAccessUnavailable(failure)
    try:
        return ArtifactRule.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise ArtifactAccessUnavailable(failure) from exc


def sync_artifact_team_access(
    *,
    artifact_id: str,
    scope: TenantScope,
    owner_user_id: str,
    workspace_id: str | None,
    access_version: int,
    allowed_team_ids: list[str] | None = None,
    allowed_user_ids: list[str] | None = None,
    notice: ShareNotice | None = None,
    settings: TurnQueueSettings | None = None,
    client: httpx.Client | None = None,
) -> ArtifactAudience:
    """Write a restricted artifact's teams and people to MindsHub, and return
    the org flag, teams and people its rule holds after the write.

    The publish lambda writes the rest of the rule (emails and the org flag,
    from the anton publisher's access block) and leaves the teams and people
    as they were, and so does this write for a list it is given as None:
    MindsHub keeps a field the caller omits. So only a person's own save
    sends a list, the one it changed, under MindsHub's names
    ``allowed_team_ids`` and ``allowed_user_ids`` (user ids of people in the
    org), with the version the lambda wrote. MindsHub checks each team
    belongs to the org and each person is a member of it, and refuses the
    whole rule otherwise; any 400 it answers is ArtifactAudienceRejected,
    naming what it refused, and a 409 or 403 is ArtifactAccessRefused. Called
    after a successful publish, so it lands second. Raises
    ArtifactAccessUnavailable when it can't write, or when its 200 doesn't
    echo the org flag and both lists (a MindsHub that drops them never stored
    them), so a restricted publish never reports an audience MindsHub doesn't
    hold.

    ``notice`` is a person's own save with "Notify people" chosen: with
    recipients, the write carries MindsHub's ``notify_recipients: true``,
    the people to tell by id and by address, and the artifact's title and
    link the email names. The lambda's write, a re-sync and a retry never
    carry any of it.
    """
    if not owner_user_id:
        raise ArtifactAccessUnavailable("Artifact ownership could not be established")
    settings = _internal_settings(scope, settings)
    failure = "Could not share the artifact with the selected teams and people"
    write = _ArtifactRuleWrite(
        artifact_id=artifact_id,
        owner_keycloak_id=str(owner_user_id),
        organization_id=str(scope.org_id),
        workspace_id=workspace_id or None,
        access_version=access_version,
    )
    if allowed_team_ids is not None:
        write.allowed_team_ids = list(allowed_team_ids)
    if allowed_user_ids is not None:
        write.allowed_user_ids = list(allowed_user_ids)
    if notice is not None and notice.recipients:
        write.notify_recipients = True
        write.notify_user_ids = list(notice.recipients.user_ids)
        write.notify_emails = list(notice.recipients.emails)
        write.artifact_title = notice.artifact_title
        write.artifact_url = notice.artifact_url
    response = _post_internal(
        "",
        write,
        settings=settings,
        client=client,
        failure=failure,
        refused="the change",
    )
    rejected = _rejection_reason(response)
    if rejected:
        raise ArtifactAudienceRejected(rejected)
    if not response.is_success:
        raise ArtifactAccessUnavailable(failure)
    try:
        return ArtifactAudience.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise ArtifactAccessUnavailable(
            "MindsHub didn't confirm the artifact's teams and people"
        ) from exc
