"""Who a restricted artifact is shared with, beyond what anton's publisher
knows, in MindsHub mode.

anton's publisher writes an artifact's rule through the publish lambda: the
mode, the outside addresses (its ``emails``) and the org flag. It knows no
teams and no people by id, so this server writes those to MindsHub itself
after the publish (sync_team_audience), the lists a person's save changed
(AudienceChanges), under MindsHub's artifact rule names
(cowork.services.sharing_contract.AudienceField). The same write carries the
people a person's save chose to tell (ShareNotify), with what the share email
names and links (ShareNotice). The publish itself plans and runs in
cowork.services.publish.publish_artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlparse

from anton.publish_access import normalize_emails

from cowork.schemas.artifact_access import (
    PUBLISHER_EMAILS_FIELD,
    ArtifactAccessRequest,
    ShareNotifyRequest,
)
from cowork.services.mindshub_client import canonical_uuid
from cowork.services.sharing_contract import (
    MAX_NOTIFY_EMAILS,
    MAX_NOTIFY_TITLE_CHARS,
    MAX_NOTIFY_URL_CHARS,
    MAX_PEOPLE_PER_RULE,
    AudienceField,
)

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope
    from cowork.services.artifact_access import ArtifactAudience, ArtifactRule

# A restricted rule's org audience beyond the emails and the org flag: teams,
# and people in the org by user id. anton's publisher knows neither, so this
# server writes a person's save of them to MindsHub itself
# (sync_team_audience).
ORG_AUDIENCE_FIELDS = (AudienceField.ALLOWED_TEAM_IDS, AudienceField.ALLOWED_USER_IDS)
# anton's publisher's flag for a restricted rule that names nobody but its
# owner.
PUBLISHER_OWNER_ONLY_FIELD = "owner_only"
# What a restricted entry of .published.json leaves out in MindsHub mode: who
# the rule shares with, which MindsHub's rule alone records
# (cowork.services.artifact_access.ArtifactRule). The entry keeps the mode,
# the password and the versions.
ENTRY_AUDIENCE_FIELDS = (PUBLISHER_EMAILS_FIELD, AudienceField.ORG_ALLOWED, PUBLISHER_OWNER_ONLY_FIELD)


class ShareNotify(NamedTuple):
    """Who a person's save asks MindsHub to email: people in the org by user
    id, and anyone else by address."""

    user_ids: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.user_ids or self.emails)


def _listed(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip()]


def share_notify(request: ShareNotifyRequest | None) -> ShareNotify:
    """Who a save's "Notify people" names: each user id in its canonical
    form, each address normalized, both deduplicated. A ValueError for a user
    id that isn't a UUID (naming it), or for more than MindsHub emails from
    one save: MAX_PEOPLE_PER_RULE ids and MAX_NOTIFY_EMAILS addresses. The
    access route reads it before anything is published."""
    if request is None:
        return ShareNotify()
    parsed_ids = {value: canonical_uuid(value) for value in _listed(request.user_ids)}
    malformed = [value for value, parsed in parsed_ids.items() if parsed is None]
    if malformed:
        raise ValueError(f"notify.user_ids must be UUIDs: {', '.join(malformed)}")
    user_ids = tuple(dict.fromkeys(parsed for parsed in parsed_ids.values() if parsed is not None))
    emails = tuple(dict.fromkeys(normalize_emails(_listed(request.emails))))
    if len(user_ids) > MAX_PEOPLE_PER_RULE:
        raise ValueError(f"notify can name at most {MAX_PEOPLE_PER_RULE} people.")
    if len(emails) > MAX_NOTIFY_EMAILS:
        raise ValueError(f"notify can name at most {MAX_NOTIFY_EMAILS} email addresses.")
    return ShareNotify(user_ids=user_ids, emails=emails)


def notify_within(notify: ShareNotify, audience: ArtifactRule) -> ShareNotify:
    """Only people the saved rule actually names, by id or by address: a
    notify list never reaches anyone the share doesn't."""
    ids = {str(v).lower() for v in audience.allowed_user_ids}
    emails = {str(v).lower() for v in audience.allowed_emails}
    return ShareNotify(
        user_ids=tuple(v for v in notify.user_ids if v in ids),
        emails=tuple(v for v in notify.emails if v.lower() in emails),
    )


def for_publisher(access: ArtifactAccessRequest | None) -> dict[str, Any] | None:
    """``access`` in the shape anton's publisher reads
    (ArtifactAccessRequest.publisher_access), which knows no teams and no
    people by id: a restricted one leaves them out. A selection of teams or
    people alone is owner-only there, so the lambda writes a restricted rule
    that sync_team_audience then widens; teams or people named where they
    can't apply (outside MindsHub mode) stay owner-only too, never the
    public rule anton falls back to for an empty selection. The one place a
    publish's access becomes anton's (cowork.services.publish._plan_access).
    """
    if access is None:
        return None
    if access.restricted:
        named = any(getattr(access, field) for field in ORG_AUDIENCE_FIELDS)
        access = access.model_copy(update={field.value: None for field in ORG_AUDIENCE_FIELDS})
        if named and not access.allowed_emails and not access.org_allowed:
            access = access.model_copy(update={PUBLISHER_OWNER_ONLY_FIELD: True})
    return access.publisher_access()


class AudienceChanges(NamedTuple):
    """The team and people lists a restricted publish changes on the
    artifact's rule (cowork.services.publish._plan_access), for
    sync_team_audience to write. None leaves that list as MindsHub holds it;
    false when neither changes."""

    allowed_team_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None

    def __bool__(self) -> bool:
        return self.allowed_team_ids is not None or self.allowed_user_ids is not None


class ShareNotice(NamedTuple):
    """What MindsHub's share email names and links, and who it goes to."""

    artifact_title: str
    artifact_url: str
    recipients: ShareNotify


def share_notice(
    published_dir: Path, published_key: str, view_url: str, *, recipients: ShareNotify
) -> ShareNotice:
    """The share email for ``recipients``: the artifact's title (its metadata
    name, else its folder or file name) and the published URL the recipient
    opens, within MindsHub's bounds. A URL that isn't an https link is left
    out; whether its host is one the email may link to is MindsHub's
    allowlist's call."""
    title = ""
    try:
        meta = json.loads((published_dir / "metadata.json").read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            title = str(meta.get("title") or meta.get("name") or "")
    except (OSError, ValueError):
        pass
    title = " ".join((title or published_dir.name or published_key or "").split())
    url = view_url if isinstance(view_url, str) else ""
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = None
    if parsed is None or parsed.scheme != "https" or not parsed.hostname:
        url = ""
    return ShareNotice(
        artifact_title=title[:MAX_NOTIFY_TITLE_CHARS],
        artifact_url=url if len(url) <= MAX_NOTIFY_URL_CHARS else "",
        recipients=recipients,
    )


def sync_team_audience(
    *,
    artifact_id: str,
    scope: TenantScope,
    access_version: int,
    changes: AudienceChanges,
    notice: ShareNotice | None = None,
) -> ArtifactAudience:
    """Write the teams and people a person's save changed (``changes``) to
    MindsHub, after the lambda wrote the rest of the rule, at the version it
    wrote (sync_artifact_team_access), and return the org flag, teams and
    people the rule holds now. A list ``changes`` leaves None is MindsHub's
    to keep. ``notice`` (a person's save with "Notify people" chosen) asks
    MindsHub to email its recipients; None never notifies."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable, sync_artifact_team_access
    from cowork.services.artifact_autopublish import _active_workspace_id

    if not artifact_id:
        raise ArtifactAccessUnavailable("The artifact has no identity to share with teams")
    return sync_artifact_team_access(
        artifact_id=artifact_id,
        scope=scope,
        owner_user_id=str(scope.user_id or ""),
        workspace_id=_active_workspace_id(scope),
        access_version=access_version or 1,
        allowed_team_ids=changes.allowed_team_ids,
        allowed_user_ids=changes.allowed_user_ids,
        notice=notice,
    )
