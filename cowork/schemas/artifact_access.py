"""Wire shapes of an artifact's sharing: the access a publish takes, one save
of the Share dialog included (ArtifactAccessRequest), and the access state
the owner reads back (ArtifactAccessView) or a publish answers
(ArtifactPublishResult).

Who a restricted artifact is shared with goes by MindsHub's artifact rule
names (cowork.services.sharing_contract.AudienceField). The access state is
camelCase on the wire, as every response model here is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_core import ErrorDetails

from cowork.models.artifact_owner_choice import AccessMode
from cowork.schemas.base import CamelRequest, CamelResponse
from cowork.services.sharing_contract import AudienceField

if TYPE_CHECKING:
    from cowork.services.artifact_access import ArtifactRule

# anton's publisher's name for the people outside the organization, by
# address: what AudienceField.ALLOWED_EMAILS is called in the access it reads.
PUBLISHER_EMAILS_FIELD = "emails"
# ArtifactAccessRequest's "Notify people", which applies to one save and is
# never part of the access a publish stores.
NOTIFY_FIELD = "notify"
_ALLOWED_EMAILS_REFUSAL = (
    f"{PUBLISHER_EMAILS_FIELD} is not an access field; send {AudienceField.ALLOWED_EMAILS}."
)


class ShareNotifyRequest(CamelRequest):
    """The people a save added and chose to tell ("Notify people"): people in
    the organization by user id, and anyone else by address. MindsHub emails
    them, for this save only."""

    user_ids: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)


class ArtifactAccessRequest(CamelRequest):
    """The access one publish of an artifact takes
    (cowork.services.publish.publish_artifact): a save of its Share dialog
    (PUT .../access), and what every other caller builds: the desktop
    publish route and the agent's publish tool from what they were asked,
    and a live sync, an autopublish or an update from the access the
    artifact keeps (publish.kept_access).

    ``mode`` is who the link opens for; a password save carries
    ``password``. A restricted save names its audience under MindsHub's
    artifact rule names: ``org_allowed``, ``allowed_team_ids`` and
    ``allowed_user_ids`` (MindsHub mode only), and ``allowed_emails`` for
    people outside the organization. In MindsHub mode a field the save
    leaves out (None) keeps what the artifact's rule holds. ``owner_only``
    keeps a rule that names nobody restricted to its owner. ``notify`` names
    the people a Share dialog save added and chose to tell; it is never
    stored, and the publish takes it on its own.

    Any other field is ignored, except a top-level ``emails`` (anton's name
    for the outside addresses), which an app tab from an older build still
    sends: it is refused, so that tab can't shrink an audience unnoticed.
    anton's own access reads in with of_publisher, and leaves with
    publisher_access.
    """

    mode: AccessMode | None = None
    password: str | None = None
    allowed_emails: list[str] | None = None
    org_allowed: bool | None = None
    allowed_team_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None
    owner_only: bool | None = None
    notify: ShareNotifyRequest | None = None

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_publishers_emails(cls, data: Any) -> Any:
        if isinstance(data, dict) and PUBLISHER_EMAILS_FIELD in data:
            raise ValueError(_ALLOWED_EMAILS_REFUSAL)
        return data

    @property
    def restricted(self) -> bool:
        return self.mode is AccessMode.RESTRICTED

    @classmethod
    def of_publisher(cls, access: Mapping[str, Any]) -> ArtifactAccessRequest:
        """``access`` in anton's publisher's shape (``mode``, ``password``,
        ``emails``, ``org_allowed``, ``owner_only``), as anton's
        access_from_owner_side rebuilds it from an entry and the desktop
        publish route takes it, read as this model: its ``emails`` under
        ``allowed_emails``."""
        fields = dict(access)
        if PUBLISHER_EMAILS_FIELD in fields:
            fields[AudienceField.ALLOWED_EMAILS.value] = fields.pop(PUBLISHER_EMAILS_FIELD)
        return cls.model_validate(fields)

    def publisher_access(self) -> dict[str, Any]:
        """This access as anton's publisher reads it: the fields it sets,
        ``allowed_emails`` under the publisher's ``emails``, and never
        ``notify``. publish_artifact hands it over once, at the call into
        anton's resolve_access."""
        access = self.model_dump(mode="json", exclude_none=True, exclude={NOTIFY_FIELD})
        if AudienceField.ALLOWED_EMAILS in access:
            access[PUBLISHER_EMAILS_FIELD] = access.pop(AudienceField.ALLOWED_EMAILS)
        return access


# How each kind of wrong value reads to the person, by pydantic's error type.
_WRONG_TYPE = {
    "list_type": "must be a list",
    "dict_type": "must be an object",
    "model_type": "must be an object",
    "model_attributes_type": "must be an object",
    "string_type": "must be text",
    "bool_type": "must be true or false",
    "bool_parsing": "must be true or false",
}


def access_refusal(errors: Iterable[ErrorDetails]) -> str:
    """Why an access object isn't an ArtifactAccessRequest, from its
    validation ``errors``, located from inside the object: each error as a
    sentence, a validator's own message as it is, else the field and what is
    wrong with it. What PUT .../access answers as its 400."""
    reasons: list[str] = []
    for error in errors:
        cause = (error.get("ctx") or {}).get("error")
        if error["type"] == "value_error" and cause is not None:
            reasons.append(str(cause))
            continue
        where = ".".join(str(part) for part in error["loc"]) or "access"
        wrong = _WRONG_TYPE.get(error["type"])
        reasons.append(f"{where} {wrong}." if wrong else f"{where}: {error['msg']}.")
    return " ".join(dict.fromkeys(reasons))


class ArtifactAudienceView(CamelResponse):
    """Who a restricted artifact is shared with, under the app's names.
    ``ownerOnly`` is a restricted rule that names nobody."""

    access_emails: list[str] = Field(default_factory=list)
    org_allowed: bool = False
    owner_only: bool = False
    allowed_team_ids: list[str] = Field(default_factory=list)
    allowed_user_ids: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, rule: ArtifactRule | None) -> ArtifactAudienceView:
        """Who ``rule`` shares with; nobody for None (a publish that isn't
        restricted)."""
        if rule is None:
            return ArtifactAudienceView()
        return ArtifactAudienceView(
            access_emails=list(rule.allowed_emails),
            org_allowed=rule.org_allowed,
            owner_only=rule.names_nobody,
            allowed_team_ids=list(rule.allowed_team_ids),
            allowed_user_ids=list(rule.allowed_user_ids),
        )


class ArtifactAccessView(ArtifactAudienceView):
    """An artifact's access state for its owner: the Share dialog's GET
    .../access and the artifact card's access fields. The plaintext password
    and the email list are the owner's only, so only the owner is answered
    with them."""

    access_mode: AccessMode = AccessMode.PUBLIC
    access_protected: bool = False
    access_password: str = ""
    # The key the renderer addresses comment requests by; "" when
    # unpublished or published before the key was kept.
    artifact_key: str = ""

    def restricted_to(self, rule: ArtifactRule) -> ArtifactAccessView:
        """This state restricted to ``rule``'s audience: what an artifact
        whose rule MindsHub holds is shared with, whatever its entry says."""
        return ArtifactAccessView(
            access_mode=AccessMode.RESTRICTED,
            artifact_key=self.artifact_key,
            **ArtifactAudienceView.of(rule).model_dump(),
        )


class ArtifactPublishResult(ArtifactAudienceView):
    """What a publish answers: the link, who it opens for, and anton's
    publisher's own result (less the bundle it uploaded)."""

    status: Literal["ok"] = "ok"
    url: str
    access_mode: AccessMode = AccessMode.PUBLIC
    access_protected: bool = False
    # The key the renderer addresses comment requests by.
    artifact_key: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
