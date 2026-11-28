"""The website access mode: anton's contract this server relies on, the
mode's breadth, and the access a website publish may carry."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cowork.models.artifact_owner_choice import AccessMode
from cowork.schemas.artifact_access import WEBSITE_FIELDS_REFUSAL, ArtifactAccessRequest
from cowork.services import publish
from cowork.services.artifact_owner_choice import OwnerChoice, Widening, widening
from cowork.services.artifacts import published_access_view

SITE_URL = "https://hk1-staging.mindshub.co/"


def test_the_installed_anton_carries_the_website_mode():
    """What publish_artifact relies on: anton keeps `website` through
    resolve_access and a kept entry, never stores confirmed_by_owner, and
    sends confirmed_by_owner only when the caller's access carries it, never
    from the previous entry, which the pod can write."""
    from anton.publish_access import WEBSITE_MODE, access_from_owner_side, resolve_access
    from anton.publisher import build_access_payload

    assert WEBSITE_MODE == "website"
    effective, _pwd_version, _access_version, owner_side = resolve_access(None, {"mode": "website"}, None)
    assert effective == {"mode": "website"}
    assert owner_side["mode"] == "website"
    assert "confirmed_by_owner" not in owner_side
    assert access_from_owner_side({"mode": "website", "confirmed_by_owner": True}) == {"mode": "website"}
    assert build_access_payload({"mode": "website"}) == {"mode": "website"}
    assert build_access_payload({"mode": "website", "confirmed_by_owner": True}) == {
        "mode": "website", "confirmed_by_owner": True,
    }
    assert build_access_payload(
        {"mode": "website"}, previous={"mode": "website", "confirmed_by_owner": True},
    ) == {"mode": "website"}


def test_website_is_an_access_mode():
    assert AccessMode("website") is AccessMode.WEBSITE


@pytest.mark.parametrize("recorded", [AccessMode.RESTRICTED, AccessMode.PASSWORD, AccessMode.PUBLIC])
def test_a_website_is_wider_than_every_other_mode(recorded):
    """A re-publish that would turn a narrower recorded choice into a
    website is refused like any widening; a narrower re-publish of a
    website is not."""
    assert widening(mode=AccessMode.WEBSITE, choice=OwnerChoice(recorded, published=True)) is (
        Widening.WIDER_THAN_OWNER_CHOICE
    )
    assert widening(mode=recorded, choice=OwnerChoice(AccessMode.WEBSITE, published=True)) is None
    assert OwnerChoice(AccessMode.WEBSITE, True).narrowest(OwnerChoice(recorded, True)) == OwnerChoice(recorded, True)


def test_a_website_republish_of_a_website_is_not_wider():
    assert widening(mode=AccessMode.WEBSITE, choice=OwnerChoice(AccessMode.WEBSITE, published=True)) is None


def test_a_website_save_carries_its_mode_alone():
    assert ArtifactAccessRequest.model_validate({"mode": "website"}).publisher_access() == {"mode": "website"}


@pytest.mark.parametrize(
    "extra",
    [
        {"password": "pw"},
        {"allowed_emails": ["a@example.com"]},
        {"org_allowed": False},
        {"allowed_team_ids": []},
        {"allowed_user_ids": ["u-1"]},
        {"owner_only": True},
        {"notify": {"emails": []}},
    ],
    ids=["password", "emails", "org", "teams", "people", "owner-only", "notify"],
)
def test_a_website_with_any_other_access_field_is_refused(extra):
    with pytest.raises(ValidationError) as refused:
        ArtifactAccessRequest.model_validate({"mode": "website", **extra})
    assert WEBSITE_FIELDS_REFUSAL in str(refused.value)


def test_confirmed_by_owner_from_a_client_is_ignored():
    """The request schema has no such field: a client that sends it, under
    either name, changes nothing."""
    access = ArtifactAccessRequest.model_validate(
        {"mode": "website", "confirmed_by_owner": True, "confirmedByOwner": True}
    )
    assert access.publisher_access() == {"mode": "website"}


def test_an_entry_that_carries_confirmed_by_owner_is_kept_without_it():
    """A .published.json entry is pod-writable; what a re-publish keeps
    from it never carries the owner's confirmation."""
    kept = publish.kept_access({"report_id": "r1", "mode": "website", "confirmed_by_owner": True}, org_mode=True)
    assert kept.mode is AccessMode.WEBSITE
    assert kept.publisher_access() == {"mode": "website"}


def test_a_website_entry_reads_as_a_website_on_the_card(tmp_path):
    folder = tmp_path / "rep"
    folder.mkdir()
    primary = folder / "index.html"
    primary.write_text("<html></html>")
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": SITE_URL, "published": True, "mode": "website", "artifact_key": "artifact/k",
    }}))

    view = published_access_view(folder, primary)

    assert (view.access_mode, view.access_protected, view.artifact_key) == (AccessMode.WEBSITE, False, "artifact/k")
    # The card names nobody, as for public: empty lists and false, never null,
    # which the UI would read as "nothing chosen".
    card = view.model_dump(by_alias=True, mode="json")
    assert {name: card[name] for name in (
        "accessEmails", "orgAllowed", "ownerOnly", "allowedTeamIds", "allowedUserIds", "accessPassword",
    )} == {
        "accessEmails": [], "orgAllowed": False, "ownerOnly": False, "allowedTeamIds": [], "allowedUserIds": [],
        "accessPassword": "",
    }


def test_a_republish_keeps_a_website_entry(tmp_path):
    """The UI republish and autopublish read the kept access this way."""
    folder = tmp_path / "rep"
    folder.mkdir()
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    (folder / ".published.json").write_text(json.dumps({"index.html": {
        "report_id": "r1", "url": SITE_URL, "published": True, "mode": "website",
    }}))

    kept = publish.published_artifact_access(folder, artifacts_base=tmp_path, org_mode=True)

    assert kept.publisher_access() == {"mode": "website"}
