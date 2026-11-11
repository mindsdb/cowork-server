"""ArtifactCard sends the keys its builder set, in the card's order."""
from __future__ import annotations

from fastapi.encoders import jsonable_encoder

from cowork.schemas.artifact_access import AccessMode, ArtifactAccessView, ArtifactStatusView
from cowork.schemas.artifact_card import ArtifactCard
from cowork.services import artifacts as a
from _artifact_cards import make_card


def test_a_card_sends_only_the_keys_it_was_given_in_declared_order():
    card = make_card(published_url="https://x", modified=True)

    assert list(card.to_json())[-4:] == ["originConversationId", "publishedUrl", "modified", "serveUrl"]
    assert "accessPassword" not in card.to_json()
    assert "draftUrl" not in card.to_json()


def test_an_assigned_field_goes_out():
    card = make_card()
    card.capabilities = {"edit": True}
    card.received_from = None

    assert list(card.to_json())[-2:] == ["capabilities", "receivedFrom"]
    assert card.to_json()["receivedFrom"] is None


def test_the_access_fields_follow_the_access_view():
    access_fields = list(ArtifactAccessView.model_fields)
    card_fields = list(ArtifactCard.model_fields)

    start = card_fields.index(access_fields[0])
    assert card_fields[start:start + len(access_fields)] == access_fields
    for name in access_fields:
        assert ArtifactCard.model_fields[name].annotation == ArtifactAccessView.model_fields[name].annotation


def test_defaults_match_what_the_status_view_reads_for_a_missing_key():
    """ArtifactStatusView.of_card read card.get(key, default) before the card
    was a model; each default here must be that value."""
    card = make_card()

    assert (card.published_url, card.modified, card.access_mode, card.access_protected) == ("", False, AccessMode.PUBLIC, False)
    assert (card.access_emails, card.org_allowed, card.owner_only, card.artifact_key) == ([], False, False, "")


def test_of_access_matches_the_loose_file_status():
    access = ArtifactAccessView(access_mode=AccessMode.PASSWORD, access_protected=True, access_password="s", artifact_key="k")

    status = ArtifactStatusView.of_access(access, published_url="https://x")

    assert status == ArtifactStatusView(
        published_url="https://x", modified=False, access_mode=AccessMode.PASSWORD,
        access_protected=True, access_emails=[], org_allowed=False, owner_only=False, artifact_key="k",
    )


def test_a_published_url_that_is_not_text_reads_as_unpublished(tmp_path):
    primary = tmp_path / "index.html"
    primary.write_text("<html></html>")
    for url in (5, ["x"], "bad" + chr(0xD800)):
        published_map = {"index.html": {"url": url, "published": True}}

        assert a._published_url_for(tmp_path, primary, published_map=published_map) == ""
    assert a._published_url_for(tmp_path, primary, published_map={"index.html": {"url": "https://x"}}) == "https://x"


def test_every_way_of_dumping_a_card_sends_only_its_set_fields():
    card = make_card()
    card.capabilities = {"edit": True}

    assert jsonable_encoder(card) == card.to_json()
    assert "accessPassword" not in jsonable_encoder(card)
    assert "access_password" not in card.model_dump()
