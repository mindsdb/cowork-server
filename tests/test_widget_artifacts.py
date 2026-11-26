"""Widget artifacts (ENG-4684): metadata.json `display: "widget"` is an
interactive piece that collects input in the chat. It shows inline exactly as
`display: "inline"` does (card key, revisionId, state.json), but it is no
deliverable: it is never published or shared, and the user's Artifacts list
leaves it out unless asked."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import cowork.services.artifact_autopublish  # noqa: F401  (see test_artifact_access_endpoints)
from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.api.v1.endpoints import artifacts as ep
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import artifact_autopublish, publish
from cowork.services.artifact_metadata import (
    WIDGET_PUBLISH_REFUSAL,
    ArtifactMetadata,
    is_widget,
    shows_inline,
)
from cowork.services.artifact_permissions import artifact_capabilities, with_display_limits
from cowork.services.artifacts import html_artifacts
from cowork.services.task_objects import cards_for_slugs, publish_and_card_turn_artifacts
from _async_stub import awaitable
from test_artifact_card_shape import _stream_event, _with_display, session  # noqa: F401  (fixture)
from test_artifacts_api_tenancy import _project_with_artifact, local_mode  # noqa: F401  (fixtures)

ORG_SCOPE = TenantScope(
    org_mode=True, org_id="0a0a0a0a-0000-4000-8000-000000000001", user_id="0b0b0b0b-0000-4000-8000-000000000001"
)
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"


# ── one helper for everything inline ────────────────────────────────────


def test_inline_and_widget_both_show_inline_and_only_widget_is_a_widget():
    assert shows_inline("inline") and shows_inline("widget")
    assert is_widget("widget") and not is_widget("inline")
    for other in ("Widget", "card", "", None, 1, ["widget"], {"widget": 1}):
        assert not shows_inline(other)
        assert not is_widget(other)


@pytest.mark.parametrize("value", ["inline", "widget"])
def test_metadata_reads_both_inline_displays(value):
    assert ArtifactMetadata.model_validate({"display": value}).display == value


@pytest.mark.parametrize("value", ["Widget", " widget", "widgets", ["widget"]])
def test_metadata_reads_any_other_widget_spelling_as_missing(value):
    assert ArtifactMetadata.model_validate({"display": value}).display is None


# ── the chat card ───────────────────────────────────────────────────────


async def test_a_widget_card_carries_display_and_revision_like_an_inline_one(session, tmp_path, local_mode):
    from cowork.services.artifact_revisions import capture_agent_revision

    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="picker")
    _with_display(folder, "widget")
    first = capture_agent_revision(folder, conversation_id="conversation-1")

    [card] = cards_for_slugs(folder.parent, ["picker"], scope=LOCAL_SCOPE, project_id=str(row.id), project_name="mine")
    streamed = await _stream_event(card)

    for sent in (card.to_json(), streamed):
        assert sent["display"] == "widget"
        assert sent["revisionId"] == first["id"]
        keys = list(sent)
        assert keys.index("display") == keys.index("primary") + 1
        assert keys.index("revisionId") == keys.index("display") + 1

    # The turn's own capture wins over the head, as for an inline card.
    [captured] = cards_for_slugs(
        folder.parent, ["picker"], scope=LOCAL_SCOPE, project_id=str(row.id), project_name="mine",
        revisions={"picker": "rev-from-the-turn"},
    )
    assert captured.revision_id == "rev-from-the-turn"


async def test_an_org_turn_cards_a_widget_it_edited_though_widgets_never_republish(tmp_path, monkeypatch):
    """In org mode an edited artifact cards only once its republish succeeds;
    a widget never publishes, so an edit of one still cards. Another edited
    artifact that didn't republish still doesn't."""
    base = tmp_path / "artifacts"
    for slug, meta in (("picker", {"display": "widget"}), ("report", {})):
        folder = base / slug
        folder.mkdir(parents=True)
        (folder / "index.html").write_text("<html></html>")
        (folder / "metadata.json").write_text(json.dumps({"slug": slug, "type": "html-app", **meta}))

    async def nothing_republished(*_a, **_kw):
        return set()

    monkeypatch.setattr(artifact_autopublish, "autopublish_project_artifacts", nothing_republished)
    seen: list[list[str]] = []

    def fake_cards(artifacts_base, slugs, **_kw):
        seen.append(list(slugs))
        return [
            SimpleNamespace(folder=str(base / slug), display="widget" if slug == "picker" else None)
            for slug in slugs
        ]

    monkeypatch.setattr("cowork.services.task_objects.cards_for_slugs", fake_cards)

    cards = await publish_and_card_turn_artifacts(
        base, new_slugs=[], touched_slugs={"picker", "report"}, scope=ORG_SCOPE, project_id="p1",
    )

    assert seen == [["picker", "report"]]
    assert [c.display for c in cards] == ["widget"]


# ── the Artifacts list ──────────────────────────────────────────────────


async def test_the_artifacts_list_leaves_widgets_out_unless_asked(session, tmp_path, local_mode):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="picker")
    _with_display(folder, "widget")
    inline = folder.parent / "chart"
    inline.mkdir()
    (inline / "index.html").write_text("<html></html>")
    (inline / "metadata.json").write_text(json.dumps({"slug": "chart", "name": "chart", "type": "html-app", "display": "inline"}))

    listed = await ep.list_artifacts(LOCAL_SCOPE, project_id=row.id, project_path=None)
    assert [c["slug"] for c in listed] == ["chart"]
    assert "canPublish" not in listed[0]["capabilities"]

    everything = await ep.list_artifacts(LOCAL_SCOPE, project_id=row.id, project_path=None, include_widgets=True)
    by_slug = {c["slug"]: c for c in everything}
    assert set(by_slug) == {"chart", "picker"}
    assert by_slug["picker"]["display"] == "widget"
    assert by_slug["picker"]["capabilities"]["canPublish"] is False
    assert by_slug["picker"]["capabilities"]["canShare"] is False


def test_the_include_widgets_query_param_defaults_to_off(session, tmp_path, local_mode):
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="picker")
    _with_display(folder, "widget")
    client = TestClient(create_app())

    assert client.get("/api/v1/artifacts/", params={"project_id": str(row.id)}).json() == []
    [card] = client.get("/api/v1/artifacts/", params={"project_id": str(row.id), "include_widgets": "true"}).json()
    assert card["display"] == "widget"


def test_capabilities_say_a_widget_cant_be_published_or_shared():
    caps = artifact_capabilities(SimpleNamespace(scope=LOCAL_SCOPE), None, "picker", display="widget")
    assert caps["canPublish"] is False and caps["canShare"] is False
    assert caps["canEdit"] is True
    for display in (None, "inline"):
        plain = artifact_capabilities(SimpleNamespace(scope=LOCAL_SCOPE), None, "chart", display=display)
        assert "canPublish" not in plain and "canShare" not in plain
    assert with_display_limits({"canEdit": True}, "inline") == {"canEdit": True}


# ── publishing and sharing ──────────────────────────────────────────────


@pytest.fixture
def widget_folder(tmp_path):
    base = tmp_path / ORG_SCOPE.org_id / "proj" / ".anton" / "artifacts"
    folder = base / "picker"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html><body>pick</body></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": "picker", "type": "html-app", "display": "widget"}))
    return folder


def test_publish_artifact_refuses_a_widget_before_anything_is_built(widget_folder, monkeypatch):
    def no_publisher(*_a, **_kw):
        raise AssertionError("nothing is built for a widget")

    monkeypatch.setattr(publish, "_a_hard_linked_file", no_publisher)

    with pytest.raises(publish.WidgetPublishRefused) as refused:
        publish.publish_artifact(
            widget_folder, artifacts_base=widget_folder.parent, api_key="k", publish_url="https://api.example",
        )
    # A ValueError: 400 on the REST paths, a failed tool result for the agent.
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == WIDGET_PUBLISH_REFUSAL


def test_autopublish_and_the_publishable_list_skip_a_widget(widget_folder, monkeypatch):
    decision = artifact_autopublish.needs_publish(widget_folder, widget_folder.parent)
    assert decision.reason == artifact_autopublish.REASON_NOT_PUBLISHABLE

    monkeypatch.setattr("cowork.services.artifacts._scan_artifact_dirs", lambda scope: [widget_folder.parent])
    assert html_artifacts(scope=None) == []
    (widget_folder / "metadata.json").write_text(json.dumps({"slug": "picker", "type": "html-app", "display": "inline"}))
    assert [row["title"] for row in html_artifacts(scope=None)] == ["Index"]


@pytest.fixture
def as_widget_owner(monkeypatch, widget_folder):
    monkeypatch.setattr(
        aw, "_owner_workspace",
        awaitable(lambda scope, project_ref, artifact_id: (
            SimpleNamespace(project_id="proj-1"), widget_folder,
            ArtifactMetadata.model_validate({"slug": "picker", "display": "widget"}), {},
        )),
    )

    def never(*_a, **_kw):
        raise AssertionError("a widget is never published")

    monkeypatch.setattr("cowork.services.publish.publish_artifact", never)
    return widget_folder


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_publish_changes_refuses_a_widget(as_widget_owner):
    with pytest.raises(HTTPException) as refused:
        await aw.republish_artifact("proj", ARTIFACT_ID, ORG_SCOPE, principal=None)
    assert refused.value.status_code == 400
    assert refused.value.detail == WIDGET_PUBLISH_REFUSAL


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_sharing_refuses_a_widget(as_widget_owner):
    body = aw._AccessBody(access={"mode": "public"})
    with pytest.raises(HTTPException) as refused:
        await aw.set_artifact_access("proj", ARTIFACT_ID, body, ORG_SCOPE, SimpleNamespace(), principal=None)
    assert refused.value.status_code == 400
    assert refused.value.detail == WIDGET_PUBLISH_REFUSAL
