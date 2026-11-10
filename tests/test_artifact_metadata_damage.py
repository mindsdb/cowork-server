"""One artifact's damaged metadata.json neither hides its siblings nor breaks
the paths that read it."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest
from anton.core.artifacts.models import extend_legacy_id

from cowork.api.v1.endpoints import search
from cowork.api.v1.endpoints.artifact_workspace import _workspace_state
from cowork.api.v1.endpoints.search import _RowMatches, search_cowork
from cowork.common.paths import pinned_dir
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services import artifacts as a
from cowork.services.artifact_export import _record_primary
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_ownership import provenance_origin
from cowork.services.artifact_revisions import capture_agent_revision
from cowork.services.task_objects import _artifact_owner, cards_for_slugs, turn_artifact_changes

GOOD_ID = "0123456789abcdef0123456789abcdef"
DAMAGED = {
    "type-list": {"type": ["html"]},
    "primary-int": {"primary": 5},
    "name-int": {"name": 5},
    "name-surrogate": {"name": "bad" + chr(0xD800)},
}


def _json(cards):
    """Cards as they go out, whether the service returns dicts or models."""
    return [card.to_json() if hasattr(card, "to_json") else card for card in cards]


def _artifact(base, slug, **meta):
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html><body>hi</body></html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": slug, "type": "html-app", **meta}))
    return folder


def _raw_artifact(base, slug, text):
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(text)
    return folder


@pytest.fixture
def source(tmp_path):
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return a.ProjectArtifacts(base=base, project_id=None, project_name="proj")


@pytest.fixture
def tree(source):
    _artifact(source.base, "good", name="Good")
    for slug, meta in DAMAGED.items():
        _artifact(source.base, slug, **meta)
    return source


def test_the_list_keeps_every_artifact_with_the_fallbacks_of_a_missing_field(tree):
    cards = {card["slug"]: card for card in _json(a.list_artifacts([tree]))}

    assert set(cards) == {"good", *DAMAGED}
    # A missing type reads as "mixed", whose kind is "Bundle" (KIND_BY_TYPE).
    # Never change the card builder to fit this test: every artifact without
    # a type would change kind on the wire.
    assert (cards["type-list"]["type"], cards["type-list"]["kind"]) == ("mixed", "Bundle")
    assert cards["name-int"]["title"] == "name-int"
    assert cards["name-surrogate"]["title"] == "name-surrogate"
    assert cards["primary-int"]["primary"] is None
    for card in cards.values():
        json.dumps(card, ensure_ascii=False).encode("utf-8")


def test_inline_cards_build_for_damaged_artifacts(tree):
    cards = _json(cards_for_slugs(tree.base, ["good", *DAMAGED], scope=LOCAL_SCOPE, project_id=None, project_name="proj"))

    assert [card["slug"] for card in cards] == ["good", *DAMAGED]


@pytest.mark.parametrize("slug", ["primary-int", "type-list", "name-int"])
def test_the_workspace_reads_a_damaged_artifact(tree, slug):
    folder = tree.base / slug
    artifact_id, metadata = ensure_full_id(folder)

    state = _workspace_state(tree, folder, metadata, artifact_id, None, with_export=True)

    assert state.export_source.source_name == "index.html"


def test_turn_end_records_every_artifact(tree):
    changes = turn_artifact_changes(conversation_id=uuid4(), artifacts_base=tree.base, before=set(), before_mtimes={})

    assert set(changes.created) == {"good", *DAMAGED}


def test_a_capture_reads_a_numeric_primary(tree):
    assert capture_agent_revision(tree.base / "primary-int", conversation_id="c-1") is not None


async def test_search_answers_beside_damaged_artifacts(tree, monkeypatch):
    async def run_db(fn, *, scope):
        return _RowMatches(before_artifacts=[], after_artifacts=[])

    async def run_efs(fn, *, scope, deadline=True):
        return fn()

    monkeypatch.setattr(search, "run_db", run_db)
    monkeypatch.setattr(search, "run_efs", run_efs)
    monkeypatch.setattr(search, "_sources_for_scan", lambda *, scope: [tree])

    found = await search_cowork(LOCAL_SCOPE, q="Good", limit=25)

    assert "Good" in [hit["title"] for hit in found["results"]]


def test_a_metadata_array_holds_no_card(source):
    _artifact(source.base, "good", name="Good")
    folder = _raw_artifact(source.base, "array", "[]")

    assert a.card_for_folder(folder, scope=LOCAL_SCOPE) is None
    assert [card["slug"] for card in _json(a.list_artifacts([source]))] == ["good"]
    assert _artifact_owner(folder) is None


def test_a_metadata_document_nested_too_deeply_holds_no_card(source):
    folder = _raw_artifact(source.base, "deep", '{"a": ' + "[" * 100_000 + "]" * 100_000 + "}")

    assert a.card_for_folder(folder, scope=LOCAL_SCOPE) is None
    assert provenance_origin(folder) is None


def test_a_valid_id_with_a_deeply_nested_created_at_still_lists(source):
    _raw_artifact(source.base, "deep-created", json.dumps({
        "id": GOOD_ID, "slug": "deep-created", "type": "html-app", "createdAt": json.loads("[" * 300 + "]" * 300),
    }))

    assert [card["id"] for card in _json(a.list_artifacts([source]))] == [GOOD_ID]


FULLSTACK = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})


def test_a_fullstack_app_with_a_numeric_primary_is_publishable(source):
    folder = _artifact(source.base, "app", type="fullstack-stateless-app", primary=5)

    candidates = a._publishable_in_folder(folder, source.base, FULLSTACK)

    assert [c.path.name for c in candidates] == ["index.html"]


def test_a_type_list_is_not_publishable_as_fullstack(source):
    folder = _artifact(source.base, "pages", type=["fullstack-stateless-app"])

    candidates = a._publishable_in_folder(folder, source.base, FULLSTACK)

    assert [c.path.name for c in candidates] == ["index.html"]


# ── identity ────────────────────────────────────────────────────────────────

def test_a_numeric_id_widens_as_before(source):
    folder = _artifact(source.base, "numeric", id=5, createdAt="2026-01-01T00:00:00Z")

    artifact_id, _metadata = ensure_full_id(folder)

    assert artifact_id == extend_legacy_id("5", "2026-01-01T00:00:00Z")


def test_a_damaged_hex_id_holds_no_id_and_its_sibling_lists(source):
    _artifact(source.base, "good", name="Good")
    folder = _artifact(source.base, "damaged", id="0123456789abcdef0123")

    with pytest.raises(ValueError):
        ensure_full_id(folder)
    assert [card["slug"] for card in _json(a.list_artifacts([source]))] == ["good"]


# ── writers ─────────────────────────────────────────────────────────────────

def _document(folder):
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def test_id_widening_changes_only_the_id(source):
    folder = _artifact(source.base, "legacy", id="a1b2c3d4", stableId="", name=5, custom={"a": 1})

    artifact_id, _metadata = ensure_full_id(folder)

    assert _document(folder) == {
        "slug": "legacy", "type": "html-app", "id": artifact_id, "name": 5, "custom": {"a": 1},
    }
    assert list(_document(folder))[:3] == ["slug", "type", "id"]


def test_id_widening_skips_a_document_it_cannot_write_back(source):
    folder = _artifact(source.base, "legacy", id="a1b2c3d4", files=[{"path": "bad" + chr(0xD800)}])
    before = (folder / "metadata.json").read_bytes()

    artifact_id, _metadata = ensure_full_id(folder)

    assert len(artifact_id) == 32
    assert (folder / "metadata.json").read_bytes() == before


def test_recording_a_primary_changes_only_the_primary(source):
    folder = _artifact(source.base, "export", name=5, custom={"a": 1})

    with pinned_dir(folder, nofollow_base=True) as pinned:
        _record_primary(folder=pinned, primary="index.html")

    assert _document(folder) == {
        "slug": "export", "type": "html-app", "name": 5, "custom": {"a": 1}, "primary": "index.html",
    }


@pytest.mark.parametrize("text", ["[]", json.dumps({"files": [{"path": "bad" + chr(0xD800)}]})])
def test_recording_a_primary_skips_a_document_it_cannot_use(source, text):
    folder = _raw_artifact(source.base, "export", text)
    before = (folder / "metadata.json").read_bytes()

    with pinned_dir(folder, nofollow_base=True) as pinned:
        _record_primary(folder=pinned, primary="index.html")

    assert (folder / "metadata.json").read_bytes() == before


def test_recording_a_backend_port_changes_only_the_port(source):
    folder = _artifact(source.base, "app", type="fullstack-stateless-app", port=5000, name=5, custom={"a": 1})

    a._record_backend_port(folder, 8123)

    assert _document(folder) == {
        "slug": "app", "type": "fullstack-stateless-app", "port": 8123, "name": 5, "custom": {"a": 1},
    }


# ── status ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["[]", json.dumps({"type": ["html"], "primary": 5, "name": 5})])
def test_the_desktop_status_answers_for_damaged_metadata(tmp_path, monkeypatch, text):
    from fastapi.testclient import TestClient

    from cowork.api.v1.endpoints import artifacts as ep
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    base = tmp_path / "proj" / ".anton" / "artifacts"
    folder = _raw_artifact(base, "damaged", text)
    root = a.ProjectArtifacts(
        base=base, project_id=None, project_name="proj",
        trusted_anchor=tmp_path / "proj", root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(ep, "artifacts_sources_for_scan", lambda *, scope: [root])
    try:
        res = TestClient(create_app()).get("/api/v1/artifacts/status", params={"path": str(folder / "index.html")})
    finally:
        get_app_settings.cache_clear()

    assert res.status_code == 200


from cowork.services.artifact_audience import ShareNotify, share_notice
from cowork.services.preview_proxy import _read_backend_port


@pytest.mark.parametrize("text", ["[]", '{"port": "8080"}', '{"port": true}', '{"port": 70000}'])
def test_a_backend_port_that_is_not_a_port_reads_as_none(tmp_path, text):
    (tmp_path / "metadata.json").write_text(text)

    assert _read_backend_port(tmp_path) is None


def test_a_backend_port_reads(tmp_path):
    (tmp_path / "metadata.json").write_text('{"port": 8080}')

    assert _read_backend_port(tmp_path) == 8080


def test_a_share_notice_titles_a_numeric_name_by_its_folder(tmp_path):
    folder = tmp_path / "quarterly"
    folder.mkdir()
    (folder / "metadata.json").write_text('{"name": 5}')

    notice = share_notice(folder, "index.html", "https://view.example/r/1", recipients=ShareNotify(emails=("a@b.co",)))

    assert notice.artifact_title == "quarterly"


def test_a_published_url_that_is_not_text_lists_as_unpublished(source):
    _artifact(source.base, "good", name="Good")
    folder = _artifact(source.base, "odd-url", name="Odd")
    (folder / ".published.json").write_text(json.dumps({"index.html": {"url": 5, "published": True}}))

    cards = {card["slug"]: card for card in _json(a.list_artifacts([source]))}
    inline = _json(cards_for_slugs(source.base, ["odd-url"], scope=LOCAL_SCOPE, project_id=None, project_name="proj"))

    assert set(cards) == {"good", "odd-url"}
    assert cards["odd-url"]["publishedUrl"] == ""
    assert inline[0]["publishedUrl"] == ""
