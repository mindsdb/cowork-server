"""The artifacts service works over roots it is given, not roots it discovers.

In org mode projects live at `<root>/<org_id>/<project>`, one level deeper than
the module-level scan looks, so discovery returns nothing there. The card also
carries projectId/projectName because the client must stop deriving an
artifact's project from its filesystem path.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from cowork.services import artifacts as a
from cowork.services import artifact_identity


def _make_artifact(base, slug, *, files: dict[str, str], meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def source(tmp_path):
    base = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    return a.ProjectArtifacts(base=base, project_id="p-1", project_name="Proj")


def test_list_artifacts_reads_the_given_root(source):
    _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                   meta={"slug": "dash", "name": "Dash", "type": "html-app"})

    cards = a.list_artifacts([source])

    assert [c["slug"] for c in cards] == ["dash"]


def test_card_carries_project_identity(source):
    _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                   meta={"slug": "dash", "name": "Dash", "type": "html-app"})

    card = a.list_artifacts([source])[0]

    assert card["projectId"] == "p-1"
    assert card["projectName"] == "Proj"


def test_list_artifacts_merges_several_roots(tmp_path):
    first = a.ProjectArtifacts(base=tmp_path / "a" / ".anton" / "artifacts",
                               project_id="p-1", project_name="A")
    second = a.ProjectArtifacts(base=tmp_path / "b" / ".anton" / "artifacts",
                                project_id="p-2", project_name="B")
    _make_artifact(first.base, "one", files={"a.md": "1"}, meta={"slug": "one", "type": "document"})
    _make_artifact(second.base, "two", files={"b.md": "2"}, meta={"slug": "two", "type": "document"})

    cards = a.list_artifacts([first, second])

    assert {c["slug"]: c["projectName"] for c in cards} == {"one": "A", "two": "B"}


def test_list_artifacts_empty_for_no_sources():
    assert a.list_artifacts([]) == []


def test_list_artifacts_skips_a_missing_root(tmp_path):
    absent = a.ProjectArtifacts(base=tmp_path / "gone" / ".anton" / "artifacts",
                                project_id="p-9", project_name="Gone")
    assert a.list_artifacts([absent]) == []


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns


def test_list_derives_legacy_id_without_mutating_metadata(source):
    metadata = {
        "id": "a1b2c3d4",
        "slug": "dash",
        "createdAt": "2026-08-28T12:00:00Z",
        "primary": "index.html",
        "type": "html-app",
    }
    folder = _make_artifact(
        source.base,
        "dash",
        files={"index.html": "<html></html>"},
        meta=metadata,
    )
    metadata_path = folder / "metadata.json"
    before = metadata_path.read_bytes()
    before_identity = _file_identity(metadata_path)

    card = a.list_artifacts([source])[0]

    assert UUID(card["id"]).hex == card["id"]
    assert card["id"].startswith("a1b2c3d4")
    assert card["artifactKey"] == f"artifact/{UUID(card['id'])}"
    assert f"/drafts/{source.project_id}/{card['id']}/" in card["draftUrl"]
    assert metadata_path.read_bytes() == before
    assert _file_identity(metadata_path) == before_identity


def test_list_uses_stable_id_without_persisting_the_normalized_metadata(source):
    stable_id = "55555555-5555-4555-8555-555555555555"
    metadata = {
        "id": "a1b2c3d4",
        "stableId": stable_id,
        "slug": "dash",
        "createdAt": "2026-08-28T12:00:00Z",
        "type": "document",
    }
    folder = _make_artifact(
        source.base,
        "dash",
        files={"brief.md": "hello"},
        meta=metadata,
    )
    metadata_path = folder / "metadata.json"
    before = metadata_path.read_bytes()

    card = a.list_artifacts([source])[0]

    assert card["id"] == UUID(stable_id).hex
    assert metadata_path.read_bytes() == before
    assert json.loads(before)["stableId"] == stable_id


def test_list_derives_an_id_for_idless_metadata_without_persisting_it(source):
    folder = _make_artifact(
        source.base,
        "idless",
        files={"brief.md": "hello"},
        meta={
            "slug": "idless",
            "createdAt": "2026-08-28T12:00:00Z",
            "type": "document",
        },
    )
    metadata_path = folder / "metadata.json"

    card = a.list_artifacts([source])[0]

    assert UUID(card["id"]).hex == card["id"]
    assert "id" not in json.loads(metadata_path.read_text(encoding="utf-8"))


def test_list_call_graph_does_not_reach_mutating_identity_or_card_helpers(
    source, monkeypatch
):
    _make_artifact(
        source.base,
        "dash",
        files={"index.html": "<html></html>"},
        meta={"id": "a1b2c3d4", "slug": "dash", "type": "html-app"},
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("read-only listing reached a mutating helper")

    monkeypatch.setattr(artifact_identity, "ensure_full_id", unexpected)
    monkeypatch.setattr(artifact_identity, "_atomic_json", unexpected)
    monkeypatch.setattr(artifact_identity.os, "utime", unexpected)
    monkeypatch.setattr(a, "_is_modified", unexpected)
    monkeypatch.setattr(a, "_write_published_map_pinned", unexpected)

    assert a.list_artifacts([source])[0]["slug"] == "dash"


def test_list_does_not_self_heal_the_published_map(source, monkeypatch):
    folder = _make_artifact(
        source.base,
        "dash",
        files={"index.html": "<html></html>"},
        meta={
            "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "slug": "dash",
            "primary": "index.html",
            "type": "html-app",
        },
    )
    published_path = folder / ".published.json"
    published_path.write_text(
        json.dumps(
            {
                "index.html": {
                    "published": True,
                    "report_id": "report-id",
                    "published_mtime": 1,
                    "last_md5": "unchanged",
                }
            }
        ),
        encoding="utf-8",
    )
    before = published_path.read_bytes()
    before_identity = _file_identity(published_path)
    monkeypatch.setattr(
        "cowork.services.publish.compute_publish_md5",
        lambda *_args, **_kwargs: "unchanged",
    )
    monkeypatch.setattr(
        a,
        "_write_published_map_pinned",
        lambda *_args, **_kwargs: pytest.fail("listing attempted publish-map heal"),
    )

    card = a.list_artifacts([source])[0]

    assert card["modified"] is False
    assert published_path.read_bytes() == before
    assert _file_identity(published_path) == before_identity


def test_list_still_reports_a_real_published_content_change(source, monkeypatch):
    folder = _make_artifact(
        source.base,
        "dash",
        files={"index.html": "<html>changed</html>"},
        meta={
            "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "slug": "dash",
            "primary": "index.html",
            "type": "html-app",
        },
    )
    (folder / ".published.json").write_text(
        json.dumps(
            {
                "index.html": {
                    "published": True,
                    "report_id": "report-id",
                    "published_mtime": 1,
                    "last_md5": "before",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cowork.services.publish.compute_publish_md5",
        lambda *_args, **_kwargs: "after",
    )

    assert a.list_artifacts([source])[0]["modified"] is True


def test_read_only_list_card_matches_the_canonical_card_shape(source, monkeypatch):
    folder = _make_artifact(
        source.base,
        "dash",
        files={"index.html": "<html></html>"},
        meta={
            "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "slug": "dash",
            "name": "Dashboard",
            "primary": "index.html",
            "type": "html-app",
        },
    )
    monkeypatch.setattr(a.time, "time", lambda: folder.stat().st_mtime + 1)

    listed = a.list_artifacts([source])[0]
    canonical = a.card_for_folder(
        folder,
        project_id=source.project_id,
        project_name=source.project_name,
    )

    assert listed == canonical


def test_serve_url_is_empty_in_org_mode(source, monkeypatch):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    monkeypatch.setattr(a, "_org_mode", lambda: True)

    card = a.card_for_folder(folder, project_id="p-1", project_name="Proj")

    assert card["serveUrl"] == ""


def test_serve_url_still_built_in_desktop_mode(tmp_path, monkeypatch):
    # Paired with the test above: an unconditional `return ""` would pass that
    # one and silently break desktop preview.
    from cowork.common.settings.app_settings import get_app_settings

    projects_root = Path(get_app_settings().project.root_dir)
    base = projects_root / "served-proj" / ".anton" / "artifacts"
    folder = _make_artifact(base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    monkeypatch.setattr(a, "_org_mode", lambda: False)

    assert a.serve_url_for(folder / "index.html").startswith("/api/v1/artifacts/serve/")


def test_owner_side_access_fields_are_absent_in_org_mode(source, monkeypatch):
    # accessPassword is plaintext. Stripping it only at the list endpoint would
    # still leak it through inline chat cards, which call card_for_folder too.
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({
        "index.html": {"report_id": "rid", "url": "u", "published": True,
                       "mode": "password", "access_password": "s3cret",
                       "requires_password": True, "emails": ["x@example.com"]},
    }))
    monkeypatch.setattr(a, "_org_mode", lambda: True)

    card = a.card_for_folder(folder, project_id="p-1", project_name="Proj")

    assert "accessPassword" not in card
    assert "accessEmails" not in card
    assert card["accessMode"] == "password"  # the badge still needs the mode


def test_owner_side_access_fields_remain_in_desktop_mode(source, monkeypatch):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({
        "index.html": {"report_id": "rid", "url": "u", "published": True,
                       "mode": "password", "access_password": "s3cret",
                       "requires_password": True},
    }))
    monkeypatch.setattr(a, "_org_mode", lambda: False)

    card = a.card_for_folder(folder)

    assert card["accessPassword"] == "s3cret"


def test_load_published_map_reads_record(source):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({"index.html": {"report_id": "rid"}}))

    assert a.load_published_map(folder)["index.html"]["report_id"] == "rid"


def test_load_published_map_empty_when_absent(source):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    assert a.load_published_map(folder) == {}


def test_delete_artifact_removes_folder_after_unpublish(source, monkeypatch):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    (folder / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "rid", "url": "u", "published": True}})
    )
    calls = []
    monkeypatch.setattr(
        "cowork.services.publish.unpublish_artifact",
        lambda art, **kw: calls.append((art, kw.get("api_key"))) or {"status": "ok"},
    )

    a.delete_artifact(folder, artifacts_base=source.base, api_key="k",
                      publish_url="https://api.staging.mindshub.ai")

    assert calls and calls[0][1] == "k"
    assert not folder.exists()


def test_delete_artifact_without_publish_record_just_removes_it(source, monkeypatch):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    calls = []
    monkeypatch.setattr("cowork.services.publish.unpublish_artifact",
                        lambda art, **kw: calls.append(art))

    a.delete_artifact(folder, artifacts_base=source.base, api_key="k",
                      publish_url="https://api.staging.mindshub.ai")

    assert calls == []
    assert not folder.exists()


def test_delete_artifact_keeps_folder_when_unpublish_fails(source, monkeypatch):
    folder = _make_artifact(source.base, "dash", files={"index.html": "<html></html>"},
                            meta={"slug": "dash", "type": "html-app"})
    (folder / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "rid", "published": True}})
    )

    def boom(art, **kw):
        raise RuntimeError("Unpublishing failed: HTTP Error 500")

    monkeypatch.setattr("cowork.services.publish.unpublish_artifact", boom)

    with pytest.raises(RuntimeError):
        a.delete_artifact(folder, artifacts_base=source.base, api_key="k",
                          publish_url="https://api.staging.mindshub.ai")
    assert folder.exists()


def test_delete_logs_orphan_when_primary_file_is_gone(source, monkeypatch, caplog):
    folder = _make_artifact(source.base, "dash", files={"other.txt": "x"},
                            meta={"slug": "dash", "type": "document"})
    (folder / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "rid", "url": "https://view.example/r/1",
                                   "published": True}})
    )
    called = []
    monkeypatch.setattr("cowork.services.publish.unpublish_artifact",
                        lambda art, **kw: called.append(art))

    with caplog.at_level("WARNING"):
        a.delete_artifact(folder, artifacts_base=source.base, api_key="k",
                          publish_url="https://api.staging.mindshub.ai")

    assert called == []  # can't unpublish a record whose file is gone
    assert "orphaned_publish" in caplog.text
    assert not folder.exists()


def test_delete_artifact_rejects_folder_outside_the_given_root(source, tmp_path):
    outside = tmp_path / "elsewhere" / "evil"
    outside.mkdir(parents=True)
    (outside / "metadata.json").write_text(json.dumps({"slug": "evil", "type": "document"}))

    with pytest.raises(FileNotFoundError):
        a.delete_artifact(outside, artifacts_base=source.base, api_key="k",
                          publish_url="https://api.staging.mindshub.ai")
    assert outside.exists()
