"""Deciding whether an artifact needs (re)publishing.

The source of truth is the artifact's own `.published.json` — no new state is
introduced anywhere. That is what makes retry free: a publish that failed left no
record, so the next turn simply sees "new" again.
"""
from __future__ import annotations

import json
import os

import pytest

from cowork.services.artifact_autopublish import needs_publish
from cowork.services.artifacts import content_mtime
from cowork.services.publish import compute_publish_md5


def _make(base, slug, *, files: dict[str, str], meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def _publish_record(folder, base, key, **overrides):
    entry = {
        "report_id": "rid-1",
        "url": "https://view.example/r/1",
        "published": True,
        "last_md5": compute_publish_md5(folder, artifacts_base=base),
        "published_mtime": content_mtime(folder),
    }
    entry.update(overrides)
    (folder / ".published.json").write_text(json.dumps({key: entry}))


def _touch_forward(path, delta_s=120):
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + delta_s))


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    root.mkdir(parents=True)
    return root


def test_unpublished_static_artifact_is_new(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == ("new", None)


def test_static_primary_named_report_html_still_counts(base):
    # _zip_html renames a single file to index.html INSIDE the archive, so the
    # criterion is the suffix, never the on-disk name.
    folder = _make(base, "rep", files={"report.html": "<html></html>"},
                   meta={"slug": "rep", "type": "html-app"})
    assert needs_publish(folder, base).action == "new"


def test_markdown_artifact_is_publishable(base):
    folder = _make(base, "doc", files={"notes.md": "# hi"},
                   meta={"slug": "doc", "type": "document"})
    assert needs_publish(folder, base).action == "new"


def test_metadata_only_folder_has_no_content(base):
    folder = base / "empty"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"slug": "empty", "type": "document"}))
    (folder / "README.md").write_text("nothing yet")

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "no_content")


def test_dataset_only_artifact_is_not_publishable(base):
    folder = _make(base, "data", files={"rows.csv": "a,b\n1,2"},
                   meta={"slug": "data", "type": "dataset"})

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "not_publishable")


def test_fullstack_with_required_layout_is_new(base):
    folder = _make(
        base, "app",
        files={"backend.py": "print('x')", "static/index.html": "<html></html>",
               "requirements.txt": "flask"},
        meta={"slug": "app", "type": "fullstack-stateless-app"},
    )

    d = needs_publish(folder, base)

    assert d.action == "new"
    assert d.is_fullstack is True


def test_fullstack_without_static_index_is_not_publishable(base):
    folder = _make(
        base, "app",
        files={"backend.py": "print('x')", "static/main.html": "<html></html>"},
        meta={"slug": "app", "type": "fullstack-stateless-app"},
    )

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "not_publishable")


def test_fullstack_without_backend_is_not_publishable(base):
    folder = _make(base, "app", files={"static/index.html": "<html></html>"},
                   meta={"slug": "app", "type": "fullstack-stateless-app"})

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "not_publishable")


def test_published_and_unchanged_is_skipped(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    _publish_record(folder, base, "report.html")

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "unchanged")


def test_changed_content_is_changed(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    _publish_record(folder, base, "report.html")

    (folder / "report.html").write_text("<html>v2</html>")
    _touch_forward(folder / "report.html")

    assert needs_publish(folder, base).action == "changed"


def test_touch_without_content_change_stays_unchanged(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    _publish_record(folder, base, "report.html")

    _touch_forward(folder / "report.html")  # mtime gate fires, md5 does not

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "unchanged")


def test_soft_deleted_record_is_not_resurrected(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    _publish_record(folder, base, "report.html", published=False)

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "unpublished")


def test_record_without_report_id_is_new_again(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({"report.html": {"url": ""}}))

    assert needs_publish(folder, base).action == "new"


def test_decision_carries_content_mtime(base):
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    assert needs_publish(folder, base).content_mtime > 0


def test_unresolvable_md5_does_not_republish_on_a_guess(base, monkeypatch):
    # None means "can't tell". Treating it as changed would republish every turn.
    folder = _make(base, "rep", files={"report.html": "<html>v1</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    _publish_record(folder, base, "report.html")
    _touch_forward(folder / "report.html")
    monkeypatch.setattr(
        "cowork.services.publish.compute_publish_md5", lambda *a, **k: None
    )

    d = needs_publish(folder, base)

    assert (d.action, d.reason) == (None, "unchanged")
