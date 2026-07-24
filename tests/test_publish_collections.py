"""publish_artifact must surface StatePublishBlocked as its own message, not the
generic 'check your credentials' error; unpublish_artifact must clear the state
snapshot so a later publish with a changed collection set is treated as fresh."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import cowork.services.publish as publish_mod
from anton.publisher import StatePublishBlocked, _STATE_SNAPSHOT


def _make_stateful(tmp_path: Path) -> Path:
    root = tmp_path / "art-1"
    (root / "static").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "id": "art-1", "type": "fullstack-stateful-app",
        "primary": "static/index.html"}), encoding="utf-8")
    (root / "static" / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (root / "backend.py").write_text("# backend\n", encoding="utf-8")
    return root


def test_publish_artifact_surfaces_block(tmp_path, monkeypatch):
    root = _make_stateful(tmp_path)
    monkeypatch.setattr(publish_mod, "get_user_settings",
                        lambda: type("S", (), {"minds_api_key": "k"})())
    monkeypatch.setattr(publish_mod, "_resolve_publish_endpoint",
                        lambda s: ("https://view.test", "k"))
    monkeypatch.setattr(publish_mod, "resolve_artifact_path", lambda p, **k: root)

    def _blocked(*a, **k):
        raise StatePublishBlocked("Collections ['b'] ... orphaned.")

    with patch("anton.publisher.publish", _blocked):
        with pytest.raises(RuntimeError) as ei:
            publish_mod.publish_artifact(str(root))
    # The block message must survive — NOT the generic credentials text.
    assert "orphaned" in str(ei.value)
    assert "credentials" not in str(ei.value).lower()


def test_unpublish_artifact_clears_snapshot(tmp_path, monkeypatch):
    root = _make_stateful(tmp_path)
    (root / ".published.json").write_text(json.dumps({
        "index.html": {"report_id": "r1", "url": "u", "last_md5": "m",
                       "published": True}}), encoding="utf-8")
    snap = root / _STATE_SNAPSHOT
    snap.write_text(json.dumps({"collections": ["comments"]}), encoding="utf-8")

    monkeypatch.setattr(publish_mod, "get_user_settings",
                        lambda: type("S", (), {"minds_api_key": "k"})())
    monkeypatch.setattr(publish_mod, "_resolve_publish_endpoint",
                        lambda s: ("https://view.test", "k"))
    monkeypatch.setattr(publish_mod, "resolve_artifact_path", lambda p, **k: root)

    with patch("anton.publisher.unpublish", return_value={}):
        publish_mod.unpublish_artifact(str(root))

    assert not snap.exists()  # baseline reset even though local record is soft-kept
