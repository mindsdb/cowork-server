"""Tests for the published-artifact `modified` badge + update flow (2026-06-23)."""

import json
import os
from pathlib import Path

from cowork.services.artifacts import _content_mtime


def _touch(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def test_content_mtime_is_max_over_user_files(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"id": "a", "type": "html-app", "primary": "index.html"}), encoding="utf-8"
    )
    a = tmp_path / "index.html"
    a.write_text("<h1>hi</h1>", encoding="utf-8")
    b = tmp_path / "data.csv"
    b.write_text("x,y\n1,2\n", encoding="utf-8")
    _touch(a, 1000.0)
    _touch(b, 2000.0)
    # Housekeeping files must not count toward the content mtime.
    pub = tmp_path / ".published.json"
    pub.write_text("{}", encoding="utf-8")
    _touch(pub, 9999.0)
    assert _content_mtime(tmp_path) == 2000


def test_content_mtime_empty_folder_is_zero(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"id": "a", "type": "mixed"}), encoding="utf-8"
    )
    assert _content_mtime(tmp_path) == 0


# ---------------------------------------------------------------------------
# Task 2: compute_publish_md5
# ---------------------------------------------------------------------------

import hashlib
from contextlib import ExitStack
from unittest.mock import patch

import cowork.services.publish as publish_mod
from anton.publisher import _zip_html


def _make_static_html(tmp_path: Path, body: str = "<h1>hi</h1>") -> Path:
    """A folder-based static HTML artifact: metadata.json + index.html at root."""
    root = tmp_path / "static-art"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"id": "static-art", "type": "html-app", "primary": "index.html"}),
        encoding="utf-8",
    )
    (root / "index.html").write_text(body, encoding="utf-8")
    return root


def _patch_scan(container: Path):
    return patch("cowork.services.artifacts._scan_artifact_dirs", lambda: [container])


def test_compute_publish_md5_matches_zip_md5(tmp_path: Path):
    root = _make_static_html(tmp_path)
    # Reference md5: exactly what the lambda stores — md5 of the zip bytes
    # anton produces for the primary file.
    expected = hashlib.md5(_zip_html(root / "index.html")).hexdigest()
    with _patch_scan(tmp_path):
        got = publish_mod.compute_publish_md5(str(root))
    assert got == expected


def test_compute_publish_md5_unresolvable_returns_none(tmp_path: Path):
    # No _patch_scan -> resolve_artifact_path raises -> None (can't tell).
    got = publish_mod.compute_publish_md5(str(tmp_path / "nope"))
    assert got is None


# ---------------------------------------------------------------------------
# Task 3: publish_artifact records published_mtime
# ---------------------------------------------------------------------------

from pydantic import SecretStr


class _FakeUserSettings:
    minds_api_key = SecretStr("test-key")
    publish_url = "https://4nton.ai"


class _FakeAppSettings:
    class connector:  # noqa: N801
        vault_dir = "/tmp/does-not-matter"


def _patched_publish(container: Path, view_url="https://4nton.ai/a/uuid-1",
                     report_id="uuid-1", md5="m1"):
    stack = ExitStack()
    stack.enter_context(_patch_scan(container))
    stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
    stack.enter_context(patch.object(publish_mod, "get_app_settings", lambda: _FakeAppSettings()))
    stack.enter_context(patch.object(publish_mod, "_load_state", lambda: {}))
    stack.enter_context(patch.object(publish_mod, "_save_state", lambda state: None))
    stack.enter_context(patch("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object()))
    stack.enter_context(
        patch("anton.publisher.publish",
              lambda *a, **k: {"view_url": view_url, "report_id": report_id, "md5": md5})
    )
    return stack


def test_publish_records_published_mtime(tmp_path: Path):
    root = _make_static_html(tmp_path)
    expected_mtime = publish_mod._content_mtime(root)
    with _patched_publish(tmp_path):
        publish_mod.publish_artifact(str(root))
    entry = json.loads((root / ".published.json").read_text(encoding="utf-8"))["index.html"]
    assert entry["published_mtime"] == expected_mtime
    assert entry["last_md5"] == "m1"


# ---------------------------------------------------------------------------
# Task 4: card_for_folder `modified` flag
# ---------------------------------------------------------------------------

from cowork.services.artifacts import card_for_folder


def _publish_static(tmp_path: Path) -> Path:
    """Make + publish a static artifact, returning its folder."""
    root = _make_static_html(tmp_path)
    with _patched_publish(tmp_path, md5=hashlib.md5(_zip_html(root / "index.html")).hexdigest()):
        publish_mod.publish_artifact(str(root))
    return root


def test_modified_false_when_unchanged(tmp_path: Path):
    root = _publish_static(tmp_path)
    with _patch_scan(tmp_path):
        card = card_for_folder(root)
    assert card["modified"] is False


def test_modified_false_for_unpublished(tmp_path: Path):
    root = _make_static_html(tmp_path)  # never published
    with _patch_scan(tmp_path):
        card = card_for_folder(root)
    assert card["modified"] is False


def test_modified_true_after_content_change(tmp_path: Path):
    root = _publish_static(tmp_path)
    # Change content AND bump mtime past published_mtime.
    idx = root / "index.html"
    idx.write_text("<h1>CHANGED</h1>", encoding="utf-8")
    _touch(idx, publish_mod._content_mtime(root) + 100)
    with _patch_scan(tmp_path):
        card = card_for_folder(root)
    assert card["modified"] is True


def test_touch_without_change_self_heals(tmp_path: Path):
    root = _publish_static(tmp_path)
    pub = root / ".published.json"
    old = json.loads(pub.read_text(encoding="utf-8"))["index.html"]["published_mtime"]
    # Bump mtime but keep identical content (md5 will match).
    idx = root / "index.html"
    _touch(idx, old + 100)
    with _patch_scan(tmp_path):
        card = card_for_folder(root)
    assert card["modified"] is False
    # Self-heal: published_mtime advanced so the next listing hits the cheap gate.
    healed = json.loads(pub.read_text(encoding="utf-8"))["index.html"]["published_mtime"]
    assert healed >= old + 100


# ---------------------------------------------------------------------------
# Task 5: update_artifact service
# ---------------------------------------------------------------------------


def test_update_reuses_report_id_and_refreshes_state(tmp_path: Path):
    root = _make_static_html(tmp_path)
    # Pre-existing published record (public).
    (root / ".published.json").write_text(
        json.dumps({"index.html": {
            "report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
            "last_md5": "old", "published": True, "mode": "public",
            "published_mtime": 1,
        }}),
        encoding="utf-8",
    )
    seen = {}

    def _spy_publish(*a, **k):
        seen["report_id"] = k.get("report_id")
        seen["access"] = k.get("access")
        return {"view_url": "https://4nton.ai/a/uuid-1", "report_id": "uuid-1", "md5": "new"}

    with ExitStack() as stack:
        stack.enter_context(_patch_scan(tmp_path))
        stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
        stack.enter_context(patch.object(publish_mod, "get_app_settings", lambda: _FakeAppSettings()))
        stack.enter_context(patch.object(publish_mod, "_load_state", lambda: {}))
        stack.enter_context(patch.object(publish_mod, "_save_state", lambda state: None))
        stack.enter_context(patch("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object()))
        stack.enter_context(patch("anton.publisher.publish", _spy_publish))
        out = publish_mod.update_artifact(str(root))

    assert seen["report_id"] == "uuid-1"          # same anchor -> AWS update
    assert out["status"] == "ok"
    entry = json.loads((root / ".published.json").read_text(encoding="utf-8"))["index.html"]
    assert entry["last_md5"] == "new"             # hash refreshed
    assert entry["published_mtime"] == publish_mod._content_mtime(root)  # snapshot refreshed
    assert entry["mode"] == "public"


def test_update_preserves_password_access(tmp_path: Path):
    root = _make_static_html(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {
            "report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
            "last_md5": "old", "published": True, "mode": "password",
            "requires_password": True, "access_password": "s3cret", "pwd_version": 2,
            "published_mtime": 1,
        }}),
        encoding="utf-8",
    )
    seen = {}

    def _spy_publish(*a, **k):
        seen["access"] = k.get("access")
        return {"view_url": "https://4nton.ai/a/uuid-1", "report_id": "uuid-1", "md5": "new"}

    with ExitStack() as stack:
        stack.enter_context(_patch_scan(tmp_path))
        stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
        stack.enter_context(patch.object(publish_mod, "get_app_settings", lambda: _FakeAppSettings()))
        stack.enter_context(patch.object(publish_mod, "_load_state", lambda: {}))
        stack.enter_context(patch.object(publish_mod, "_save_state", lambda state: None))
        stack.enter_context(patch("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object()))
        stack.enter_context(patch("anton.publisher.publish", _spy_publish))
        publish_mod.update_artifact(str(root))

    assert seen["access"] == {"mode": "password", "password": "s3cret"}
    entry = json.loads((root / ".published.json").read_text(encoding="utf-8"))["index.html"]
    assert entry["mode"] == "password"
    assert entry["access_password"] == "s3cret"


def test_update_unpublished_raises(tmp_path: Path):
    import pytest
    root = _make_static_html(tmp_path)  # no .published.json at all
    with _patch_scan(tmp_path):
        with pytest.raises(FileNotFoundError):
            publish_mod.update_artifact(str(root))


# ---------------------------------------------------------------------------
# Task 6: POST /publish/update endpoint
# ---------------------------------------------------------------------------


def test_update_endpoint_delegates(tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cowork.api.v1.endpoints import publish as publish_ep

    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    client = TestClient(app)

    captured = {}

    def _fake_update(path):
        captured["path"] = path
        return {"status": "ok", "url": "https://4nton.ai/a/uuid-1"}

    with patch.object(publish_ep, "_update", _fake_update):
        res = client.post("/api/v1/publish/update", json={"path": "/some/art"})
    assert res.status_code == 200
    assert res.json()["url"] == "https://4nton.ai/a/uuid-1"
    assert captured["path"] == "/some/art"


def test_update_endpoint_404_when_not_published(tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cowork.api.v1.endpoints import publish as publish_ep

    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    client = TestClient(app)

    def _raise(path):
        raise FileNotFoundError("No published version to update")

    with patch.object(publish_ep, "_update", _raise):
        res = client.post("/api/v1/publish/update", json={"path": "/some/art"})
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Task 6: artifact_status — the preview viewer's live in-place refresh (ENG-468)
# ---------------------------------------------------------------------------

from cowork.services.artifacts import artifact_status


def test_artifact_status_unpublished(tmp_path: Path):
    root = _make_static_html(tmp_path)  # never published
    with _patch_scan(tmp_path):
        s = artifact_status(str(root))
    assert s["publishedUrl"] == ""
    assert s["modified"] is False
    assert s["accessMode"] == "public"
    assert s["accessProtected"] is False


def test_artifact_status_published_unmodified(tmp_path: Path):
    root = _publish_static(tmp_path)
    with _patch_scan(tmp_path):
        s = artifact_status(str(root))
    assert s["publishedUrl"]            # carries the view_url
    assert s["modified"] is False


def test_artifact_status_modified_after_change(tmp_path: Path):
    root = _publish_static(tmp_path)
    idx = root / "index.html"
    idx.write_text("<h1>CHANGED</h1>", encoding="utf-8")
    _touch(idx, publish_mod._content_mtime(root) + 100)
    with _patch_scan(tmp_path):
        s = artifact_status(str(root))
    assert s["modified"] is True
    assert s["publishedUrl"]            # still published


def test_artifact_status_unknown_path_is_blank(tmp_path: Path):
    # A path that can't be resolved → blank default, never raises.
    with _patch_scan(tmp_path):
        s = artifact_status(str(tmp_path / "does-not-exist"))
    assert s["publishedUrl"] == ""
    assert s["modified"] is False
    assert s["accessMode"] == "public"
