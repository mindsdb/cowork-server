"""Tests for version history + rollback: list_versions / activate_version."""

import io
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest
from pydantic import SecretStr

import cowork.services.publish as publish_mod


def _make_fullstack(tmp_path: Path) -> Path:
    """A fullstack artifact: metadata.json at root, primary in static/."""
    root = tmp_path / "art-1"
    (root / "static").mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps({"id": "art-1", "type": "fullstack-stateless-app", "primary": "static/index.html"}),
        encoding="utf-8",
    )
    (root / "static" / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (root / "backend.py").write_text("# backend\n", encoding="utf-8")
    return root


def _write_map(root: Path, entry: dict, name: str = "index.html") -> None:
    (root / ".published.json").write_text(json.dumps({name: entry}), encoding="utf-8")


def _read_map(root: Path, name: str = "index.html") -> dict:
    return json.loads((root / ".published.json").read_text(encoding="utf-8"))[name]


class _FakeUserSettings:
    minds_api_key = SecretStr("test-key")
    minds_url = "https://api.mindshub.ai/v1"
    openai_base_url = ""
    openai_api_key = None
    publish_url = ""


def _patch_scan(container: Path):
    return patch("cowork.services.artifacts._scan_artifact_dirs", lambda: [container])


def _base_patches(container: Path):
    stack = ExitStack()
    stack.enter_context(_patch_scan(container))
    stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
    return stack


def _http_error(code: int, message: str) -> HTTPError:
    body = io.BytesIO(json.dumps({"error": message}).encode())
    return HTTPError("https://x/activate/r1", code, "err", {}, body)


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------


def test_list_versions_shapes_history_newest_first(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r1", "url": "https://x/a/r1", "published": True})

    backend_resp = {
        "report_id": "r1",
        "current_md5": "m1",
        "artifact_type": None,
        "versions": [
            {"md5": "m1", "published_at": "2026-01-01", "title": "v1"},
            {"md5": "m2", "published_at": "2026-01-02", "title": "v2"},
        ],
    }
    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.list_versions", lambda *a, **k: backend_resp))
        out = publish_mod.list_versions(str(root))

    assert out["reportId"] == "r1"
    assert out["currentMd5"] == "m1"
    # Newest publish first: m2 then m1.
    assert [v["md5"] for v in out["versions"]] == ["m2", "m1"]
    assert out["versions"][1]["isCurrent"] is True   # m1 is current
    assert out["versions"][0]["isCurrent"] is False
    assert out["versions"][0]["publishedAt"] == "2026-01-02"


def test_list_versions_forwards_report_id_and_key(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r-xyz", "url": "https://x/a/r-xyz", "published": True})
    seen = {}

    def _spy(report_id, *, api_key, publish_url, ssl_verify):
        seen["report_id"] = report_id
        seen["api_key"] = api_key
        return {"report_id": report_id, "current_md5": "m1", "versions": []}

    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.list_versions", _spy))
        publish_mod.list_versions(str(root))

    assert seen["report_id"] == "r-xyz"
    assert seen["api_key"] == "test-key"


def test_list_versions_no_record_raises(tmp_path: Path):
    root = _make_fullstack(tmp_path)  # no .published.json
    with _base_patches(tmp_path):
        with pytest.raises(FileNotFoundError):
            publish_mod.list_versions(str(root))


# ---------------------------------------------------------------------------
# activate_version
# ---------------------------------------------------------------------------


def test_activate_rewrites_published_json_for_modified_badge(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {
        "report_id": "r1", "url": "https://x/a/r1", "published": True,
        "last_md5": "m_new", "published_mtime": 1_700_000_000, "mode": "public",
    })

    def _activate(report_id, md5, *, api_key, publish_url, ssl_verify):
        return {"report_id": report_id, "current_md5": md5, "view_url": "https://x/a/r1"}

    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.activate_version", _activate))
        out = publish_mod.activate_version(str(root), "m_old")

    assert out["status"] == "ok"
    assert out["currentMd5"] == "m_old"

    entry = _read_map(root)
    # last_md5 now points at the rolled-back version, and published_mtime is
    # zeroed so the modified-badge cheap gate falls through to the md5 compare.
    assert entry["last_md5"] == "m_old"
    assert entry["published_mtime"] == 0
    assert entry["published"] is True          # still live, just an older version
    assert entry["report_id"] == "r1"          # anchor untouched


def test_activate_forwards_md5_to_publisher(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r1", "url": "https://x/a/r1", "published": True})
    seen = {}

    def _activate(report_id, md5, *, api_key, publish_url, ssl_verify):
        seen["report_id"], seen["md5"] = report_id, md5
        return {"report_id": report_id, "current_md5": md5, "view_url": ""}

    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.activate_version", _activate))
        publish_mod.activate_version(str(root), "m_target")

    assert seen == {"report_id": "r1", "md5": "m_target"}


def test_activate_missing_md5_raises_value_error(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r1", "url": "https://x/a/r1", "published": True})
    with _base_patches(tmp_path):
        with pytest.raises(ValueError):
            publish_mod.activate_version(str(root), "")


def test_activate_409_fullstack_maps_to_value_error_with_message(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r1", "url": "https://x/a/r1", "published": True})

    def _activate(*a, **k):
        raise _http_error(409, "Version rollback is not yet supported for fullstack apps")

    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.activate_version", _activate))
        with pytest.raises(ValueError) as ei:
            publish_mod.activate_version(str(root), "m_old")
    assert "fullstack" in str(ei.value)
    # .published.json must be untouched when the rollback failed upstream.
    assert "last_md5" not in _read_map(root)


def test_activate_500_maps_to_runtime_error(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    _write_map(root, {"report_id": "r1", "url": "https://x/a/r1", "published": True})

    def _activate(*a, **k):
        raise _http_error(500, "boom")

    with _base_patches(tmp_path) as stack:
        stack.enter_context(patch("anton.publisher.activate_version", _activate))
        with pytest.raises(RuntimeError):
            publish_mod.activate_version(str(root), "m_old")
