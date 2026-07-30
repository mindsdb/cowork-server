"""Tests for stable artifact URLs across republish / unpublish (2026-06-23)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr


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


def _read_map(folder: Path) -> dict:
    return json.loads((folder / ".published.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 1: _published_url_for honors the `published` flag
# ---------------------------------------------------------------------------

from cowork.services.artifacts import _published_url_for


def _write_published(folder: Path, entry: dict, name: str = "index.html") -> Path:
    (folder / ".published.json").write_text(json.dumps({name: entry}), encoding="utf-8")
    return folder / name


def test_published_url_present_when_published(tmp_path: Path):
    primary = _write_published(tmp_path, {"report_id": "r1", "url": "https://x/a/r1", "published": True})
    assert _published_url_for(tmp_path, primary) == "https://x/a/r1"


def test_published_url_empty_when_unpublished(tmp_path: Path):
    primary = _write_published(tmp_path, {"report_id": "r1", "url": "https://x/a/r1", "published": False})
    assert _published_url_for(tmp_path, primary) == ""


def test_published_url_legacy_entry_without_flag_is_published(tmp_path: Path):
    # Pre-existing entries have no `published` field but a url -> treat as live.
    primary = _write_published(tmp_path, {"report_id": "r1", "url": "https://x/a/r1"})
    assert _published_url_for(tmp_path, primary) == "https://x/a/r1"


# ---------------------------------------------------------------------------
# Task 2/3: publish_artifact / unpublish_artifact
# ---------------------------------------------------------------------------

import cowork.services.publish as publish_mod


class _FakeUserSettings:
    minds_api_key = SecretStr("test-key")
    minds_url = "https://api.mindshub.ai/v1"
    openai_base_url = ""
    openai_api_key = None
    publish_url = ""  # empty → derived from the provider endpoint


class _FakeAppSettings:
    class connector:  # noqa: N801 - mimic nested settings attr
        vault_dir = "/tmp/does-not-matter"


def _patch_scan(container: Path):
    """Make resolve_artifact_path treat `container` as a registered artifacts dir."""
    return patch("cowork.services.artifacts._scan_artifact_dirs", lambda: [container])


def _patched_publish(container: Path, view_url="https://4nton.ai/a/uuid-1", report_id="uuid-1"):
    """Context-manager stack patching everything publish_artifact touches."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(_patch_scan(container))
    stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
    stack.enter_context(patch.object(publish_mod, "get_app_settings", lambda: _FakeAppSettings()))
    stack.enter_context(patch.object(publish_mod, "_load_state", lambda: {}))
    stack.enter_context(patch.object(publish_mod, "_save_state", lambda state: None))
    stack.enter_context(patch("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object()))
    stack.enter_context(
        patch(
            "anton.publisher.publish",
            lambda *a, **k: {"view_url": view_url, "report_id": report_id, "md5": "m1"},
        )
    )
    return stack


def test_publish_artifact_marks_entry_published(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    with _patched_publish(tmp_path):
        publish_mod.publish_artifact(str(root))
    entry = _read_map(root)["index.html"]
    assert entry["report_id"] == "uuid-1"
    assert entry["url"] == "https://4nton.ai/a/uuid-1"
    assert entry["published"] is True
    # Written at the artifact ROOT, not static/ — the divergence the bug came from.
    assert (root / ".published.json").is_file()
    assert not (root / "static" / ".published.json").exists()


def test_unpublish_keeps_report_id_and_marks_unpublished(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
                                    "last_md5": "m1", "published": True, "mode": "public"}}),
        encoding="utf-8",
    )
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(_patch_scan(tmp_path))
        stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
        stack.enter_context(patch("anton.publisher.unpublish", lambda *a, **k: {"deleted": True}))
        publish_mod.unpublish_artifact(str(root))

    entry = _read_map(root)["index.html"]
    assert entry["report_id"] == "uuid-1"          # anchor survives
    assert entry["published"] is False             # but no longer live
    assert (root / ".published.json").is_file()     # file NOT deleted


def test_republish_after_unpublish_reuses_report_id(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
                                    "published": False}}),
        encoding="utf-8",
    )
    seen = {}

    def _spy_publish(*a, **k):
        seen["report_id"] = k.get("report_id")
        return {"view_url": "https://4nton.ai/a/uuid-1", "report_id": "uuid-1", "md5": "m2"}

    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(_patch_scan(tmp_path))
        stack.enter_context(patch.object(publish_mod, "get_user_settings", lambda: _FakeUserSettings()))
        stack.enter_context(patch.object(publish_mod, "get_app_settings", lambda: _FakeAppSettings()))
        stack.enter_context(patch.object(publish_mod, "_load_state", lambda: {}))
        stack.enter_context(patch.object(publish_mod, "_save_state", lambda state: None))
        stack.enter_context(patch("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object()))
        stack.enter_context(patch("anton.publisher.publish", _spy_publish))
        publish_mod.publish_artifact(str(root))

    assert seen["report_id"] == "uuid-1"           # old anchor was re-sent
    assert _read_map(root)["index.html"]["published"] is True


# ---------------------------------------------------------------------------
# Task 4: published_state read helper
# ---------------------------------------------------------------------------


def test_published_state_resolves_fullstack_root(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
                                    "published": True}}),
        encoding="utf-8",
    )
    # Passing the PRIMARY FILE inside static/ must still resolve to the root record.
    with _patch_scan(tmp_path):
        state = publish_mod.published_state(str(root / "static" / "index.html"))
    assert state == {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1", "published": True}


def test_published_state_unpublished_hides_url_keeps_report_id(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1",
                                    "published": False}}),
        encoding="utf-8",
    )
    with _patch_scan(tmp_path):
        state = publish_mod.published_state(str(root / "static" / "index.html"))
    assert state["published"] is False
    assert state["url"] == ""
    assert state["report_id"] == "uuid-1"


def test_published_state_no_record(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    with _patch_scan(tmp_path):
        state = publish_mod.published_state(str(root / "static" / "index.html"))
    assert state == {"report_id": "", "url": "", "published": False}


# ---------------------------------------------------------------------------
# Task 5: tool path delegates to the service
# ---------------------------------------------------------------------------

import cowork.harnesses.anton_harness.tools as tools_mod


class _FakeWorkspace:
    def __init__(self, base): self.base = str(base)


class _FakeSession:
    def __init__(self, base): self._workspace = _FakeWorkspace(base)


def _run(coro):
    return asyncio.run(coro)


def test_tool_ask_reports_live_url(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    with patch.object(tools_mod, "_published_state",
                      lambda p: {"report_id": "uuid-1", "url": "https://4nton.ai/a/uuid-1", "published": True}):
        out = _run(tools_mod._cowork_publish_or_preview(
            _FakeSession(tmp_path),
            {"file_path": str(root / "static" / "index.html"), "action": "ask", "title": "Dash"},
        ))
    assert "https://4nton.ai/a/uuid-1" in out


def test_tool_ask_reports_not_published(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    with patch.object(tools_mod, "_published_state",
                      lambda p: {"report_id": "uuid-1", "url": "", "published": False}):
        out = _run(tools_mod._cowork_publish_or_preview(
            _FakeSession(tmp_path),
            {"file_path": str(root / "static" / "index.html"), "action": "ask", "title": "Dash"},
        ))
    assert "NOT" in out or "not been published" in out


def test_tool_publish_delegates_and_returns_url(tmp_path: Path):
    root = _make_fullstack(tmp_path)
    captured = {}

    def _fake_publish_artifact(raw_path, access=None):
        captured["path"] = raw_path
        captured["access"] = access
        return {"status": "ok", "url": "https://4nton.ai/a/uuid-1"}

    with patch.object(tools_mod, "_publish_artifact", _fake_publish_artifact):
        out = _run(tools_mod._cowork_publish_or_preview(
            _FakeSession(tmp_path),
            {"file_path": str(root / "static" / "index.html"), "action": "publish", "title": "Dash"},
        ))
    assert "https://4nton.ai/a/uuid-1" in out
    assert captured["path"].endswith("static/index.html")


def test_tool_publish_no_api_key_returns_stop(tmp_path: Path):
    root = _make_fullstack(tmp_path)

    def _raise(raw_path, access=None):
        raise ValueError("Configure your Minds API key in Settings before publishing")

    with patch.object(tools_mod, "_publish_artifact", _raise):
        out = _run(tools_mod._cowork_publish_or_preview(
            _FakeSession(tmp_path),
            {"file_path": str(root / "static" / "index.html"), "action": "publish", "title": "Dash"},
        ))
    assert "STOP" in out and "API key" in out


def test_tool_publish_unsupported_type_is_not_treated_as_missing_key(tmp_path: Path):
    # Review #1: a non-key ValueError must not be reported as "no API key".
    root = _make_fullstack(tmp_path)

    def _raise(raw_path, access=None):
        raise ValueError("Only HTML and Markdown artifacts can be published")

    with patch.object(tools_mod, "_publish_artifact", _raise):
        out = _run(tools_mod._cowork_publish_or_preview(
            _FakeSession(tmp_path),
            {"file_path": str(root / "static" / "index.html"), "action": "publish", "title": "Dash"},
        ))
    assert "STOP" not in out
    assert "PUBLISH FAILED" in out
    assert "Only HTML and Markdown" in out


# ---------------------------------------------------------------------------
# Review fixes: readers honour `published`; published_state never raises
# ---------------------------------------------------------------------------

from cowork.services.artifacts import _published_access_for, _unpublish_folder


def test_published_access_unpublished_returns_public(tmp_path: Path):
    # Review #5: a soft-deleted password artifact must not report a lock icon.
    primary = _write_published(
        tmp_path,
        {"mode": "password", "requires_password": True, "access_password": "s3cret",
         "report_id": "r1", "url": "https://x/a/r1", "published": False},
    )
    out = _published_access_for(tmp_path, primary)
    assert out["accessMode"] == "public"
    assert out["accessProtected"] is False
    assert out["accessPassword"] == ""


def test_published_state_path_outside_artifacts_dir_returns_default(tmp_path: Path):
    # Review #6: resolve raises for unregistered paths; must return the default.
    root = _make_fullstack(tmp_path)
    # No _patch_scan here -> resolve_artifact_path raises FileNotFoundError.
    state = publish_mod.published_state(str(root / "static" / "index.html"))
    assert state == {"report_id": "", "url": "", "published": False}


def test_unpublish_folder_skips_soft_deleted(tmp_path: Path):
    # Review #3: deleting an artifact must not re-unpublish soft-deleted records.
    root = _make_fullstack(tmp_path)
    (root / ".published.json").write_text(
        json.dumps({"index.html": {"report_id": "uuid-1", "url": "https://x/a/uuid-1",
                                    "published": False}}),
        encoding="utf-8",
    )
    # The keyed file must exist on disk so the only reason to skip is the flag.
    (root / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    called = []
    with patch.object(publish_mod, "unpublish_artifact", lambda p: called.append(p)):
        _unpublish_folder(root)
    assert called == []
