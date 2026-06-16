"""Publishing Markdown artifacts.

`.md` files publish as rendered HTML pages: the service renders the markdown
to a throwaway ``index.html`` and hands *that* to the anton publisher, while
the publish registry / history keep keying off the original ``.md`` file.
These tests pin both the renderer and that wiring.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cowork.services import publish


# ─── Renderer helpers (pure) ─────────────────────────────────────────────

def test_render_markdown_produces_index_html(tmp_path: Path):
    md = tmp_path / "report.md"
    md.write_text(
        "# Sales Report\n\nSome **bold** text.\n\n- one\n- two\n\n"
        "```python\nprint(1)\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    out = publish._render_markdown_to_html(md, tmp_path)

    assert out.name == "index.html"
    html = out.read_text()
    assert "<title>Sales Report</title>" in html   # title from first heading
    assert "<strong>bold</strong>" in html          # inline markdown
    assert "<table>" in html                          # tables extension
    assert "print(1)" in html                         # fenced_code extension
    # The original markdown file is left untouched.
    assert md.read_text().startswith("# Sales Report")


def test_markdown_title_falls_back_to_filename(tmp_path: Path):
    md = tmp_path / "untitled-notes.md"
    md.write_text("no heading here, just text\n")
    assert publish._markdown_title(md, md.read_text()) == "untitled-notes"


def test_markdown_title_prefers_first_h1(tmp_path: Path):
    md = tmp_path / "x.md"
    text = "intro line\n\n# Real Title\n\nbody"
    assert publish._markdown_title(md, text) == "Real Title"


# ─── publish_artifact() flow ─────────────────────────────────────────────

def _wire_publish(monkeypatch, tmp_path, target: Path, key: str, *, is_fullstack=False):
    """Stub publish_artifact's heavy deps; return the dict capturing the
    publisher call so tests can assert what was actually uploaded."""
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        publish, "get_user_settings",
        lambda: SimpleNamespace(minds_api_key="key", publish_url="https://4nton.ai"),
    )
    monkeypatch.setattr(
        publish, "get_app_settings",
        lambda: SimpleNamespace(connector=SimpleNamespace(vault_dir=str(tmp_path / "vault"))),
    )
    monkeypatch.setattr(publish, "resolve_artifact_path", lambda raw, allow_dir=True: target)
    monkeypatch.setattr(
        publish, "_resolve_publish_target",
        lambda a: (target, target.parent, key, is_fullstack),
    )

    captured: dict = {}

    def fake_publish(src, **kw):
        captured["src"] = Path(src)
        captured["src_html"] = Path(src).read_text()
        captured["report_id"] = kw.get("report_id")
        return {"view_url": "https://4nton.ai/view/u/abc", "report_id": "rid-1", "md5": "m1"}

    monkeypatch.setattr("anton.publisher.publish", fake_publish)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", lambda *a, **k: object())
    return captured


def test_publish_markdown_uploads_rendered_html_and_keys_registry_on_md(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text("# Title\n\nbody text")

    captured = _wire_publish(monkeypatch, tmp_path, md, "report.md")
    result = publish.publish_artifact(str(md))

    # The publisher received a generated index.html, not the raw .md.
    assert captured["src"].name == "index.html"
    assert "<h1" in captured["src_html"] and "body text" in captured["src_html"]
    assert result["url"] == "https://4nton.ai/view/u/abc"

    # Registry + history key off the ORIGINAL markdown file, not the temp html.
    registry = json.loads((art / ".published.json").read_text())
    assert "report.md" in registry
    assert registry["report.md"]["report_id"] == "rid-1"


def test_republish_markdown_reuses_report_id(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text("# Title\n\nv1")

    captured = _wire_publish(monkeypatch, tmp_path, md, "report.md")
    publish.publish_artifact(str(md))          # first publish → records rid-1
    assert captured["report_id"] is None        # no prior id on first call

    md.write_text("# Title\n\nv2")
    publish.publish_artifact(str(md))          # re-publish
    assert captured["report_id"] == "rid-1"     # reused from .published.json


def test_publish_rejects_unsupported_extension(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    txt = art / "notes.txt"
    txt.write_text("plain")

    _wire_publish(monkeypatch, tmp_path, txt, "notes.txt")
    with pytest.raises(ValueError, match="HTML and Markdown"):
        publish.publish_artifact(str(txt))


def test_html_still_published_verbatim(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    page = art / "dashboard.html"
    page.write_text("<h1>dash</h1>")

    captured = _wire_publish(monkeypatch, tmp_path, page, "dashboard.html")
    publish.publish_artifact(str(page))

    # HTML is handed to the publisher as-is (no markdown temp indirection).
    assert captured["src"] == page
    assert captured["src_html"] == "<h1>dash</h1>"
