"""Publishing Markdown artifacts.

`.md` files publish as rendered HTML pages: the service renders the markdown
to a throwaway ``index.html`` and hands *that* to the anton publisher, while
the publish registry / history keep keying off the original ``.md`` file.
These tests pin both the renderer and that wiring.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import markdown
import psutil
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.db.scoped import LOCAL_SCOPE
from cowork.api.v1.endpoints import publish as publish_ep
from cowork.services import artifact_autopublish, artifact_export, artifacts, export_runner, publish

# Every render here starts a real conversion process, which imports cowork
# from the interpreter's own site-packages.
pytestmark = pytest.mark.usefixtures("child_runs_this_checkout")


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


# ─── The render runs in the export's conversion process ──────────────────

HOSTILE = "[a][" * 8_000  # Python-Markdown is quadratic on it: 30 s and more.


@pytest.fixture(autouse=True)
def _no_cached_pages(monkeypatch):
    monkeypatch.setattr(
        publish, "_RENDERED_PAGES", publish._RenderedPages(max_bytes=8 * 1024 * 1024, max_refused=256)
    )


def _counted_conversions(monkeypatch) -> list:
    """Count the conversion processes publish starts."""
    calls = []
    convert = export_runner.convert_in_child

    def counting(**kwargs):
        calls.append(kwargs["request"])
        return convert(**kwargs)

    monkeypatch.setattr(export_runner, "convert_in_child", counting)
    return calls


def test_a_published_page_is_what_rendering_in_the_server_made(tmp_path: Path):
    """The page is byte for byte what Python-Markdown made in the server's
    own process, a [TOC] marker's table of contents included."""
    text = "# Q3\n\n[TOC]\n\n## Results\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nx\n```\n"
    md = tmp_path / "report.md"
    md.write_text(text)
    body = markdown.markdown(
        text, extensions=["fenced_code", "tables", "toc", "sane_lists"], output_format="html5"
    )

    page = publish._render_markdown_to_html(md, tmp_path).read_text()

    assert page == publish._MD_HTML_TEMPLATE.format(title="Q3", body=body)
    assert 'class="toc"' in page


def test_a_hostile_file_is_refused_within_the_wall_clock_and_leaves_no_process(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 3.0)
    md = tmp_path / "report.md"
    md.write_text(HOSTILE)
    children = {process.pid for process in psutil.Process().children(recursive=True)}
    started = time.monotonic()

    with pytest.raises(publish.MarkdownNotPublishable):
        publish._render_markdown_to_html(md, tmp_path / "out")

    assert time.monotonic() - started < 3.0 + 1
    assert {process.pid for process in psutil.Process().children(recursive=True)} == children
    assert not (tmp_path / "out" / "index.html").exists()


def test_a_render_while_another_conversion_runs_is_refused_as_busy(tmp_path: Path):
    md = tmp_path / "report.md"
    md.write_text("# R\n")
    assert export_runner._child_slot.acquire(blocking=False)
    try:
        with pytest.raises(publish.MarkdownRenderBusy):
            publish._render_markdown_to_html(md, tmp_path)
    finally:
        export_runner._child_slot.release()


def test_the_same_markdown_renders_once(tmp_path: Path, monkeypatch):
    """The "changed since publish" check renders on every listing of an
    edited artifact; one render serves them all until the next edit."""
    calls = _counted_conversions(monkeypatch)
    md = tmp_path / "report.md"
    md.write_text("# R\n\ntext\n")
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()

    first = publish._render_markdown_to_html(md, tmp_path / "a")
    second = publish._render_markdown_to_html(md, tmp_path / "b")
    md.write_text("# R\n\nedited\n")
    third = publish._render_markdown_to_html(md, tmp_path / "c")

    assert first.read_text() == second.read_text() != third.read_text()
    assert len(calls) == 2


def _notes_artifact(tmp_path: Path, text: str) -> Path:
    root = tmp_path / "notes"
    root.mkdir(exist_ok=True)
    (root / "metadata.json").write_text(
        json.dumps({"id": "notes", "type": "document", "primary": "report.md"}), encoding="utf-8"
    )
    (root / "report.md").write_text(text, encoding="utf-8")
    return root


def test_the_changed_since_publish_check_cant_tell_for_a_hostile_file_and_doesnt_ask_again(
    tmp_path: Path, monkeypatch
):
    """A file the conversion can't render is "can't tell": no badge, and the
    next listing doesn't hold the server's one conversion for it again."""
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 3.0)
    # The same artifact with ordinary Markdown has a hash, so None below is the render's.
    root = _notes_artifact(tmp_path, "# R\n")
    assert publish.compute_publish_md5(root, artifacts_base=tmp_path) is not None
    (root / "report.md").write_text(HOSTILE, encoding="utf-8")
    calls = _counted_conversions(monkeypatch)
    started = time.monotonic()

    assert publish.compute_publish_md5(root, artifacts_base=tmp_path) is None
    assert time.monotonic() - started < 3.0 + 1
    again = time.monotonic()
    assert publish.compute_publish_md5(root, artifacts_base=tmp_path) is None
    assert time.monotonic() - again < 1.0
    assert len(calls) == 1


def test_a_persons_publish_renders_again_a_file_whose_check_was_stopped(
    tmp_path: Path, monkeypatch
):
    """Only background checks reuse a stopped render; a person's publish tries again."""
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 3.0)
    root = _notes_artifact(tmp_path, HOSTILE)
    assert publish.compute_publish_md5(root, artifacts_base=tmp_path) is None
    calls = _counted_conversions(monkeypatch)

    with pytest.raises(publish.MarkdownNotPublishable):
        publish._render_markdown_to_html(root / "report.md", tmp_path)

    assert len(calls) == 1


def test_the_changed_since_publish_check_reports_a_busy_slot(tmp_path: Path):
    """So autopublish can come back for it, rather than reading it as unchanged."""
    root = _notes_artifact(tmp_path, "# R\n")
    assert export_runner._child_slot.acquire(blocking=False)
    try:
        with pytest.raises(publish.MarkdownRenderBusy):
            publish.compute_publish_md5(root, artifacts_base=tmp_path)
    finally:
        export_runner._child_slot.release()


def test_a_listing_reads_a_busy_slot_as_cant_tell(tmp_path: Path, monkeypatch):
    def busy(*_args, **_kwargs):
        raise publish.MarkdownRenderBusy()

    monkeypatch.setattr(publish, "compute_publish_md5", busy)
    root = _notes_artifact(tmp_path, "# R\n")
    entry = {"report_id": "rid-1", "published": True, "last_md5": "m1", "published_mtime": 0}

    state = artifacts._modified_state(
        root, root / "report.md", 10, artifacts_base=tmp_path, published_map={"report.md": entry}
    )

    assert state == (False, False)


def test_autopublish_plans_a_busy_check_for_later_not_as_unchanged(tmp_path: Path, monkeypatch):
    root = _notes_artifact(tmp_path, "# R\n")
    entry = {"report_id": "rid-1", "published": True, "last_md5": "m1", "published_mtime": 0}
    (root / ".published.json").write_text(json.dumps({"report.md": entry}), encoding="utf-8")

    def busy(*_args, **_kwargs):
        raise publish.MarkdownRenderBusy()

    monkeypatch.setattr(publish, "compute_publish_md5", busy)

    decision = artifact_autopublish.needs_publish(root, tmp_path)
    phase = artifact_autopublish._plan(tmp_path, ["notes"])

    assert (decision.action, decision.reason) == (None, artifact_autopublish.REASON_CONVERSION_BUSY)
    assert phase == artifact_autopublish.PlannedPhase(planned=[], conversion_busy=1)


async def test_an_autopublish_the_slot_refused_is_deferred_and_asked_for_again(
    tmp_path: Path, monkeypatch
):
    asked = []
    root = _notes_artifact(tmp_path, "# R\n")

    def busy(*_args, **_kwargs):
        raise publish.MarkdownRenderBusy()

    monkeypatch.setattr(artifact_autopublish, "publish_artifact", busy)
    monkeypatch.setattr(
        artifact_autopublish, "_ask_again_later", lambda **kwargs: asked.append(kwargs)
    )
    assert artifact_autopublish.acquire(tmp_path, root.name, ttl_s=60)

    landed = await artifact_autopublish._publish_one(
        tmp_path, root.name, "key", "https://4nton.ai", 30.0, LOCAL_SCOPE, project_id="p1"
    )

    assert landed is False
    assert asked == [{"artifacts_base": tmp_path, "scope": LOCAL_SCOPE, "project_id": "p1"}]
    # The slug's lock is free for the next pass.
    assert artifact_autopublish.acquire(tmp_path, root.name, ttl_s=60)


def test_a_markdown_file_past_the_bound_is_refused_before_any_conversion(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(publish, "PUBLISH_MARKDOWN_MAX_BYTES", 64)
    calls = _counted_conversions(monkeypatch)
    md = tmp_path / "report.md"
    md.write_bytes(b"# R\n" + b"x" * 60)
    publish._render_markdown_to_html(md, tmp_path)
    md.write_bytes(b"# R\n" + b"x" * 61)

    with pytest.raises(publish.MarkdownNotPublishable) as refused:
        publish._render_markdown_to_html(md, tmp_path)

    assert refused.value.status_code == 413
    assert refused.value.detail == {
        "code": "markdown_not_publishable",
        "message": "This Markdown file is too large to publish. Publish a file under 2 MiB.",
    }
    assert len(calls) == 1


def test_the_bounded_read_is_the_text_read_text_gives(tmp_path: Path):
    """Line endings and bytes that aren't UTF-8 come out as before the bound."""
    md = tmp_path / "report.md"
    md.write_bytes(b"\xef\xbb\xbf# T\r\nline\rnext\n\xff\xfe end\n")

    assert publish._read_markdown(md) == md.read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize(
    "text",
    [
        "intro\r\n  # Title  \r\nbody",
        "a\x0b# Vertical tab\nb",
        "a\u2028# Line separator",
        "#NoSpace\n#  \n# Second",
        "\x85# After NEL",
        "",
    ],
)
def test_the_title_is_the_first_heading_splitlines_would_find(tmp_path: Path, text: str):
    def by_splitlines(md_path: Path, md_text: str) -> str:
        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or md_path.stem
        return md_path.stem

    md = tmp_path / "fallback.md"
    assert publish._markdown_title(md, text) == by_splitlines(md, text)


@pytest.mark.parametrize(
    "endpoint",
    [publish_ep.publish_artifact, publish_ep.update_artifact, artifacts_ep.artifact_status],
    ids=["publish", "update", "status"],
)
def test_the_routes_that_render_a_markdown_page_run_in_the_threadpool(endpoint):
    """A plain `def` route runs in FastAPI's threadpool, off the event loop."""
    assert not inspect.iscoroutinefunction(endpoint)


@pytest.mark.parametrize(
    ("refusal", "status", "code"),
    [
        (publish.MarkdownNotPublishable(publish.MARKDOWN_TOO_SLOW_MESSAGE), 413, "markdown_not_publishable"),
        (publish.MarkdownRenderBusy(), 429, "conversion_busy"),
        (publish.MarkdownConversionFailed(), 500, "markdown_conversion_failed"),
    ],
    ids=["stopped", "busy", "failed"],
)
@pytest.mark.parametrize("route", ["/api/v1/publish/", "/api/v1/publish/update"])
def test_a_markdown_refusal_answers_with_its_status_and_code(refusal, status, code, route):
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    target = "_publish" if route.endswith("/") else "_update"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            publish_ep,
            "_desktop_context",
            lambda raw, *, scope: (Path(raw), Path(raw).parent, "key", "https://4nton.ai"),
        )

        def refuse(*_args, **_kwargs):
            raise refusal

        patch.setattr(publish_ep, target, refuse)
        res = TestClient(app).post(route, json={"path": "/some/report.md"})

    assert res.status_code == status
    assert res.json()["detail"]["code"] == code


async def test_the_server_keeps_answering_while_a_hostile_markdown_page_publishes(
    tmp_path: Path, monkeypatch
):
    """Criterion 5 through the route the app calls: the render waits off the loop."""
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 3.0)
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text(HOSTILE)
    _wire_publish(monkeypatch, tmp_path, md, "report.md")
    monkeypatch.setattr(
        publish_ep, "_desktop_context", lambda raw, *, scope: (md, tmp_path, "key", "https://4nton.ai")
    )
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")

    @app.get("/live")
    async def live():
        return {"ok": True}

    loop = asyncio.get_running_loop()
    latencies: list[float] = []
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        publishing = asyncio.create_task(client.post("/api/v1/publish/", json={"path": str(md)}))
        while not publishing.done():
            started = loop.time()
            assert (await client.get("/live")).status_code == 200
            latencies.append(loop.time() - started)
            await asyncio.sleep(0.05)
        answered = await publishing

    assert answered.status_code == 413, answered.text
    assert answered.json()["detail"]["code"] == "markdown_not_publishable"
    assert len(latencies) > 10
    assert max(latencies) < 0.5, f"liveness took {max(latencies):.3f}s"


# ─── publish_artifact() flow ─────────────────────────────────────────────

def _wire_publish(monkeypatch, tmp_path, target: Path, key: str, *, is_fullstack=False):
    """Stub publish_artifact's heavy deps; return the dict capturing the
    publisher call so tests can assert what was actually uploaded."""
    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        publish, "get_user_settings",
        lambda: SimpleNamespace(
            minds_api_key="key",
            minds_url="https://api.mindshub.ai/v1",
            openai_base_url="",
            openai_api_key=None,
            publish_url="",
        ),
    )
    monkeypatch.setattr(
        publish, "get_app_settings",
        lambda: SimpleNamespace(
            tenancy_mode="local", connector=SimpleNamespace(vault_dir=str(tmp_path / "vault"))
        ),
    )
    monkeypatch.setattr(
        publish, "resolve_artifact_path", lambda raw, *, scope, allow_dir=True: target
    )
    monkeypatch.setattr(
        publish, "_resolve_publish_target",
        lambda a, container_dirs=None: (target, target.parent, key, is_fullstack),
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
    result = publish.publish_artifact(md, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    # The publisher received a generated index.html, not the raw .md.
    assert captured["src"].name == "index.html"
    assert "<h1" in captured["src_html"] and "body text" in captured["src_html"]
    assert result["url"] == "https://4nton.ai/view/u/abc"

    # Registry + history key off the ORIGINAL markdown file, not the temp html.
    registry = json.loads((art / ".published.json").read_text())
    assert "report.md" in registry
    assert registry["report.md"]["report_id"] == "rid-1"


def test_publish_refuses_an_artifact_holding_a_hard_link(tmp_path, monkeypatch):
    """anton bundles an artifact's files by path, so a hard link there to the
    project's `.anton/.env` would publish its current secrets on every publish."""
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text("# Title\n\nbody text")
    saved = tmp_path / ".env"
    saved.write_text("SECRET_TOKEN=do-not-publish")
    os.link(saved, art / "env.txt")

    captured = _wire_publish(monkeypatch, tmp_path, md, "report.md")
    with pytest.raises(ValueError, match="second name"):
        publish.publish_artifact(md, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    assert captured == {}
    assert not (art / ".published.json").exists()


def test_republish_markdown_reuses_report_id(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text("# Title\n\nv1")

    captured = _wire_publish(monkeypatch, tmp_path, md, "report.md")
    publish.publish_artifact(md, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")          # first publish → records rid-1
    assert captured["report_id"] is None        # no prior id on first call

    md.write_text("# Title\n\nv2")
    publish.publish_artifact(md, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")          # re-publish
    assert captured["report_id"] == "rid-1"     # reused from .published.json


def test_publish_rejects_unsupported_extension(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    txt = art / "notes.txt"
    txt.write_text("plain")

    _wire_publish(monkeypatch, tmp_path, txt, "notes.txt")
    with pytest.raises(ValueError, match="HTML and Markdown"):
        publish.publish_artifact(txt, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")


def test_html_still_published_verbatim(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    page = art / "dashboard.html"
    page.write_text("<h1>dash</h1>")

    captured = _wire_publish(monkeypatch, tmp_path, page, "dashboard.html")
    publish.publish_artifact(page, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai")

    # HTML is handed to the publisher as-is (no markdown temp indirection).
    assert captured["src"] == page
    assert captured["src_html"] == "<h1>dash</h1>"


def test_the_page_cache_holds_its_memory_budget_least_recently_used_out_first():
    """A page counts the memory it holds: one emoji makes every character four bytes."""
    ascii_page, wide_page = "a" * 100, "\U0001F4CA" * 25
    assert sys.getsizeof(wide_page) > sys.getsizeof(ascii_page) > len(wide_page.encode("utf-8"))
    pages = publish._RenderedPages(
        max_bytes=sys.getsizeof(ascii_page) + sys.getsizeof(wide_page), max_refused=2
    )
    pages.keep("a", ascii_page)
    pages.keep("wide", wide_page)
    assert pages.page("a") == ascii_page  # now the most recently used
    pages.keep("b", "b" * 100)  # the budget holds two such pages: the wide one goes
    pages.keep("huge", "x" * 10_000)  # past the whole budget: never kept

    assert [pages.page(key) for key in ("a", "wide", "b", "huge")] == [
        ascii_page,
        None,
        "b" * 100,
        None,
    ]
    for key in ("r1", "r2", "r3"):
        pages.refuse(key)
    assert [pages.was_refused(key) for key in ("r1", "r2", "r3")] == [False, True, True]


def test_an_autopublish_doesnt_render_again_a_page_the_conversion_stopped(tmp_path, monkeypatch):
    """A never-published hostile page: the first background publish runs its
    30 s and is refused; the next is refused at once and starts no process."""
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", 3.0)
    art = tmp_path / "art"
    art.mkdir()
    md = art / "report.md"
    md.write_text(HOSTILE)
    _wire_publish(monkeypatch, tmp_path, md, "report.md")
    calls = _counted_conversions(monkeypatch)

    for _ in range(2):
        with pytest.raises(publish.MarkdownNotPublishable):
            publish.publish_artifact(
                md, artifacts_base=tmp_path, api_key="key", publish_url="https://4nton.ai",
                background=True,
            )

    assert len(calls) == 1


def test_two_publishes_of_one_folder_take_turns(tmp_path, monkeypatch):
    """Publishes run in worker threads; overlapping, both would read no report
    id and upload two pages."""
    art = tmp_path / "art"
    art.mkdir()
    page = art / "dashboard.html"
    page.write_text("<h1>dash</h1>")
    _wire_publish(monkeypatch, tmp_path, page, "dashboard.html")
    inside, most, guard = [0], [0], threading.Lock()
    report_ids = []

    def slow_publish(src, **kw):
        with guard:
            inside[0] += 1
            most[0] = max(most[0], inside[0])
        time.sleep(0.3)
        with guard:
            inside[0] -= 1
        report_ids.append(kw.get("report_id"))
        return {"view_url": "https://4nton.ai/view/u/abc", "report_id": "rid-1", "md5": "m1"}

    monkeypatch.setattr("anton.publisher.publish", slow_publish)
    threads = [
        threading.Thread(
            target=publish.publish_artifact,
            args=(page,),
            kwargs={"artifacts_base": tmp_path, "api_key": "key", "publish_url": "https://4nton.ai"},
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert most[0] == 1
    assert report_ids == [None, "rid-1"]
