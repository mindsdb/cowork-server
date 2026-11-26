"""On-artifact comment marker layer injection.

The layer is injected into the top-level HTML document served for preview ONLY
when the renderer opts in via the activation query flag. Asset requests and
flag-less requests must stream untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cowork.server import app
from cowork.services.comments_layer import ACTIVATION_PARAM, LAYER_JS, inject_layer

client = TestClient(app)

_HTML = "<html><head></head><body><h1>Report</h1></body></html>"


def test_inject_layer_before_body_close():
    out = inject_layer(_HTML)
    assert "anton-comments" in out
    assert out.index("<script>") < out.index("</body>")


def test_inject_layer_appends_when_no_body():
    out = inject_layer("<div>x</div>")
    assert out.startswith("<div>x</div>")
    assert out.rstrip().endswith("</script>")


def test_layer_js_has_no_script_terminator():
    # A literal </script> in the payload would break out of the injected tag.
    assert "</script>" not in LAYER_JS


def test_layer_supports_edit_delete_protocol():
    # Owner-only edit/delete affordances + the outbound message types they post,
    # plus the fields (author_user_id, edited_at) and viewer gating they rely on.
    for token in [
        "'edit'", "'delete'", "'edit-reply'", "'delete-reply'",
        "author_user_id", "edited_at", "isMine", "viewer",
        "Delete this comment thread?", "Delete this reply?",
    ]:
        assert token in LAYER_JS, token


def test_layer_reports_anchor_states():
    # Classify anchor visibility and report it outward (to mark items "hidden"
    # in the inbox). anchored is not reported; the scroll path is excluded.
    for token in [
        "isShown",
        "'anchor-states'",
        "function reportAnchors",
        "function scheduleReport",
        "getClientRects().length",
        "attributeFilter: ['style', 'class', 'hidden']",
    ]:
        assert token in LAYER_JS, token


def test_layer_hidden_popover_and_hover():
    # A hidden anchor (resolves but is not shown) opens a centered popover
    # with a separate hidden notice; hover does not draw a zero-size box.
    assert "isShown(m.target)" in LAYER_JS
    assert "to see it in place" in LAYER_JS                 # hidden notice (contiguous chunk)
    # focus scrolls only once the page shows the element (reveal, ENG-4383)
    assert "if (!found) { openThread(focusId, at); return; }" in LAYER_JS
    assert "!hel || !isShown(hel)" in LAYER_JS             # hl-on suppresses a hidden one


def test_orphan_notice_skipped_for_general_comments():
    # A thread created without a selector is an intentional general comment —
    # the "element was removed or changed" notice must not render for it.
    assert "orphan && c.selector" in LAYER_JS


def _make_project(tmp: str):
    project_dir = Path(tmp) / "proj"
    artifacts = project_dir / ".anton" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "index.html").write_text(_HTML, encoding="utf-8")
    (artifacts / "styles.css").write_text("body{color:red}", encoding="utf-8")
    return project_dir


def test_serve_injects_only_with_flag():
    with tempfile.TemporaryDirectory() as tmp:
        # Resolved because the stub must honour the contract of the function it
        # replaces — see `_registered_project_dirs`. Without it this fails on
        # macOS only, where tempfile's /var path is a symlink.
        project_dir = _make_project(tmp).resolve()
        with patch(
            "cowork.services.artifacts._registered_project_dirs",
            return_value=[project_dir],
        ), patch(
            "cowork.services.artifacts._projects_root",
            return_value=project_dir.parent,
        ):
            # Entry document with the flag → layer injected.
            r = client.get(f"/api/v1/artifacts/serve/proj/index.html?{ACTIVATION_PARAM}=1")
            assert r.status_code == 200
            assert "anton-comments" in r.text

            # Same document without the flag → untouched.
            r2 = client.get("/api/v1/artifacts/serve/proj/index.html")
            assert r2.status_code == 200
            assert "anton-comments" not in r2.text

            # Non-HTML asset with the flag → untouched (only text/html is wrapped).
            r3 = client.get(f"/api/v1/artifacts/serve/proj/styles.css?{ACTIVATION_PARAM}=1")
            assert r3.status_code == 200
            assert "anton-comments" not in r3.text


def test_serve_html_gets_sandbox_csp():
    """Artifact HTML is project-generated and can carry attacker-influenced
    content, rendered without sanitization by design (arbitrary preview is
    the feature). The in-app iframe already sandboxes it, but that only
    applies when the content is framed — ArtifactViewer's "open in browser"
    loads this same URL as a direct top-level navigation, which no
    client-side sandbox attribute can constrain. A CSP `sandbox` response
    header must cover that path too, on every HTML response regardless of
    whether the comment layer is active, and must not appear on non-HTML
    assets."""
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = _make_project(tmp).resolve()
        with patch(
            "cowork.services.artifacts._registered_project_dirs",
            return_value=[project_dir],
        ), patch(
            "cowork.services.artifacts._projects_root",
            return_value=project_dir.parent,
        ):
            csp = "sandbox allow-scripts allow-popups allow-forms allow-modals"

            r = client.get(f"/api/v1/artifacts/serve/proj/index.html?{ACTIVATION_PARAM}=1")
            assert r.headers.get("content-security-policy") == csp

            r2 = client.get("/api/v1/artifacts/serve/proj/index.html")
            assert r2.headers.get("content-security-policy") == csp

            r3 = client.get("/api/v1/artifacts/serve/proj/styles.css")
            assert "content-security-policy" not in r3.headers


# ── Images in comments (ENG-4410) ────────────────────────────────────────────


def _rendered_layer_js() -> str:
    import re

    html = inject_layer(_HTML)
    return next(s for s in re.findall(r"<script>(.*?)</script>", html, re.S) if "antonCommentsLayer" in s)


def test_layer_sends_pasted_images_with_create_and_reply():
    for token in [
        "send({type:'create', selector: cssPath(el), text: text, images: images})",
        "send({type:'reply', id: id, text: text, images: images})",
        "addEventListener('paste'",
        "addEventListener('drop'",
        "IMG_MAX = 5 * 1024 * 1024, IMG_COUNT = 4",
    ]:
        assert token in LAYER_JS, token


def test_layer_shows_only_images_the_app_pushes():
    # Pictures come only from the parent window, keyed by a stored image id,
    # as raster Blobs: a script in the page posting as the app plants nothing.
    for token in [
        "d.type === 'images'",
        "if (ev.source !== parent) return;",
        "!(blob instanceof Blob)",
        "IMG_TYPES.indexOf(String(blob.type || '').toLowerCase()) < 0",
    ]:
        assert token in LAYER_JS, token


def test_layer_edit_keeps_image_markers_out_of_the_box():
    assert "var words = splitText(oldText).body, marks = marksOf(oldText);" in LAYER_JS
    assert "var full = marks ? (text ? text + '\\n\\n' + marks : marks) : text;" in LAYER_JS


def _node():
    import shutil

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    return node


def test_rendered_layer_js_parses(tmp_path):
    import subprocess

    script = tmp_path / "layer.js"
    script.write_text(_rendered_layer_js())
    subprocess.run([_node(), "--check", str(script)], check=True)


def test_layer_reads_the_markers_the_server_reads():
    # The layer's marker pattern and the server's (services/comment_images)
    # must agree on what names an image, or a thread would show one thing and
    # hand the agent another.
    import json
    import re
    import subprocess

    from cowork.services.comment_images import comment_image_ids

    js = _rendered_layer_js()
    pattern = re.search(r"function imgMarks\(\)\{ return (/.*?/g); \}", js).group(1)
    image = "0123456789abcdef0123456789abcdef.png"
    text = f"Make it like this\n\n![shot (1).png](cowork-image:{image})\n![x](cowork-image:not-an-id.png)"
    out = subprocess.run(
        [_node(), "-e", f"const t={json.dumps(text)};console.log(JSON.stringify([...t.matchAll({pattern})].map(m=>m[1])))"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert json.loads(out) == comment_image_ids(text) == [image]


# ── Going to a comment, and editing it from its author line (ENG-4383) ───────


def test_layer_takes_the_page_to_a_comment_before_opening_it():
    # A comment on another slide, tab or section: controls that point at it
    # are clicked, <details> opened, and the deck walked with the arrow keys
    # (bounded, stopping when a key changes nothing) before the card opens;
    # the "not visible right now" notice is what's left when that fails.
    for token in [
        "function reveal(el, done)",
        "function openControls(el)",
        "[aria-controls=\"",
        "det.open = true",
        "walk(el, 'ArrowRight', WALK_MAX",
        "walk(el, 'ArrowLeft', WALK_MAX * 2",
        "if (!moved) { done(false); return; }",
        "reveal(el, function(found){",
        "if (!found) { openThread(focusId, at); return; }",
    ]:
        assert token in LAYER_JS, token


def test_layer_offers_a_hover_edit_on_your_own_comment():
    for token in [
        "class=\"act-m-edit-root\"",
        "aria-label=\"Edit comment\"",
        "var rootEdit = p.querySelector('.act-m-edit-root');",
        "send({type:'edit', id: id, text: t});",
    ]:
        assert token in LAYER_JS, token


def test_layer_saves_an_edit_on_click_away_and_keeps_the_card():
    # Clicking anywhere else (or the card closing) keeps an edit in progress;
    # saving shows the text in place and leaves the card open (ENG-4383).
    for token in [
        "var editing = null;",
        "if (editing) { editing.commit(); return; }",
        "if (editing) editing.commit();",
        "ta.addEventListener('blur'",
        "textEl.textContent = splitText(full).body;",
        "function(t){ c.text = t; send({type:'edit', id: id, text: t}); }",
        "function(t){ rep.text = t; send({type:'edit-reply', id: id, replyId: rid, text: t}); }",
    ]:
        assert token in LAYER_JS, token
    assert "mutate({type:'edit'" not in LAYER_JS


def test_keys_typed_in_a_comment_card_stay_in_the_card():
    # A deck that moves on Space or the arrows (and swallows them) must not
    # hear keys typed in the card, or a comment can't have spaces. Escape still
    # reaches the layer, which closes the card with it.
    for token in [
        "['keydown', 'keypress', 'keyup'].forEach(function(type){",
        "pop.addEventListener(type, function(e){ if (e.key !== 'Escape') e.stopPropagation(); });",
    ]:
        assert token in LAYER_JS, token
