"""In-frame preview shim: placement, escaping and the line-offset constant.

The shim must be the document's first script — it substitutes storage APIs that
generated pages read during top-level initialisation, and a page that reads them
before the shim runs is exactly the failure this exists to prevent.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from cowork.services.preview_shim import SHIM_JS, _script_tag, inject_shim


def test_shim_goes_first_inside_head():
    out = inject_shim("<html><head><script>var a=1;</script></head><body></body></html>")
    assert out.index("anton-preview") < out.index("var a=1;")


def test_shim_survives_attributes_on_head():
    out = inject_shim('<html><head data-x="1"><title>t</title></head></html>')
    assert out.index("anton-preview") < out.index("<title>")


def test_shim_lands_after_html_open_when_there_is_no_head():
    out = inject_shim('<!doctype html><html lang="en"><body>x</body></html>')
    assert out.index("<!doctype html>") < out.index("anton-preview")
    assert out.index('<html lang="en">') < out.index("anton-preview")


def test_shim_never_precedes_the_doctype():
    # A script before the doctype puts the document into quirks mode and moves
    # the page's layout under the user.
    # Explicit escape: a literal BOM in the source is invisible to a reader.
    out = inject_shim("\ufeff<!DOCTYPE html><body>x</body>")
    assert out.index("<!DOCTYPE html>") < out.index("anton-preview")


def test_shim_prepends_to_a_bare_fragment():
    out = inject_shim("<div>x</div>")
    assert out.index("anton-preview") < out.index("<div>x</div>")


def test_shim_js_has_no_script_terminator():
    # A literal </script> in the payload would break out of the injected tag.
    # Same rule and same precedent as comments_layer.LAYER_JS.
    assert "</script>" not in SHIM_JS


def test_a_script_terminator_would_be_escaped_if_one_ever_appeared():
    # The guard above is a review rule; this is the mechanism that holds when a
    # future edit slips one in.
    out = _script_tag("var s = '</script>';", 0)
    assert "<\\/script>" in out
    assert out.count("</script>") == 1


def test_line_offset_matches_the_lines_the_injection_adds():
    # Every inline script below the injection shifts by exactly this many lines;
    # the shim subtracts the same number before reporting an error's position.
    out = inject_shim("<html><head></head><body></body></html>")
    added = out.count("\n")
    assert f"LINE_OFFSET = {added}" in out


def test_reporter_is_installed_before_the_storage_guard():
    # Diagnostics must survive a real origin, where the storage half returns early.
    assert SHIM_JS.index("addEventListener('error'") < SHIM_JS.index("nativeStorageWorks")


def test_every_reported_url_field_goes_through_the_one_cut():
    # On the draft host the document sits under /_t/<token>/. A field that
    # skipped the cut would carry the token into the agent's repair prompt.
    assert r"\/_t\/v1\.[^\/?#]+\/" in SHIM_JS
    assert r"\/_t\/v1\.[^\/?#\s]+\/" in SHIM_JS
    for field in (
        "file = local(file);",
        "message: local(ev.message),",
        "stack: ev.error && ev.error.stack ? local(ev.error.stack) : ''",
        "var url = local(node.src || node.href || '');",
        "message: 'Unhandled promise rejection: ' + local(message),",
        "stack: reason && reason.stack ? local(reason.stack) : ''",
        "blockedURI: local(ev.blockedURI)",
    ):
        assert field in SHIM_JS, field
    assert "String(ev.error.stack)" not in SHIM_JS
    assert "String(ev.blockedURI" not in SHIM_JS


# The cut itself, run in node on real addresses: the string checks above only
# show where it is applied. CI runners have node; without it the tests skip.
TOKEN = "v1.eyJhcnQiOiI0NCJ9.QNm0tQAs4c7pfMfsflQzCy1joIUTm9Y8nZLLRckRVt4"
DRAFT = f"https://draft-staging.mindshub.co/_t/{TOKEN}/"
LOOPBACK = "http://127.0.0.1:26866/api/v1/artifacts/preview-asset/abc123/"


def _cut(href: str, values: list) -> list[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    start = SHIM_JS.index("  var HREF = ")
    end = SHIM_JS.index("  function post(")
    script = (
        f"var window = {{location: {{href: {json.dumps(href)}}}}};\n"
        + SHIM_JS[start:end]
        + f"\nprocess.stdout.write(JSON.stringify({json.dumps(values)}.map(local)));\n"
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def test_the_token_root_is_cut_even_above_a_nested_primary():
    stack = f"Error: x\n    at f ({DRAFT}static/app.js:1:2)\n    at {DRAFT}pages/index.html?__antonComments=1:9:3"
    assert _cut(f"{DRAFT}pages/index.html?__antonComments=1", [f"{DRAFT}static/x.png", stack, "inline", None]) == [
        "static/x.png",
        "Error: x\n    at f (static/app.js:1:2)\n    at pages/index.html?__antonComments=1:9:3",
        "inline",
        "",
    ]


def test_a_token_root_without_its_origin_is_cut_on_the_draft_host():
    # A page's own error text can carry location.pathname, which starts at /_t/.
    path = f"/_t/{TOKEN}/"
    stack = f"Error: x\n    at f ({path}static/app.js:1:2)"
    assert _cut(
        f"{DRAFT}pages/index.html",
        [f"bad {path}pages/index.html", f"two {path}a and {path}b", stack, f"/_t/{TOKEN}"],
    ) == [
        "bad pages/index.html",
        "two a and b",
        "Error: x\n    at f (static/app.js:1:2)",
        f"/_t/{TOKEN}",
    ]


def test_a_token_path_in_text_is_left_alone_off_the_draft_host():
    text = f"bad /_t/{TOKEN}/index.html"
    assert _cut(f"{LOOPBACK}index.html", [text]) == [text]
    assert _cut("about:srcdoc", [text]) == [text]


def test_a_desktop_document_keeps_its_folder_cut():
    assert _cut(f"{LOOPBACK}index.html", [f"{LOOPBACK}static/app.js", "https://cdn.example/x.js"]) == [
        "static/app.js",
        "https://cdn.example/x.js",
    ]


def test_a_folder_named_t_in_a_desktop_artifact_is_not_a_token_root():
    # The loopback address has the artifact's own folders in it; only the draft
    # host puts /_t/v1.<token>/ first.
    folder = "http://127.0.0.1:26866/api/v1/artifacts/drafts/p/a/_t/x/"
    assert _cut(
        f"{folder}pages/index.html?sig=1",
        [f"{folder}pages/app.js", f"{folder}static/x.png", "see /_t/x/pages/app.js", "/_t/x/"],
    ) == [
        "app.js",
        f"{folder}static/x.png",
        "see /_t/x/pages/app.js",
        "/_t/x/",
    ]


def test_a_t_folder_below_a_desktop_document_keeps_its_path():
    base = "http://127.0.0.1:26866/api/v1/artifacts/drafts/p/a/"
    assert _cut(f"{base}index.html?sig=1", [f"{base}_t/x/app.js", "at /_t/x/app.js"]) == [
        "_t/x/app.js",
        "at /_t/x/app.js",
    ]


def test_a_t_folder_inside_a_draft_host_artifact_is_not_cut():
    # Only the token segment (v1.<...>) goes; the artifact's own _t folder stays.
    assert _cut(
        f"{DRAFT}_t/x/index.html",
        [f"{DRAFT}_t/x/app.js", f"at /_t/{TOKEN}/_t/x/app.js", "at /_t/x/app.js"],
    ) == ["_t/x/app.js", "at _t/x/app.js", "at /_t/x/app.js"]


def test_a_slash_in_the_query_or_hash_does_not_move_the_folder():
    assert _cut(f"{LOOPBACK}index.html?next=/a/b", [f"{LOOPBACK}static/app.js"]) == ["static/app.js"]
    assert _cut(f"{LOOPBACK}index.html#/route/x", [f"{LOOPBACK}static/app.js"]) == ["static/app.js"]


def test_srcdoc_cuts_nothing():
    assert _cut("about:srcdoc", ["about:srcdoc", "https://cdn.example/x.js"]) == [
        "about:srcdoc",
        "https://cdn.example/x.js",
    ]
