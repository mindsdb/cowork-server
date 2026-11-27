"""In-frame preview shim: placement, escaping and the line-offset constant.

The shim must be the document's first script — it substitutes storage APIs that
generated pages read during top-level initialisation, and a page that reads them
before the shim runs is exactly the failure this exists to prevent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from cowork.services import pdf_viewer
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


def test_a_pdf_element_the_viewer_shows_is_not_reported_as_a_failed_file():
    # Chrome fires error on the hidden original <object> of a replaced PDF.
    element = (
        "const element = (tagName, attrs) => ({ tagName, src: attrs.src || '',"
        " getAttribute: (name) => (name in attrs ? attrs[name] : null) });\n"
    )
    fire = "listeners.error.forEach((fn) => fn({ target: %s }));\n"
    scenario = (
        element
        + fire % "element('OBJECT', { data: 'report.pdf' })"
        + fire % "element('EMBED', { src: 'docs/report.PDF' })"
        + fire % "element('IMG', { src: 'chart.png' })"
        + fire % "element('OBJECT', { data: 'chart.svg' })"
    )
    result = _run_pdf_shim("/_t/v1.x/_anton_pdf/", observer="record", scenario=scenario)
    resources = [(m["tagName"], m["url"]) for m in result["posted"] if m.get("type") == "resource"]
    assert resources == [("IMG", "chart.png"), ("OBJECT", "")]


# Each reported field that can hold an address, and the value that feeds it.
# Whitespace and line breaks may change; a field that stops going through
# local() may not.
_CUT_FIELDS = {
    "error file": r"\bfile\s*=\s*local\(\s*file\b",
    "error message": r"\bmessage\s*:\s*local\(\s*ev\.message\b",
    "error stack": r"\bstack\s*:[^,}]*\blocal\(\s*ev\.error\.stack\b",
    "resource url": r"\burl\s*=\s*local\(\s*node\.src\b",
    "rejection message": r"\bmessage\s*:[^,}]*\blocal\(\s*message\b",
    "rejection stack": r"\bstack\s*:[^,}]*\blocal\(\s*reason\.stack\b",
    "csp blocked uri": r"\bblockedURI\s*:\s*local\(\s*ev\.blockedURI\b",
}


def test_every_reported_url_field_goes_through_the_one_cut():
    # On the draft host the document sits under /_t/<token>/. A field that
    # skipped the cut would carry the token into the agent's repair prompt.
    for field, pattern in _CUT_FIELDS.items():
        assert re.search(pattern, SHIM_JS), field
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


# The reporters, run in node against stand-in browser objects: what reaches
# the parent frame, and that the page's own console, fetch and XHR behave
# exactly as without the shim.
_HARNESS = r"""
const posted = [];
const nativeErrors = [];
const listeners = {};
const fetchQueue = [];
const page = {};
globalThis.ErrorEvent = class ErrorEvent {};
class FakeXHR {
  constructor() { this.l = {}; this.responseType = ''; this.status = 0; this.responseURL = ''; }
  open(method, url) { this.opened = [method, url]; }
  send() {
    setTimeout(() => {
      this.status = this.nextStatus;
      (this.l.loadend || []).forEach((fn) => fn());
    }, 0);
  }
  addEventListener(type, fn) { (this.l[type] = this.l[type] || []).push(fn); }
  get responseText() {
    if (this.responseType && this.responseType !== 'text') throw new Error('InvalidStateError');
    return this.body || '';
  }
}
const fakeWindow = {
  location: { href: 'about:srcdoc' },
  parent: { postMessage: (m) => posted.push(JSON.parse(JSON.stringify(m))) },
  addEventListener: (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); },
  localStorage: {},
  console: { error: function () { nativeErrors.push(Array.from(arguments).map(String)); } },
  fetch: function () { const next = fetchQueue.shift(); return next instanceof Error ? Promise.reject(next) : Promise.resolve(next); },
  XMLHttpRequest: FakeXHR,
};
const fakeDocument = { addEventListener: (type, fn) => { (listeners['doc:' + type] = listeners['doc:' + type] || []).push(fn); } };
(function (window, document) {
__SHIM__
})(fakeWindow, fakeDocument);
const window = fakeWindow;
const settle = () => new Promise((r) => setTimeout(r, 20));
(async () => {
__SCENARIO__
  await settle();
  process.stdout.write(JSON.stringify({ posted, nativeErrors, page }));
})().catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


def _node_shim() -> str:
    """SHIM_JS with the placeholders that prepare_preview_html fills, and no PDF viewer."""
    return SHIM_JS.replace("__LINE_OFFSET__", "0").replace("__PDF_ASSETS__", '""')


def _run_shim(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _HARNESS.replace("__SHIM__", _node_shim()).replace("__SCENARIO__", scenario)
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    out = json.loads(done.stdout)
    assert out["posted"][0] == {"type": "document-start", "source": "anton-preview"}
    out["reports"] = out["posted"][1:]
    return out


def test_an_unhandled_rejection_is_its_own_type():
    out = _run_shim("listeners.unhandledrejection[0]({ reason: new Error('boom') });")
    [report] = out["reports"]
    assert report["type"] == "rejection"
    assert report["message"] == "Unhandled promise rejection: boom"
    assert "boom" in report["stack"]


def test_console_error_with_an_error_or_error_text_is_reported_and_still_logged():
    out = _run_shim(
        "window.console.error(new TypeError('rows.map is not a function'));\n"
        "window.console.error('Uncaught ReferenceError: chart is not defined');\n"
    )
    assert [(r["type"], r["message"]) for r in out["reports"]] == [
        ("console", "TypeError: rows.map is not a function"),
        ("console", "Uncaught ReferenceError: chart is not defined"),
    ]
    assert "rows.map" in out["reports"][0]["stack"]
    # The page's own console still got both calls, arguments untouched.
    assert out["nativeErrors"] == [
        ["TypeError: rows.map is not a function"],
        ["Uncaught ReferenceError: chart is not defined"],
    ]


def test_console_noise_is_not_reported():
    out = _run_shim(
        "window.console.error('[Violation] TypeError: handler took 300ms');\n"
        "window.console.error('Warning: Each child in a list should have a unique key prop.');\n"
        "window.console.error('Download the React DevTools for a better development experience');\n"
        "window.console.error('could not load the chart', 42);\n"
    )
    assert out["reports"] == []
    assert len(out["nativeErrors"]) == 4


def test_a_repeated_failure_is_reported_once():
    out = _run_shim(
        "for (let i = 0; i < 5; i++) window.console.error(new TypeError('same'));\n"
        "for (let i = 0; i < 5; i++) {\n"
        "  fetchQueue.push(new Response('down', { status: 503 }));\n"
        "  await window.fetch('/api/poll?t=' + i);\n"
        "}\n"
    )
    assert [r["type"] for r in out["reports"]] == ["console", "http"]
    assert len(out["nativeErrors"]) == 5


def test_a_5xx_fetch_is_reported_and_the_page_still_reads_its_body():
    out = _run_shim(
        "const big = 'Traceback (most recent call last):\\n  ' + 'x'.repeat(1000);\n"
        "fetchQueue.push(new Response(big, { status: 500 }));\n"
        "const res = await window.fetch('/api/sales', { method: 'post' });\n"
        "page.status = res.status;\n"
        "page.length = (await res.text()).length;\n"
        "fetchQueue.push(new Response('nope', { status: 404 }));\n"
        "fetchQueue.push(new Response('ok', { status: 200 }));\n"
        "await window.fetch('/api/missing');\n"
        "await window.fetch(new Request('http://localhost/api/fine'));\n"
    )
    assert out["page"] == {"status": 500, "length": len("Traceback (most recent call last):\n  ") + 1000}
    [report] = out["reports"]
    assert report["type"] == "http"
    assert report["url"] == "/api/sales" and report["status"] == 500 and report["method"] == "POST"
    head = "POST /api/sales → 500: "
    assert report["message"].startswith(head + "Traceback (most recent call last): xxx")
    # The body is cut to 300 characters (whitespace collapsed first).
    assert len(report["message"]) == len(head) + 300


def test_a_failed_fetch_rejects_for_the_page_as_before():
    out = _run_shim(
        "fetchQueue.push(new TypeError('Failed to fetch'));\n"
        "try { await window.fetch('/api/x'); } catch (e) { page.caught = e.message; }\n"
    )
    assert out["page"] == {"caught": "Failed to fetch"}
    assert out["reports"] == []


def test_a_5xx_xhr_is_reported_with_its_text_body():
    out = _run_shim(
        "const xhr = new window.XMLHttpRequest();\n"
        "xhr.open('get', '/api/table');\n"
        "xhr.nextStatus = 502; xhr.body = 'bad gateway';\n"
        "xhr.send();\n"
        "const binary = new window.XMLHttpRequest();\n"
        "binary.responseType = 'json';\n"
        "binary.open('POST', '/api/json');\n"
        "binary.nextStatus = 500;\n"
        "binary.send();\n"
        "const fine = new window.XMLHttpRequest();\n"
        "fine.open('GET', '/api/fine'); fine.nextStatus = 200; fine.send();\n"
        "await settle();\n"
        "page.opened = xhr.opened;\n"
    )
    # The page's own open() still ran with its arguments.
    assert out["page"] == {"opened": ["get", "/api/table"]}
    assert [(r["type"], r["message"]) for r in out["reports"]] == [
        ("http", "GET /api/table → 502: bad gateway"),
        # A non-text responseType has no responseText to read: no body.
        ("http", "POST /api/json → 500:"),
    ]


def test_new_reports_still_count_against_the_report_cap():
    out = _run_shim("for (let i = 0; i < 40; i++) window.console.error(new Error('e' + i));")
    assert len(out["reports"]) == 20


def test_pdf_embeds_are_spliced_into_the_shim():
    assert pdf_viewer.EMBED_JS in SHIM_JS
    assert pdf_viewer.EMBEDS_MARKER not in SHIM_JS


def test_the_shim_carries_no_document_tags():
    # comments_layer.inject_layer looks for the last </body>, the cowork
    # renderer for the first <head>: inside the shim they would land in a script.
    assert not re.search(r"</?(?:html|head|body)\b|<!doctype", SHIM_JS, re.I)


def test_the_comment_layer_lands_after_the_shim_on_a_page_without_a_closing_body():
    from cowork.services.preview_html import prepare_preview_html

    out = prepare_preview_html("<html><head></head><body><p>text</p>", comments=True)
    assert out.index("</script>") < out.index("anton-comments")


def test_pdf_embeds_are_installed_after_the_reporter_and_before_the_storage_guard():
    assert (
        SHIM_JS.index("addEventListener('error'")
        < SHIM_JS.index("installPdfEmbeds(window, {")
        < SHIM_JS.index("var nativeStorageWorks")
    )


def test_a_shim_template_without_the_marker_is_refused():
    with pytest.raises(ValueError, match="preview_shim.js"):
        pdf_viewer.splice_embeds("(function () {})();", "preview_shim.js")


def test_pdf_assets_are_filled_per_response():
    html = "<html><head></head><body></body></html>"
    with_assets = inject_shim(html, pdf_assets="/_t/v1.abc/_anton_pdf/")
    assert 'var PDF_ASSETS = "/_t/v1.abc/_anton_pdf/";' in with_assets
    assert 'var PDF_ASSETS = "";' in inject_shim(html)
    assert "__PDF_ASSETS__" not in with_assets
    # The value has no line break, so the line offset is the same.
    assert with_assets.count("\n") == inject_shim(html).count("\n")


def test_pdf_assets_cannot_close_the_script():
    out = inject_shim("<head></head>", pdf_assets="</script><b>")
    assert "</script><b>" not in out
    assert chr(92) + "u003c/script>" in out


def _run_pdf_shim(pdf_assets, *, observer, scenario=""):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    markup = inject_shim("<html><head></head></html>", pdf_assets=pdf_assets)
    js = markup[markup.index("<script>") + len("<script>"): markup.rindex("</script>")]
    harness = (
        "const posted = []; const observed = []; const listeners = {};\n"
        "globalThis.ErrorEvent = class ErrorEvent {};\n"
        "class MO { constructor() { if (" + json.dumps(observer) + " === 'throw') throw new Error('no observer'); }"
        " observe(target, options) { observed.push([...options.attributeFilter]); } }\n"
        "const window = { location: { href: 'https://draft-staging.mindshub.co/_t/v1.x/index.html',"
        " origin: 'https://draft-staging.mindshub.co' }, parent: { postMessage: (m) => posted.push(m) },"
        " addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },"
        " MutationObserver: MO, URL, Blob, navigator: {},"
        " get localStorage() { throw new Error('SecurityError'); } };\n"
        "window.document = { baseURI: window.location.href, addEventListener() {}, documentElement: null };\n"
        "const document = window.document;\n"
        + js
        + "\n"
        + scenario
        + "\nprocess.stdout.write(JSON.stringify({ observed, posted,"
        " storage: typeof Object.getOwnPropertyDescriptor(window, 'localStorage').value }));\n"
    )
    done = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def test_the_shim_installs_pdf_embeds_only_with_an_address():
    assert _run_pdf_shim("/_t/v1.x/_anton_pdf/", observer="record")["observed"] == [["src", "data", "type", "srcdoc"]]
    assert _run_pdf_shim(None, observer="record")["observed"] == []


def test_a_failing_pdf_embed_install_is_reported_and_the_shim_goes_on():
    result = _run_pdf_shim("/_t/v1.x/_anton_pdf/", observer="throw")
    errors = [m for m in result["posted"] if m.get("type") == "error"]
    assert [e["message"] for e in errors] == ["PDF viewer: no observer"]
    assert result["storage"] == "object", "the storage substitutes are still installed"

# The view reporter, run in node against a stand-in DOM and a fake clock: what
# the shim tells the app about where the user is, so the app's inline editor
# can open on the same slide (ENG-4384).
_VIEW_HARNESS = r"""
const posted = [];
const listeners = {};
let now = 0, nextId = 1;
const timers = [];
const fakeSetTimeout = (fn, ms) => { const id = nextId++; timers.push({ id, at: now + (ms || 0), fn }); return id; };
const fakeClearTimeout = (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); };
const advance = (ms) => {
  const end = now + ms;
  for (;;) {
    timers.sort((a, b) => a.at - b.at || a.id - b.id);
    const t = timers[0];
    if (!t || t.at > end) break;
    timers.shift();
    now = t.at;
    t.fn();
  }
  now = end;
};
const observers = new Set();
class FakeMutationObserver {
  constructor(cb) { this.cb = cb; }
  observe() { observers.add(this); }
  disconnect() { observers.delete(this); }
}
const mutate = (target) => { for (const o of [...observers]) o.cb([{ target }]); };
const R = (top, left = 20) => ({ top, left, right: left + 200, bottom: top + 40, width: 200, height: 40 });
class El {
  constructor(tag, opts = {}) {
    this.localName = tag; this.nodeType = 1; this.id = opts.id || '';
    this.sel = [tag].concat(opts.sel || []); if (this.id) this.sel.push('#' + this.id);
    this.classes = opts.classes || []; this.classes.forEach((c) => this.sel.push('.' + c));
    this.text = opts.text || ''; this.rect = opts.rect || null; this.visible = opts.visible !== false;
    this.children = []; this.parentElement = null; this.isConnected = true;
    this.scrollTop = 0; this.scrollLeft = 0; this.dataset = {}; this.style = {};
    this.classList = { contains: (c) => this.classes.includes(c) };
  }
  add(child) { child.parentElement = this; this.children.push(child); return child; }
  get previousElementSibling() {
    const p = this.parentElement; if (!p) return null;
    const i = p.children.indexOf(this); return i > 0 ? p.children[i - 1] : null;
  }
  matches(sel) { return sel.split(',').map((s) => s.trim()).some((s) => this.sel.includes(s)); }
  closest(sel) { for (let e = this; e && e.nodeType === 1; e = e.parentElement) if (e.matches(sel)) return e; return null; }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(''); }
  getBoundingClientRect() { return this.rect || { top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  checkVisibility() { return this.visible; }
}
const all = (root) => [root].concat(...root.children.map(all));
const html = new El('html');
const body = html.add(new El('body'));
const fakeDocument = {
  body, documentElement: html,
  addEventListener: (type, fn) => { (listeners['doc:' + type] = listeners['doc:' + type] || []).push(fn); },
  getElementById: (id) => all(html).find((e) => e.id === id) || null,
  querySelectorAll: (sel) => all(html).filter((e) => e.matches(sel)),
};
const fakeWindow = {
  location: { href: 'about:srcdoc', hash: '' },
  parent: { postMessage: (m) => posted.push(JSON.parse(JSON.stringify(m))) },
  addEventListener: (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); },
  innerWidth: 1000, innerHeight: 800, scrollX: 0, scrollY: 0,
  localStorage: {},
};
const fire = (type, ev) => (listeners[type] || []).forEach((fn) => fn(ev));
const key = (k, more = {}) => fire('keydown', Object.assign({ isTrusted: true, key: k, target: body }, more));
const click = (target, more = {}) => fire('click', Object.assign({ isTrusted: true, button: 0, target }, more));
const window = fakeWindow, document = fakeDocument;
// A deck: two slides (the second off screen) and a nav with Prev / Next.
const deck = body.add(new El('main', { id: 'deck' }));
const h1 = deck.add(new El('section')).add(new El('h1', { text: 'One', rect: R(20) }));
const h2 = deck.add(new El('section')).add(new El('h1', { text: 'Two' }));
const nav = body.add(new El('nav'));
const prev = nav.add(new El('button', { text: 'Prev' }));
const next = nav.add(new El('button', { text: 'Next' }));
const nextIcon = next.add(new El('span'));
const show = (from, to) => { from.rect = null; to.rect = R(20); mutate(to); };
// Synthetic events for a restore's replay, and the page's own handlers
// (bound after the shim, as a page's are).
class FakeEvent {
  constructor(type, init = {}) { Object.assign(this, init); this.type = type; this.defaultPrevented = false; }
  preventDefault() { this.defaultPrevented = true; }
  stopImmediatePropagation() { this.stopped = true; }
}
globalThis.KeyboardEvent = class extends FakeEvent {};
globalThis.MouseEvent = class extends FakeEvent {};
const pageHandlers = [];
const reloads = [];
El.prototype.dispatchEvent = function (ev) {
  if (!('target' in ev)) ev.target = this;
  fire(ev.type, ev);
  if (!ev.stopped) pageHandlers.forEach((fn) => fn(ev));
  return !ev.defaultPrevented;
};
fakeDocument.readyState = 'complete';
fakeWindow.location.reload = () => reloads.push(1);
fakeWindow.scrollTo = (x, y) => { fakeWindow.scrollX = x; fakeWindow.scrollY = y; };
const restore = (view, more = {}) => fire('message', Object.assign({
  isTrusted: true, source: fakeWindow.parent,
  data: { source: 'anton-preview-parent', type: 'restore', view },
}, more));
(function (window, document, setTimeout, clearTimeout, MutationObserver) {
__SHIM__
})(fakeWindow, fakeDocument, fakeSetTimeout, fakeClearTimeout, FakeMutationObserver);
__SCENARIO__
advance(20000);
process.stdout.write(JSON.stringify({
  posted, reloads: reloads.length,
  theme: { dataset: html.dataset.theme || null, colorScheme: html.style.colorScheme || null },
}));
"""


def _views(scenario: str) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace("__SHIM__", _node_shim()).replace("__SCENARIO__", scenario)
    done = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    views = [m for m in json.loads(done.stdout)["posted"] if m["type"] == "view"]
    for view in views:
        assert view["source"] == "anton-preview"
    return views


def test_a_key_the_page_reacts_to_is_reported_with_the_heading_it_lands_on():
    views = _views("key('ArrowRight'); show(h1, h2);")
    assert views[-1] == {
        "type": "view",
        "source": "anton-preview",
        "hash": "",
        "scrollX": 0,
        "scrollY": 0,
        "container": None,
        "inputs": [{"type": "key", "key": "ArrowRight", "shiftKey": False}],
        "overflow": False,
        "heading": "Two",
    }


def test_only_trusted_unmodified_nav_keys_the_page_reacted_to_are_kept():
    views = _views(
        "const field = body.add(new El('input'));\n"
        "key('ArrowRight', { isTrusted: false }); mutate(h2); advance(300);\n"
        "key('ArrowRight', { ctrlKey: true }); mutate(h2); advance(300);\n"
        "key('a'); mutate(h2); advance(300);\n"
        "key('ArrowRight', { target: field }); mutate(h2); advance(300);\n"
        # Space on a focused button also clicks it; the click is recorded instead.
        "key(' ', { target: nextIcon }); mutate(h2); advance(300);\n"
        # Scrolling a long page with the arrow keys changes no DOM.
        "key('ArrowDown'); advance(300);\n"
    )
    assert views == []


def test_a_click_names_the_control_by_a_path_the_editor_can_resolve():
    views = _views(
        "const inDeck = deck.add(new El('button', { text: 'Start' }));\n"
        "click(nextIcon); show(h1, h2); advance(300);\n"
        "click(inDeck); mutate(h1); advance(300);\n"
    )
    assert views[-1]["inputs"] == [
        {"type": "click", "path": "nav:0/button:1", "label": "Next"},
        {"type": "click", "path": "#deck/button:0", "label": "Start"},
    ]


def test_a_click_the_page_ignored_or_an_untrusted_one_is_not_kept():
    views = _views(
        "click(h1); advance(300);\n"
        "click(next, { isTrusted: false }); mutate(h2); advance(300);\n"
        "click(next, { button: 2 }); mutate(h2); advance(300);\n"
    )
    assert views == []


def test_the_comment_layer_is_left_out_of_paths_and_inputs():
    views = _views(
        # comments_layer.py appends its own elements to <body>.
        "const hl = body.add(new El('div', { classes: ['act-hl'] }));\n"
        "const mk = body.add(new El('div', { id: 'act-mk' }));\n"
        "const panel = body.add(new El('div'));\n"
        "const pin = mk.add(new El('div', { classes: ['act-marker'] }));\n"
        "click(pin); mutate(h2); advance(300);\n"
        "click(next); mutate(hl); advance(300);\n"
        "body.classes.push('act-mode');\n"
        "click(next); mutate(h2); advance(300);\n"
        "body.classes.pop();\n"
        "click(panel); mutate(panel); advance(300);\n"
    )
    assert views[-1]["inputs"] == [{"type": "click", "path": "div:0", "label": ""}]


def test_too_many_inputs_report_an_overflow_and_none_of_them():
    views = _views(
        "for (let i = 0; i < 51; i++) { key('ArrowRight'); mutate(h2); advance(300); }"
    )
    assert views[-1]["overflow"] is True
    assert views[-1]["inputs"] == []
    assert all(len(view["inputs"]) <= 50 for view in views)


def test_hash_window_and_container_scroll_are_reported():
    views = _views(
        "window.location.hash = '#pricing'; fire('hashchange', {}); advance(1000);\n"
        "const pane = body.add(new El('div', { id: 'pane' }));\n"
        "pane.scrollTop = 300; window.scrollY = 50;\n"
        "fire('scroll', { target: pane }); advance(300);\n"
    )
    assert views[0]["hash"] == "#pricing"
    assert views[-1]["scrollY"] == 50
    assert views[-1]["container"] == {"path": "#pane", "top": 300, "left": 0}


def test_an_unchanged_view_is_reported_once():
    views = _views(
        "window.location.hash = '#a'; fire('hashchange', {}); advance(1000);\n"
        "fire('hashchange', {}); advance(1000);\n"
        "fire('scroll', { target: fakeDocument }); advance(300);\n"
    )
    assert len(views) == 1


def test_the_heading_is_the_topmost_one_actually_on_screen():
    views = _views(
        "const lower = body.add(new El('h2', { text: '  Lower\\n  heading ', rect: R(300) }));\n"
        "const upper = body.add(new El('h3', { text: 'Upper', rect: R(100) }));\n"
        "const faded = body.add(new El('h2', { text: 'Faded', rect: R(5), visible: false }));\n"
        "const below = body.add(new El('h2', { text: 'Below', rect: R(900) }));\n"
        "h1.rect = null;\n"
        "fire('hashchange', {});\n"
    )
    assert views[-1]["heading"] == "Upper"
    views = _views("h1.rect = null; fire('hashchange', {});")
    assert views[-1]["heading"] == ""


def test_the_view_reporter_cannot_stop_the_storage_half():
    # A page in the opaque preview origin still gets its storage substitutes.
    assert SHIM_JS.index("View reporter") < SHIM_JS.index("var nativeStorageWorks")


# Keeping the place across a reload the app made (ENG-4383): the app sends the
# last view back to the reloaded document, and the shim replays it.

# The deck in the harness moves to slide Two on ArrowRight (read through the
# legacy keyCode) or on a click on Next.
_DECK = """
pageHandlers.push((ev) => {
  if (ev.type === 'keydown' && ev.keyCode === 39) show(h1, h2);
  if (ev.type === 'click' && ev.target === next) show(h1, h2);
});
"""
_KEY = {"type": "key", "key": "ArrowRight", "shiftKey": False}
_CLICK = {"type": "click", "path": "nav:0/button:1", "label": "Next"}


def _run_restore(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace("__SHIM__", _node_shim()).replace("__SCENARIO__", _DECK + scenario)
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    out = json.loads(done.stdout)
    out["restored"] = [m["status"] for m in out["posted"] if m["type"] == "view-restored"]
    out["views"] = [m for m in out["posted"] if m["type"] == "view"]
    return out


def test_a_restore_replays_the_view_and_reports_the_replayed_inputs_as_its_own():
    out = _run_restore(f"restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});")
    assert out["restored"] == ["ok"]
    assert out["reloads"] == 0
    # The next save restores the whole way again: the replayed input is now
    # the document's own.
    assert out["views"][-1]["inputs"] == [_KEY]
    assert out["views"][-1]["heading"] == "Two"


def test_a_restore_replays_a_click_by_its_path():
    out = _run_restore(f"restore({json.dumps({'inputs': [_CLICK], 'heading': 'Two'})});")
    assert out["restored"] == ["ok"]
    assert out["views"][-1]["inputs"] == [_CLICK]


def test_inputs_after_a_restore_add_to_the_replayed_ones():
    out = _run_restore(
        f"restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});\n"
        "advance(5000); key('ArrowLeft'); mutate(h1); advance(1000);"
    )
    assert out["views"][-1]["inputs"] == [_KEY, {"type": "key", "key": "ArrowLeft", "shiftKey": False}]


def test_a_restore_that_lands_elsewhere_reloads_to_the_start():
    # "One" is still on the page and "Two" was expected: the replay went
    # somewhere else, and a fresh document beats a random slide.
    out = _run_restore(
        "pageHandlers.length = 0;\n"
        f"restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});"
    )
    assert out["restored"] == ["mismatch"]
    assert out["reloads"] == 1
    assert out["views"] == []


def test_a_heading_the_change_rewrote_does_not_count_as_a_mismatch():
    out = _run_restore(f"restore({json.dumps({'inputs': [_KEY], 'heading': 'Old title of two'})});")
    assert out["restored"] == ["ok"]
    assert out["reloads"] == 0


def test_a_missing_control_is_a_mismatch():
    view = {"inputs": [{"type": "click", "path": "main:3/button:0", "label": "Gone"}], "heading": ""}
    out = _run_restore(f"restore({json.dumps(view)});")
    assert out["restored"] == ["mismatch"]
    assert out["reloads"] == 1


def test_a_restore_is_taken_from_the_parent_only_and_once_per_document():
    view = json.dumps({"inputs": [_KEY], "heading": "Two"})
    out = _run_restore(
        f"restore({view}, {{ source: {{}} }});\n"
        f"restore({view}, {{ isTrusted: false }});\n"
        f"restore({view}, {{ data: {{ source: 'anton-preview', type: 'restore', view: {view} }} }});\n"
        "advance(5000);"
    )
    assert out["restored"] == []
    assert out["views"] == []
    out = _run_restore(f"restore({view}); advance(5000); restore({view}); advance(5000);")
    assert out["restored"] == ["ok"]


def test_a_malformed_restore_replays_nothing():
    for view in (
        {"inputs": [{"type": "key", "key": "Delete"}]},
        {"inputs": [{"type": "click", "path": "div:0/#x"}]},
        {"inputs": [_KEY] * 51},
        "nope",
    ):
        out = _run_restore(f"restore({json.dumps(view)});")
        assert out["restored"] == ["invalid"]
        assert out["views"] == []


def test_the_user_waits_while_a_restore_replays_and_is_not_recorded():
    out = _run_restore(
        f"restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});\n"
        "const held = new FakeEvent('click', { isTrusted: true, button: 0, target: prev });\n"
        "fire('click', held); mutate(h1);\n"
        "posted.push({ type: 'probe', held: held.defaultPrevented && !!held.stopped });\n"
    )
    assert out["restored"] == ["ok"]
    assert next(m for m in out["posted"] if m["type"] == "probe")["held"] is True
    assert out["views"][-1]["inputs"] == [_KEY]


def test_hash_and_scroll_are_restored():
    view = {"hash": "#pricing", "scrollY": 900, "container": {"path": "#pane", "top": 40, "left": 0}}
    out = _run_restore(
        "const pane = body.add(new El('div', { id: 'pane' }));\n"
        f"restore({json.dumps(view)});\n"
        "advance(5000); posted.push({ type: 'probe', hash: window.location.hash, y: window.scrollY, top: pane.scrollTop });"
    )
    assert out["restored"] == ["ok"]
    probe = next(m for m in out["posted"] if m["type"] == "probe")
    assert probe == {"type": "probe", "hash": "#pricing", "y": 900, "top": 40}


def test_the_shim_reports_the_page_height_and_whether_it_scrolls():
    """The app fits a page taller than its frame that scrolls nowhere (a deck
    sized from the frame's width, overflow hidden): the shim tells it the
    page's full height and whether the root or body hide their overflow,
    after load and on every resize, best effort."""
    js = SHIM_JS
    block = js[js.index("type: 'size'") - 2400 : js.index("size reports are best effort")]
    # A locked page counts only what starts on screen: a deck's hidden
    # slides stacked below the frame don't make it look cut off.
    assert "? visibleBottom(height)" in js
    assert "if (box.top >= frameHeight - 1 || box.bottom <= 0) continue;" in js
    assert "var locked = hidesOverflow(root) || hidesOverflow(body);" in js
    assert "window.addEventListener('load', scheduleSize)" in js
    assert "window.addEventListener('resize', scheduleSize)" in js


# The app's theme, for a page shown inside a chat reply (ENG-4684): the parent
# posts `theme` on the restore's channel, as often as the app's theme changes.
_THEME = """
const theme = (mode, more = {}) => fire('message', Object.assign({
  isTrusted: true, source: fakeWindow.parent,
  data: { source: 'anton-preview-parent', type: 'theme', mode },
}, more));
"""


def _run_theme(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace("__SHIM__", _node_shim()).replace(
        "__SCENARIO__", _THEME + scenario
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)["theme"]


def test_a_theme_sets_the_root_data_theme_and_color_scheme():
    assert _run_theme("theme('dark');") == {"dataset": "dark", "colorScheme": "dark"}
    assert _run_theme("theme('light');") == {"dataset": "light", "colorScheme": "light"}


def test_a_theme_is_taken_every_time_it_changes():
    assert _run_theme("theme('dark'); theme('light'); theme('dark');") == {
        "dataset": "dark", "colorScheme": "dark",
    }


@pytest.mark.parametrize("mode", ["'Dark'", "' dark'", "'auto'", "'constructor'", "''", "1", "null", "{}"])
def test_any_other_mode_is_ignored(mode):
    assert _run_theme(f"theme({mode});") == {"dataset": None, "colorScheme": None}
    # And it leaves a theme already set alone.
    assert _run_theme(f"theme('dark'); theme({mode});") == {"dataset": "dark", "colorScheme": "dark"}


def test_a_theme_is_taken_from_the_parent_only_with_its_tag():
    out = _run_theme(
        "theme('dark', { source: {} });\n"
        "theme('dark', { isTrusted: false });\n"
        "theme('dark', { data: { source: 'anton-preview', type: 'theme', mode: 'dark' } });\n"
        "theme('dark', { data: { type: 'theme', mode: 'dark' } });\n"
        "theme('dark', { data: 'dark' });"
    )
    assert out == {"dataset": None, "colorScheme": None}


def test_a_theme_that_came_early_is_set_again_once_the_page_has_loaded():
    # A theme before there is a root, and one the page's own script then
    # overwrote: both hold once the document is parsed and loaded.
    out = _run_theme(
        "fakeDocument.documentElement = null; theme('dark'); fakeDocument.documentElement = html;\n"
        "fire('doc:DOMContentLoaded', {});"
    )
    assert out == {"dataset": "dark", "colorScheme": "dark"}
    out = _run_theme(
        "theme('dark'); html.dataset.theme = 'light'; html.style.colorScheme = 'light';\n"
        "fire('load', {});"
    )
    assert out == {"dataset": "dark", "colorScheme": "dark"}


def test_a_theme_does_not_take_the_restore():
    # The two share a channel: a theme is no restore, and a restore after it
    # still replays.
    out = _run_restore(_THEME + f"theme('dark'); restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});")
    assert out["restored"] == ["ok"]


# The page's own state, for a page shown inside a chat reply (ENG-4684):
# `window.anton` posts `state` and `submit`, and takes the app's `state` on
# the restore's channel. A scenario reports what the page saw as a `probe`.
_STATE = """
const warnings = [];
console.warn = (text) => warnings.push(String(text));
const sendState = (state, more = {}) => fire('message', Object.assign({
  isTrusted: true, source: fakeWindow.parent,
  data: { source: 'anton-preview-parent', type: 'state', state },
}, more));
const probe = (what) => posted.push({ type: 'probe', what: JSON.parse(JSON.stringify(what === undefined ? null : what)) });
"""


def _run_state(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace("__SHIM__", _node_shim()).replace(
        "__SCENARIO__", _STATE + scenario + "\nprobe({ warnings });"
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    posted = json.loads(done.stdout)["posted"]
    *probes, last = [m["what"] for m in posted if m["type"] == "probe"]
    sent = [m for m in posted if m["type"] in ("state", "submit")]
    for message in sent:
        assert message["source"] == "anton-preview"
    return {"sent": sent, "probes": probes, "warnings": last["warnings"]}


def test_set_state_posts_the_state_and_get_state_returns_a_copy():
    out = _run_state(
        "const list = { people: [{ name: 'Ana', contact: true }] };\n"
        "probe(window.anton.getState());\n"
        "probe(window.anton.setState(list));\n"
        "list.people[0].contact = false;\n"
        "const got = window.anton.getState(); got.people = [];\n"
        "probe(window.anton.getState());"
    )
    assert out["sent"] == [
        {
            "type": "state", "state": {"people": [{"name": "Ana", "contact": True}]}, "handed": False,
            "source": "anton-preview",
        }
    ]
    assert out["probes"] == [None, True, {"people": [{"name": "Ana", "contact": True}]}]
    assert out["warnings"] == []


@pytest.mark.parametrize(
    "value",
    ["'x'.repeat(32 * 1024)", "'é'.repeat(16 * 1024)", "undefined", "() => 1", "(() => { const o = {}; o.o = o; return o; })()"],
)
def test_set_state_refuses_what_is_not_json_or_over_32_kb_with_a_warning(value):
    out = _run_state(
        "window.anton.setState({ kept: 1 });\n"
        f"probe(window.anton.setState({value}));\n"
        "probe(window.anton.getState());"
    )
    assert [m["state"] for m in out["sent"]] == [{"kept": 1}]
    assert out["probes"] == [False, {"kept": 1}]
    assert len(out["warnings"]) == 1 and out["warnings"][0].startswith("anton.setState:")


def test_a_state_just_under_32_kb_is_sent():
    # The serialized string, quotes included, is exactly 32 KB.
    out = _run_state("probe(window.anton.setState('x'.repeat(32 * 1024 - 2)));")
    assert out["probes"] == [True]
    assert len(out["sent"]) == 1


def test_the_apps_state_reaches_on_state_and_get_state():
    out = _run_state(
        "const seen = [];\n"
        "window.anton.onState((s) => seen.push(s));\n"
        "sendState({ people: ['Ana'] });\n"
        "sendState({ people: ['Ana', 'Bo'] });\n"
        "probe(seen); probe(window.anton.getState());"
    )
    assert out["probes"] == [[{"people": ["Ana"]}, {"people": ["Ana", "Bo"]}], {"people": ["Ana", "Bo"]}]
    # Taking the app's state sends nothing back.
    assert out["sent"] == []


def test_a_state_the_app_sent_before_on_state_is_passed_on_the_next_tick():
    out = _run_state(
        "sendState({ saved: true });\n"
        "const seen = [];\n"
        "window.anton.onState((s) => seen.push(s));\n"
        "const off = window.anton.onState((s) => seen.push('removed'));\n"
        "off();\n"
        "probe(seen.slice());\n"
        "advance(1);\n"
        "probe(seen);"
    )
    assert out["probes"] == [[], [{"saved": True}]]


def test_a_state_is_taken_from_the_parent_only_with_its_tag():
    out = _run_state(
        "const seen = [];\n"
        "window.anton.onState((s) => seen.push(s));\n"
        "sendState({ a: 1 }, { source: {} });\n"
        "sendState({ a: 1 }, { isTrusted: false });\n"
        "sendState({ a: 1 }, { data: { source: 'anton-preview', type: 'state', state: { a: 1 } } });\n"
        "sendState({ a: 1 }, { data: { type: 'state', state: { a: 1 } } });\n"
        "sendState({ a: 1 }, { data: 'state' });\n"
        "sendState(undefined);\n"
        "sendState('x'.repeat(40000));\n"
        "probe(seen); probe(window.anton.getState());"
    )
    assert out["probes"] == [[], None]


def test_a_listeners_error_does_not_stop_the_others():
    out = _run_state(
        "const seen = [];\n"
        "window.anton.onState(() => { throw new Error('page bug'); });\n"
        "window.anton.onState((s) => seen.push(s));\n"
        "sendState({ ok: 1 });\n"
        "probe(seen);"
    )
    assert out["probes"] == [[{"ok": 1}]]


def test_submit_sends_the_given_state_or_the_last_one_with_a_short_message():
    out = _run_state(
        "window.anton.setState({ people: ['Ana'] });\n"
        "probe(window.anton.submit());\n"
        "probe(window.anton.submit({ people: ['Ana', 'Bo'] }, 'Contact these two'));\n"
        "probe(window.anton.getState());\n"
        "window.anton.submit(undefined, 'y'.repeat(600));\n"
        "window.anton.submit(undefined, 42);"
    )
    submits = [m for m in out["sent"] if m["type"] == "submit"]
    assert submits[0] == {"type": "submit", "state": {"people": ["Ana"]}, "source": "anton-preview"}
    assert submits[1] == {
        "type": "submit", "state": {"people": ["Ana", "Bo"]}, "message": "Contact these two",
        "source": "anton-preview",
    }
    assert submits[2]["message"] == "y" * 500
    assert "message" not in submits[3]
    assert out["probes"] == [True, True, {"people": ["Ana", "Bo"]}]


def test_submit_refuses_a_state_over_32_kb():
    out = _run_state("probe(window.anton.submit('x'.repeat(40000), 'done'));")
    assert out["probes"] == [False]
    assert out["sent"] == []
    assert len(out["warnings"]) == 1 and out["warnings"][0].startswith("anton.submit:")


def test_the_state_does_not_take_the_restore_or_the_theme():
    out = _run_restore(
        _THEME + _STATE
        + "sendState({ a: 1 }); theme('dark');"
        + f" restore({json.dumps({'inputs': [_KEY], 'heading': 'Two'})});"
    )
    assert out["restored"] == ["ok"]
    assert out["theme"] == {"dataset": "dark", "colorScheme": "dark"}


# ENG-4684: two frames of one artifact (the chat card's and the viewer's)
# bounced a state between them without end: each page set again what it was
# handed, and set a default before the app's state reached it.


def test_a_state_set_before_the_apps_state_arrived_says_so():
    out = _run_state(
        "window.anton.setState({ picked: ['all'] });\n"
        "sendState({ picked: ['li'] });\n"
        "window.anton.setState({ picked: [] });"
    )
    assert [(m["state"], m["handed"]) for m in out["sent"]] == [({"picked": ["all"]}, False), ({"picked": []}, True)]


def test_a_state_set_again_while_the_apps_state_is_taken_is_not_sent_back():
    out = _run_state(
        "window.anton.onState((s) => { window.anton.setState({ picked: s.picked, at: 1 }); });\n"
        "sendState({ picked: ['li'] });\n"
        "probe(window.anton.getState());\n"
        "window.anton.setState({ picked: [] });"
    )
    # The echo stays in the page; the user's next change goes out.
    assert out["probes"] == [{"picked": ["li"], "at": 1}]
    assert [m["state"] for m in out["sent"]] == [{"picked": []}]


def test_a_click_the_page_answered_with_a_state_is_not_kept_for_a_restore():
    views = _views(
        _STATE
        + "click(next); mutate(h2); window.anton.setState({ picked: ['li'] }); advance(300);\n"
        "click(prev); mutate(h1); advance(300);\n"
    )
    # The state brings the tick back; only the navigation is the place.
    assert views[-1]["inputs"] == [{"type": "click", "path": "nav:0/button:0", "label": "Prev"}]


def test_a_restore_replay_sets_no_state_for_the_app():
    out = _run_restore(
        _STATE
        + "pageHandlers.push((ev) => { if (ev.type === 'click' && ev.target === next) window.anton.setState({ picked: ['x'] }); });\n"
        + f"restore({json.dumps({'inputs': [_CLICK], 'heading': 'Two'})});\n"
        + "advance(5000); window.anton.setState({ picked: ['y'] });"
    )
    assert out["restored"] == ["ok"]
    assert [m["state"] for m in out["posted"] if m["type"] == "state"] == [{"picked": ["y"]}]


# ENG-4684: a page that never calls setState still tells the app what its
# form shows. The shim posts a `snapshot` of the page's controls and text,
# which the app saves as snapshot.json, and puts a saved one back for a page
# with no onState listener. The stand-in DOM below gains form controls, text
# nodes and attributes; a scenario builds its form in `form`.
_FORM = """
class TextNode {
  constructor(value, parent) { this.nodeType = 3; this.nodeValue = value; this.parentElement = parent; }
  get previousSibling() { const k = this.parentElement.childNodes; const i = k.indexOf(this); return i > 0 ? k[i - 1] : null; }
}
Object.defineProperty(El.prototype, 'childNodes', { get() {
  const head = this.text ? [this._t || (this._t = new TextNode(this.text, this))] : [];
  return head.concat(this.nodes || this.children);
} });
Object.defineProperty(El.prototype, 'previousSibling', { get() {
  const p = this.parentElement; if (!p) return null;
  const k = p.childNodes; const i = k.indexOf(this); return i > 0 ? k[i - 1] : null;
} });
Object.defineProperty(El.prototype, 'textContent', { get() {
  return this.childNodes.map((n) => n.nodeType === 3 ? n.nodeValue : n.textContent).join('');
} });
const baseAdd = El.prototype.add;
El.prototype.add = function (child) {
  if (!this.nodes) this.nodes = this.children.slice();
  this.nodes.push(child);
  return baseAdd.call(this, child);
};
El.prototype.addText = function (value) {
  if (!this.nodes) this.nodes = this.children.slice();
  this.nodes.push(new TextNode(value, this));
  return this;
};
El.prototype.getAttribute = function (name) { return this.attrs && name in this.attrs ? this.attrs[name] : null; };
El.prototype.querySelector = function (sel) { return all(this).slice(1).find((e) => e.matches(sel)) || null; };
globalThis.Event = class extends FakeEvent {};
const form = body.add(new El('form'));
const field = (parent, tag, props = {}, opts = {}) => {
  const el = parent.add(new El(tag, opts));
  el.attrs = opts.attrs || {};
  Object.assign(el, { name: '', value: '', checked: false, type: tag === 'input' ? 'text' : undefined }, props);
  return el;
};
const option = (value, selected = false) => ({ value, selected });
const user = (el, value) => {
  if (value !== undefined) el.value = value;
  fire('input', { isTrusted: true, type: 'input', target: el });
};
const sendSnapshot = (snapshot, more = {}) => fire('message', Object.assign({
  isTrusted: true, source: fakeWindow.parent,
  data: { source: 'anton-preview-parent', type: 'snapshot', snapshot },
}, more));
const pageEvents = [];
pageHandlers.push((ev) => { if (ev.type === 'input' || ev.type === 'change') pageEvents.push(ev.type + ':' + (ev.target.name || ev.target.id)); });
"""


def _run_snapshot(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace("__SHIM__", _node_shim()).replace(
        "__SCENARIO__", _STATE + _FORM + scenario + "\nprobe({ pageEvents });"
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=True)
    posted = json.loads(done.stdout)["posted"]
    *probes, last = [m["what"] for m in posted if m["type"] == "probe"]
    snapshots = [m for m in posted if m["type"] == "snapshot"]
    for message in snapshots:
        assert set(message) == {"type", "snapshot", "source"}
        assert message["source"] == "anton-preview"
    return {
        "snapshots": [m["snapshot"] for m in snapshots], "probes": probes, "pageEvents": last["pageEvents"],
        "states": [m for m in posted if m["type"] == "state"],
    }


_SLIDER = (
    "form.add(new El('label', { text: 'Threshold', attrs: { for: 'thr' } })).attrs = { for: 'thr' };\n"
    "const slider = field(form, 'input', { type: 'range', value: '0.5' }, { id: 'thr' });\n"
)


def test_a_moved_slider_is_reported_with_its_label_and_value():
    out = _run_snapshot(
        _SLIDER
        + "advance(1100);\n"  # the one sent after load
        + "user(slider, '0.40'); advance(100); user(slider, '0.45'); advance(100); user(slider, '0.47');\n"
        + "advance(399); probe('before'); advance(1);"
    )
    first, moved = out["snapshots"]
    assert first["controls"] == [{"label": "Threshold", "name": "thr", "type": "range", "value": "0.5"}]
    # Debounced: three inputs inside 400 ms send one snapshot, with the last value.
    assert moved["controls"] == [{"label": "Threshold", "name": "thr", "type": "range", "value": "0.47"}]
    assert "Threshold" in moved["text"] and "Prev" in moved["text"]


def test_change_events_count_too_and_nothing_is_sent_twice():
    out = _run_snapshot(
        _SLIDER
        + "advance(1100);\n"
        + "fire('change', { isTrusted: true, type: 'change', target: slider }); advance(500);\n"
        + "slider.value = '0.9'; fire('change', { isTrusted: true, type: 'change', target: slider }); advance(500);\n"
        + "user(slider); advance(500);"
    )
    # The unchanged change sends nothing, and neither does the repeated input.
    assert [s["controls"][0]["value"] for s in out["snapshots"]] == ["0.5", "0.9"]


def test_labels_come_from_for_wrapping_aria_placeholder_or_the_text_before():
    out = _run_snapshot(
        "const wrap = form.add(new El('label', { text: 'Your name ' }));\n"
        "field(wrap, 'input', { name: 'who', value: 'Ana' });\n"
        "field(form, 'input', { name: 'aria', value: 'a' }, { attrs: { 'aria-label': '  Aria   label ' } });\n"
        "form.add(new El('span', { id: 'lb1', text: 'Labelled' })); form.add(new El('span', { id: 'lb2', text: 'by two' }));\n"
        "field(form, 'input', { name: 'byid', value: 'b' }, { attrs: { 'aria-labelledby': 'lb1 lb2' } });\n"
        "field(form, 'textarea', { name: 'notes', value: 'hi' }, { attrs: { placeholder: 'Notes here' } });\n"
        "const row = form.add(new El('div')); row.add(new El('span', { text: 'Budget (USD)' }));\n"
        "field(row, 'input', { type: 'number', name: 'budget', value: '120' });\n"
        "const row2 = form.add(new El('div')); row2.addText('Speed: ');\n"
        "field(row2, 'input', { type: 'range', name: 'speed', value: '3' });\n"
        "form.add(new El('p', { text: 'L'.repeat(100) }));\n"
        "field(form, 'input', { name: 'long', value: '' }, { attrs: { 'aria-label': 'x'.repeat(120) } });\n"
        "advance(1100);"
    )
    labels = {c["name"]: c["label"] for c in out["snapshots"][0]["controls"]}
    assert labels == {
        "who": "Your name", "aria": "Aria label", "byid": "Labelled by two", "notes": "Notes here",
        "budget": "Budget (USD)", "speed": "Speed:", "long": "x" * 80,
    }


def test_a_control_after_another_control_takes_no_label_from_it():
    out = _run_snapshot(
        "const row = form.add(new El('div')); row.add(new El('span', { text: 'First' }));\n"
        "field(row, 'input', { name: 'a', value: '1' });\n"
        "field(row, 'input', { name: 'b', value: '2' });\n"
        "form.add(new El('p', { text: 'w'.repeat(250) }));\n"
        "field(form, 'input', { name: 'c', value: '3' });\n"
        "advance(1100);"
    )
    assert [(c["name"], c["label"]) for c in out["snapshots"][0]["controls"]] == [
        ("a", "First"), ("b", ""), ("c", ""),
    ]


def test_radio_checkbox_and_select_shapes():
    out = _run_snapshot(
        "const set = form.add(new El('fieldset')); set.add(new El('legend', { text: 'Size' }));\n"
        "for (const v of ['s', 'm', 'l']) field(set, 'input', { type: 'radio', name: 'size', value: v, checked: v === 'm' });\n"
        "field(form, 'input', { type: 'radio', name: 'none', value: 'x' }, { attrs: { 'aria-label': 'Pick' } });\n"
        "field(form, 'input', { type: 'checkbox', name: 'agree', value: 'on', checked: true }, { attrs: { 'aria-label': 'Agree' } });\n"
        "field(form, 'input', { type: 'checkbox', name: 'spam', value: 'on' }, { attrs: { 'aria-label': 'Spam' } });\n"
        "field(form, 'select', { name: 'one', value: 'b', options: [option('a'), option('b', true)] }, { attrs: { 'aria-label': 'One' } });\n"
        "field(form, 'select', { name: 'many', multiple: true, options: [option('a', true), option('b'), option('c', true)] }, { attrs: { 'aria-label': 'Many' } });\n"
        "advance(1100);"
    )
    assert out["snapshots"][0]["controls"] == [
        {"label": "Size", "name": "size", "type": "radio", "value": "m"},
        {"label": "Pick", "name": "none", "type": "radio", "value": ""},
        {"label": "Agree", "name": "agree", "type": "checkbox", "value": True},
        {"label": "Spam", "name": "spam", "type": "checkbox", "value": False},
        {"label": "One", "name": "one", "type": "select-one", "value": "b"},
        {"label": "Many", "name": "many", "type": "select-multiple", "value": ["a", "c"]},
    ]


def test_secrets_hidden_fields_files_and_the_comment_layer_are_left_out():
    out = _run_snapshot(
        "for (const t of ['password', 'hidden', 'file']) field(form, 'input', { type: t, name: t, value: 'secret' });\n"
        "const layer = body.add(new El('div', { classes: ['act-pop'], text: 'Comment UI' }));\n"
        "const note = field(layer, 'textarea', { name: 'comment', value: 'mine' });\n"
        "field(form, 'input', { name: 'kept', value: 'k' });\n"
        "advance(1100);\n"
        "user(note, 'typed'); advance(500);\n"
        "fire('input', { isTrusted: true, type: 'input', target: form.children[0] }); advance(500);"
    )
    assert len(out["snapshots"]) == 1
    snap = out["snapshots"][0]
    assert [c["name"] for c in snap["controls"]] == ["kept"]
    assert "Comment UI" not in snap["text"]


def test_a_control_with_no_name_or_id_is_named_by_its_path_and_at_most_60_are_kept():
    out = _run_snapshot(
        "field(form, 'input', { value: 'anon' });\n"
        "for (let i = 0; i < 70; i++) field(form, 'input', { name: 'f' + i, value: String(i) });\n"
        "body.add(new El('p', { text: 'z'.repeat(3000) }));\n"
        "advance(1100);"
    )
    snap = out["snapshots"][0]
    assert len(snap["controls"]) == 60
    assert snap["controls"][0]["name"] == "form:0/input:0"
    assert snap["controls"][-1]["name"] == "f58"
    assert len(snap["text"]) == 2000


def test_a_saved_snapshot_is_put_back_and_the_page_hears_it_once():
    out = _run_snapshot(
        _SLIDER
        + "const box = field(form, 'input', { type: 'checkbox', name: 'agree' });\n"
        "for (const v of ['s', 'm']) field(form, 'input', { type: 'radio', name: 'size', value: v, checked: v === 's' });\n"
        "const many = field(form, 'select', { name: 'many', multiple: true, options: [option('a', true), option('b')] });\n"
        "sendSnapshot({ controls: [\n"
        "  { label: 'Threshold', name: 'thr', type: 'range', value: '0.47' },\n"
        "  { label: '', name: 'agree', type: 'checkbox', value: true },\n"
        "  { label: '', name: 'size', type: 'radio', value: 'm' },\n"
        "  { label: '', name: 'many', type: 'select-multiple', value: ['b'] },\n"
        "], text: 'ignored' });\n"
        "probe([slider.value, box.checked, form.children.filter((e) => e.name === 'size').map((e) => e.checked), many.options.map((o) => o.selected)]);\n"
        "advance(5000);"
    )
    assert out["probes"][0] == ["0.47", True, [False, True], [False, True]]
    assert out["pageEvents"] == [
        "input:thr", "change:thr", "input:agree", "change:agree", "input:size", "change:size",
        "input:many", "change:many",
    ]
    # Not echoed: neither the restore's events nor the after-load send post it back.
    assert out["snapshots"] == []


def test_input_after_a_restore_is_reported_again():
    out = _run_snapshot(
        _SLIDER
        + "sendSnapshot({ controls: [{ label: 'Threshold', name: 'thr', type: 'range', value: '0.47' }], text: '' });\n"
        "advance(2000); user(slider, '0.2'); advance(500);"
    )
    assert [s["controls"][0]["value"] for s in out["snapshots"]] == ["0.2"]


def test_a_page_with_on_state_restores_itself_and_the_snapshot_is_left_alone():
    out = _run_snapshot(
        _SLIDER
        + "window.anton.onState(() => {});\n"
        "sendSnapshot({ controls: [{ label: 'Threshold', name: 'thr', type: 'range', value: '0.47' }], text: '' });\n"
        "advance(5000); probe(slider.value);"
    )
    assert out["probes"] == ["0.5"]
    assert out["pageEvents"] == []
    assert out["snapshots"] == []


@pytest.mark.parametrize(
    "more",
    [
        "{ source: {} }",
        "{ isTrusted: false }",
        "{ data: { source: 'anton-preview', type: 'snapshot', snapshot: good } }",
        "{ data: { type: 'snapshot', snapshot: good } }",
        "{ data: { source: 'anton-preview-parent', type: 'snapshot', snapshot: { controls: [{ name: 'thr', type: 'range', value: 4 }] } } }",
        "{ data: { source: 'anton-preview-parent', type: 'snapshot', snapshot: { controls: 'thr' } } }",
        "{ data: { source: 'anton-preview-parent', type: 'snapshot', snapshot: { controls: Array(61).fill({ name: 'thr', type: 'range', value: '1' }) } } }",
    ],
)
def test_a_snapshot_is_taken_from_the_parent_only_with_its_tag_and_shape(more):
    out = _run_snapshot(
        _SLIDER
        + "const good = { controls: [{ label: '', name: 'thr', type: 'range', value: '0.47' }], text: '' };\n"
        + f"sendSnapshot(good, {more});\n"
        "advance(5000); probe(slider.value);"
    )
    assert out["probes"] == ["0.5"]
    assert out["pageEvents"] == []


def test_the_snapshot_does_not_take_the_state_or_the_restore():
    out = _run_snapshot(
        _SLIDER
        + "const seen = []; window.anton.onState((s) => seen.push(s));\n"
        "sendSnapshot({ controls: [], text: '' });\n"
        "advance(1); probe(seen); probe(window.anton.getState());"
    )
    assert out["probes"] == [[], None]
    assert out["states"] == []
