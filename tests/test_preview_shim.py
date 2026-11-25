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


def _run_shim(scenario: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _HARNESS.replace("__SHIM__", SHIM_JS.replace("__LINE_OFFSET__", "0")).replace("__SCENARIO__", scenario)
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
    this.scrollTop = 0; this.scrollLeft = 0;
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
process.stdout.write(JSON.stringify({ posted, reloads: reloads.length }));
"""


def _views(scenario: str) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _VIEW_HARNESS.replace(
        "__SHIM__", SHIM_JS.replace("__LINE_OFFSET__", "0")
    ).replace("__SCENARIO__", scenario)
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
    script = _VIEW_HARNESS.replace("__SHIM__", SHIM_JS.replace("__LINE_OFFSET__", "0")).replace(
        "__SCENARIO__", _DECK + scenario
    )
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
