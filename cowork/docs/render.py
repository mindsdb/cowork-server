"""Views of a doc: the agent's outline and full read, and HTML for export."""

from __future__ import annotations

import html
import math
import re
from typing import Any

from collections.abc import Callable

from .markdown import block_to_markdown
from .model import (
    LIST_TYPES,
    TEXT_TYPES,
    Block,
    Doc,
    Tab,
    chart_source_table,
    find_tab,
    image_src_ok,
    name_key,
    to_number,
)

_SHORT = {
    "paragraph": "p", "heading": "h", "bullet": "bullet", "numbered": "numbered",
    "task": "task", "quote": "quote", "callout": "callout", "code": "code",
    "divider": "divider", "table": "table", "chart": "chart", "metrics": "metrics",
    "image": "image", "pending": "pending",
}


def _clip(text: str, n: int = 90) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _block_line(b: dict[str, Any]) -> str:
    t = b["type"]
    label = _SHORT[t] + (str(b["level"]) if t == "heading" else "")
    if t in LIST_TYPES and b.get("indent"):
        label += f">{b['indent']}"
    if t == "task":
        label += " [x]" if b.get("done") else " [ ]"
    if t in TEXT_TYPES:
        detail = _clip(b.get("text", ""))
        if t == "callout":
            detail = f"({b.get('tone')}) " + (f"{b['title']}: " if b.get("title") else "") + detail
    elif t == "table":
        cols = ", ".join(_col_label(c) for c in b["columns"])
        detail = f"{len(b['rows'])} rows · {cols}"
    elif t == "chart":
        src = b.get("source")
        detail = f"{b['chart']} “{b.get('title', '')}”" + (
            f" from table {src['table']} (x {src['x']}; y {', '.join(src['y'])})" if src
            else f" · {len(b.get('x', []))} points × {len(b.get('series', []))} series")
    elif t == "metrics":
        detail = " · ".join(f"{i['label']} {i['value']}" for i in b["items"])
    elif t == "code":
        detail = f"{b.get('language') or 'text'}, {len(b.get('code', '').splitlines())} lines"
    elif t == "image":
        detail = b.get("alt") or b.get("src", "")
    elif t == "pending":
        detail = "(placeholder) " + b.get("intent", "")
    else:
        detail = ""
    return f"{b['id']}  {label}  {detail}".rstrip()


def _col_label(c: dict[str, Any]) -> str:
    if c["kind"] == "status":
        return f"{c['name']} (status: {'/'.join(o['name'] for o in c.get('options', []))})"
    if c["kind"] in ("number", "date"):
        return f"{c['name']} ({c['kind']})"
    return c["name"]


def doc_header(doc: dict[str, Any]) -> str:
    tabs = "; ".join(f"“{t['title']}” {t['id']} ({len(t['blocks'])} blocks)" for t in doc["tabs"])
    return f"Doc “{doc['title']}” · id {doc['id']} · rev {doc['rev']} · tabs: {tabs}"


def outline(doc: dict[str, Any], tab_ref: str | None = None, *, all_tabs: bool = False) -> str:
    tabs = doc["tabs"] if all_tabs else [find_tab(doc, tab_ref)]
    lines = [doc_header(doc)]
    for tab in tabs:
        lines.append(f"\nTab “{tab['title']}” {tab['id']}:")
        if not tab["blocks"]:
            lines.append("(empty)")
        lines.extend(_block_line(b) for b in tab["blocks"])
    return "\n".join(lines)


def full(doc: dict[str, Any], tab_ref: str | None = None, *, ids: list[str] | None = None) -> str:
    """Each block as Markdown under its id: what the agent edits against."""
    tab = find_tab(doc, tab_ref)
    lines = [doc_header(doc), f"\nTab “{tab['title']}” {tab['id']} — each block is `[id] type` then its Markdown:"]
    wanted = set(ids or [])
    for b in tab["blocks"]:
        if wanted and b["id"] not in wanted:
            continue
        head = f"[{b['id']}] {b['type']}"
        if b["type"] == "table":
            head += " · columns: " + ", ".join(_col_label(c) for c in b["columns"])
            head += f" · rows 0–{len(b['rows']) - 1} under the header" if b["rows"] else " · no rows"
        lines.append(head)
        lines.append(block_to_markdown(b))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def changed_lines(doc: dict[str, Any], touched: dict[str, list[str]]) -> str:
    """Outline lines for the blocks one write produced, grouped by tab."""
    out = []
    for tab in doc["tabs"]:
        ids = set(touched.get(tab["id"], []))
        if not ids:
            continue
        out.append(f"Tab “{tab['title']}” {tab['id']}:")
        out.extend(_block_line(b) for b in tab["blocks"] if b["id"] in ids)
    return "\n".join(out)


# ---------------------------------------------------------------- HTML

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])|(?<![_\w])_(?!\s)(.+?)(?<!\s)_(?![_\w])")
_STRIKE = re.compile(r"~~(.+?)~~")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+)\)")
_CHIP = re.compile(r"\{\{\s*(date|status|user)\s*:\s*([^}]+?)\s*\}\}")
_HELD = re.compile(r"\x00(\d+)\x00")


def inline_html(text: str) -> str:
    """A block's inline Markdown as HTML: code, links, bold, italic, strike and chips.

    Code spans and link addresses are set aside before anything else runs, so
    escaping and emphasis never reach inside them (``https://x.com/_p_/``
    keeps its underscores; ``?a=1&b=2`` is escaped once).
    """
    held: list[str] = []

    def hold(markup: str) -> str:
        held.append(markup)
        return f"\x00{len(held) - 1}\x00"

    s = (text or "").replace("\x00", "")
    s = _INLINE_CODE.sub(lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"), s)
    # The link's words stay in the text (they may be bold); only its address is held.
    s = _LINK.sub(lambda m: hold(f'<a href="{html.escape(m.group(2))}">') + m.group(1) + hold("</a>"), s)
    s = html.escape(s, quote=False)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", s)
    s = _STRIKE.sub(r"<s>\1</s>", s)

    def chip(m: re.Match) -> str:
        kind, value = m.group(1), m.group(2)
        if kind == "date":
            value = _date_label(value)
        return f'<span class="chip chip-{kind}">{value}</span>'

    s = _CHIP.sub(chip, s)
    return _HELD.sub(lambda m: held[int(m.group(1))], s)


def _chart_rows(doc_tab: dict[str, Any], b: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """A chart's labels and series, from its own data or its source table."""
    src = b.get("source")
    if not src:
        return list(b.get("x", [])), list(b.get("series", []))
    table = chart_source_table(doc_tab, b)
    if table is None:
        return [], []
    cols = {name_key(c["name"]): c for c in table["columns"]}
    xcol = cols.get(name_key(src["x"]))
    if xcol is None:
        return [], []
    labels = [r["cells"].get(xcol["id"], "") for r in table["rows"]]
    series = []
    for name in src["y"]:
        c = cols.get(name_key(name))
        if c is None:
            continue
        series.append({"name": c["name"], "values": [to_number(r["cells"].get(c["id"], "")) for r in table["rows"]]})
    return labels, series


def _chart_table_html(labels: list[str], series: list[dict[str, Any]]) -> str:
    head = "<tr><th></th>" + "".join(f"<th>{html.escape(s['name'])}</th>" for s in series) + "</tr>"
    rows = []
    for i, label in enumerate(labels):
        cells = "".join(
            f"<td class='num'>{'' if s['values'][i] is None else _fmt(s['values'][i])}</td>" for s in series
        )
        rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    return f"<table class='chart-data'>{head}{''.join(rows)}</table>"


def _fmt(n: float) -> str:
    return f"{int(n):,}" if float(n).is_integer() else f"{n:,.2f}"


_CURRENCY = re.compile(r"^[$€£¥]$")


def _nice_scale(values: list[float], count: int = 4) -> tuple[float, float, list[float]]:
    """The app's value axis (chartGeometry.niceScale): from 0 (or below) past the largest
    value, on round ticks."""
    lo, hi = min(0.0, *values), max(0.0, *values)
    if lo == hi:
        hi = lo + 1
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw or 1))
    norm = raw / mag
    step = (10 if norm >= 5 else 5 if norm >= 2 else 2 if norm >= 1 else 1) * mag
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    ticks, v = [], lo
    while v <= hi + step / 2:
        ticks.append(round(v, 10))
        v += step
    return lo, hi, ticks


def _axis_format(values: list[float], unit: str | None) -> Callable[[float], str]:
    """The app's number format (chartGeometry.formatter): compact from 10,000, a currency
    unit as a prefix."""
    prefix = unit.strip() if unit and _CURRENCY.match(unit.strip()) else ""
    compact = max((abs(v) for v in values), default=0) >= 10000

    def fmt(n: float) -> str:
        a = abs(n)
        if compact:
            for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
                if a >= div:
                    text = f"{a / div:.1f}".rstrip("0").rstrip(".") + suffix
                    break
            else:
                text = f"{a:.1f}".rstrip("0").rstrip(".")
        else:
            text = f"{a:,.2f}".rstrip("0").rstrip(".")
        return f"{'−' if n < 0 else ''}{prefix}{text}"
    return fmt


def _chart_svg(b: dict[str, Any], labels: list[str], series: list[dict[str, Any]]) -> str:
    """A plain bar or line chart for HTML export (the app draws the interactive one), on the
    app's axis: round ticks, compact numbers, a value margin as wide as its longest label, and
    every n-th category label when they would run into each other."""
    # The editor's categorical order (light), so an export matches what the person saw.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    vals = [v for s in series for v in s["values"] if v is not None]
    if not labels or not vals:
        return ""
    low, top, ticks = _nice_scale(vals)
    fmt = _axis_format(vals, b.get("unit"))
    tick_text = [fmt(v) for v in ticks]
    # 10px labels average about 6px a character; the margin fits the longest, with a gap.
    w, h, pad_b, pad_t = 640, 260, 36, 16
    pad_l = max(36, 6 * max(len(t) for t in tick_text) + 12)
    span = (top - low) or 1
    plot_w, plot_h = w - pad_l - 8, h - pad_b - pad_t

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - low) / span)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(b.get("title", ""))}">']
    for v, text in zip(ticks, tick_text):
        parts.append(f'<line x1="{pad_l}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e2dc"/>'
                     f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="#777">{html.escape(text)}</text>')
    step = plot_w / len(labels)
    line_like = b.get("chart") in ("line", "area")
    for si, s in enumerate(series):
        color = palette[si % len(palette)]
        if line_like:
            pts = [(pad_l + step * (i + 0.5), y(v)) for i, v in enumerate(s["values"]) if v is not None]
            parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (
                color, " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)))
        else:
            bw = step * 0.7 / max(1, len(series))
            for i, v in enumerate(s["values"]):
                if v is None:
                    continue
                x0 = pad_l + step * i + step * 0.15 + bw * si
                y0, y1 = sorted((y(v), y(0)))
                parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1, y1 - y0):.1f}" fill="{color}" rx="2"/>')
    # As the app: a label every `every` categories, each clipped to the room it has.
    every = max(1, math.ceil(len(labels) * 64 / plot_w))
    room = max(6, int(step * every / 7))
    for i, label in enumerate(labels):
        if i % every == 0:
            parts.append(f'<text x="{pad_l + step * (i + 0.5):.1f}" y="{h - pad_b + 16}" font-size="10" text-anchor="middle" fill="#555">{html.escape(_clip(label, room))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _list_html(items: list[Block]) -> str:
    """Consecutive list blocks → nested <ul>/<ol>, honouring indent."""
    out: list[str] = []
    stack: list[str] = []
    for b in items:
        tag = "ol" if b["type"] == "numbered" else "ul"
        depth = b.get("indent", 0) + 1
        while len(stack) > depth:
            out.append(f"</li></{stack.pop()}>")
        if len(stack) == depth and stack[-1] != tag:
            out.append(f"</li></{stack.pop()}>")
        if len(stack) == depth:
            out.append("</li>")
        while len(stack) < depth:
            cls = ' class="tasks"' if b["type"] == "task" else ""
            out.append(f"<{tag}{cls}>")
            stack.append(tag)
        text = inline_html(b.get("text", ""))
        if b["type"] == "task":
            text = ("☑ " if b.get("done") else "☐ ") + text
        out.append(f"<li>{text}")
    while stack:
        out.append(f"</li></{stack.pop()}>")
    return "".join(out)


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _date_label(value: str) -> str:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", (value or "").strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        return value
    return f"{_MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}"


def _cell_html(col: dict[str, Any], value: str, rich: bool) -> str:
    if not value:
        return ""
    if col["kind"] == "status" and rich:
        tone = next((o["tone"] for o in col.get("options", []) if o["name"].lower() == value.lower()), "gray")
        return f"<span class='chip tone-{tone}'>{html.escape(value)}</span>"
    if col["kind"] == "date":
        return html.escape(_date_label(value))
    return inline_html(value)


# Each block's HTML. ``rich`` is the page a browser shows (charts drawn, status
# chips, metric cards); without it, plain markup the Word and PDF converters
# understand (a chart as its data table, images from the web only).
def _heading_html(tab: Tab, b: Block, rich: bool) -> str:
    return f"<h{b['level']}>{inline_html(b['text'])}</h{b['level']}>"


def _paragraph_html(tab: Tab, b: Block, rich: bool) -> str:
    return f"<p>{inline_html(b['text'])}</p>"


def _quote_html(tab: Tab, b: Block, rich: bool) -> str:
    return f"<blockquote>{inline_html(b['text'])}</blockquote>"


def _callout_html(tab: Tab, b: Block, rich: bool) -> str:
    title = f"<strong>{inline_html(b['title'])}</strong> " if b.get("title") else ""
    # The text sits in its own paragraph: Word's converter appends bare text
    # inside a div to whatever paragraph came before (a heading).
    return f"<div class='callout callout-{b['tone']}'><p>{title}{inline_html(b['text'])}</p></div>"


def _code_html(tab: Tab, b: Block, rich: bool) -> str:
    return f"<pre><code>{html.escape(b.get('code', ''))}</code></pre>"


def _divider_html(tab: Tab, b: Block, rich: bool) -> str:
    return "<hr/>"


_WEB_ADDRESS = re.compile(r"^https?://", re.I)


def _image_html(tab: Tab, b: Block, rich: bool) -> str:
    cap = f"<figcaption>{html.escape(b['caption'])}</figcaption>" if b.get("caption") else ""
    # The page shows any image a doc may hold. The Word and PDF inliner reads
    # the doc's own folder and never fetches, so an export keeps an image beside
    # the doc or inline, and gives a web image's words instead.
    keep = image_src_ok(b["src"]) and (rich or not _WEB_ADDRESS.match(b["src"].strip()))
    if keep:
        return f"<figure><img src='{html.escape(b['src'])}' alt='{html.escape(b.get('alt', ''))}'/>{cap}</figure>"
    if b.get("alt") or b.get("caption"):
        return f"<p><em>{html.escape(b.get('alt') or b.get('caption'))}</em></p>"
    return ""


def _table_html(tab: Tab, b: Block, rich: bool) -> str:
    def num(c: dict[str, Any]) -> str:
        return " class=num" if c["kind"] == "number" else ""

    head = "".join(f"<th{num(c)}>{inline_html(c['name'])}</th>" for c in b["columns"])
    rows = "".join(
        "<tr>" + "".join(f"<td{num(c)}>{_cell_html(c, r['cells'].get(c['id'], ''), rich)}</td>" for c in b["columns"])
        + "</tr>"
        for r in b["rows"])
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    return f"<div class='table-wrap'>{table}</div>" if rich else table


def _chart_html(tab: Tab, b: Block, rich: bool) -> str:
    labels, series = _chart_rows(tab, b)
    unit = f" ({html.escape(b['unit'])})" if b.get("unit") else ""
    title = f"<figcaption><strong>{html.escape(b.get('title', ''))}</strong>{unit}</figcaption>"
    body = _chart_svg(b, labels, series) if rich else _chart_table_html(labels, series)
    return f"<figure class='chart'>{title}{body}</figure>"


def _metrics_html(tab: Tab, b: Block, rich: bool) -> str:
    def tile(m: dict[str, Any], delta_class: str) -> str:
        delta = f"<div class='{delta_class}'>{html.escape(m['delta'])}</div>" if m.get("delta") else ""
        return f"<div class='m-label'>{html.escape(m['label'])}</div><div class='m-value'>{html.escape(m['value'])}</div>{delta}"

    if rich:
        cards = "".join(f"<div class='metric'>{tile(m, 'm-delta m-' + m.get('tone', 'neutral'))}</div>" for m in b["items"])
        return f"<div class='metrics'>{cards}</div>"
    cells = "".join(f"<td>{tile(m, 'm-delta')}</td>" for m in b["items"])
    return f"<table class='metrics'><tr>{cells}</tr></table>"


_BLOCK_HTML: dict[str, Callable[[Tab, Block, bool], str]] = {
    "heading": _heading_html,
    "paragraph": _paragraph_html,
    "quote": _quote_html,
    "callout": _callout_html,
    "code": _code_html,
    "divider": _divider_html,
    "image": _image_html,
    "table": _table_html,
    "chart": _chart_html,
    "metrics": _metrics_html,
    # A pending section is the agent's placeholder: nothing to show a reader.
    "pending": lambda tab, b, rich: "",
}


def tab_html(tab: Tab, *, rich: bool = True) -> str:
    out: list[str] = []
    blocks = tab["blocks"]
    i = 0
    while i < len(blocks):
        if blocks[i]["type"] in LIST_TYPES:
            j = i
            while j < len(blocks) and blocks[j]["type"] in LIST_TYPES:
                j += 1
            out.append(_list_html(blocks[i:j]))
            i = j
            continue
        markup = _BLOCK_HTML[blocks[i]["type"]](tab, blocks[i], rich)
        if markup:
            out.append(markup)
        i += 1
    return "\n".join(out)


# The page a doc opens as in a browser and is shared as: the editor's look,
# standalone (no scripts, no external assets), light and dark, and printable.
PAGE_CSS = """
:root{color-scheme:light dark;--bg:#fafafa;--surface:#fff;--surface-2:#f4f4f4;--line:#e4e4e1;--ink:#0e0f10;--ink-2:#3a3d40;
--ink-3:#6b6f73;--accent:#1f9cb0;--accent-ink:#115562;
--gray-bg:#eeeff0;--gray:#4a4f55;--blue-bg:#e3eefb;--blue:#1c5296;--green-bg:#e1f3e6;--green:#1c6236;
--yellow-bg:#fbf0d5;--yellow:#6e5010;--red-bg:#fbe5e2;--red:#922b20;--purple-bg:#eee7fa;--purple:#553789}
@media (prefers-color-scheme:dark){:root{--bg:#0f1113;--surface:#16191c;--surface-2:#1d2125;--line:#2a2f34;--ink:#f2f4f5;
--ink-2:#c9cdd2;--ink-3:#8f969d;--accent:#22d3ee;--accent-ink:#7fe3f3;
--gray-bg:#2c2f33;--gray:#c9cdd2;--blue-bg:#18304d;--blue:#a9cbf5;--green-bg:#173826;--green:#9fdcb2;
--yellow-bg:#3b3014;--yellow:#f0d08a;--red-bg:#45201c;--red:#f4ada4;--purple-bg:#2e2346;--purple:#cdb8f3}}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;color:var(--ink-2);font:15px/1.65 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
.page{max-width:780px;margin:0 auto;padding:56px 32px 96px}
.page-title{font-size:32px;line-height:1.2;font-weight:650;letter-spacing:-.02em;color:var(--ink);margin:0 0 6px}
.meta{font-size:13px;color:var(--ink-3);margin:0 0 28px}
nav.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 28px;padding-bottom:14px;border-bottom:1px solid var(--line)}
nav.tabs a{font-size:13px;color:var(--ink-2);text-decoration:none;padding:4px 10px;border-radius:6px;background:var(--surface-2)}
nav.tabs a:hover{color:var(--ink)}
section.tab+section.tab{margin-top:48px;padding-top:28px;border-top:1px solid var(--line)}
h2.tab-title{font-size:22px;margin:0 0 14px;color:var(--ink)}
h1,h2,h3{color:var(--ink);letter-spacing:-.01em}
h1{font-size:24px;line-height:1.3;margin:32px 0 8px}h2{font-size:19px;margin:28px 0 8px}h3{font-size:16px;margin:20px 0 6px}
p{margin:6px 0}
a{color:var(--accent-ink);text-underline-offset:2px}
strong{color:var(--ink);font-weight:620}
ul,ol{padding-left:1.4em;margin:6px 0}
li{margin:3px 0}
ul.tasks{list-style:none;padding-left:.2em}
code{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;background:var(--surface-2);
padding:.12em .36em;border-radius:5px}
pre{background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto}
pre code{background:none;padding:0}
blockquote{border-left:3px solid var(--line);margin:8px 0;padding:2px 0 2px 14px}
hr{border:0;border-top:1px solid var(--line);margin:20px 0}
.callout{border-radius:10px;padding:12px 14px;margin:10px 0;background:var(--surface-2)}.callout p{margin:0}
.callout-info{background:var(--blue-bg)}.callout-success{background:var(--green-bg)}
.callout-warning{background:var(--yellow-bg)}.callout-danger{background:var(--red-bg)}
.chip{display:inline-block;padding:0 7px;border-radius:6px;font-size:.86em;line-height:1.6;background:var(--gray-bg);color:var(--gray);
white-space:nowrap}
.tone-blue{background:var(--blue-bg);color:var(--blue)}.tone-green{background:var(--green-bg);color:var(--green)}
.tone-yellow{background:var(--yellow-bg);color:var(--yellow)}.tone-red{background:var(--red-bg);color:var(--red)}
.tone-purple{background:var(--purple-bg);color:var(--purple)}.tone-gray{background:var(--gray-bg);color:var(--gray)}
.table-wrap{margin:10px 0;overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);border-right:1px solid var(--line)}
tr>:last-child{border-right:0}tbody tr:last-child td{border-bottom:0}
th{background:var(--surface-2);color:var(--ink-3);font-weight:600;font-size:12.5px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:12px 0}
figure.chart{border:1px solid var(--line);border-radius:10px;padding:14px 16px 10px;background:var(--surface)}
figure.chart figcaption{font-size:14px;color:var(--ink);margin-bottom:6px}
figure.chart svg text{fill:var(--ink-3)}figure.chart svg line{stroke:var(--line)}
figure img{max-width:100%;border-radius:10px;border:1px solid var(--line)}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}
.metric{padding:12px 14px;border-radius:10px;background:var(--surface-2)}
.m-label{font-size:12.5px;color:var(--ink-3)}.m-value{font-size:24px;font-weight:650;color:var(--ink);line-height:1.25}
.m-delta{font-size:12px;color:var(--ink-3)}.m-good{color:var(--green)}.m-bad{color:var(--red)}
footer{margin-top:56px;font-size:12px;color:var(--ink-3)}
@media print{html{background:#fff}.page{padding:0}nav.tabs{display:none}.table-wrap{overflow:visible}}
@media (max-width:640px){.page{padding:32px 18px 64px}.page-title{font-size:26px}}
"""


def export_body_html(doc: Doc) -> str:
    """The doc as body markup for the Word and PDF converters, each extra tab under its title."""
    multi = len(doc["tabs"]) > 1
    parts = []
    for n, tab in enumerate(doc["tabs"]):
        body = tab_html(tab, rich=False)
        starts_with_h1 = bool(tab["blocks"]) and tab["blocks"][0]["type"] == "heading" and tab["blocks"][0]["level"] == 1
        if multi and not starts_with_h1:
            body = f"<h1{' class=tab' if n else ''}>{html.escape(tab['title'])}</h1>\n" + body
        parts.append(body)
    return "\n".join(parts)


def doc_html(doc: Doc) -> str:
    """The doc as a standalone page (doc.html): what opens in a browser and what sharing publishes."""
    multi = len(doc["tabs"]) > 1
    sections = []
    for n, tab in enumerate(doc["tabs"]):
        head = f"<h2 class='tab-title'>{html.escape(tab['title'])}</h2>" if multi else ""
        sections.append(f"<section class='tab' id='tab-{n + 1}'>{head}{tab_html(tab)}</section>")
    nav = ""
    if multi:
        nav = "<nav class='tabs'>" + "".join(
            f"<a href='#tab-{n + 1}'>{html.escape(t['title'])}</a>" for n, t in enumerate(doc["tabs"])) + "</nav>"
    updated = (doc.get("updated_at") or "")[:10]
    meta = f"<p class='meta'>Updated {html.escape(_date_label(updated))}</p>" if updated else ""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(doc['title'])}</title><style>{PAGE_CSS}</style></head><body>"
        f"<main class='page'><h1 class='page-title'>{html.escape(doc['title'])}</h1>{meta}{nav}"
        + "".join(sections)
        + "<footer>Made with MindsHub Cowork</footer></main></body></html>"
    )
