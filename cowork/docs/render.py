"""Views of a doc: the agent's outline and full read, and HTML for export."""

from __future__ import annotations

import base64
import html
import math
import re
import unicodedata
from datetime import date
from pathlib import Path
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


def clip(text: str, n: int = 90) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _block_line(b: dict[str, Any]) -> str:
    t = b["type"]
    label = _SHORT[t] + (str(b["level"]) if t == "heading" else "")
    if t in LIST_TYPES and b.get("indent"):
        label += f">{b['indent']}"
    if t == "task":
        label += " [x]" if b.get("done") else " [ ]"
    if t in TEXT_TYPES:
        detail = clip(b.get("text", ""))
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

# Inline Markdown is read as CommonMark reads it, for the marks a doc uses:
# backslash escapes, `code`, [links](https://…), {{chips}}, *emphasis*,
# **strong** (with * or _) and ~~strike~~. The editor (docModel.inlineToHtml)
# reads it with the same steps; tests/fixtures/docs_inline/cases.json holds
# the cases both must render alike.
#
# One pass from the left finds escapes, code spans, chips and links. They bind
# tighter than emphasis, so a mark inside them never opens or closes emphasis.
# Emphasis is paired by CommonMark's delimiter-run rules: inside each link's
# words on their own (at the link's "]"), then over the rest, so tags always
# nest. Every look ahead is a cached "next match from here", which keeps all
# of it linear in the length of the text.
#
# Looser than CommonMark: a run of * or ~~ opens unless white space follows it
# and closes unless white space precedes it, so "**Note:**text" and
# "价格**$5**起" are bold, as the editor writes them. "_" keeps CommonMark's
# rule, so snake_case stays as typed.
#
# Narrower than CommonMark: a code span is one backtick each side, kept
# verbatim; only these marks escape; a link is inline, has words, and its
# address is http(s):// or mailto: plus at least one character, up to ")";
# strike is exactly two tildes; no raw HTML, entities, autolinks or hard breaks.
_ESCAPABLE = r"[*`~\[\]{}_\\]"
_TOKEN = re.compile(rf"\\{_ESCAPABLE}|[`\[\]{{]|\*+|_+|~+")
_BACKTICK = re.compile("`")
_BRACE = re.compile(r"\}")
# CommonMark's white space: tab, line feed, form feed, carriage return and the Unicode Zs spaces.
_SPACE_CHARS = "\t\n\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_SPACE = f"[{re.escape(_SPACE_CHARS)}]"
_CHIP_HEAD = re.compile(r"\{\{" + _SPACE + r"*(date|status|user)" + _SPACE + "*:")
_LINK_HEAD = re.compile(r"\]\((?:https?://|mailto:)")
_HREF_END = re.compile(r"[)\x00-\x20\x7f]")  # where CommonMark ends an address not in <…>
_HREF_ESCAPE = re.compile(rf"\\({_ESCAPABLE})")
_EM, _STRONG, _STRIKE = ("<em>", "</em>"), ("<strong>", "</strong>"), ("<s>", "</s>")


def _char_class(ch: str) -> int:
    """A character beside a run of marks: white space (0), punctuation (1: CommonMark's
    Unicode P and S categories) or anything else (2).

    P and S come from Python's Unicode tables; the editor's may be newer and class a
    character assigned since then apart.
    """
    return 0 if ch in _SPACE_CHARS else 1 if unicodedata.category(ch)[0] in "PS" else 2


def _flanks(before: int, after: int, underscore: bool) -> tuple[bool, bool]:
    """Whether a run can open and close emphasis, by the classes of the characters around it."""
    space_before, punct_before, space_after, punct_after = before == 0, before == 1, after == 0, after == 1
    if not underscore:
        return not space_after, not space_before
    left = not space_after and (not punct_after or space_before or punct_before)
    right = not space_before and (not punct_before or space_after or punct_after)
    # "_" never opens or closes inside a word.
    return left and (not right or punct_before), right and (not left or punct_after)


_FLANKS = {u: tuple(_flanks(b, a, u) for b in range(3) for a in range(3)) for u in (False, True)}


def _escape(text: str) -> str:
    """Text for HTML, escaped as the editor's escapeHtml does."""
    return html.escape(text, quote=False).replace('"', "&quot;")


def _text_html(text: str) -> str:
    """Plain text as HTML, a line feed as the line break the editor shows."""
    return _escape(text).replace("\n", "<br>")


class _Next:
    """The first match of ``pattern`` in ``s`` at or after a position (``len(s)`` if none).

    Asked for positions that only grow, it searches each stretch of ``s`` once.
    """

    def __init__(self, pattern: re.Pattern[str], s: str) -> None:
        self._pattern, self._s = pattern, s
        self._from, self._at = 1, 0  # an empty span: the first ask searches

    def __call__(self, pos: int) -> int:
        if not self._from <= pos <= self._at:
            m = self._pattern.search(self._s, pos)
            self._from, self._at = pos, m.start() if m else len(self._s)
        return self._at


class _Atom:
    """Text replaced by markup of its own: an escape, a code span, a chip or a link's end."""

    __slots__ = ("start", "end", "markup")

    def __init__(self, start: int, end: int, markup: str) -> None:
        self.start, self.end, self.markup = start, end, markup

    def html(self) -> str:
        return self.markup


class _Bracket:
    """A "[" that opens a link once its "](address)" is found."""

    __slots__ = ("start", "end", "runs", "href")

    def __init__(self, start: int, runs: _Run) -> None:
        self.start, self.end, self.runs, self.href = start, start + 1, runs, None

    def html(self) -> str:
        return "[" if self.href is None else f'<a href="{_escape(self.href)}">'


class _Run:
    """A run of *, _ or ~~ that may open or close emphasis (CommonMark's delimiter run).

    Live runs form a list through ``prev``/``next``. Pairing spends marks from
    the run's inner side: an opener's from its end, a closer's from its start.
    """

    __slots__ = ("start", "end", "mark", "length", "left", "can_open", "can_close", "prev", "next", "opens", "closes")

    def __init__(self, start: int, end: int, mark: str, can_open: bool, can_close: bool, prev: _Run | None) -> None:
        self.start, self.end, self.mark = start, end, mark
        self.length = self.left = end - start
        self.can_open, self.can_close = can_open, can_close
        self.prev, self.next = prev, None
        # The tags it opens after its marks and closes before them, innermost first; most runs pair with nothing.
        self.opens: list[str] | None = None
        self.closes: list[str] | None = None

    def pairs_with(self, closer: _Run) -> bool:
        # CommonMark's rule of 3: "*a**b*" is one emphasis, not two.
        return self.mark == closer.mark and self.can_open and not (
            (self.can_close or closer.can_open) and closer.length % 3 and (self.length + closer.length) % 3 == 0
        )

    def html(self) -> str:
        marks = self.mark * self.left
        closes = "".join(self.closes) if self.closes else ""
        return closes + marks + "".join(reversed(self.opens)) if self.opens else closes + marks

    def drop(self) -> None:
        self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev


def _pair_runs(bottom: _Run) -> None:
    """Pair the live runs above ``bottom`` into emphasis, then drop them (CommonMark's "process emphasis")."""
    # Per kind of closer, the start at or before which no opener pairs with it.
    floor: dict[tuple[str, bool, int], int] = {}
    closer = bottom.next
    while closer is not None:
        if not closer.can_close:
            closer = closer.next
            continue
        kind = (closer.mark, closer.can_open, closer.length % 3)
        lowest = floor.get(kind, bottom.start)
        opener = closer.prev
        while opener.start > lowest and not opener.pairs_with(closer):
            opener = opener.prev
        if opener.start <= lowest:
            floor[kind] = closer.prev.start
            if not closer.can_open:
                closer.drop()
            closer = closer.next
            continue
        used = 2 if opener.left >= 2 and closer.left >= 2 else 1
        open_tag, close_tag = _STRIKE if closer.mark == "~" else _STRONG if used == 2 else _EM
        opener.left -= used
        closer.left -= used
        if opener.opens is None:
            opener.opens = []
        opener.opens.append(open_tag)
        if closer.closes is None:
            closer.closes = []
        closer.closes.append(close_tag)
        opener.next, closer.prev = closer, opener  # the runs between pair with nothing now
        if not opener.left:
            opener.drop()
        if not closer.left:
            closer.drop()
            closer = closer.next
    bottom.next = None


def _read_chip(s: str, i: int, brace: _Next) -> _Atom | None:
    if not (head := _CHIP_HEAD.match(s, i)):
        return None
    q = brace(head.end())  # the value runs to the first "}"
    if q == head.end() or not s.startswith("}}", q):
        return None
    kind, raw = head.group(1), s[head.end():q]
    value = raw.strip(_SPACE_CHARS) or raw[-1]  # at least one character, as typed
    label = _date_label(value) if kind == "date" else value
    return _Atom(i, q + 2, f'<span class="chip chip-{kind}">{_escape(label)}</span>')


def _link_end(s: str, q: int, href_end: _Next) -> tuple[str, int] | None:
    """The address and end of a "](address)" at ``q``, or None."""
    if not (head := _LINK_HEAD.match(s, q)):
        return None
    e = href_end(head.end())
    if e == head.end() or not s.startswith(")", e):
        return None
    return _HREF_ESCAPE.sub(r"\1", s[q + 2:e]), e + 1


def _read_inline(s: str) -> list[_Atom | _Bracket | _Run]:
    """What replaces parts of ``s``, in text order: atoms (escapes, code spans, chips, a
    link's end) with their HTML, "[" brackets, and delimiter runs."""
    n = len(s)
    items: list[_Atom | _Bracket | _Run] = []
    base = top = _Run(-1, -1, "", False, False, None)  # below every live run
    backtick, brace, href_end = _Next(_BACKTICK, s), _Next(_BRACE, s), _Next(_HREF_END, s)
    brackets: list[_Bracket] = []
    pos = 0
    while m := _TOKEN.search(s, pos):
        i, pos = m.span()
        c = s[i]
        if c in "*_~":
            if c == "~" and pos - i != 2:
                continue
            before = _char_class(s[i - 1]) if i else 0  # the text's edges count as white space
            after = _char_class(s[pos]) if pos < n else 0
            can_open, can_close = _FLANKS[c == "_"][before * 3 + after]
            if can_open or can_close:
                run = _Run(i, pos, c, can_open, can_close, top)
                top.next = run
                top = run
                items.append(run)
        elif c == "\\":
            items.append(_Atom(i, pos, s[i + 1]))  # no escapable mark is special to HTML
        elif c == "`":
            k = backtick(pos)  # the code runs to the next backtick
            if pos < k < n:
                items.append(_Atom(i, k + 1, f"<code>{_escape(s[pos:k])}</code>"))
                pos = k + 1
        elif c == "{":
            if chip := _read_chip(s, i, brace):
                items.append(chip)
                pos = chip.end
        elif c == "[":
            brackets.append(_Bracket(i, top))
            items.append(brackets[-1])
        elif brackets:  # "]": the last "[" pairs with it or with none
            opener = brackets.pop()
            # A link needs words: "[](…)" would be a link no one sees.
            if i > opener.end and (link := _link_end(s, i, href_end)):
                opener.href, pos = link
                items.append(_Atom(i, pos, "</a>"))
                _pair_runs(opener.runs)
                top = opener.runs
                brackets.clear()  # links never nest: a "[" before this link opens none
    _pair_runs(base)
    for item in items:
        if item.__class__ is _Run:  # unlinked, the runs free without the cycle collector
            item.prev = item.next = None
    return items


def inline_html(text: str) -> str:
    """A block's inline Markdown as HTML: escapes, code, links, emphasis, strike and chips.

    Every text run is escaped, so text never becomes markup, and only
    http(s):// and mailto: addresses become links.
    """
    s = (text or "").replace("\x00", "")
    if not _TOKEN.search(s):
        return _text_html(s)
    out, done = [], 0
    for item in _read_inline(s):
        if done < item.start:
            out.append(_text_html(s[done:item.start]))
        out.append(item.html())
        done = item.end
    out.append(_text_html(s[done:]))
    return "".join(out)


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


# The editor's categorical order (light), so an export matches what the person saw.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _legend_html(entries: list[tuple[str, str]]) -> str:
    """Which colour is which series (or, for a donut, which slice)."""
    items = "".join(
        f"<span><i style='background:{color}'></i>{html.escape(name)}</span>" for name, color in entries
    )
    return f"<div class='legend'>{items}</div>"


def _stack_totals(labels: list[str], series: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Each label's stacked extent (chartGeometry.stackTotals): the sum of its negative
    values and the sum of its positive ones."""
    totals = []
    for i in range(len(labels)):
        vals = [s["values"][i] or 0 for s in series]
        totals.append((sum(v for v in vals if v < 0), sum(v for v in vals if v >= 0)))
    return totals


def _donut_svg(b: dict[str, Any], labels: list[str], series: list[dict[str, Any]]) -> str:
    """The first series as slices clockwise from 12 o'clock (chartGeometry.donutArcs), with a
    legend naming each slice and its value."""
    if not labels or not series:
        return ""
    values = [max(0.0, v or 0.0) for v in series[0]["values"]]
    total = sum(values)
    if total <= 0:
        return ""
    r, inner = 100.0, 60.0

    def at(rad: float, ang: float) -> tuple[float, float]:
        return r + rad * math.cos(ang), r + rad * math.sin(ang)

    parts = [f'<svg viewBox="0 0 {2 * r:.0f} {2 * r:.0f}" width="200" role="img" aria-label="{html.escape(b.get("title", ""))}">']
    angle = -math.pi / 2
    for i, v in enumerate(values):
        sweep = v / total * 2 * math.pi
        if sweep <= 0:
            continue
        a0, a1 = angle, angle + sweep
        angle = a1
        if sweep >= 2 * math.pi - 1e-6:
            # A whole ring can't be one arc (its ends meet): two half-open circles.
            d = (f"M{r},0A{r},{r} 0 1 1 {r - 0.01},0Z "
                 f"M{r},{r - inner}A{inner},{inner} 0 1 0 {r + 0.01},{r - inner}Z")
        else:
            large = 1 if sweep > math.pi else 0
            (x0, y0), (x1, y1) = at(r, a0), at(r, a1)
            (x2, y2), (x3, y3) = at(inner, a1), at(inner, a0)
            d = (f"M{x0:.2f},{y0:.2f}A{r},{r} 0 {large} 1 {x1:.2f},{y1:.2f}"
                 f"L{x2:.2f},{y2:.2f}A{inner},{inner} 0 {large} 0 {x3:.2f},{y3:.2f}Z")
        parts.append(f'<path d="{d}" fill="{_color(i)}" fill-rule="evenodd"/>')
    parts.append("</svg>")
    fmt = _axis_format(values, b.get("unit"))
    return "".join(parts) + _legend_html(
        [(f"{label} {fmt(v)}", _color(i)) for i, (label, v) in enumerate(zip(labels, values)) if v > 0]
    )


def _chart_svg(b: dict[str, Any], labels: list[str], series: list[dict[str, Any]]) -> str:
    """A plain chart of each kind the editor offers, for HTML export (the app draws the
    interactive one), on the app's axis: round ticks, compact numbers, a value margin as wide
    as its longest label, and every n-th category label when they would run into each other.
    Bars stand side by side, or on one another (stacked), or lie along the value axis (hbar);
    a donut is drawn by _donut_svg. More than one series gets a legend."""
    kind = b.get("chart", "bar")
    if kind == "donut":
        return _donut_svg(b, labels, series)
    vals = [v for s in series for v in s["values"] if v is not None]
    if not labels or not vals:
        return ""
    stacked = kind == "stacked"
    low, top, ticks = _nice_scale([v for pair in _stack_totals(labels, series) for v in pair] if stacked else vals)
    fmt = _axis_format(vals, b.get("unit"))
    tick_text = [fmt(v) for v in ticks]
    span = (top - low) or 1
    aria = html.escape(b.get("title", ""))
    if kind == "hbar":
        svg = _hbar_svg(labels, series, low, span, ticks, tick_text, aria)
    else:
        # 10px labels average about 6px a character; the margin fits the longest, with a gap.
        w, h, pad_b, pad_t = 640, 260, 36, 16
        pad_l = max(36, 6 * max(len(t) for t in tick_text) + 12)
        plot_w, plot_h = w - pad_l - 8, h - pad_b - pad_t

        def y(v: float) -> float:
            return pad_t + plot_h * (1 - (v - low) / span)

        parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{aria}">']
        for v, text in zip(ticks, tick_text):
            parts.append(f'<line x1="{pad_l}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e2dc"/>'
                         f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="#777">{html.escape(text)}</text>')
        step = plot_w / len(labels)
        line_like = kind in ("line", "area")
        # Where each label's next stacked bar starts, above zero and below it.
        ups, downs = [0.0] * len(labels), [0.0] * len(labels)
        for si, s in enumerate(series):
            color = _color(si)
            if line_like:
                pts = [(pad_l + step * (i + 0.5), y(v)) for i, v in enumerate(s["values"]) if v is not None]
                if kind == "area" and pts:
                    base = y(max(low, min(0, top)))
                    parts.append('<polygon fill="%s" fill-opacity="0.15" points="%s"/>' % (color, " ".join(
                        f"{px:.1f},{py:.1f}" for px, py in [(pts[0][0], base), *pts, (pts[-1][0], base)])))
                parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (
                    color, " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)))
                continue
            bw = step * 0.7 if stacked else step * 0.7 / max(1, len(series))
            for i, v in enumerate(s["values"]):
                if v is None:
                    continue
                x0 = pad_l + step * i + step * 0.15 + (0 if stacked else bw * si)
                base = 0.0
                if stacked:
                    base = ups[i] if v >= 0 else downs[i]
                    if v >= 0:
                        ups[i] += v
                    else:
                        downs[i] += v
                y0, y1 = sorted((y(base + v), y(base)))
                parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1, y1 - y0):.1f}" fill="{color}" rx="2"/>')
        # As the app: a label every `every` categories, each clipped to the room it has.
        every = max(1, math.ceil(len(labels) * 64 / plot_w))
        room = max(6, int(step * every / 7))
        for i, label in enumerate(labels):
            if i % every == 0:
                parts.append(f'<text x="{pad_l + step * (i + 0.5):.1f}" y="{h - pad_b + 16}" font-size="10" text-anchor="middle" fill="#555">{html.escape(clip(label, room))}</text>')
        parts.append("</svg>")
        svg = "".join(parts)
    if len(series) > 1:
        svg += _legend_html([(s["name"], _color(si)) for si, s in enumerate(series)])
    return svg


def _hbar_svg(labels: list[str], series: list[dict[str, Any]], low: float, span: float,
              ticks: list[float], tick_text: list[str], aria: str) -> str:
    """Bars lying along the value axis: a row per label, the labels down the left, the value
    ticks along the bottom."""
    w, pad_t, pad_b = 640, 8, 24
    row = 28.0 if len(labels) <= 8 else max(14.0, 224 / len(labels))
    h = pad_t + row * len(labels) + pad_b
    pad_l = min(160, max(48, 6 * max(len(label) for label in labels) + 12))
    plot_w = w - pad_l - 16

    def x(v: float) -> float:
        return pad_l + plot_w * (v - low) / span

    parts = [f'<svg viewBox="0 0 {w} {h:.0f}" width="100%" role="img" aria-label="{aria}">']
    for v, text in zip(ticks, tick_text):
        parts.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{pad_t}" y2="{h - pad_b:.1f}" stroke="#e5e2dc"/>'
                     f'<text x="{x(v):.1f}" y="{h - pad_b + 14:.1f}" font-size="10" text-anchor="middle" fill="#777">{html.escape(text)}</text>')
    bh = row * 0.7 / max(1, len(series))
    for si, s in enumerate(series):
        for i, v in enumerate(s["values"]):
            if v is None:
                continue
            y0 = pad_t + row * i + row * 0.15 + bh * si
            x0, x1 = sorted((x(v), x(0)))
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{max(1, x1 - x0):.1f}" height="{bh:.1f}" fill="{_color(si)}" rx="2"/>')
    room = max(6, int((pad_l - 12) / 6))
    for i, label in enumerate(labels):
        parts.append(f'<text x="{pad_l - 6}" y="{pad_t + row * (i + 0.5) + 4:.1f}" font-size="10" text-anchor="end" fill="#555">{html.escape(clip(label, room))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _list_html(items: list[Block], rich: bool) -> str:
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
        if b["type"] == "task" and rich:
            # The editor's check box, drawn by PAGE_CSS in the list's indent.
            done = " done" if b.get("done") else ""
            out.append(f"<li class='task{done}'><span class='task-box'></span><span class='task-text'>{text}</span>")
            continue
        if b["type"] == "task":
            text = ("☑ " if b.get("done") else "☐ ") + text
        out.append(f"<li>{text}")
    while stack:
        out.append(f"</li></{stack.pop()}>")
    return "".join(out)


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _date_label(value: str) -> str:
    """``2026-10-21`` as ``Oct 21, 2026``; anything but a real day of a four-digit year unchanged.

    The editor's formatDate refuses the same values (it formats the rest in the app's language).
    """
    m = re.match(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$", (value or "").strip(_SPACE_CHARS))
    try:
        day = date(int(m[1]), int(m[2]), int(m[3])) if m and m[1] >= "1000" else None
    except ValueError:
        day = None
    return f"{_MONTHS[day.month - 1]} {day.day}, {day.year}" if day else value


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
    # An empty <p> has no height and its margins collapse into the neighbours',
    # so a blank line the editor shows would vanish from the page.
    if not b["text"].strip():
        return "<p><br></p>"
    return f"<p>{inline_html(b['text'])}</p>"


def _quote_html(tab: Tab, b: Block, rich: bool) -> str:
    return f"<blockquote>{inline_html(b['text'])}</blockquote>"


# The editor's callout icons (lucide 1.31: info, circle-check, triangle-alert,
# octagon-alert, sticky-note) at 16px with its 1px line.
_CALLOUT_ICONS = {
    "info": "<circle cx='12' cy='12' r='10'/><path d='M12 16v-4'/><path d='M12 8h.01'/>",
    "success": "<circle cx='12' cy='12' r='10'/><path d='m9 12 2 2 4-4'/>",
    "warning": ("<path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3'/>"
                "<path d='M12 9v4'/><path d='M12 17h.01'/>"),
    "danger": ("<path d='M12 16h.01'/><path d='M12 8v4'/><path d='M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 "
               "2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 "
               "1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 "
               "8.688 2z'/>"),
    "note": ("<path d='M21 9a2.4 2.4 0 0 0-.706-1.706l-3.588-3.588A2.4 2.4 0 0 0 15 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 "
             "2 2h14a2 2 0 0 0 2-2z'/><path d='M15 3v5a1 1 0 0 0 1 1h5'/>"),
}


def _callout_html(tab: Tab, b: Block, rich: bool) -> str:
    if rich:
        tone = b["tone"] if b["tone"] in _CALLOUT_ICONS else "info"
        icon = ("<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
                "stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
                f"{_CALLOUT_ICONS[tone]}</svg>")
        head = f"<div class='callout-title'>{inline_html(b['title'])}</div>" if b.get("title") else ""
        return (f"<div class='callout callout-{tone}'><span class='callout-icon'>{icon}</span>"
                f"<div class='callout-body'>{head}<p>{inline_html(b['text'])}</p></div></div>")
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
        return (f"<figure class='image'><img src='{html.escape(b['src'])}' alt='{html.escape(b.get('alt', ''))}'/>"
                f"{cap}</figure>")
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
            out.append(_list_html(blocks[i:j], rich))
            i = j
            continue
        markup = _BLOCK_HTML[blocks[i]["type"]](tab, blocks[i], rich)
        if markup:
            out.append(markup)
        i += 1
    return "\n".join(out)


# Inter as the editor draws it (the app's InterVariable), cut by
# scripts/build_doc_page_fonts.py and carried inline: a doc's page is one file
# wherever it goes (published, exported as HTML, sent to the PDF renderer).
# Licensed under the SIL OFL 1.1 (page_fonts/OFL.txt).
_PAGE_FONTS = Path(__file__).with_name("page_fonts")
FONT_STACK = 'Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'


def _font_face(style: str, name: str) -> str:
    data = base64.b64encode((_PAGE_FONTS / name).read_bytes()).decode("ascii")
    return (f"@font-face{{font-family:Inter;font-style:{style};font-weight:400 700;font-display:swap;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")


FONT_CSS = _font_face("normal", "inter.woff2") + _font_face("italic", "inter-italic.woff2")
# A PDF export draws the doc's plain markup in the page's font.
PDF_CSS = FONT_CSS + f"body{{font-family:{FONT_STACK}}}"

# The page a doc opens as in a browser and is shared as: the editor's look,
# standalone (no scripts, no external assets), light and dark, and printable.
# The editor (cowork, components/doc/doc.css) is the reference: the numbers
# both share are pinned in tests/fixtures/docs_typography/rules.json.
PAGE_CSS = FONT_CSS + """
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
body{margin:0;color:var(--ink-2);font:15px/1.65 """ + FONT_STACK + """;font-optical-sizing:auto;
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
p,h1,h2,h3,blockquote,figure,pre,ul,ol{margin:0}
.page{--doc-gutter:56px;max-width:760px;margin:0 auto;padding:36px var(--doc-gutter) 96px;font-size:15px;line-height:1.65}
.page-title{font-size:30px;line-height:1.2;font-weight:650;letter-spacing:-.02em;color:var(--ink);margin:0 0 6px}
.meta{font-size:12.5px;color:var(--ink-3);margin:0 0 22px}
nav.tabs{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 22px;padding-bottom:14px;border-bottom:1px solid var(--line)}
nav.tabs a{font-size:12.5px;border-radius:6px;color:var(--ink-2);text-decoration:none;padding:5px 8px;background:var(--surface-2)}
nav.tabs a:hover{color:var(--ink)}
section.tab+section.tab{margin-top:48px;padding-top:28px;border-top:1px solid var(--line)}
.tab>p{padding:3px 0}
.tab>h1,.tab>h2,.tab>h3{color:var(--ink);font-weight:650;letter-spacing:-.01em}
.tab>h1{font-size:24px;line-height:1.3;margin-top:30px;padding:3px 0}
.tab>h2{font-size:19px;line-height:1.35;margin-top:26px;padding:3px 0}
.tab>h3{font-size:16px;line-height:1.4;margin-top:18px;padding:2px 0;font-weight:620}
.tab>:is(h1,h2,h3):first-child,.tab>.tab-title+:is(h1,h2,h3){margin-top:4px}
.tab>h2.tab-title{font-size:22px;line-height:1.3;margin:0 0 14px;padding:0}
a{color:var(--accent-ink);text-underline-offset:2px}
strong{color:var(--ink);font-weight:620}
code{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;background:var(--surface-2);
padding:.12em .36em;border-radius:5px;color:var(--ink)}
.tab ul,.tab ol{padding-left:26px}
.tab li>ul,.tab li>ol{padding-left:24px}
.tab li{padding:2px 0}
.tab ul.tasks{list-style:none;padding-left:24px}
.task-box{position:relative;display:inline-block;width:16px;height:16px;margin-left:-24px;margin-right:8px;vertical-align:-3px;
border:1.5px solid var(--line);border-radius:4px}
.task.done>.task-box{background:var(--accent);border-color:var(--accent)}
.task.done>.task-box::after{content:"";position:absolute;left:4px;top:1px;width:4px;height:8px;border:solid #fff;
border-width:0 1.5px 1.5px 0;transform:rotate(45deg)}
.task.done>.task-text{color:var(--ink-3);text-decoration:line-through}
.tab>blockquote{border-left:3px solid var(--line);margin:4px 0;padding:2px 0 2px 14px}
.tab>hr{border:0;border-top:1px solid var(--line);margin:18px 0}
.callout{display:flex;gap:10px;border-radius:10px;padding:12px 14px;margin:8px 0;background:var(--surface-2)}
.callout-icon{flex-shrink:0;margin-top:3px;color:var(--ink-3)}
.callout-body{flex:1;min-width:0}
.callout-title{font-weight:620;color:var(--ink)}
.callout-info{background:var(--blue-bg)}.callout-info .callout-icon{color:var(--blue)}
.callout-success{background:var(--green-bg)}.callout-success .callout-icon{color:var(--green)}
.callout-warning{background:var(--yellow-bg)}.callout-warning .callout-icon{color:var(--yellow)}
.callout-danger{background:var(--red-bg)}.callout-danger .callout-icon{color:var(--red)}
.chip{display:inline-flex;align-items:center;padding:0 7px;height:1.5em;border-radius:6px;font-size:.86em;line-height:1;
vertical-align:baseline;background:var(--gray-bg);color:var(--gray);white-space:nowrap}
.tone-blue{background:var(--blue-bg);color:var(--blue)}.tone-green{background:var(--green-bg);color:var(--green)}
.tone-yellow{background:var(--yellow-bg);color:var(--yellow)}.tone-red{background:var(--red-bg);color:var(--red)}
.tone-purple{background:var(--purple-bg);color:var(--purple)}.tone-gray{background:var(--gray-bg);color:var(--gray)}
.tab>pre{margin:8px 0;border-radius:10px;background:var(--surface-2);border:1px solid var(--line);overflow:hidden}
pre code{display:block;padding:12px 14px;font-size:12.5px;line-height:1.6;tab-size:2;background:none;border-radius:0;
white-space:pre;overflow-x:auto}
.tab>figure.image{margin:10px 0}
figure.image img{display:block;max-width:100%;border-radius:10px;border:1px solid var(--line)}
figure.image figcaption{font-size:12.5px;color:var(--ink-3);margin-top:6px}
.table-wrap{margin:10px 0;overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;line-height:1.5}
th,td{min-width:140px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);border-right:1px solid var(--line)}
tr>:last-child{border-right:0}tbody tr:last-child td{border-bottom:0}
th{padding:7px 12px;background:var(--surface-2);color:var(--ink-3);font-weight:600;font-size:12.5px}
td{padding:7px 12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
figure.chart{margin:12px 0;border:1px solid var(--line);border-radius:10px;padding:14px 16px 10px;background:var(--surface)}
figure.chart figcaption{font-size:14px;color:var(--ink);margin-bottom:8px}
figure.chart svg text{fill:var(--ink-3)}figure.chart svg line{stroke:var(--line)}
figure.chart .legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;font-size:11px;color:var(--ink-3)}
figure.chart .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:10px 0}
.metric{padding:12px 14px;border-radius:10px;background:var(--surface-2);min-width:0}
.m-label{font-size:12.5px;color:var(--ink-3)}
.m-value{font-size:24px;font-weight:650;letter-spacing:-.01em;color:var(--ink);line-height:1.25;margin-top:2px}
.m-delta{font-size:11px;color:var(--ink-3);margin-top:2px}.m-good{color:var(--green)}.m-bad{color:var(--red)}
footer{margin-top:56px;font-size:12px;color:var(--ink-3)}
@media print{html{background:#fff}.page{padding:0}nav.tabs{display:none}.table-wrap{overflow:visible}}
@media (max-width:619.98px){.page{--doc-gutter:48px;padding-top:24px}}
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
