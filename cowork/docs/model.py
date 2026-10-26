"""Doc artifact model: tabs of typed blocks, addressed by stable ids.

A doc is plain JSON so the server can store it as an artifact file and the
renderer can draw it without a translation layer:

    {"schema": "cowork.doc/1", "id", "title", "rev", "tabs": [
        {"id": "t_…", "title", "blocks": [{"id": "b_…", "type", "v", …}]}]}

Every block carries ``v``: the doc ``rev`` that last changed it. Edits that
quote a revision (``if_rev``) are refused when a block they touch moved on
since, which is how an agent edit and a person typing in the same doc avoid
silently overwriting each other.
"""

from __future__ import annotations

import copy
import math
import re
import secrets
import string
from datetime import date
from typing import Any

# A doc, a tab and a block are plain JSON objects (see the module docstring);
# these name which one a function takes or returns.
Doc = dict[str, Any]
Tab = dict[str, Any]
Block = dict[str, Any]

SCHEMA = "cowork.doc/1"

TEXT_TYPES = ("paragraph", "heading", "bullet", "numbered", "task", "quote", "callout")
LIST_TYPES = ("bullet", "numbered", "task")
BLOCK_TYPES = TEXT_TYPES + (
    "code",
    "divider",
    "table",
    "chart",
    "metrics",
    "image",
    "pending",
)
CALLOUT_TONES = ("info", "success", "warning", "danger", "note")
CHART_KINDS = ("bar", "hbar", "stacked", "line", "area", "donut")
COLUMN_KINDS = ("text", "number", "status", "date")
STATUS_TONES = ("gray", "blue", "green", "yellow", "red", "purple")

MAX_TABS = 40
MAX_BLOCKS_PER_TAB = 3000
MAX_TEXT = 20_000
MAX_TABLE_ROWS = 2000
MAX_TABLE_COLS = 40
MAX_INDENT = 4
MAX_METRICS = 8
# One colour each in the chart palette.
MAX_CHART_SERIES = 8

_ALPHABET = string.ascii_lowercase + string.digits


class DocError(Exception):
    """A refused change. ``code`` is stable; ``message`` says how to fix it."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        # Read when shown: a caller may add context to ``message`` after construction.
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.detail}


def new_id(prefix: str) -> str:
    return prefix + "".join(secrets.choice(_ALPHABET) for _ in range(8))


# ---------------------------------------------------------------- statuses

_TONE_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("green", ("done", "complete", "completed", "shipped", "live", "yes", "pass", "passed",
               "approved", "on track", "healthy", "resolved", "won", "low", "recommended", "selected",
               "accepted", "go")),
    ("blue", ("in progress", "active", "doing", "ongoing", "building", "started", "open")),
    ("red", ("blocked", "at risk", "off track", "fail", "failed", "no", "rejected", "high",
             "critical", "overdue", "lost", "cancelled", "canceled")),
    ("yellow", ("review", "in review", "pending", "waiting", "medium", "paused", "on hold",
                "partial", "needs work", "maybe", "draft", "shortlisted", "at review")),
    ("gray", ("not started", "todo", "to do", "planned", "backlog", "n/a", "none", "idea",
              "proposed")),
]


_TONE_OF_WORD = {w: tone for tone, words in reversed(_TONE_WORDS) for w in words}
_TONE_SUBSTRINGS = [(tone, [w for w in words if len(w) > 3]) for tone, words in _TONE_WORDS]


def tone_for(value: str) -> str:
    v = value.strip().lower()
    if v in _TONE_OF_WORD:
        return _TONE_OF_WORD[v]
    for tone, words in _TONE_SUBSTRINGS:
        if any(w in v for w in words):
            return tone
    return "gray"


# ---------------------------------------------------------------- validation


_WEB_IMAGE = re.compile(r"^(https?://|data:image/(png|jpe?g|gif|webp);base64,)", re.I)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)


def is_web_image(src: str) -> bool:
    return bool(_WEB_IMAGE.match((src or "").strip()))


def image_src_ok(src: str) -> bool:
    """A web address, an inline image, or a file beside the doc.

    Never a path elsewhere on the machine: the Word and PDF exporters read
    whatever an image points at, so an absolute, home or parent path would
    copy a local file into an export the person may then share. Any other
    scheme (file:, javascript:, a drive letter) is refused for the same reason.
    """
    s = (src or "").strip()
    if is_web_image(s):
        return True
    if not s or _SCHEME.match(s) or s.startswith(("/", "\\", "~")):
        return False
    return ".." not in re.split(r"[\\/]", s)


def require_image_src(src: str) -> None:
    """Refuse a write that points an image somewhere it may not (see image_src_ok).

    Checked on writes, not on load: a doc saved before this rule still opens,
    and rendering and exports leave its image out.
    """
    if not image_src_ok(src):
        raise DocError("bad_block", "an image's src must be a web address (https://…), a data: image, "
                       "or a file beside the doc; a path elsewhere on this computer can't be used.")


def clean_text(value: Any, field: str = "text") -> str:
    """``value`` as a block's text: a string within MAX_TEXT, or a ``too_long`` refusal."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if len(value) > MAX_TEXT:
        raise DocError("too_long", f"{field} is {len(value)} characters; the limit is {MAX_TEXT}.")
    return value


def _indent(value: Any) -> int:
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        n = 0
    return max(0, min(MAX_INDENT, n))


def normalize_column(col: dict[str, Any]) -> dict[str, Any]:
    kind = col.get("kind") or "text"
    if kind not in COLUMN_KINDS:
        raise DocError("bad_column", f"column kind {kind!r} is not one of {', '.join(COLUMN_KINDS)}.")
    out: dict[str, Any] = {
        "id": col.get("id") or new_id("c_"),
        "name": clean_text(col.get("name"), "column name"),
        "kind": kind,
    }
    if kind == "status":
        options = []
        seen = set()
        for opt in col.get("options") or []:
            if isinstance(opt, str):
                opt = {"name": opt}
            name = clean_text(opt.get("name"), "option").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            tone = opt.get("tone") or tone_for(name)
            options.append({"name": name, "tone": tone if tone in STATUS_TONES else "gray"})
        out["options"] = options
    if col.get("width"):
        try:
            out["width"] = max(60, min(800, int(col["width"])))
        except (TypeError, ValueError):
            pass
    return out


def normalize_block(raw: dict[str, Any], rev: int = 0) -> Block:
    """Validate one block and fill defaults. Raises DocError on a bad shape."""
    if not isinstance(raw, dict):
        raise DocError("bad_block", "a block must be an object.")
    btype = raw.get("type")
    if btype not in BLOCK_TYPES:
        raise DocError("bad_block", f"block type {btype!r} is not one of {', '.join(BLOCK_TYPES)}.")
    b: Block = {"id": raw.get("id") or new_id("b_"), "type": btype, "v": raw.get("v", rev)}
    if raw.get("by") in ("agent", "user"):
        b["by"] = raw["by"]
    if btype in TEXT_TYPES:
        b["text"] = clean_text(raw.get("text"))
    fields = _TYPE_FIELDS.get(btype)
    if fields is not None:
        b.update(fields(raw))
    return b


def _heading_fields(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        level = int(raw.get("level") or 2)
    except (TypeError, ValueError):
        level = 2
    return {"level": max(1, min(3, level))}


def _list_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {"indent": _indent(raw.get("indent"))}


def _task_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {**_list_fields(raw), "done": bool(raw.get("done"))}


def _callout_fields(raw: dict[str, Any]) -> dict[str, Any]:
    tone = raw.get("tone") or "info"
    out: dict[str, Any] = {"tone": tone if tone in CALLOUT_TONES else "info"}
    if raw.get("title"):
        out["title"] = clean_text(raw.get("title"), "callout title")
    return out


def _code_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {"language": clean_text(raw.get("language") or "", "language")[:40],
            "code": clean_text(raw.get("code"), "code")}


def _image_fields(raw: dict[str, Any]) -> dict[str, Any]:
    src = clean_text(raw.get("src"), "src").strip()
    if not src:
        raise DocError("bad_block", "an image block needs a src.")
    out = {"src": src, "alt": clean_text(raw.get("alt"), "alt")}
    if raw.get("caption"):
        out["caption"] = clean_text(raw.get("caption"), "caption")
    return out


def _pending_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {"intent": clean_text(raw.get("intent") or "Writing…", "intent")}


def _table_fields(raw: dict[str, Any]) -> dict[str, Any]:
    cols = [normalize_column(c) for c in raw.get("columns") or []]
    if not cols:
        raise DocError("bad_table", "a table needs at least one column.")
    col_ids = [c["id"] for c in cols]
    rows = []
    for row in raw.get("rows") or []:
        cells_in = row.get("cells") if isinstance(row, dict) else row
        cells: dict[str, str] = {}
        if isinstance(cells_in, list):
            for i, cid in enumerate(col_ids):
                cells[cid] = clean_text(cells_in[i] if i < len(cells_in) else "", "cell")
        elif isinstance(cells_in, dict):
            for cid in col_ids:
                cells[cid] = clean_text(cells_in.get(cid, ""), "cell")
        rows.append({"id": (row.get("id") if isinstance(row, dict) else None) or new_id("r_"),
                     "cells": cells})
    table = {"columns": cols, "rows": rows}
    check_table_limits(table)
    sync_column_values(table)
    return table


def _metrics_fields(raw: dict[str, Any]) -> dict[str, Any]:
    items = []
    for it in raw.get("items") or []:
        if not isinstance(it, dict):
            continue
        item = {"label": clean_text(it.get("label"), "label"), "value": clean_text(it.get("value"), "value")}
        for k in ("delta", "note"):
            if it.get(k) not in (None, ""):
                item[k] = clean_text(it.get(k), k)
        tone = it.get("tone")
        if tone in ("good", "bad", "neutral"):
            item["tone"] = tone
        items.append(item)
    if not items:
        raise DocError("bad_block", "a metrics block needs at least one item with a label and value.")
    if len(items) > MAX_METRICS:
        raise DocError("bad_block", f"a metrics block holds at most {MAX_METRICS} tiles; this one has {len(items)}. "
                       "Put the rest in a table, or in a second metrics block.")
    return {"items": items}


def check_table_limits(table: dict[str, Any]) -> None:
    """A table past the limits would save, then refuse to load: every later write fails."""
    if len(table["columns"]) > MAX_TABLE_COLS:
        raise DocError("bad_table", f"a table holds at most {MAX_TABLE_COLS} columns.")
    if len(table["rows"]) > MAX_TABLE_ROWS:
        raise DocError("bad_table", f"a table holds at most {MAX_TABLE_ROWS} rows.")


def normalize_chart(raw: dict[str, Any]) -> dict[str, Any]:
    kind = raw.get("chart") or raw.get("kind") or "bar"
    if kind not in CHART_KINDS:
        raise DocError("bad_chart", f"chart kind {kind!r} is not one of {', '.join(CHART_KINDS)}.")
    out: dict[str, Any] = {"chart": kind, "title": clean_text(raw.get("title"), "chart title")}
    for k in ("unit", "caption", "y_label", "x_label"):
        if raw.get(k):
            out[k] = clean_text(raw.get(k), k)
    source = raw.get("source")
    if source:
        if not isinstance(source, dict) or not source.get("table"):
            raise DocError("bad_chart", "a chart source is {\"table\": <table block id>, \"x\": <column>, \"y\": [<columns>]}.")
        y = source.get("y") or []
        if isinstance(y, str):
            y = [y]
        out["source"] = {"table": str(source["table"]), "x": clean_text(source.get("x"), "x"),
                         "y": [clean_text(v, "y") for v in y]}
        return out
    x = raw.get("x")
    series = raw.get("series")
    if not isinstance(x, list) or not isinstance(series, list) or not series:
        raise DocError(
            "bad_chart",
            "a chart needs either source {table, x, y} or data: \"x\": [labels], "
            "\"series\": [{\"name\", \"values\": [numbers]}].",
        )
    if len(series) > MAX_CHART_SERIES:
        raise DocError("bad_chart", f"a chart draws at most {MAX_CHART_SERIES} series; this one has {len(series)}.")
    labels = [clean_text(v, "x label") for v in x]
    clean_series = []
    for s in series:
        if not isinstance(s, dict):
            raise DocError("bad_chart", "each series is {\"name\": str, \"values\": [numbers]}.")
        values = s.get("values") or []
        if len(values) != len(labels):
            raise DocError(
                "bad_chart",
                f"series {s.get('name')!r} has {len(values)} values for {len(labels)} x labels.",
            )
        nums = []
        for v in values:
            nums.append(None if v is None or v == "" else to_number(v))
        clean_series.append({"name": clean_text(s.get("name") or "Series", "series name"), "values": nums})
    out["x"] = labels
    out["series"] = clean_series
    return out


_TYPE_FIELDS = {
    "heading": _heading_fields,
    "bullet": _list_fields,
    "numbered": _list_fields,
    "task": _task_fields,
    "callout": _callout_fields,
    "code": _code_fields,
    "image": _image_fields,
    "pending": _pending_fields,
    "table": _table_fields,
    "chart": normalize_chart,
    "metrics": _metrics_fields,
}


# A number as it is written in a cell, and nothing else: an optional sign, an
# optional currency, digits (plain, or grouped by commas in threes) with an
# optional decimal part, then an optional % or k/K/M/B. "500 MB", "1e6",
# "12-15" and "5pm" are text. The app's docModel.toNumber reads by the same rule.
_NUM_RE = re.compile(r"([+\-−]?)[$€£]?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|[kKMB])?")
_EXPONENT = {"k": 3, "K": 3, "M": 6, "B": 9}
# A cell holds the inline Markdown subset: "**1,200**" is a number made bold.
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(.+)\1")


def is_number(value: str) -> bool:
    return to_number(value) is not None


def to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            n = float(value)
        except OverflowError:
            return None
    else:
        text = str(value or "").strip()
        emphasis = _EMPHASIS_RE.fullmatch(text)
        match = _NUM_RE.fullmatch(emphasis.group(2).strip() if emphasis else text)
        if match is None:
            return None
        sign, digits, decimals, suffix = match.groups()
        # Scaled as written, in decimal: "$1.1M" is 1100000, not 1100000.0000000002.
        n = float(f"{digits.replace(',', '')}{decimals or ''}e{_EXPONENT.get(suffix or '', 0)}")
        if sign in ("-", "−"):
            n = -n
    return n if math.isfinite(n) else None


_MONTH_NUMBERS = {name: n for n, names in enumerate(
    [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
     ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
     ("december", "dec")], start=1) for name in names}
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# "November 4, 2026", "Nov. 4 2026", "Wednesday, November 4th, 2026"; "4 November 2026", "4 Nov, 2026".
# Numeric forms other than ISO ("11/04/2026") are left alone: which part is the month is a guess.
_MONTH_DAY_YEAR_RE = re.compile(r"(?:[a-z]+,?\s+)?([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")
_DAY_MONTH_YEAR_RE = re.compile(r"(?:[a-z]+,?\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?,?\s+(\d{4})")


def to_iso_date(value: Any) -> str | None:
    """A cell's date as YYYY-MM-DD, or None when it isn't written as a whole date.

    The same rule as the editor's toIsoDate: a date column stores ISO, so it
    sorts by date and the editor's date picker can read it.
    """
    text = str(value or "").strip()
    emphasis = _EMPHASIS_RE.fullmatch(text)
    text = (emphasis.group(2).strip() if emphasis else text).lower()
    if m := _ISO_DATE_RE.fullmatch(text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    elif (m := _MONTH_DAY_YEAR_RE.fullmatch(text)) and m.group(1) in _MONTH_NUMBERS:
        year, month, day = int(m.group(3)), _MONTH_NUMBERS[m.group(1)], int(m.group(2))
    elif (m := _DAY_MONTH_YEAR_RE.fullmatch(text)) and m.group(2) in _MONTH_NUMBERS:
        year, month, day = int(m.group(3)), _MONTH_NUMBERS[m.group(2)], int(m.group(1))
    else:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def sync_column_values(table: dict[str, Any]) -> None:
    """Keep cells in step with their column: status choices, and dates in ISO."""
    sync_status_options(table)
    for col in table["columns"]:
        if col["kind"] != "date":
            continue
        for row in table["rows"]:
            iso = to_iso_date(row["cells"].get(col["id"], ""))
            if iso:
                row["cells"][col["id"]] = iso


def sync_status_options(table: dict[str, Any]) -> None:
    """Every value typed into a status column becomes one of its options."""
    for col in table["columns"]:
        if col["kind"] != "status":
            continue
        known = {o["name"].lower() for o in col["options"]}
        for row in table["rows"]:
            v = row["cells"].get(col["id"], "").strip()
            if v and v.lower() not in known:
                known.add(v.lower())
                col["options"].append({"name": v, "tone": tone_for(v)})


# ---------------------------------------------------------------- doc


def new_doc(title: str, doc_id: str | None = None) -> dict[str, Any]:
    title = clean_text(title, "title").strip() or "Untitled doc"
    return {"schema": SCHEMA, "id": doc_id or new_id("d_"), "title": title, "rev": 0, "tabs": []}


def new_tab(title: str, blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"id": new_id("t_"), "title": clean_text(title, "tab title").strip() or "Untitled", "blocks": blocks or []}


def normalize_doc(raw: dict[str, Any]) -> dict[str, Any]:
    """Load-time validation for a doc read off disk or sent whole by a client."""
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise DocError("bad_doc", f"not a {SCHEMA} document.")
    try:
        rev = int(raw.get("rev") or 0)
    except (TypeError, ValueError):
        raise DocError("bad_doc", "the doc's rev is not a number.") from None
    doc = {
        "schema": SCHEMA,
        "id": str(raw.get("id") or new_id("d_")),
        "title": clean_text(raw.get("title"), "title") or "Untitled doc",
        "rev": rev,
        "tabs": [],
    }
    for k in ("created_at", "updated_at"):
        if raw.get(k):
            doc[k] = raw[k]
    for tab in raw.get("tabs") or []:
        if not isinstance(tab, dict):
            raise DocError("bad_doc", "each tab is an object with a title and blocks.")
        blocks = [normalize_block(b, doc["rev"]) for b in tab.get("blocks") or []]
        doc["tabs"].append({"id": tab.get("id") or new_id("t_"),
                            "title": clean_text(tab.get("title"), "tab title") or "Untitled",
                            "blocks": blocks})
    if not doc["tabs"]:
        doc["tabs"].append(new_tab(doc["title"]))
    return doc


def clone(doc: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(doc)


def find_tab(doc: dict[str, Any], tab_ref: str | None) -> dict[str, Any]:
    tabs = doc["tabs"]
    if not tabs:
        raise DocError("no_tab", "this doc has no tabs; add one with {\"op\": \"add_tab\"}.")
    if not tab_ref:
        return tabs[0]
    for t in tabs:
        if t["id"] == tab_ref:
            return t
    lowered = tab_ref.strip().lower()
    for t in tabs:
        if t["title"].strip().lower() == lowered:
            return t
    raise DocError(
        "no_tab",
        f"no tab {tab_ref!r} in this doc.",
        tabs=[{"id": t["id"], "title": t["title"]} for t in tabs],
    )


def _one_slip(given: str, real: str) -> bool:
    """``given`` is ``real`` with one character dropped, added or changed."""
    if given[:2] != real[:2] or abs(len(given) - len(real)) > 1:
        return False
    if len(given) == len(real):
        return sum(a != b for a, b in zip(given, real)) == 1
    short, long_ = sorted((given, real), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def block_index(tab: dict[str, Any], block_id: str) -> int:
    for i, b in enumerate(tab["blocks"]):
        if b["id"] == block_id:
            return i
    # A model copying an id now and then slips a character (b_7kq16vg for
    # b_7kq16vg6) and repeats the slip on every retry, so the edit never lands.
    # Ids are 8 random characters: one id a single slip away is the block meant;
    # more than one is ambiguous and refused like any unknown id.
    if len(block_id) >= 8:
        near = [i for i, b in enumerate(tab["blocks"]) if _one_slip(block_id, b["id"])]
        if len(near) == 1:
            return near[0]
    raise DocError("no_block", f"no block {block_id!r} in tab {tab['title']!r}; read the tab for current ids.")


def text_fields(b: Block) -> list[tuple[dict[str, Any], str]]:
    """Where a block keeps words, in reading order, as (holder, key) pairs.

    One list for finding a block by its words and for changing them in place
    (``edit_text``), so what a search finds is always what an edit can reach.
    """
    t = b["type"]
    if t == "callout":
        return [(b, k) for k in ("title", "text") if k in b]
    if t in TEXT_TYPES:
        return [(b, "text")]
    if t == "code":
        return [(b, "code")]
    if t == "table":
        fields = [(c, "name") for c in b["columns"]]
        for r in b["rows"]:
            fields.extend((r["cells"], cid) for cid in r["cells"])
        return fields
    if t == "chart":
        return [(b, k) for k in ("title", "caption") if k in b]
    if t == "image":
        return [(b, k) for k in ("alt", "caption") if k in b]
    if t == "metrics":
        return [(item, k) for item in b["items"] for k in ("label", "value", "delta", "note") if k in item]
    if t == "pending":
        return [(b, "intent")]
    return []


def block_text_for_search(b: Block) -> str:
    # Joined by newlines so a search can't match across two fields.
    return "\n".join(str(holder.get(key) or "") for holder, key in text_fields(b))


def name_key(name: Any) -> str:
    """A column name as it is matched: case and spacing don't matter."""
    return " ".join(str(name or "").split()).lower()


def chart_source_table(tab: dict[str, Any], chart: dict[str, Any]) -> dict[str, Any] | None:
    """The table a chart draws from: its recorded id, else the table holding its columns.

    A chart keeps its table's id, but the id can go missing: a section rewritten
    whole mints a new table, a table is pasted again, an id is copied in from
    another doc. Rather than show a broken chart, fall back to the nearest table
    above it (then anywhere in the tab) that has the chart's x and y columns.
    """
    src = chart.get("source") or {}
    blocks = tab.get("blocks", [])
    for b in blocks:
        if b.get("id") == src.get("table") and b.get("type") == "table":
            return b
    wanted = [name_key(src.get("x"))] + [name_key(y) for y in src.get("y") or []]
    wanted = [w for w in wanted if w]
    if not wanted:
        return None
    try:
        at = next(i for i, b in enumerate(blocks) if b.get("id") == chart.get("id"))
    except StopIteration:
        at = len(blocks)
    above = [b for b in reversed(blocks[:at]) if b.get("type") == "table"]
    seen = {b["id"] for b in above}
    for b in above + [t for t in blocks if t.get("type") == "table" and t["id"] not in seen]:
        names = {name_key(c.get("name")) for c in b.get("columns", [])}
        if all(w in names for w in wanted):
            return b
    return None
