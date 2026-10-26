"""Edit operations on a doc, shared by the agent's tools and the editor UI.

``apply_ops`` is atomic: every op lands or none does, and a refusal names the
op (by index) and how to fix it. Each op is a dict with an ``op`` key:

Blocks (in the tab named by the call, or the op's own ``tab``):
  insert    {position, markdown | blocks}
  replace   {target, markdown | blocks}           target = id | [ids] | {"find": text, "nth"?}
  update    {id, set: {text?, level?, done?, tone?, type?, indent?, code?, language?, …}}
  edit_text {id?, find, replace, nth?}            in-place words change, keeps the block
  delete    {target}
  move      {ids, position}

Tables (``table`` = the table block's id):
  update_cell    {table, row, col, value}          row = index (0 = first body row) or row id;
                                                   col = name, index or column id
  add_rows       {table, rows: [[…]] | [{col: value}], index?}
  delete_rows    {table, rows: [index | row id, …]}
  add_column     {table, name, kind?, options?, values?, index?}
  update_column  {table, col, name?, kind?, options?, width?}
  delete_column  {table, col}
  sort_rows      {table, col, descending?}

Tabs and doc:
  add_tab    {title, markdown?, index?}  → the new tab's id is in the result
  rename_tab {tab, title}
  delete_tab {tab}
  move_tab   {tab, index}
  rename_doc {title}

Positions: "start" | "end" | {"after": id} | {"before": id} | {"end_of_section": heading id}.
``index`` counts from 1; left out (or 0) it means the end.
"""

from __future__ import annotations

import re
from typing import Any

from . import markdown as mdmod
from .model import (
    COLUMN_KINDS,
    MAX_BLOCKS_PER_TAB,
    MAX_TABS,
    TEXT_TYPES,
    Block,
    Doc,
    DocError,
    Tab,
    block_index,
    block_text_for_search,
    chart_source_table,
    check_table_limits,
    clean_text,
    clone,
    find_tab,
    name_key,
    new_doc,
    new_id,
    new_tab,
    normalize_block,
    normalize_column,
    require_image_src,
    sync_column_values,
    text_fields,
    to_iso_date,
    to_number,
)

MAX_OPS = 200

_CLIENT_ID = re.compile(r"^[bcr]_[a-z0-9]{8}$")
_TAB_ID = re.compile(r"^t_[a-z0-9]{8}$")

# Props a client may set on a block via `update`, per type. `type` changes are
# allowed only between text-bearing types (a paragraph becomes a heading, …).
_SETTABLE: dict[str, tuple[str, ...]] = {
    "paragraph": ("text",),
    "heading": ("text", "level"),
    "bullet": ("text", "indent"),
    "numbered": ("text", "indent"),
    "task": ("text", "indent", "done"),
    "quote": ("text",),
    "callout": ("text", "tone", "title"),
    "code": ("code", "language"),
    "image": ("src", "alt", "caption"),
    "chart": ("chart", "title", "unit", "caption", "source", "x", "series", "x_label", "y_label"),
    "metrics": ("items",),
    "table": ("columns", "rows"),
    "pending": ("intent",),
    "divider": (),
}


class _Ctx:
    def __init__(self, doc: dict[str, Any], default_tab: str | None, rev: int, by: str,
                 if_rev: int | None) -> None:
        self.doc = doc
        self.default_tab = default_tab
        self.rev = rev
        self.by = by
        # The person's editor sends what it means ("" empties a block); only
        # the agent's ops come with blanks filled in that must be sifted out.
        self.literal = by == "user"
        self.if_rev = if_rev
        self.touched: dict[str, set[str]] = {}  # tab id -> block ids written
        self.removed: dict[str, set[str]] = {}

    def tab(self, op: dict[str, Any]) -> dict[str, Any]:
        return find_tab(self.doc, op.get("tab") or self.default_tab)

    def guard(self, block: dict[str, Any]) -> None:
        v = block.get("v", 0)
        # `v == self.rev` is this batch's own write (an earlier op in the same
        # call): never a conflict with the caller's read.
        if self.if_rev is not None and v > self.if_rev and v != self.rev:
            who = {"user": "the person", "agent": "another agent call"}.get(block.get("by", ""), "someone")
            raise DocError(
                "stale",
                f"block {block['id']} was changed by {who} at rev {v}, after the rev {self.if_rev} you read. "
                "Read it again (read_doc with ids), keep their change, and resend against the rev that read returns.",
                ids=[block["id"]],
                rev=v,
            )

    def mark(self, tab: dict[str, Any], b: dict[str, Any]) -> None:
        b["v"] = self.rev
        b["by"] = self.by
        self.touched.setdefault(tab["id"], set()).add(b["id"])


def _blocks_from(op: dict[str, Any], ctx: _Ctx, field_hint: str) -> list[dict[str, Any]]:
    if "markdown" in op and op["markdown"] is not None:
        blocks = mdmod.parse_markdown(str(op["markdown"]), ctx.rev, ctx.by)
    elif isinstance(op.get("blocks"), list):
        # The editor mints ids itself so it can move the caret before the
        # server answers; keep one that is well formed and not already taken.
        taken = {b["id"] for t in ctx.doc["tabs"] for b in t["blocks"]}
        blocks = []
        for raw in op["blocks"]:
            if not isinstance(raw, dict):
                raise DocError("bad_op", f"{field_hint}: each block is an object with a \"type\".")
            bid = raw.get("id")
            keep = isinstance(bid, str) and _CLIENT_ID.match(bid) and bid not in taken
            b = normalize_block({**raw, "id": bid if keep else None, "by": ctx.by}, ctx.rev)
            if b["type"] == "image":
                require_image_src(b["src"])
            taken.add(b["id"])
            blocks.append(b)
    else:
        raise DocError("bad_op", f"{field_hint} needs \"markdown\" (a string) or \"blocks\" (a list).")
    return blocks


def _block_position(tab: Tab, pos: Any) -> int:
    """Where blocks land for a ``position`` (see the module docstring): an index into the tab."""
    blocks = tab["blocks"]
    if pos in (None, "end"):
        return len(blocks)
    if pos == "start":
        return 0
    if isinstance(pos, dict):
        if "after" in pos:
            return block_index(tab, str(pos["after"])) + 1
        if "before" in pos:
            return block_index(tab, str(pos["before"]))
        if "end_of_section" in pos:
            hi = block_index(tab, str(pos["end_of_section"]))
            head = blocks[hi]
            if head["type"] != "heading":
                raise DocError("bad_position", f"end_of_section takes a heading id; {head['id']} is a {head['type']}.")
            j = hi + 1
            while j < len(blocks) and not (blocks[j]["type"] == "heading" and blocks[j]["level"] <= head["level"]):
                j += 1
            return j
    raise DocError("bad_position",
                   "position is \"start\", \"end\", {\"after\": id}, {\"before\": id} or {\"end_of_section\": heading id}.")


def _nth(value: Any) -> int:
    """Which occurrence an op means: a whole number from 1 (left out: the first)."""
    try:
        n = int(value or 1)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        raise DocError("bad_op", "\"nth\" is which occurrence, counting from 1.")
    return n


def _resolve_targets(tab: Tab, target: Any) -> list[int]:
    if isinstance(target, str):
        return [block_index(tab, target)]
    if isinstance(target, list) and target:
        return sorted({block_index(tab, str(t)) for t in target})
    if isinstance(target, dict) and target.get("find"):
        needle = str(target["find"])
        nth = _nth(target.get("nth"))
        hits = [i for i, b in enumerate(tab["blocks"]) if needle in block_text_for_search(b)]
        if len(hits) < nth:
            raise DocError(
                "find_none",
                f"no block in tab {tab['title']!r} contains {needle!r}"
                + (f" {nth} times" if nth > 1 else "")
                + "; quote words exactly as they stand in your last read.",
            )
        return [hits[nth - 1]]
    raise DocError("bad_target", "target is a block id, a list of ids, or {\"find\": \"exact words\"}.")


def _check_size(tab: dict[str, Any]) -> None:
    if len(tab["blocks"]) > MAX_BLOCKS_PER_TAB:
        raise DocError("too_big", f"a tab holds at most {MAX_BLOCKS_PER_TAB} blocks; split it into tabs.")


# ---------------------------------------------------------------- block ops


def _op_insert(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    at = _block_position(tab, op.get("position"))
    blocks = _blocks_from(op, ctx, "insert")
    resolve_chart_sources(tab, blocks, at)
    tab["blocks"][at:at] = blocks
    for b in blocks:
        ctx.mark(tab, b)
    _check_size(tab)
    return {"tab": tab["id"], "ids": [b["id"] for b in blocks]}


def _op_replace(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    idx = _resolve_targets(tab, op.get("target"))
    for i in idx:
        if tab["blocks"][i]["type"] != "pending":
            ctx.guard(tab["blocks"][i])
    blocks = _blocks_from(op, ctx, "replace")
    first = idx[0]
    old = [tab["blocks"][i] for i in idx]
    # One block replaced by one block of a compatible kind keeps its id, so
    # the person's cursor, scroll anchor and highlights stay put.
    if len(old) == 1 and len(blocks) == 1 and old[0]["type"] != "pending" and (
        old[0]["type"] == blocks[0]["type"]
        or (old[0]["type"] in TEXT_TYPES and blocks[0]["type"] in TEXT_TYPES)
    ):
        blocks[0]["id"] = old[0]["id"]
        # The person's editor sends a table with the ids it means (an undo
        # puts a table back as it was); an agent's rewrite keeps the old ones.
        if old[0]["type"] == "table" and blocks[0]["type"] == "table" and not ctx.literal:
            _carry_table_ids(old[0], blocks[0])
    else:
        # A section rewritten whole (heading, prose, its one table) keeps that
        # table's id, so charts elsewhere that draw from it keep drawing.
        old_tables = [b for b in old if b["type"] == "table"]
        new_tables = [b for b in blocks if b["type"] == "table"]
        if len(old_tables) == 1 and len(new_tables) == 1:
            new_tables[0]["id"] = old_tables[0]["id"]
            _carry_table_ids(old_tables[0], new_tables[0])
    for i in reversed(idx):
        removed = tab["blocks"].pop(i)
        ctx.removed.setdefault(tab["id"], set()).add(removed["id"])
    resolve_chart_sources(tab, blocks, first)
    tab["blocks"][first:first] = blocks
    for b in blocks:
        ctx.removed.get(tab["id"], set()).discard(b["id"])
        ctx.mark(tab, b)
    _check_size(tab)
    return {"tab": tab["id"], "ids": [b["id"] for b in blocks], "replaced": [b["id"] for b in old]}


def _carry_table_ids(old: Block, new: Block) -> None:
    """A table rewritten whole keeps column and row ids where names/positions match."""
    old_by_name = {name_key(c["name"]): c for c in old["columns"]}
    remap = {}
    for c in new["columns"]:
        prev = old_by_name.get(name_key(c["name"]))
        if prev:
            remap[c["id"]] = prev["id"]
            c["id"] = prev["id"]
    for r in new["rows"]:
        r["cells"] = {remap.get(k, k): v for k, v in r["cells"].items()}
    for i, r in enumerate(new["rows"]):
        if i < len(old["rows"]):
            r["id"] = old["rows"][i]["id"]


_FALSY_VALUES = frozenset({"done", "indent"})


def _blank(v: Any) -> bool:
    """A value that says nothing: None, "", 0, false, [], or an object of those."""
    if v is None or v is False:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v == 0
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, tuple)):
        return not v
    if isinstance(v, dict):
        return all(_blank(x) for x in v.values())
    return False


# A blank that still means something to its op: "" replaces the found words
# with nothing, "" empties a cell, 0 is the first row or column.
_BLANK_MEANS_SOMETHING = {
    "edit_text": {"replace"},
    "update_cell": {"value", "row", "col"},
    "update_column": {"col"},
    "delete_column": {"col"},
    "sort_rows": {"col", "descending"},
    "update": {"set"},  # sifted field by field in _op_update ("done": false unticks)
}


def _without_filled_blanks(op: dict[str, Any]) -> dict[str, Any]:
    """The op without the empty fields a model filled in for it.

    Some models send every field the tool declares on every op ("name": "",
    "options": [], "index": 0). Taken at their word those rename a column to
    nothing, wipe a status column's choices, or put new rows first.
    """
    keep = _BLANK_MEANS_SOMETHING.get(op.get("op"), set())
    return {k: v for k, v in op.items() if k == "op" or k in keep or not _blank(v)}


def _index_from_one(op: dict[str, Any], length: int) -> int:
    """Where an add or move lands in a list of ``length``: "index" counts from 1; none (or 0) is the end."""
    at = op.get("index")
    try:
        at = int(at) if at is not None else 0
    except (TypeError, ValueError):
        raise DocError("bad_op", "\"index\" is a position counting from 1 (1 = first).") from None
    return length if at <= 0 else min(length, at - 1)


def _op_update(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    i = block_index(tab, str(op.get("id")))
    b = tab["blocks"][i]
    ctx.guard(b)
    raw = op.get("set")
    if not isinstance(raw, dict) or not raw:
        raise DocError("bad_op", "update needs \"set\": {field: value, …}.")
    if ctx.literal:
        new_type = raw.get("type") or b["type"]
    else:
        new_type = raw.get("type") if not _blank(raw.get("type")) else b["type"]
    if new_type != b["type"]:
        if b["type"] not in TEXT_TYPES or new_type not in TEXT_TYPES:
            raise DocError("bad_op", f"a {b['type']} block cannot become a {new_type}; replace it instead.")
    allowed = set(_SETTABLE.get(new_type, ())) | {"type"}
    # Some models fill in every field the tool declares, whatever the block
    # ("tone": "", "chart": "bar", "items": []) and cannot stop: refusing
    # those sent the agent round the same update until the turn ran out. So
    # what this block does not take is left out, and from the agent an empty
    # value never overwrites one (to empty a block, replace or delete it). A
    # false "done" and a zero "indent" are real values. The person's edits
    # (ctx.literal) take every value as sent: deleting a block's text empties it.
    changes = {k: v for k, v in raw.items()
               if k in allowed and (ctx.literal or not _blank(v) or (k in _FALSY_VALUES and v is not None))}
    if new_type == b["type"]:
        changes.pop("type", None)
    if not changes and new_type == b["type"]:
        raise DocError("bad_op", f"nothing to change: a {new_type} block takes "
                       f"{', '.join(sorted(allowed - {'type'})) or 'no fields'}.")
    merged = {**b, **changes, "type": new_type}
    if new_type == "chart" and ("x" in changes or "series" in changes):
        merged.pop("source", None)
    if new_type == "chart" and "source" in changes:
        merged.pop("x", None)
        merged.pop("series", None)
    nb = normalize_block(merged, ctx.rev)
    if nb["type"] == "image" and "src" in changes:
        require_image_src(nb["src"])
    nb["id"] = b["id"]
    tab["blocks"][i] = nb
    ctx.mark(tab, nb)
    return {"tab": tab["id"], "ids": [nb["id"]]}


def _op_edit_text(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    find = op.get("find")
    if not find:
        raise DocError("bad_op", "edit_text needs \"find\" (exact words) and \"replace\".")
    replace = clean_text(op.get("replace", ""), "replace")
    nth = _nth(op.get("nth"))
    if op.get("id"):
        candidates = [block_index(tab, str(op["id"]))]
    else:
        candidates = [i for i, b in enumerate(tab["blocks"]) if find in block_text_for_search(b)]
    count = 0
    for i in candidates:
        b = tab["blocks"][i]
        for holder, key in text_fields(b):
            text = holder.get(key, "")
            start = 0
            while True:
                pos = text.find(find, start)
                if pos < 0:
                    break
                count += 1
                if count == nth:
                    ctx.guard(b)
                    holder[key] = clean_text(text[:pos] + replace + text[pos + len(find):], "the edited text")
                    if b["type"] == "table":
                        sync_column_values(b)
                    ctx.mark(tab, b)
                    return {"tab": tab["id"], "ids": [b["id"]]}
                start = pos + len(find)
    raise DocError(
        "find_none",
        f"{find!r} was found {count} time(s)" + (f", not {nth}" if count else "")
        + f" in tab {tab['title']!r}. edit_text searches text blocks, callout titles, table headers and cells, "
        "chart titles and captions, image alt text and captions, and metric tiles; quote the words exactly as "
        "your last read shows them. To change a chart's source table, columns or kind, use update with set "
        "{source} or {chart}.",
    )


def _op_delete(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    idx = _resolve_targets(tab, op.get("target") or op.get("ids") or op.get("id"))
    ids = []
    for i in reversed(idx):
        ctx.guard(tab["blocks"][i])
        removed = tab["blocks"].pop(i)
        ids.append(removed["id"])
        ctx.removed.setdefault(tab["id"], set()).add(removed["id"])
    return {"tab": tab["id"], "deleted": list(reversed(ids))}


def _op_move(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = ctx.tab(op)
    ids = op.get("ids") or ([op["id"]] if op.get("id") else [])
    if not ids:
        raise DocError("bad_op", "move needs \"ids\": [block ids] and a \"position\".")
    idx = _resolve_targets(tab, [str(x) for x in ids])
    moving = [tab["blocks"][i] for i in idx]
    pos = op.get("position")
    if isinstance(pos, dict):
        anchor = pos.get("after") or pos.get("before") or pos.get("end_of_section")
        if anchor in ids:
            raise DocError("bad_position", "a block cannot move relative to itself.")
    for i in reversed(idx):
        tab["blocks"].pop(i)
    at = _block_position(tab, pos)
    tab["blocks"][at:at] = moving
    for b in moving:
        ctx.mark(tab, b)
    return {"tab": tab["id"], "ids": [b["id"] for b in moving]}


def resolve_chart_sources(tab: Tab, new_blocks: list[Block], at: int) -> None:
    """`"table": "previous"` in a chart source → the nearest table above it."""
    for j, b in enumerate(new_blocks):
        if b["type"] != "chart" or not b.get("source"):
            continue
        ref = b["source"]["table"]
        if ref not in ("previous", "prev", "above"):
            continue
        found = None
        for k in range(j - 1, -1, -1):
            if new_blocks[k]["type"] == "table":
                found = new_blocks[k]["id"]
                break
        if not found:
            for k in range(at - 1, -1, -1):
                if tab["blocks"][k]["type"] == "table":
                    found = tab["blocks"][k]["id"]
                    break
        if not found:
            raise DocError("bad_chart", "chart source \"previous\" needs a table above the chart.")
        b["source"]["table"] = found


def link_inline_charts(tab: Tab, chart_ids: set[str] | None = None) -> list[Block]:
    """Point a chart that carries its own numbers at the table holding the same numbers.

    The prompt asks for charts drawn from their table, so the numbers live in
    one place and the person's edits move the chart; models still copy them
    in. A chart links only when each of its series is, row for row, a number
    column of one table in the tab (the nearest above first); its labels come
    from the column whose cells read as them, else the first other column.
    Returns the charts it linked.
    """
    def same(a: float | None, b: Any) -> bool:
        return a is not None and isinstance(b, (int, float)) and abs(a - b) <= 1e-9 * max(1.0, abs(b))

    linked = []
    blocks = tab["blocks"]
    for i, b in enumerate(blocks):
        if b["type"] != "chart" or b.get("source") or (chart_ids is not None and b["id"] not in chart_ids):
            continue
        labels, series = b.get("x") or [], b.get("series") or []
        if len(labels) < 2 or not series:
            continue
        tables = [t for t in reversed(blocks[:i]) if t["type"] == "table"] + \
                 [t for t in blocks[i + 1:] if t["type"] == "table"]
        for t in tables:
            if len(t["rows"]) != len(labels):
                continue
            numbers = {c["id"]: [to_number(r["cells"].get(c["id"], "")) for r in t["rows"]] for c in t["columns"]}
            ys = []
            for sr in series:
                col = next((c for c in t["columns"] if c not in ys
                            and all(same(n, v) for n, v in zip(numbers[c["id"]], sr.get("values") or []))
                            and len(sr.get("values") or []) == len(labels)), None)
                if col is None:
                    break
                ys.append(col)
            if len(ys) != len(series):
                continue
            want = [str(x).strip().lower() for x in labels]
            others = [c for c in t["columns"] if c not in ys]
            x_col = next((c for c in others if [r["cells"].get(c["id"], "").strip().lower() for r in t["rows"]] == want),
                         others[0] if others else None)
            if x_col is None:
                continue
            b["source"] = {"table": t["id"], "x": x_col["name"], "y": [c["name"] for c in ys]}
            b.pop("x", None)
            b.pop("series", None)
            linked.append(b)
            break
    return linked


def _rename_in_charts(tab: dict[str, Any], table_id: str, before: str, after: str, ctx: _Ctx) -> None:
    """A renamed column stays plotted: charts drawing from it follow the new name."""
    low = name_key(before)
    for b in tab["blocks"]:
        src = b.get("source") if b["type"] == "chart" else None
        if not src or src.get("table") != table_id:
            continue
        changed = False
        if name_key(src.get("x")) == low:
            src["x"] = after
            changed = True
        ys = [after if name_key(y) == low else y for y in src.get("y", [])]
        if ys != src.get("y", []):
            src["y"] = ys
            changed = True
        if changed:
            ctx.mark(tab, b)


def _repair_chart_sources(ctx: _Ctx) -> None:
    """After a call, re-link any chart whose table id no longer exists (see chart_source_table)."""
    for tab in ctx.doc["tabs"]:
        ids = {b["id"] for b in tab["blocks"] if b["type"] == "table"}
        for b in tab["blocks"]:
            src = b.get("source") if b["type"] == "chart" else None
            if not src or src.get("table") in ids:
                continue
            found = chart_source_table(tab, b)
            if found is not None:
                src["table"] = found["id"]
                ctx.mark(tab, b)


# ---------------------------------------------------------------- table ops


def _table(ctx: _Ctx, op: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tab = ctx.tab(op)
    tid = op.get("table") or op.get("id")
    if not tid:
        raise DocError("bad_op", f"{op['op']} needs \"table\": the table block's id.")
    b = tab["blocks"][block_index(tab, str(tid))]
    if b["type"] != "table":
        raise DocError("bad_op", f"block {b['id']} is a {b['type']}, not a table.")
    ctx.guard(b)
    return tab, b


def _col(table: dict[str, Any], ref: Any) -> dict[str, Any]:
    cols = table["columns"]
    if isinstance(ref, int) or (isinstance(ref, str) and ref.lstrip("-").isdigit()):
        n = int(ref)
        if -len(cols) <= n < len(cols):
            return cols[n]
    else:
        for c in cols:
            if c["id"] == ref:
                return c
        low = name_key(ref)
        for c in cols:
            if name_key(c["name"]) == low:
                return c
    raise DocError("no_column", f"no column {ref!r}; columns are: {', '.join(c['name'] for c in cols)}.",
                   columns=[c["name"] for c in cols])


def _row_index(table: dict[str, Any], ref: Any) -> int:
    rows = table["rows"]
    if isinstance(ref, int) or (isinstance(ref, str) and ref.lstrip("-").isdigit()):
        n = int(ref)
        if -len(rows) <= n < len(rows):
            return n % len(rows)
    else:
        for i, r in enumerate(rows):
            if r["id"] == ref:
                return i
    raise DocError("no_row", f"no row {ref!r}; the table has {len(rows)} body rows (0 = first under the header).",
                   rows=len(rows))


def _op_update_cell(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    c = _col(t, op.get("col"))
    r = t["rows"][_row_index(t, op.get("row"))]
    r["cells"][c["id"]] = clean_text(op.get("value", ""), "cell")
    sync_column_values(t)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]], "row": r["id"], "col": c["id"]}


def _op_add_rows(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    rows_in = op.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        raise DocError("bad_op", "add_rows needs \"rows\": a list of rows, each a list of cell values or {column: value}.")
    new_rows = []
    for row in rows_in:
        cells = {c["id"]: "" for c in t["columns"]}
        row_id = None
        if isinstance(row, dict) and isinstance(row.get("cells"), dict):
            # The editor's shape: {id, cells: {column id or name: value}}.
            rid = row.get("id")
            if isinstance(rid, str) and _CLIENT_ID.match(rid) and all(r["id"] != rid for r in t["rows"]):
                row_id = rid
            row = row["cells"]
        if isinstance(row, list):
            for i, c in enumerate(t["columns"]):
                if i < len(row):
                    cells[c["id"]] = clean_text(row[i], "cell")
        elif isinstance(row, dict):
            for k, v in row.items():
                cells[_col(t, k)["id"]] = clean_text(v, "cell")
        else:
            raise DocError("bad_op", "each row is a list of cell values or an object of column: value.")
        new_rows.append({"id": row_id or new_id("r_"), "cells": cells})
    at = _index_from_one(op, len(t["rows"]))
    t["rows"][at:at] = new_rows
    check_table_limits(t)
    sync_column_values(t)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]], "rows": [r["id"] for r in new_rows]}


def _op_delete_rows(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    refs = op.get("rows")
    if not isinstance(refs, list) or not refs:
        raise DocError("bad_op", "delete_rows needs \"rows\": [row index or row id, …].")
    idx = sorted({_row_index(t, r) for r in refs}, reverse=True)
    for i in idx:
        t["rows"].pop(i)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]]}


def _op_add_column(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    col = normalize_column({"name": op.get("name") or "Column", "kind": op.get("kind") or "text",
                            "options": op.get("options")})
    at = _index_from_one(op, len(t["columns"]))
    t["columns"].insert(at, col)
    values = op.get("values") or []
    for i, r in enumerate(t["rows"]):
        r["cells"][col["id"]] = clean_text(values[i], "cell") if i < len(values) else ""
    check_table_limits(t)
    sync_column_values(t)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]], "col": col["id"]}


def _op_update_column(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    c = _col(t, op.get("col"))
    if "name" in op:
        before = c["name"]
        c["name"] = clean_text(op["name"], "column name")
        _rename_in_charts(tab, t["id"], before, c["name"], ctx)
    if "kind" in op and op["kind"] == "text" and c["kind"] != "text" and any(k in op for k in ("name", "options", "width")):
        # "text" is the first kind, the one a model fills in on a rename; a
        # status column turned to text loses its choices, so only an op that
        # asks for nothing else converts it.
        op = {k: v for k, v in op.items() if k != "kind"}
    if "kind" in op:
        if op["kind"] not in COLUMN_KINDS:
            raise DocError("bad_column", f"column kind is one of {', '.join(COLUMN_KINDS)}.")
        c["kind"] = op["kind"]
        if c["kind"] == "status" and "options" not in c:
            c["options"] = []
        if c["kind"] != "status":
            c.pop("options", None)
    if "options" in op and c["kind"] == "status":
        c["options"] = normalize_column({"name": c["name"], "kind": "status", "options": op["options"]})["options"]
    if "width" in op:
        c.update({k: v for k, v in normalize_column({**c, "width": op["width"]}).items() if k == "width"})
    sync_column_values(t)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]], "col": c["id"]}


def _op_delete_column(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    c = _col(t, op.get("col"))
    if len(t["columns"]) == 1:
        raise DocError("bad_op", "a table keeps at least one column; delete the table block instead.")
    t["columns"] = [x for x in t["columns"] if x["id"] != c["id"]]
    for r in t["rows"]:
        r["cells"].pop(c["id"], None)
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]]}


def _op_sort_rows(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab, t = _table(ctx, op)
    c = _col(t, op.get("col"))
    desc = bool(op.get("descending"))

    def key(r: dict[str, Any]) -> tuple[int, Any]:
        v = r["cells"].get(c["id"], "")
        if c["kind"] == "number":
            n = to_number(v)
            return (0, n) if n is not None else (1, 0)
        if c["kind"] == "status":
            names = [o["name"].lower() for o in c.get("options", [])]
            return (0, names.index(v.lower())) if v.lower() in names else (1, 0)
        if c["kind"] == "date":
            iso = to_iso_date(v)
            return (0, iso) if iso else (1, 0)
        return (0 if v else 1, v.lower())

    filled = [r for r in t["rows"] if key(r)[0] == 0]
    empty = [r for r in t["rows"] if key(r)[0] == 1]
    filled.sort(key=lambda r: key(r)[1], reverse=desc)
    t["rows"] = filled + empty
    ctx.mark(tab, t)
    return {"tab": tab["id"], "ids": [t["id"]]}


# ---------------------------------------------------------------- tab/doc ops


def _op_add_tab(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    if len(ctx.doc["tabs"]) >= MAX_TABS:
        raise DocError("too_big", f"a doc holds at most {MAX_TABS} tabs.")
    if isinstance(op.get("blocks"), list):
        # A tab put back as it was (an undo): its id when free, and its blocks.
        tab = new_tab(op.get("title") or "Untitled", _blocks_from(op, ctx, "add_tab"))
        tid = op.get("id")
        if isinstance(tid, str) and _TAB_ID.match(tid) and all(t["id"] != tid for t in ctx.doc["tabs"]):
            tab["id"] = tid
    else:
        tab = tab_from_markdown(op.get("title") or "Untitled", str(op.get("markdown") or ""), rev=ctx.rev, by=ctx.by)
    at = _index_from_one(op, len(ctx.doc["tabs"]))
    ctx.doc["tabs"].insert(at, tab)
    for b in tab["blocks"]:
        ctx.mark(tab, b)
    ctx.touched.setdefault(tab["id"], set())
    return {"tab": tab["id"], "ids": [b["id"] for b in tab["blocks"]]}


def _op_rename_tab(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = find_tab(ctx.doc, op.get("tab"))
    title = clean_text(op.get("title"), "tab title").strip()
    if not title:
        raise DocError("bad_op", "rename_tab needs a non-empty \"title\".")
    tab["title"] = title
    ctx.touched.setdefault(tab["id"], set())
    return {"tab": tab["id"]}


def _op_delete_tab(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = find_tab(ctx.doc, op.get("tab"))
    if len(ctx.doc["tabs"]) == 1:
        raise DocError("bad_op", "a doc keeps at least one tab.")
    for b in tab["blocks"]:
        ctx.guard(b)  # a tab the person typed in since the caller read it is not theirs to drop
    ctx.doc["tabs"] = [t for t in ctx.doc["tabs"] if t["id"] != tab["id"]]
    return {"tab": tab["id"], "deleted_tab": True}


def _op_move_tab(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    tab = find_tab(ctx.doc, op.get("tab"))
    tabs = [t for t in ctx.doc["tabs"] if t["id"] != tab["id"]]
    if not op.get("index"):
        raise DocError("bad_op", "move_tab needs \"index\": the position counting from 1 (1 = first).")
    tabs.insert(_index_from_one(op, len(tabs)), tab)
    ctx.doc["tabs"] = tabs
    return {"tab": tab["id"]}


def _op_rename_doc(op: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    title = clean_text(op.get("title"), "title").strip()
    if not title:
        raise DocError("bad_op", "rename_doc needs a non-empty \"title\".")
    old = ctx.doc["title"]
    ctx.doc["title"] = title
    # A one-tab doc's tab is the doc's page: it carries the doc's name.
    if len(ctx.doc["tabs"]) == 1 and ctx.doc["tabs"][0]["title"].strip().lower() == old.strip().lower():
        ctx.doc["tabs"][0]["title"] = title
    return {}


_OPS = {
    "insert": _op_insert,
    "replace": _op_replace,
    "update": _op_update,
    "edit_text": _op_edit_text,
    "delete": _op_delete,
    "move": _op_move,
    "update_cell": _op_update_cell,
    "add_rows": _op_add_rows,
    "delete_rows": _op_delete_rows,
    "add_column": _op_add_column,
    "update_column": _op_update_column,
    "delete_column": _op_delete_column,
    "sort_rows": _op_sort_rows,
    "add_tab": _op_add_tab,
    "rename_tab": _op_rename_tab,
    "delete_tab": _op_delete_tab,
    "move_tab": _op_move_tab,
    "rename_doc": _op_rename_doc,
}
OP_NAMES = tuple(_OPS)


def apply_ops(
    doc: Doc,
    ops: list[dict[str, Any]],
    *,
    tab: str | None = None,
    by: str = "agent",
    if_rev: int | None = None,
) -> tuple[Doc, dict[str, Any]]:
    """Apply ``ops`` to a copy of ``doc``; return (new doc, result).

    The result is ``{"rev", "results": [per-op dict], "touched": {tab id: [ids]},
    "removed": {tab id: [ids]}}``. Raises DocError (with ``op`` = the failing
    index) and leaves ``doc`` untouched when any op is refused.
    """
    if not isinstance(ops, list) or not ops:
        raise DocError("bad_op", "ops is a non-empty list of operations.")
    if len(ops) > MAX_OPS:
        raise DocError("too_big", f"send at most {MAX_OPS} ops per call.")
    work = clone(doc)
    rev = int(work.get("rev") or 0) + 1
    ctx = _Ctx(work, tab, rev, by, if_rev)
    results = []
    for n, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in _OPS:
            raise DocError("bad_op", f"op {n}: \"op\" is one of {', '.join(OP_NAMES)}.", op=n)
        try:
            res = _OPS[op["op"]](op if ctx.literal else _without_filled_blanks(op), ctx)
        except DocError as err:
            err.detail.setdefault("op", n)
            err.message = f"op {n} ({op['op']}, counting from 0): {err.message}"
            raise
        except (TypeError, AttributeError, ValueError) as exc:
            # A field of the wrong shape ("tab": 1, "ids": 5) fails deep in an
            # op; it is the caller's to fix, not a server error.
            raise DocError("bad_op", f"op {n} ({op['op']}, counting from 0): a field has the wrong type "
                           f"or shape ({exc}).", op=n) from exc
        results.append({"op": n, **res})
    _repair_chart_sources(ctx)
    if not ctx.literal:
        for tab_id, ids in ctx.touched.items():
            tab_ = find_tab(work, tab_id)
            for chart in link_inline_charts(tab_, ids):
                ctx.mark(tab_, chart)
    work["rev"] = rev
    return work, {
        "rev": rev,
        "results": results,
        "touched": {k: sorted(v) for k, v in ctx.touched.items()},
        "removed": {k: sorted(v) for k, v in ctx.removed.items()},
    }


def _without_title_heading(blocks: list[Block], title: str) -> list[Block]:
    """A tab's title is drawn above its content; a leading `# Title` would repeat it."""
    if blocks and blocks[0]["type"] == "heading" and blocks[0]["level"] == 1 \
            and blocks[0]["text"].strip().lower() == title.strip().lower():
        return blocks[1:]
    return blocks


def tab_from_markdown(title: str, markdown: str, *, rev: int, by: str, strip_title: bool = False) -> Tab:
    """A new tab holding ``markdown``'s blocks, its charts' "previous" tables resolved."""
    tab = new_tab(title)
    blocks = mdmod.parse_markdown(markdown or "", rev, by)
    if strip_title:
        blocks = _without_title_heading(blocks, tab["title"])
    resolve_chart_sources(tab, blocks, 0)
    tab["blocks"] = blocks
    _check_size(tab)
    return tab


def build_new_doc(
    title: str,
    *,
    markdown: str | None = None,
    tabs: list[dict[str, Any]] | None = None,
    by: str = "agent",
) -> Doc:
    doc = new_doc(title)
    doc["rev"] = 1
    specs = tabs if tabs else [{"title": doc["title"], "markdown": markdown or ""}]
    for spec in specs:
        if not isinstance(spec, dict):
            raise DocError("bad_tabs", "each tab is {\"title\": …, \"markdown\": …}.")
        doc["tabs"].append(tab_from_markdown(spec.get("title") or doc["title"], spec.get("markdown") or "",
                                             rev=1, by=by, strip_title=True))
        if by != "user":
            link_inline_charts(doc["tabs"][-1])
    return doc


__all__ = ["apply_ops", "build_new_doc", "link_inline_charts", "tab_from_markdown", "MAX_OPS", "OP_NAMES"]
