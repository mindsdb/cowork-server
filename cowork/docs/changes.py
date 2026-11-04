"""What changed between two versions of a doc, said in one line a person can scan.

``describe(old, new)`` names the change that comes first in the doc, where it
is, and how many came with it: "Budget: Paid search 72 → 80, and 2 more cells",
"Added the Risks tab", "3 changes across Budget and Risks". It compares the two
docs block by block (blocks keep their ids), so it reads the same whoever made
the change and however: the person typing, the agent, or a restore. Pure.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from .model import TEXT_TYPES, Block, Doc, Tab

# Room in a history row: a quote of a passage, a cell's value, a row's or tab's name.
QUOTE_CHARS = 28
VALUE_CHARS = 20
NAME_CHARS = 20

# Bookkeeping a block carries that isn't its content: who last wrote it, and when (its rev).
_BOOKKEEPING = ("v", "by")

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CHIP = re.compile(r"\{\{(?:date:)?([^}]*)\}\}")
_EMPHASIS = re.compile(r"\*+|`+|~~")


class Change(NamedTuple):
    tab: str | None  # the tab it is in; None for the doc's own title and its tabs
    text: str
    cell: bool = False  # one table cell or metric, which "and 2 more cells" counts


def describe(old: Doc, new: Doc) -> str:
    changes = doc_changes(old, new)
    if not changes:
        return "No visible changes"
    tabs = list(dict.fromkeys(c.tab for c in changes if c.tab is not None))
    if len(tabs) > 1:
        return f"{len(changes)} changes across {_join_tabs(tabs)}"
    head, rest = changes[0], changes[1:]
    text = head.text
    if rest:
        noun = "cell" if all(c.cell for c in rest) else "change"
        text += f", and {len(rest)} more {noun}{'' if len(rest) == 1 else 's'}"
    if head.tab is not None and len(new["tabs"]) > 1:
        return f"{_short(head.tab, NAME_CHARS)}: {text}"
    return text[:1].upper() + text[1:]


def doc_changes(old: Doc, new: Doc) -> list[Change]:
    """Every change, in reading order: the doc's title and tabs first, then each tab's blocks."""
    out: list[Change] = []
    if old.get("title") != new.get("title"):
        out.append(Change(None, f"renamed the doc {_quote(old.get('title'))} → {_quote(new.get('title'))}"))
    old_tabs = {t["id"]: t for t in old["tabs"]}
    new_ids = {t["id"] for t in new["tabs"]}
    # A one-tab doc shows no tabs: its tab is the doc, named by the doc's title.
    tabs_shown = len(old["tabs"]) > 1 or len(new["tabs"]) > 1
    if tabs_shown:
        for tab in new["tabs"]:
            was = old_tabs.get(tab["id"])
            if was is None:
                out.append(Change(None, f"added the {_short(tab['title'], NAME_CHARS)} tab"))
            elif was["title"] != tab["title"]:
                out.append(Change(None, f"renamed the {_short(was['title'], NAME_CHARS)} tab to "
                                        f"{_short(tab['title'], NAME_CHARS)}"))
        out.extend(Change(None, f"removed the {_short(t['title'], NAME_CHARS)} tab")
                   for t in old["tabs"] if t["id"] not in new_ids)
        moved = _first_moved([t["id"] for t in old["tabs"]], [t["id"] for t in new["tabs"]])
        if moved is not None:
            out.append(Change(None, f"moved the {_short(_by_id(new['tabs'], moved)['title'], NAME_CHARS)} tab"))
    for tab in new["tabs"]:
        if tab["id"] in old_tabs:
            out.extend(_tab_changes(old_tabs[tab["id"]], tab))
    return out


# ---------------------------------------------------------------- blocks


def _tab_changes(old: Tab, new: Tab) -> list[Change]:
    where = new["title"]
    before = {b["id"]: b for b in old["blocks"]}
    after_ids = {b["id"] for b in new["blocks"]}
    out: list[Change] = []
    for b in new["blocks"]:
        was = before.get(b["id"])
        if was is None:
            if _worth_naming(b):
                out.append(Change(where, f"added {_name(b)}"))
        elif _content(was) != _content(b):
            out.extend(_block_changes(where, was, b))
    out.extend(Change(where, f"removed {_name(b)}") for b in old["blocks"]
               if b["id"] not in after_ids and _worth_naming(b))
    moved = _first_moved([b["id"] for b in old["blocks"]], [b["id"] for b in new["blocks"]])
    if moved is not None:
        out.append(Change(where, f"moved {_name(_by_id(new['blocks'], moved))}"))
    return out


def _block_changes(where: str, old: Block, new: Block) -> list[Change]:
    kind = new["type"]
    if kind != old["type"]:
        return [Change(where, f"edited {_name(new)}")]
    if kind == "table":
        return _table_changes(where, old, new)
    if kind == "metrics":
        return _metrics_changes(where, old, new)
    if kind == "task" and old.get("done") != new.get("done") and _plain(old.get("text")) == _plain(new.get("text")):
        return [Change(where, f"{'ticked' if new.get('done') else 'unticked'} {_quote(new.get('text'))}")]
    if kind in ("chart", "image"):
        return [Change(where, f"changed {_name(new)}")]
    if kind == "code":
        return [Change(where, "edited code")]
    return [Change(where, f"edited {_name(new) if _plain(new.get('text')) else _name(old)}")]


def _name(b: Block) -> str:
    """A block as a history row names it: its words, quoted, or what it is."""
    kind = b["type"]
    if kind in TEXT_TYPES:
        words = _plain(b.get("text")) or _plain(b.get("title"))
        return _quote(words) if words else "an empty line"
    if kind == "chart":
        return f"the chart {_quote(b['title'])}" if _plain(b.get("title")) else "a chart"
    return {"table": "a table", "metrics": "the metrics", "image": "an image", "code": "code",
            "divider": "a divider"}.get(kind, "a section")


def _worth_naming(b: Block) -> bool:
    """An empty line or a section the agent has still to write is no change worth a row."""
    if b["type"] == "pending":
        return False
    return b["type"] not in TEXT_TYPES or bool(_plain(b.get("text")) or _plain(b.get("title")))


def _content(b: Block) -> dict[str, Any]:
    return {k: v for k, v in b.items() if k not in _BOOKKEEPING}


# ---------------------------------------------------------------- tables and metrics


def _table_changes(where: str, old: Block, new: Block) -> list[Change]:
    out: list[Change] = []
    old_cols = {c["id"]: c for c in old["columns"]}
    new_col_ids = {c["id"] for c in new["columns"]}
    for col in new["columns"]:
        was = old_cols.get(col["id"])
        if was is None:
            out.append(Change(where, f"added column {_quote(col['name'])}"))
        elif was["name"] != col["name"]:
            out.append(Change(where, f"renamed {_quote(was['name'])} → {_quote(col['name'])}"))
        elif was.get("kind") != col.get("kind"):
            out.append(Change(where, f"made {_quote(col['name'])} a {col.get('kind')} column"))
    out.extend(Change(where, f"removed column {_quote(c['name'])}") for c in old["columns"] if c["id"] not in new_col_ids)

    label_col = new["columns"][0]["id"]
    old_rows = {r["id"]: r for r in old["rows"]}
    new_row_ids = {r["id"] for r in new["rows"]}
    added = [r for r in new["rows"] if r["id"] not in old_rows]
    removed = [r for r in old["rows"] if r["id"] not in new_row_ids]
    out.extend(_rows_change(where, "added", added, label_col))
    out.extend(_rows_change(where, "removed", removed, label_col))

    wide = len(new["columns"]) > 2
    shared_cols = [c for c in new["columns"] if c["id"] in old_cols]
    for row in new["rows"]:
        was = old_rows.get(row["id"])
        if was is None:
            continue
        for col in shared_cols:
            before, after = was["cells"].get(col["id"], ""), row["cells"].get(col["id"], "")
            if _plain(before) != _plain(after):
                out.append(_cell_change(where, row, col, before, after, label_col=label_col, wide=wide))
    if not added and not removed and _first_moved([r["id"] for r in old["rows"]], [r["id"] for r in new["rows"]]):
        out.append(Change(where, "sorted a table"))
    return out


def _rows_change(where: str, verb: str, rows: list[dict[str, Any]], label_col: str) -> list[Change]:
    if not rows:
        return []
    if len(rows) > 1:
        return [Change(where, f"{verb} {len(rows)} rows")]
    label = _plain(rows[0]["cells"].get(label_col, ""))
    return [Change(where, f"{verb} row {_quote(label)}" if label else f"{verb} a row")]


def _cell_change(where: str, row: dict[str, Any], col: dict[str, Any], before: str, after: str, *,
                 label_col: str, wide: bool) -> Change:
    """"Paid search 72 → 80"; in a wider table, "Paid search · Budget 72 → 80"."""
    if col["id"] == label_col:
        if not _plain(before) or not _plain(after):
            return Change(where, f"named a row {_quote(after)}" if _plain(after) else "cleared a row's name", cell=True)
        return Change(where, f"renamed row {_quote(before)} → {_quote(after)}", cell=True)
    label = _short(_plain(row["cells"].get(label_col, "")), NAME_CHARS)
    column = _short(col["name"], NAME_CHARS)
    head = f"{label} · {column}" if wide and label else (label or column)
    if not _plain(before):
        return Change(where, f"{head} set to {_short(after, VALUE_CHARS)}", cell=True)
    if not _plain(after):
        return Change(where, f"{head} cleared", cell=True)
    return Change(where, f"{head} {_short(before, VALUE_CHARS)} → {_short(after, VALUE_CHARS)}", cell=True)


def _metrics_changes(where: str, old: Block, new: Block) -> list[Change]:
    before, after = old["items"], new["items"]
    if [i["label"] for i in before] != [i["label"] for i in after]:
        return [Change(where, "edited the metrics")]
    out = [Change(where, f"{_short(a['label'], NAME_CHARS)} {_short(b['value'], VALUE_CHARS)} → "
                         f"{_short(a['value'], VALUE_CHARS)}", cell=True)
           for b, a in zip(before, after) if _plain(b["value"]) != _plain(a["value"])]
    if not out:  # a delta, a note or a tone
        return [Change(where, "edited the metrics")]
    return out


# ---------------------------------------------------------------- words


def _plain(text: Any) -> str:
    """Text as it reads on the page: no Markdown marks, chips as their value, one space apart."""
    t = _LINK.sub(r"\1", str(text or ""))
    t = _CHIP.sub(lambda m: m.group(1).removeprefix("pending:").strip(), t)
    return " ".join(_EMPHASIS.sub("", t).split())


def _short(text: Any, limit: int) -> str:
    t = _plain(text)
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _quote(text: Any) -> str:
    """“The first few words…” of a passage, cut at a word."""
    t = _plain(text)
    if len(t) > QUOTE_CHARS:
        cut = t[:QUOTE_CHARS].rsplit(" ", 1)[0] if " " in t[:QUOTE_CHARS] else t[:QUOTE_CHARS]
        t = cut.rstrip(" ,.;:—-") + "…"
    return f"“{t}”"


def _join_tabs(titles: list[str]) -> str:
    names = [_short(t, NAME_CHARS) for t in titles]
    if len(names) > 3:
        return f"{', '.join(names[:2])} and {len(names) - 2} more tabs"
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _first_moved(before: list[str], after: list[str]) -> str | None:
    """The id that moved, among the ids both lists share, if any did (the first, when several did)."""
    shared = set(before) & set(after)
    was = [i for i in before if i in shared]
    now = [i for i in after if i in shared]
    at = next((n for n, (w, a) in enumerate(zip(was, now)) if w != a), None)
    if at is None:
        return None
    # At the first difference, either the new one moved up to here or the old one moved down;
    # when both read true (two neighbours swapped), the one that moved up is what was moved.
    for candidate in (now[at], was[at]):
        if [i for i in was if i != candidate] == [i for i in now if i != candidate]:
            return candidate
    return now[at]


def _by_id(items: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
    return next(i for i in items if i["id"] == item_id)


__all__ = ["Change", "describe", "doc_changes"]
