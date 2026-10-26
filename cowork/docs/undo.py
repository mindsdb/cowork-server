"""Undo for the person's edits: the ops that put back what one write changed.

The editor's Cmd+Z used to be the browser's own, which knows only typing and
runs across every block on the page: it once emptied a table cell typed into
minutes before instead of undoing the last thing done, and a deleted tab or
column could not come back at all. So each of the person's writes answers
with ``undo``: ops that restore what that write changed and nothing else. The
write path computes them under the doc's lock, from the doc just before and
just after, so another writer's changes in between are never part of them.
The editor checks that what they would touch still reads as the write left
it before sending them (``undo`` is itself a write, so it answers with the
ops that redo it).
"""

from __future__ import annotations

from typing import Any

from .model import Block, Doc, Tab


def _plain(b: Block) -> Block:
    """A block as it reads: without who wrote it, and when."""
    return {k: v for k, v in b.items() if k not in ("v", "by")}


def _position(order: list[str], i: int, present: list[str]) -> Any:
    """Where block ``order[i]`` goes back: after the one before it, else first."""
    if i:
        return {"after": order[i - 1]}
    return {"before": present[0]} if present else "start"


def _undo_in_tab(before: Tab, after: Tab, changed: set[str]) -> list[dict[str, Any]]:
    tid = before["id"]
    ops: list[dict[str, Any]] = []
    if before["title"] != after["title"]:
        ops.append({"op": "rename_tab", "tab": tid, "title": before["title"]})
    was = {b["id"]: b for b in before["blocks"]}
    now = {b["id"]: b for b in after["blocks"]}
    added = [b["id"] for b in after["blocks"] if b["id"] in changed and b["id"] not in was]
    if added:
        ops.append({"op": "delete", "tab": tid, "target": added})
    left = [b["id"] for b in after["blocks"] if b["id"] not in added]
    order = [b["id"] for b in before["blocks"]]
    for i, bid in enumerate(order):
        if bid not in changed:
            continue
        old = was[bid]
        if bid not in now:
            ops.append({"op": "insert", "tab": tid, "position": _position(order, i, left), "blocks": [_plain(old)]})
            continue
        if _plain(old) != _plain(now[bid]):
            ops.append({"op": "replace", "tab": tid, "target": bid, "blocks": [_plain(old)]})
        at = left.index(bid)
        if (left[at - 1] if at else None) != (order[i - 1] if i else None):
            ops.append({"op": "move", "tab": tid, "ids": [bid],
                        "position": _position(order, i, [x for x in left if x != bid])})
    return ops


def undo_ops(before: Doc, after: Doc, result: dict[str, Any]) -> list[dict[str, Any]]:
    """The ops that turn ``after`` back into ``before`` where this write (``result``) changed it."""
    ops: list[dict[str, Any]] = []
    if before["title"] != after["title"]:
        ops.append({"op": "rename_doc", "title": before["title"]})
    was = {t["id"]: t for t in before["tabs"]}
    now = {t["id"]: t for t in after["tabs"]}
    for t in after["tabs"]:
        if t["id"] not in was:
            ops.append({"op": "delete_tab", "tab": t["id"]})
    for i, t in enumerate(before["tabs"]):
        if t["id"] not in now:
            ops.append({"op": "add_tab", "id": t["id"], "title": t["title"], "index": i + 1,
                        "blocks": [_plain(b) for b in t["blocks"]]})
    named = set(result.get("touched") or {}) | set(result.get("removed") or {})
    for tid in [t["id"] for t in before["tabs"] if t["id"] in named and t["id"] in now]:
        changed = set((result.get("touched") or {}).get(tid, [])) | set((result.get("removed") or {}).get(tid, []))
        ops += _undo_in_tab(was[tid], now[tid], changed)
    order = [t["id"] for t in before["tabs"]]
    if [x for x in (t["id"] for t in after["tabs"]) if x in was] != [x for x in order if x in now]:
        ops += [{"op": "move_tab", "tab": tid, "index": i + 1} for i, tid in enumerate(order)]
    return ops


__all__ = ["undo_ops"]
