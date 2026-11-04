"""Undo of the person's edits (cowork.docs.undo): each write answers the ops that put it back."""
from __future__ import annotations

import pytest

from cowork.docs.ops import apply_ops, build_new_doc
from cowork.docs.undo import undo_ops

MD = ("Intro words here.\n\n## Plan\n\nFirst step is short.\n\n- one item\n- two item\n\n"
      "| Task | Owner | When |\n| --- | --- | --- |\n| Copy | Dana | 2026-11-04 |\n| Checkout | Marco | 2026-11-13 |\n\n"
      "Closing line.")


def _doc():
    doc = build_new_doc("Plan", markdown=MD, by="user")
    doc, _ = apply_ops(doc, [{"op": "add_tab", "title": "Notes", "markdown": "Second tab text."}], by="user")
    return doc


def _reads(doc):
    """The doc as it reads: titles, tabs in order, blocks in order, without who wrote them when."""
    return (doc["title"], [(t["id"], t["title"], [{k: v for k, v in b.items() if k not in ("v", "by")}
                                                  for b in t["blocks"]]) for t in doc["tabs"]])


def _ids(doc, kind=None, tab=0):
    return [b["id"] for b in doc["tabs"][tab]["blocks"] if kind is None or b["type"] == kind]


def _table(doc):
    return next(b for b in doc["tabs"][0]["blocks"] if b["type"] == "table")


EDITS = {
    "type": lambda d: [{"op": "update", "id": _ids(d, "paragraph")[0], "set": {"text": "Intro words here, more."}}],
    "empty a block": lambda d: [{"op": "update", "id": _ids(d, "paragraph")[1], "set": {"text": ""}}],
    "split with Enter": lambda d: [{"op": "update", "id": _ids(d, "paragraph")[1], "set": {"text": "First step"}},
                                   {"op": "insert", "position": {"after": _ids(d, "paragraph")[1]},
                                    "blocks": [{"id": "b_split001", "type": "paragraph", "text": " is short."}]}],
    "merge with Backspace": lambda d: [{"op": "update", "id": _ids(d, "bullet")[0], "set": {"text": "one itemtwo item"}},
                                       {"op": "delete", "target": [_ids(d, "bullet")[1]]}],
    "turn into a heading": lambda d: [{"op": "update", "id": _ids(d, "paragraph")[0],
                                       "set": {"type": "heading", "level": 2}}],
    "slash insert": lambda d: [{"op": "insert", "position": {"before": _ids(d, "paragraph")[2]},
                                "blocks": [{"id": "b_divider1", "type": "divider"}]},
                               {"op": "update", "id": _ids(d, "paragraph")[2], "set": {"text": ""}}],
    "delete the first block": lambda d: [{"op": "delete", "target": [_ids(d)[0]]}],
    "move a block up": lambda d: [{"op": "move", "ids": [_ids(d, "paragraph")[2]], "position": "start"}],
    "edit a cell": lambda d: [{"op": "update_cell", "table": _table(d)["id"], "row": _table(d)["rows"][1]["id"],
                               "col": _table(d)["columns"][1]["id"], "value": "Priya"}],
    "add a row": lambda d: [{"op": "add_rows", "table": _table(d)["id"], "index": 1,
                             "rows": [{"id": "r_newrow01", "cells": {}}]}],
    "delete a row": lambda d: [{"op": "delete_rows", "table": _table(d)["id"], "rows": [_table(d)["rows"][0]["id"]]}],
    "delete a column": lambda d: [{"op": "delete_column", "table": _table(d)["id"], "col": _table(d)["columns"][1]["id"]}],
    "add a column": lambda d: [{"op": "add_column", "table": _table(d)["id"], "name": "Notes"}],
    "rename and retype a column": lambda d: [{"op": "update_column", "table": _table(d)["id"],
                                              "col": _table(d)["columns"][2]["id"], "name": "Due", "kind": "text"}],
    "sort rows": lambda d: [{"op": "sort_rows", "table": _table(d)["id"], "col": _table(d)["columns"][0]["id"]}],
    "add a tab": lambda d: [{"op": "add_tab", "title": "Scratch"}],
    "rename a tab": lambda d: [{"op": "rename_tab", "tab": d["tabs"][1]["id"], "title": "Launch week"}],
    "delete a tab with content": lambda d: [{"op": "delete_tab", "tab": d["tabs"][1]["id"]}],
    "move a tab": lambda d: [{"op": "move_tab", "tab": d["tabs"][1]["id"], "index": 1}],
    "rename the doc": lambda d: [{"op": "rename_doc", "title": "Plan v2"}],
}


@pytest.mark.parametrize("name", list(EDITS))
def test_an_edit_and_its_undo_leave_the_doc_as_it_was_and_redo_brings_it_back(name):
    before = _doc()
    after, result = apply_ops(before, EDITS[name](before), by="user")
    assert _reads(after) != _reads(before)
    undo = undo_ops(before, after, result)
    undone, undo_result = apply_ops(after, undo, by="user")
    assert _reads(undone) == _reads(before)
    redone, _ = apply_ops(undone, undo_ops(after, undone, undo_result), by="user")
    assert _reads(redone) == _reads(after)


def test_an_undo_leaves_what_someone_else_changed_meanwhile():
    """Only what the write changed is put back: the agent's text in another block stays."""
    before = _doc()
    after, result = apply_ops(before, EDITS["edit a cell"](before), by="user")
    undo = undo_ops(before, after, result)
    agent, _ = apply_ops(after, [{"op": "update", "id": _ids(after, "paragraph")[0], "set": {"text": "Agent's intro."}}])
    undone, _ = apply_ops(agent, undo, by="user")
    assert next(b for b in undone["tabs"][0]["blocks"] if b["type"] == "paragraph")["text"] == "Agent's intro."
    assert _table(undone)["rows"][1]["cells"] == _table(before)["rows"][1]["cells"]
