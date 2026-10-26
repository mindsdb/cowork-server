"""Edit operations: atomic batches, the stale guard, tables, tabs, and what a model fills in."""
from __future__ import annotations

import copy

import pytest

from cowork.docs.model import MAX_TABLE_COLS, MAX_TABLE_ROWS, MAX_TEXT, DocError
from cowork.docs.ops import apply_ops, build_new_doc

from tests._doc_samples import PLAN


def _doc():
    return build_new_doc("Q3 launch plan", markdown=PLAN)


def _blocks(doc, kind):
    return [b for t in doc["tabs"] for b in t["blocks"] if b["type"] == kind]


def _block(doc, block_id):
    return next(b for t in doc["tabs"] for b in t["blocks"] if b["id"] == block_id)


def test_markdown_becomes_typed_blocks():
    doc = _doc()
    types = [b["type"] for b in doc["tabs"][0]["blocks"]]
    # The leading "# <title>" is the tab's own title, drawn above the page.
    assert types == [
        "paragraph", "heading", "table", "chart", "callout", "task", "task", "bullet",
        "numbered", "numbered", "metrics", "pending",
    ]
    table = _blocks(doc, "table")[0]
    kinds = {c["name"]: c["kind"] for c in table["columns"]}
    assert kinds == {"Workstream": "text", "Owner": "text", "Status": "status", "Due": "date", "Budget ($k)": "number"}
    status = next(c for c in table["columns"] if c["kind"] == "status")
    assert {o["name"]: o["tone"] for o in status["options"]} == {
        "In progress": "blue", "Not started": "gray", "Done": "green",
    }
    chart = _blocks(doc, "chart")[0]
    assert chart["source"]["table"] == table["id"]
    callout = _blocks(doc, "callout")[0]
    assert (callout["tone"], callout["title"]) == ("warning", "Legal review")
    assert _blocks(doc, "bullet")[0]["indent"] == 1
    assert _blocks(doc, "task")[1]["done"] is True


def test_filling_a_pending_section_lands_where_it_was_planned():
    doc = _doc()
    pending = _blocks(doc, "pending")[0]
    new, result = apply_ops(doc, [{"op": "replace", "target": pending["id"],
                                   "markdown": "## Risks\n\n- Legal review\n- Staffing"}])
    blocks = new["tabs"][0]["blocks"]
    assert [b["type"] for b in blocks[-3:]] == ["heading", "bullet", "bullet"]
    assert new["rev"] == doc["rev"] + 1
    assert all(b["by"] == "agent" and b["v"] == new["rev"] for b in blocks[-3:])
    assert pending["id"] in result["removed"][new["tabs"][0]["id"]]
    assert not _blocks(new, "pending")


def test_replacing_one_block_with_one_keeps_its_id():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    new, _ = apply_ops(doc, [{"op": "replace", "target": lead["id"], "markdown": "Q3 ships to every workspace."}])
    assert new["tabs"][0]["blocks"][0]["id"] == lead["id"]
    assert new["tabs"][0]["blocks"][0]["text"] == "Q3 ships to every workspace."


def test_table_ops_address_rows_by_index_and_columns_by_name():
    doc = _doc()
    table = _blocks(doc, "table")[0]
    new, _ = apply_ops(doc, [
        {"op": "update_cell", "table": table["id"], "row": 1, "col": "status", "value": "Blocked"},
        {"op": "add_rows", "table": table["id"], "rows": [{"Workstream": "Docs", "Status": "In review"}]},
        {"op": "add_column", "table": table["id"], "name": "Risk", "kind": "status", "values": ["Low", "High"]},
        {"op": "sort_rows", "table": table["id"], "col": "Budget ($k)", "descending": True},
    ])
    t = _blocks(new, "table")[0]
    status = next(c for c in t["columns"] if c["name"] == "Status")
    assert {"Blocked", "In review"} <= {o["name"] for o in status["options"]}
    names = [r["cells"][t["columns"][0]["id"]] for r in t["rows"]]
    assert names[:3] == ["Connector", "Launch comms", "Pricing"] and names[-1] == "Docs"
    assert [c["name"] for c in t["columns"]][-1] == "Risk"


def test_edit_text_changes_words_in_place():
    doc = _doc()
    new, result = apply_ops(doc, [{"op": "edit_text", "find": "500", "replace": "750"}])
    assert "750" in new["tabs"][0]["blocks"][0]["text"]
    assert result["touched"][new["tabs"][0]["id"]] == [doc["tabs"][0]["blocks"][0]["id"]]


def test_a_refused_op_changes_nothing():
    doc = _doc()
    with pytest.raises(DocError) as err:
        apply_ops(doc, [
            {"op": "edit_text", "find": "500", "replace": "750"},
            {"op": "delete", "target": "b_missing"},
        ])
    assert err.value.code == "no_block"
    assert err.value.detail["op"] == 1
    assert "500" in doc["tabs"][0]["blocks"][0]["text"]


def test_an_id_copied_with_one_character_slipped_still_finds_its_block():
    doc = _doc()
    blocks = doc["tabs"][0]["blocks"]
    blocks[0]["id"], blocks[1]["id"] = "b_7kq16vg6", "b_aaaaaaaa"
    for slip in ("b_7kq16vg", "b_7kq16vg6x", "b_7kq16vg7", "b_7kq1vg6"):
        new, _ = apply_ops(doc, [{"op": "update", "id": slip, "set": {"text": "Moved to Q4."}}])
        assert new["tabs"][0]["blocks"][0]["text"] == "Moved to Q4."
        assert new["tabs"][0]["blocks"][1] == blocks[1]

    # Two slips, a short or foreign id, or two blocks a slip away: refused.
    for unknown in ("b_7kq61vg6", "b_7kq16v", "b_12", "c_7kq16vg6"):
        with pytest.raises(DocError) as err:
            apply_ops(doc, [{"op": "delete", "target": unknown}])
        assert err.value.code == "no_block"
    blocks[1]["id"] = "b_7kq16vg5"
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "delete", "target": "b_7kq16vg"}])
    assert err.value.code == "no_block"


def test_an_edit_against_an_older_read_is_refused_as_stale():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    # The person edits the lead (rev 2) after the agent read rev 1.
    doc, _ = apply_ops(doc, [{"op": "update", "id": lead["id"], "set": {"text": "Edited by hand."}}], by="user")
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "edit_text", "id": lead["id"], "find": "hand", "replace": "agent"}], if_rev=1)
    assert err.value.code == "stale"
    # Blocks the person did not touch are still editable against the old read.
    heading = _blocks(doc, "heading")[0]
    apply_ops(doc, [{"op": "update", "id": heading["id"], "set": {"text": "Workstreams and owners"}}], if_rev=1)


def test_columns_can_be_named_by_index_from_zero():
    doc = _doc()
    table = _blocks(doc, "table")[0]
    first = table["columns"][0]["id"]
    new, _ = apply_ops(doc, [
        {"op": "sort_rows", "table": table["id"], "col": 0, "descending": True},
        {"op": "update_column", "table": table["id"], "col": 0, "name": "Track"},
    ])
    t = _blocks(new, "table")[0]
    names = [r["cells"][first] for r in t["rows"]]
    assert names == sorted(names, reverse=True) and t["columns"][0]["name"] == "Track"
    new, _ = apply_ops(new, [{"op": "delete_column", "table": table["id"], "col": 0}])
    assert first not in [c["id"] for c in _blocks(new, "table")[0]["columns"]]


def test_an_edit_cannot_grow_text_past_what_a_load_accepts():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    doc, _ = apply_ops(doc, [{"op": "update", "id": lead["id"], "set": {"text": "x" * (MAX_TEXT - 1) + "!"}}])
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "edit_text", "id": lead["id"], "find": "!", "replace": "!!"}])
    assert err.value.code == "too_long"


def test_a_tab_the_person_typed_in_after_the_read_is_not_deleted_from_it():
    doc = _doc()
    doc, result = apply_ops(doc, [{"op": "add_tab", "title": "Notes", "markdown": "First thought."}])
    tab_id = result["results"][0]["tab"]
    read_at = doc["rev"]
    note = next(t for t in doc["tabs"] if t["id"] == tab_id)["blocks"][0]
    doc, _ = apply_ops(doc, [{"op": "update", "tab": tab_id, "id": note["id"], "set": {"text": "Typed since."}}],
                       by="user")
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "delete_tab", "tab": tab_id}], if_rev=read_at)
    assert err.value.code == "stale"
    apply_ops(doc, [{"op": "delete_tab", "tab": tab_id}], if_rev=doc["rev"])


def test_tabs_can_be_added_renamed_and_removed():
    doc = _doc()
    new, result = apply_ops(doc, [{"op": "add_tab", "title": "Sources", "markdown": "- [Thread](https://example.com)"}])
    tab_id = result["results"][0]["tab"]
    new, _ = apply_ops(new, [{"op": "rename_tab", "tab": "Sources", "title": "References"}])
    assert [t["title"] for t in new["tabs"]] == ["Q3 launch plan", "References"]
    new, _ = apply_ops(new, [{"op": "delete_tab", "tab": tab_id}])
    assert len(new["tabs"]) == 1
    with pytest.raises(DocError):
        apply_ops(new, [{"op": "delete_tab", "tab": new["tabs"][0]["id"]}])


def test_renaming_a_one_tab_doc_renames_its_page():
    doc = _doc()
    new, _ = apply_ops(doc, [{"op": "rename_doc", "title": "Q4 launch plan"}])
    assert new["title"] == new["tabs"][0]["title"] == "Q4 launch plan"


def test_type_changes_stay_between_text_blocks():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    new, _ = apply_ops(doc, [{"op": "update", "id": lead["id"], "set": {"type": "heading", "level": 2}}])
    assert new["tabs"][0]["blocks"][0]["type"] == "heading"
    table = _blocks(doc, "table")[0]
    with pytest.raises(DocError):
        apply_ops(doc, [{"op": "update", "id": table["id"], "set": {"type": "paragraph"}}])


def test_end_of_section_inserts_before_the_next_heading():
    doc = build_new_doc("Notes", markdown="## A\n\nOne.\n\n## B\n\nTwo.")
    heading_a = doc["tabs"][0]["blocks"][0]
    new, _ = apply_ops(doc, [{"op": "insert", "position": {"end_of_section": heading_a["id"]}, "markdown": "Also."}])
    assert [b.get("text") for b in new["tabs"][0]["blocks"]] == ["A", "One.", "Also.", "B", "Two."]


def test_editor_minted_ids_are_kept_when_well_formed_and_free():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    new, result = apply_ops(doc, [
        {"op": "insert", "position": {"after": lead["id"]}, "blocks": [{"id": "b_abcdefgh", "type": "paragraph", "text": "Mine"}]},
        {"op": "insert", "position": "end", "blocks": [{"id": lead["id"], "type": "paragraph", "text": "Taken"}]},
        {"op": "insert", "position": "end", "blocks": [{"id": "../etc", "type": "paragraph", "text": "Bad"}]},
    ], by="user")
    ids = [r["ids"][0] for r in result["results"]]
    assert ids[0] == "b_abcdefgh"
    assert ids[1] != lead["id"] and ids[2] != "../etc"
    table = _blocks(new, "table")[0]
    col = table["columns"][0]["id"]
    new, _ = apply_ops(new, [{"op": "add_rows", "table": table["id"], "rows": [{"id": "r_zzzzzzzz", "cells": {col: "New"}}]}])
    assert _blocks(new, "table")[0]["rows"][-1] == {"id": "r_zzzzzzzz", "cells": {c["id"]: ("New" if c["id"] == col else "") for c in table["columns"]}}


def test_edit_text_reaches_metric_tiles_and_callout_titles():
    doc = _doc()
    new, _ = apply_ops(doc, [
        {"op": "edit_text", "find": "+18%", "replace": "+21%"},
        {"op": "edit_text", "find": "Legal review", "replace": "Legal sign-off"},
    ])
    metrics = _blocks(new, "metrics")[0]
    assert metrics["items"][0]["delta"] == "+21%"
    assert _blocks(new, "callout")[0]["title"] == "Legal sign-off"


def test_ops_in_one_call_build_on_each_other_without_tripping_the_stale_guard():
    """Regression (found by a model eval): two ops on one table in a call, with if_rev, were refused as stale."""
    doc = _doc()
    table = _blocks(doc, "table")[0]
    new, result = apply_ops(doc, [
        {"op": "add_column", "table": table["id"], "name": "Setup ($)", "kind": "number", "values": ["5", "0", "2"], "index": 5},
        {"op": "add_column", "table": table["id"], "name": "Total ($k)", "kind": "number", "values": ["125", "45", "12"]},
        {"op": "sort_rows", "table": table["id"], "col": "Total ($k)"},
    ], if_rev=doc["rev"])
    t = _blocks(new, "table")[0]
    assert [c["name"] for c in t["columns"]][4:] == ["Setup ($)", "Budget ($k)", "Total ($k)"]
    assert [r["cells"][t["columns"][0]["id"]] for r in t["rows"]] == ["Pricing", "Launch comms", "Connector"]


def test_a_stale_refusal_names_who_changed_the_block():
    doc = _doc()
    lead = doc["tabs"][0]["blocks"][0]
    doc, _ = apply_ops(doc, [{"op": "update", "id": lead["id"], "set": {"text": "By hand."}}], by="user")
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "update", "id": lead["id"], "set": {"text": "x"}}], if_rev=1)
    assert "changed by the person at rev 2" in err.value.message
    assert err.value.detail["rev"] == 2


def test_rewriting_a_section_keeps_its_table_and_the_chart_drawing_from_it():
    doc = build_new_doc("Plan", markdown=(
        "## Costs\n\n| Vendor | Price |\n| --- | ---: |\n| Acme | 10 |\n| Beta | 20 |\n\n"
        "## Chart\n\n```chart\n{\"type\":\"bar\",\"title\":\"Cost\",\"source\":{\"table\":\"previous\",\"x\":\"Vendor\",\"y\":[\"Price\"]}}\n```"))
    heading = _blocks(doc, "heading")[0]
    table = _blocks(doc, "table")[0]
    new, _ = apply_ops(doc, [{"op": "replace", "target": [heading["id"], table["id"]],
                              "markdown": "## Costs\n\nUpdated.\n\n| Vendor | Price |\n| --- | ---: |\n| Acme | 12 |\n| Beta | 20 |\n| Corvid | 9 |"}])
    t = _blocks(new, "table")[0]
    assert t["id"] == table["id"]
    assert _blocks(new, "chart")[0]["source"]["table"] == table["id"]


def test_a_chart_whose_table_id_is_gone_relinks_to_the_table_with_its_columns():
    doc = build_new_doc("Plan", markdown="| Vendor | Price |\n| --- | ---: |\n| Acme | 10 |")
    new, _ = apply_ops(doc, [{"op": "insert", "position": "end", "markdown":
        "```chart\n{\"type\":\"bar\",\"title\":\"Cost\",\"source\":{\"table\":\"b_gone0000\",\"x\":\"Vendor\",\"y\":[\"Price\"]}}\n```"}])
    assert _blocks(new, "chart")[0]["source"]["table"] == _blocks(new, "table")[0]["id"]


def test_renaming_a_column_keeps_it_plotted():
    doc = _doc()
    table = _blocks(doc, "table")[0]
    new, _ = apply_ops(doc, [{"op": "update_column", "table": table["id"], "col": "Budget ($k)", "name": "Budget (k$)"}])
    assert _blocks(new, "chart")[0]["source"]["y"] == ["Budget (k$)"]


def test_an_edit_cannot_point_an_image_at_a_local_file():
    doc = build_new_doc("Pics", markdown="Intro.")
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "insert", "position": "end",
                         "blocks": [{"type": "image", "src": "/Users/me/Pictures/passport.png"}]}])
    assert err.value.code == "bad_block" and "web address" in err.value.message
    # Written as Markdown, the line stays text: no image points at the file.
    new, _ = apply_ops(doc, [{"op": "insert", "position": "end", "markdown": "![x](/Users/me/Pictures/passport.png)"}])
    assert not any(b["type"] == "image" for t in new["tabs"] for b in t["blocks"])


def test_an_image_kept_from_before_still_takes_edits_but_not_a_new_local_src():
    doc = build_new_doc("Pics", markdown="![logo](https://example.com/logo.png)")
    img = next(b for t in doc["tabs"] for b in t["blocks"] if b["type"] == "image")
    img["src"] = "/Users/me/old.png"  # saved before local paths were refused
    new, _ = apply_ops(doc, [{"op": "update", "id": img["id"], "set": {"caption": "Old logo"}}])
    assert next(b for t in new["tabs"] for b in t["blocks"] if b["id"] == img["id"])["caption"] == "Old logo"
    with pytest.raises(DocError):
        apply_ops(new, [{"op": "update", "id": img["id"], "set": {"src": "~/Desktop/scan.png"}}])


FILLED = {"text": "", "type": "", "level": 0, "indent": 0, "done": False, "tone": "", "title": "", "code": "",
          "language": "", "src": "", "alt": "", "caption": "", "chart": "bar", "unit": "", "x_label": "",
          "y_label": "", "source": {"table": "", "x": "", "y": []}, "items": [], "intent": ""}


def _seed_tasks():
    doc = build_new_doc("Tracker", markdown="## Next steps\n\n- [ ] Book the security review\n  - [ ] Pick a date\n- [ ] Send discount wording to legal")
    tasks = [b for t in doc["tabs"] for b in t["blocks"] if b["type"] == "task"]
    return doc, tasks


def test_ticking_a_task_works_when_the_model_fills_in_every_field():
    """Captured from MindsHub Air: it sends every declared field, and the
    refusals ("task blocks do not take level, tone, chart…") looped the turn
    until it ran out of tokens."""
    doc, tasks = _seed_tasks()
    new, _ = apply_ops(doc, [{"op": "update", "id": tasks[0]["id"],
                              "set": {**FILLED, "text": "Book the security review", "type": "task", "done": True}}])
    after = {b["id"]: b for t in new["tabs"] for b in t["blocks"]}
    assert after[tasks[0]["id"]]["done"] is True
    assert after[tasks[0]["id"]]["text"] == "Book the security review"
    assert after[tasks[2]["id"]]["done"] is False


def test_an_empty_filled_in_value_never_overwrites():
    doc = build_new_doc("Notes", markdown="> [!WARNING] Legal review\n> GA slips if legal review runs long.")
    callout = next(b for t in doc["tabs"] for b in t["blocks"] if b["type"] == "callout")
    new, _ = apply_ops(doc, [{"op": "update", "id": callout["id"], "set": {**FILLED, "tone": "danger"}}])
    got = next(b for t in new["tabs"] for b in t["blocks"] if b["id"] == callout["id"])
    assert got["tone"] == "danger"
    assert got["text"] == callout["text"] and got.get("title") == callout.get("title")


def test_unticking_and_outdenting_are_still_real_values():
    doc, tasks = _seed_tasks()
    done, _ = apply_ops(doc, [{"op": "update", "id": tasks[0]["id"], "set": {"done": True}}])
    undone, _ = apply_ops(done, [{"op": "update", "id": tasks[0]["id"], "set": {"done": False}}])
    assert next(b for t in undone["tabs"] for b in t["blocks"] if b["id"] == tasks[0]["id"])["done"] is False
    nested = tasks[1]
    assert nested.get("indent", 0) >= 1
    flat, _ = apply_ops(doc, [{"op": "update", "id": nested["id"], "set": {"indent": 0}}])
    assert next(b for t in flat["tabs"] for b in t["blocks"] if b["id"] == nested["id"]).get("indent", 0) == 0


def test_an_update_with_nothing_this_block_takes_says_what_it_takes():
    doc, tasks = _seed_tasks()
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "update", "id": tasks[0]["id"], "set": {"tone": "danger", "level": 2}}])
    assert "done" in err.value.message and "text" in err.value.message


TOP_FILLED = {"tab": "", "target": "", "markdown": "", "position": "end", "id": "", "ids": [], "set": {}, "find": "",
              "replace": "", "nth": 1, "table": "", "row": 0, "col": 0, "value": "", "rows": [], "index": 0, "name": "",
              "kind": "text", "values": [], "options": [], "descending": False, "title": ""}


def _plain(doc):
    """The doc's content without ids minted on the way (new rows, tabs) and revisions."""
    return [(t["title"], [(b["type"], b.get("text"), [(c["name"], c["kind"], [o["name"] for o in c.get("options", [])])
                                                       for c in b.get("columns", [])],
                           [list(r["cells"].values()) for r in b.get("rows", [])]) for b in t["blocks"]])
            for t in doc["tabs"]]


@pytest.mark.parametrize("name", ["add_rows", "add_column", "rename_status", "status_options", "add_tab", "edit_text_delete", "clear_cell"])
def test_an_op_means_the_same_with_every_field_filled_in(name):
    doc = build_new_doc("T", markdown="Lead. Tail words.\n\n| Task | Owner | Status |\n| --- | --- | --- |\n| Copy | Dana | Open |\n| Checkout | Marco | Done |")
    t = next(b for tab in doc["tabs"] for b in tab["blocks"] if b["type"] == "table")
    op = {
        "add_rows": {"op": "add_rows", "table": t["id"], "rows": [{"Task": "Analytics", "Owner": "Lee", "Status": "Open"}]},
        "add_column": {"op": "add_column", "table": t["id"], "name": "Due", "kind": "date", "values": ["2026-11-10", "2026-11-04"]},
        "rename_status": {"op": "update_column", "table": t["id"], "col": "Status", "name": "State"},
        "status_options": {"op": "update_column", "table": t["id"], "col": "Status", "options": ["Open", "Blocked", "Done"]},
        "add_tab": {"op": "add_tab", "title": "Appendix", "markdown": "More."},
        "edit_text_delete": {"op": "edit_text", "find": " Tail words.", "replace": ""},
        "clear_cell": {"op": "update_cell", "table": t["id"], "row": 0, "col": "Owner", "value": ""},
    }[name]
    intended, _ = apply_ops(doc, [op])
    filled, _ = apply_ops(doc, [{**TOP_FILLED, **op}])
    assert _plain(filled) == _plain(intended)


def test_an_add_goes_at_the_end_unless_a_position_from_1_is_given():
    doc = build_new_doc("T", markdown="| Task |\n| --- |\n| A |\n| B |")
    t = next(b for tab in doc["tabs"] for b in tab["blocks"] if b["type"] == "table")
    end, _ = apply_ops(doc, [{"op": "add_rows", "table": t["id"], "rows": [["C"]]}])
    first, _ = apply_ops(doc, [{"op": "add_rows", "table": t["id"], "rows": [["Z"]], "index": 1}])
    col = t["columns"][0]["id"]
    def rows(d):
        table = next(b for tab in d["tabs"] for b in tab["blocks"] if b["type"] == "table")
        return [r["cells"][col] for r in table["rows"]]

    assert rows(end) == ["A", "B", "C"]
    assert rows(first) == ["Z", "A", "B"]


def test_a_column_still_converts_to_text_when_that_is_all_the_op_asks():
    doc = build_new_doc("T", markdown="| Task | Status |\n| --- | --- |\n| A | Open |")
    t = next(b for tab in doc["tabs"] for b in tab["blocks"] if b["type"] == "table")
    new, _ = apply_ops(doc, [{**TOP_FILLED, "op": "update_column", "table": t["id"], "col": "Status", "kind": "text"}])
    assert next(b for tab in new["tabs"] for b in tab["blocks"] if b["type"] == "table")["columns"][1]["kind"] == "text"


def test_a_table_cannot_grow_past_the_limits_a_load_enforces():
    """Past them the doc would save, then refuse to load, and every later write fail."""
    doc = build_new_doc("Big", markdown="| A |\n| --- |\n" + "| x |\n" * (MAX_TABLE_ROWS - 1))
    table = _blocks(doc, "table")[0]
    apply_ops(doc, [{"op": "add_rows", "table": table["id"], "rows": [["last"]]}])
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "add_rows", "table": table["id"], "rows": [["one"], ["too many"]]}])
    assert err.value.code == "bad_table"

    wide = build_new_doc("Wide", markdown="| " + " | ".join(f"c{i}" for i in range(MAX_TABLE_COLS)) + " |\n"
                         + "| " + " | ".join("---" for _ in range(MAX_TABLE_COLS)) + " |\n")
    with pytest.raises(DocError) as err:
        apply_ops(wide, [{"op": "add_column", "table": _blocks(wide, "table")[0]["id"], "name": "One more"}])
    assert err.value.code == "bad_table"


@pytest.mark.parametrize("op", [
    {"op": "edit_text", "find": "500", "replace": "750", "nth": "x"},
    {"op": "edit_text", "find": "500", "replace": "750", "nth": -1},
    {"op": "replace", "target": {"find": "500", "nth": "first"}, "markdown": "x"},
])
def test_a_bad_occurrence_number_is_refused_not_a_crash(op):
    with pytest.raises(DocError) as err:
        apply_ops(_doc(), [op])
    assert err.value.code == "bad_op" and "nth" in err.value.message


def test_a_refusal_reads_as_its_message_with_the_op_it_came_from():
    with pytest.raises(DocError) as err:
        apply_ops(_doc(), [{"op": "delete", "target": "b_nothere1"}])
    assert str(err.value) == err.value.message and err.value.message.startswith("op 0 (delete")


def test_edit_text_reaches_chart_captions_and_image_alt_text():
    doc = build_new_doc("Pics", markdown=(
        '```chart\n{"type":"bar","title":"Revenue","caption":"Q3 numbers","x":["a"],"series":[{"name":"s","values":[1]}]}\n```\n\n'
        '![A team photo](https://x.com/p.png "Taken in May")'))
    new, _ = apply_ops(doc, [{"op": "edit_text", "find": "Q3 numbers", "replace": "Q4 numbers"},
                             {"op": "edit_text", "find": "team photo", "replace": "group photo"}])
    assert _blocks(new, "chart")[0]["caption"] == "Q4 numbers"
    assert _blocks(new, "image")[0]["alt"] == "A group photo"


# A field of the wrong type, as a model or a client may send it: each op, given
# the doc's table id.
_MALFORMED = {
    "tab_is_a_number": lambda t: {"op": "insert", "tab": 1, "markdown": "x"},
    "find_is_a_number": lambda t: {"op": "edit_text", "find": 5, "replace": "x"},
    "ids_is_a_number": lambda t: {"op": "move", "ids": 5, "position": "end"},
    "values_is_a_number": lambda t: {"op": "add_column", "table": t, "name": "Extra", "values": 5},
    "table_columns_is_a_string": lambda t: {"op": "update", "id": t, "set": {"columns": "x"}},
    "chart_y_is_a_number": lambda t: {"op": "insert", "position": "end", "blocks": [
        {"type": "chart", "source": {"table": t, "x": "Workstream", "y": 5}}]},
    "metrics_items_is_not_a_list": lambda t: {"op": "insert", "position": "end", "blocks": [
        {"type": "metrics", "items": 5}]},
}


@pytest.mark.parametrize("make_op", _MALFORMED.values(), ids=_MALFORMED.keys())
def test_a_malformed_op_is_refused_as_a_bad_op_naming_it_not_a_crash(make_op):
    """It reached the editor's HTTP call as a 500, and the agent as a failed tool call."""
    doc = _doc()
    before = copy.deepcopy(doc)
    op = make_op(_blocks(doc, "table")[0]["id"])
    with pytest.raises(DocError) as err:
        apply_ops(doc, [{"op": "rename_doc", "title": "Renamed"}, op])
    assert err.value.code == "bad_op"
    assert err.value.detail["op"] == 1
    assert err.value.message.startswith(f"op 1 ({op['op']}, counting from 0): ")
    assert doc == before


def test_a_number_column_sorts_what_isnt_a_number_last():
    doc = build_new_doc("Sizes", markdown="| Item | Size |\n| --- | --- |\n| a | 5 |\n| b | 12-15 |\n| c | 1,200 |")
    table = _blocks(doc, "table")[0]
    new, _ = apply_ops(doc, [{"op": "update_column", "table": table["id"], "col": "Size", "kind": "number"},
                             {"op": "sort_rows", "table": table["id"], "col": "Size", "descending": True}])
    t = _block(new, table["id"])
    size = t["columns"][1]["id"]
    assert [r["cells"][size] for r in t["rows"]] == ["1,200", "5", "12-15"]


def test_the_persons_empty_text_empties_the_block():
    """Deleting all of a block's text in the editor sends text "": it must land.
    The agent's filled-in blanks are still sifted out."""
    doc = build_new_doc("Notes", markdown="Keep.\n\nTemporary words.")
    para = _blocks(doc, "paragraph")[1]
    op = {"op": "update", "id": para["id"], "set": {"text": ""}}
    new, _ = apply_ops(doc, [op], by="user")
    assert _block(new, para["id"])["text"] == ""
    with pytest.raises(DocError, match="nothing to change"):
        apply_ops(doc, [op], by="agent")


def test_a_slash_command_line_becomes_its_block_from_the_editor():
    """The slash menu inserts the picked block and clears the "/table" line in one batch."""
    doc = build_new_doc("Notes", markdown="Intro.\n\n/table")
    line = _blocks(doc, "paragraph")[1]
    table = {"id": "b_new_tbl1", "type": "table", "columns": [{"id": "c_name0001", "name": "Name", "kind": "text"}],
             "rows": [{"id": "r_row00001", "cells": {"c_name0001": ""}}]}
    new, _ = apply_ops(doc, [{"op": "insert", "position": {"before": line["id"]}, "blocks": [table]},
                             {"op": "update", "id": line["id"], "set": {"text": ""}}], by="user")
    assert [b["type"] for b in new["tabs"][0]["blocks"]] == ["paragraph", "table", "paragraph"]
    assert _block(new, line["id"])["text"] == ""
    turned, _ = apply_ops(doc, [{"op": "update", "id": line["id"], "set": {"type": "task", "text": ""}}], by="user")
    assert _block(turned, line["id"])["type"] == "task" and _block(turned, line["id"])["text"] == ""


def test_turning_a_column_into_dates_stores_them_as_iso_and_sorts_by_date():
    """Found in QA: written-out dates stayed strings, sorted alphabetically, and the
    editor's date picker (ISO only) blanked them."""
    doc = build_new_doc("Plan", markdown="| Step | When |\n| --- | --- |\n| Ship | December 2, 2026 |\n"
                                         "| Scope | November 4, 2026 |\n| Review | November 13, 2026 |\n| Later | TBD |")
    table = _blocks(doc, "table")[0]
    assert table["columns"][1]["kind"] == "text"  # "TBD" isn't a date
    new, _ = apply_ops(doc, [{"op": "update_column", "table": table["id"], "col": "When", "kind": "date"},
                             {"op": "sort_rows", "table": table["id"], "col": "When"}], by="user")
    t = _block(new, table["id"])
    when = t["columns"][1]["id"]
    assert [r["cells"][when] for r in t["rows"]] == ["2026-11-04", "2026-11-13", "2026-12-02", "TBD"]
    newest, _ = apply_ops(new, [{"op": "sort_rows", "table": table["id"], "col": "When", "descending": True}])
    assert [r["cells"][when] for r in _block(newest, table["id"])["rows"]] == \
        ["2026-12-02", "2026-11-13", "2026-11-04", "TBD"]


def test_a_date_typed_into_a_date_column_is_stored_as_iso():
    doc = build_new_doc("Plan", markdown="| Step | When |\n| --- | --- |\n| Ship | 2026-12-02 |")
    table = _blocks(doc, "table")[0]
    new, _ = apply_ops(doc, [{"op": "update_cell", "table": table["id"], "row": 0, "col": "When", "value": "Dec 3, 2026"}])
    t = _block(new, table["id"])
    assert t["rows"][0]["cells"][t["columns"][1]["id"]] == "2026-12-03"


_MONTHS_TABLE = "| Month | Revenue (USD) |\n| --- | --- |\n| Jan 2026 | 279,750 |\n| Feb 2026 | 281,120 |\n| Mar 2026 | 320,390 |"


def _chart_md(values):
    import json
    return "```chart\n" + json.dumps({"type": "line", "title": "Monthly revenue", "x": ["Jan", "Feb", "Mar"],
                                      "series": [{"name": "Revenue", "values": values}]}) + "\n```"


def test_a_chart_the_agent_copies_from_a_table_draws_from_that_table():
    """Found in QA: the CSV summary's chart carried its own copy of the monthly
    table's numbers, so editing the table left the chart behind."""
    doc = build_new_doc("Sales", markdown="Intro.")
    new, _ = apply_ops(doc, [{"op": "insert", "position": "end",
                              "markdown": _MONTHS_TABLE + "\n\n" + _chart_md([279750, 281120, 320390])}])
    table, chart = _blocks(new, "table")[0], _blocks(new, "chart")[0]
    assert chart["source"] == {"table": table["id"], "x": "Month", "y": ["Revenue (USD)"]}
    assert "series" not in chart and "x" not in chart

    created = build_new_doc("Sales", markdown=_MONTHS_TABLE + "\n\n" + _chart_md([279750, 281120, 320390]))
    assert _blocks(created, "chart")[0]["source"]["table"] == _blocks(created, "table")[0]["id"]


def test_a_chart_keeps_its_own_numbers_when_no_table_holds_them_or_the_person_wrote_it():
    doc = build_new_doc("Sales", markdown="Intro.")
    other, _ = apply_ops(doc, [{"op": "insert", "position": "end",
                                "markdown": _MONTHS_TABLE + "\n\n" + _chart_md([1, 2, 3])}])
    assert _blocks(other, "chart")[0]["series"][0]["values"] == [1, 2, 3]
    mine, _ = apply_ops(doc, [{"op": "insert", "position": "end",
                               "markdown": _MONTHS_TABLE + "\n\n" + _chart_md([279750, 281120, 320390])}], by="user")
    assert not _blocks(mine, "chart")[0].get("source")
