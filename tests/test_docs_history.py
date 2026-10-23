"""Version history: what changed between versions, the history listed, a version read, restored and undone."""
from __future__ import annotations

import copy
import json
import time

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.docs import history, store
from cowork.docs.changes import describe
from cowork.docs.model import DocError
from cowork.docs.ops import apply_ops, build_new_doc
from cowork.docs.service import apply, create_doc
from cowork.docs.store import DOC_FILENAME, DOC_HTML, list_history, load_doc
from cowork.services.projects import GENERAL_PROJECT_ID

BUDGET = ("| Channel | Budget ($K) | Signups |\n| --- | --- | --- |\n"
          "| Paid search | 72 | 1050 |\n| Webinar | 28 | 520 |\n| Retargeting | 40 | 160 |")
PAIRS = "| Channel | Budget ($K) |\n| --- | --- |\n| Paid search | 72 |\n| Webinar | 28 |"


def _doc(*tabs: tuple[str, str]):
    return build_new_doc("Launch plan", tabs=[{"title": t, "markdown": md} for t, md in tabs])


def _after(doc, ops, tab=None):
    return apply_ops(copy.deepcopy(doc), ops, tab=tab, by="user")[0]


def _table(doc, tab=0):
    return next(b for b in doc["tabs"][tab]["blocks"] if b["type"] == "table")


def _cell(doc, row, col, value, tab=0):
    table = _table(doc, tab)
    return {"op": "update_cell", "table": table["id"], "row": table["rows"][row]["id"], "col": col, "value": value}


# ---------------------------------------------------------------- what changed


def test_a_cell_names_its_row_and_both_values():
    old = _doc(("Budget", PAIRS))
    assert describe(old, _after(old, [_cell(old, 0, "Budget ($K)", "80")])) == "Paid search 72 → 80"


def test_in_a_wider_table_a_cell_also_names_its_column():
    old = _doc(("Budget", BUDGET))
    assert describe(old, _after(old, [_cell(old, 0, "Budget ($K)", "80")])) == "Paid search · Budget ($K) 72 → 80"


def test_a_change_in_a_doc_with_tabs_says_which_tab():
    old = _doc(("Overview", "Lead."), ("Budget", PAIRS))
    new = _after(old, [_cell(old, 0, "Budget ($K)", "80", tab=1)], tab=old["tabs"][1]["id"])
    assert describe(old, new) == "Budget: Paid search 72 → 80"


def test_more_cells_are_counted_as_cells_and_anything_else_as_changes():
    old = _doc(("Budget", BUDGET))
    cells = _after(old, [_cell(old, 0, "Budget ($K)", "80"), _cell(old, 1, "Budget ($K)", "30"),
                         _cell(old, 2, "Signups", "200")])
    assert describe(old, cells) == "Paid search · Budget ($K) 72 → 80, and 2 more cells"
    lead = old["tabs"][0]
    mixed = _after(old, [_cell(old, 0, "Budget ($K)", "80"),
                         {"op": "insert", "markdown": "A note under the table.", "after": _table(old)["id"]}],
                   tab=lead["id"])
    assert describe(old, mixed) == "Paid search · Budget ($K) 72 → 80, and 1 more change"


def test_a_cell_set_or_cleared_says_so():
    old = _doc(("Budget", PAIRS))
    cleared = _after(old, [_cell(old, 0, "Budget ($K)", "")])
    assert describe(old, cleared) == "Paid search cleared"
    assert describe(cleared, old) == "Paid search set to 72"


def test_rows_added_and_removed_are_named_by_their_first_column():
    old = _doc(("Budget", PAIRS))
    table = _table(old)
    added = _after(old, [{"op": "add_rows", "table": table["id"], "rows": [["Podcast ads", "15"]]}])
    assert describe(old, added) == "Added row “Podcast ads”"
    several = _after(old, [{"op": "add_rows", "table": table["id"], "rows": [["A", "1"], ["B", "2"]]}])
    assert describe(old, several) == "Added 2 rows"
    removed = _after(old, [{"op": "delete_rows", "table": table["id"], "rows": [table["rows"][1]["id"]]}])
    assert describe(old, removed) == "Removed row “Webinar”"


def test_a_renamed_row_or_column_reads_old_to_new():
    old = _doc(("Budget", PAIRS))
    table = _table(old)
    assert describe(old, _after(old, [_cell(old, 0, "Channel", "Search ads")])) == "Renamed row “Paid search” → “Search ads”"
    renamed = _after(old, [{"op": "update_column", "table": table["id"], "col": "Budget ($K)", "name": "Spend ($K)"}])
    assert describe(old, renamed) == "Renamed “Budget ($K)” → “Spend ($K)”"


def test_columns_added_removed_and_changed_in_kind():
    old = _doc(("Budget", PAIRS))
    table = _table(old)
    added = _after(old, [{"op": "add_column", "table": table["id"], "name": "Owner"}])
    assert describe(old, added) == "Added column “Owner”"
    assert describe(added, old) == "Removed column “Owner”"
    kind = _after(old, [{"op": "update_column", "table": table["id"], "col": "Channel", "kind": "status"}])
    assert describe(old, kind) == "Made “Channel” a status column"


def test_a_sorted_table_says_it_was_sorted():
    old = _doc(("Budget", PAIRS))
    table = _table(old)
    assert describe(old, _after(old, [{"op": "sort_rows", "table": table["id"], "col": "Budget ($K)"}])) == "Sorted a table"


def test_edited_text_is_quoted_from_its_start_and_cut_at_a_word():
    old = _doc(("Plan", "Spend is weighted to the first two weeks of the launch window."))
    lead = old["tabs"][0]["blocks"][0]
    new = _after(old, [{"op": "edit_text", "id": lead["id"], "find": "two", "replace": "three"}])
    assert describe(old, new) == "Edited “Spend is weighted to the…”"


def test_markdown_marks_links_and_dates_read_as_they_show():
    old = _doc(("Plan", "Old."))
    lead = old["tabs"][0]["blocks"][0]
    new = _after(old, [{"op": "update", "id": lead["id"],
                        "set": {"text": "**Ship** `v2` {{date:2026-11-18}} [now](https://x.test)"}}])
    assert describe(old, new) == "Edited “Ship v2 2026-11-18 now”"


def test_a_ticked_task_says_ticked():
    old = _doc(("Plan", "- [ ] Book the security review"))
    task = old["tabs"][0]["blocks"][0]
    ticked = _after(old, [{"op": "update", "id": task["id"], "set": {"done": True}}])
    assert describe(old, ticked) == "Ticked “Book the security review”"
    assert describe(ticked, old) == "Unticked “Book the security review”"


def test_blocks_added_and_removed_are_named_by_what_they_are():
    old = _doc(("Plan", "Lead."))
    heading = _after(old, [{"op": "insert", "markdown": "## Risks\n\nReview slips."}])
    assert describe(old, heading) == "Added “Risks”, and 1 more change"
    chart = _after(old, [{"op": "insert", "markdown": '```chart\n{"type":"line","title":"Signups","x":["a","b"],'
                                                       '"series":[{"name":"s","values":[1,2]}]}\n```'}])
    assert describe(old, chart) == "Added the chart “Signups”"
    assert describe(chart, old) == "Removed the chart “Signups”"


def test_empty_lines_and_sections_still_to_write_are_not_changes():
    old = _doc(("Plan", "Lead.\n\n{{pending: Risks — what could slip}}"))
    pending = old["tabs"][0]["blocks"][1]
    blank = copy.deepcopy(old)
    blank["tabs"][0]["blocks"].append({"id": "b_blank000", "type": "paragraph", "v": 2, "text": "  "})
    assert describe(old, blank) == "No visible changes"
    filled = _after(old, [{"op": "replace", "target": pending["id"], "markdown": "## Risks\n\nReview slips."}])
    assert describe(old, filled) == "Added “Risks”, and 1 more change"


def test_only_bookkeeping_changed_is_no_visible_change():
    old = _doc(("Plan", "Lead."))
    new = copy.deepcopy(old)
    for block in new["tabs"][0]["blocks"]:
        block["v"] = 9
        block["by"] = "agent"
    assert describe(old, new) == "No visible changes"


def test_tabs_added_removed_renamed_and_moved():
    old = _doc(("Overview", "Lead."), ("Budget", PAIRS))
    added = _after(old, [{"op": "add_tab", "title": "Risks", "markdown": "Review slips."}])
    assert describe(old, added) == "Added the Risks tab"
    assert describe(added, old) == "Removed the Risks tab"
    renamed = _after(old, [{"op": "rename_tab", "tab": old["tabs"][1]["id"], "title": "Spend"}])
    assert describe(old, renamed) == "Renamed the Budget tab to Spend"
    moved = _after(old, [{"op": "move_tab", "tab": old["tabs"][1]["id"], "index": 1}])
    assert describe(old, moved) == "Moved the Budget tab"


def test_a_one_tab_doc_talks_about_the_doc_not_its_tab():
    old = _doc(("Plan", "Lead."))
    new = _after(old, [{"op": "rename_doc", "title": "Launch"}])
    assert describe(old, new) == "Renamed the doc “Launch plan” → “Launch”"


def test_changes_in_several_tabs_are_counted_across_them():
    old = _doc(("Overview", "Lead."), ("Budget", PAIRS), ("Risks", "Slips."), ("Notes", "Pilot."))
    def lead(n):
        return old["tabs"][n]["blocks"][0]["id"]
    two = _after(old, [{"op": "update", "id": lead(0), "set": {"text": "New lead."}, "tab": old["tabs"][0]["id"]},
                       {"op": "update", "id": lead(2), "set": {"text": "Fewer slips."}, "tab": old["tabs"][2]["id"]}])
    assert describe(old, two) == "2 changes across Overview and Risks"
    four = _after(two, [{"op": "update", "id": lead(3), "set": {"text": "More pilots."}, "tab": old["tabs"][3]["id"]},
                        _cell(old, 0, "Budget ($K)", "80", tab=1) | {"tab": old["tabs"][1]["id"]}])
    assert describe(old, four) == "4 changes across Overview, Budget and 2 more tabs"


def test_a_moved_block_is_the_one_named_whichever_way_it_went():
    old = _doc(("Plan", "Alpha.\n\nBravo.\n\nCharlie."))
    a, _b, c = (b["id"] for b in old["tabs"][0]["blocks"])
    down = _after(old, [{"op": "move", "id": a, "position": {"after": c}}])
    assert describe(old, down) == "Moved “Alpha.”"
    up = _after(old, [{"op": "move", "id": c, "position": {"before": a}}])
    assert describe(old, up) == "Moved “Charlie.”"


def test_metrics_name_the_tile_that_changed():
    md = '```metrics\n[{"label":"Launch budget","value":"$160K"},{"label":"Days to launch","value":"26"}]\n```'
    old = _doc(("Plan", md))
    tiles = old["tabs"][0]["blocks"][0]
    items = [dict(tiles["items"][0], value="$186K"), tiles["items"][1]]
    assert describe(old, _after(old, [{"op": "update", "id": tiles["id"], "set": {"items": items}}])) == \
        "Launch budget $160K → $186K"
    delta = [dict(tiles["items"][0], delta="+16%"), tiles["items"][1]]
    assert describe(old, _after(old, [{"op": "update", "id": tiles["id"], "set": {"items": delta}}])) == \
        "Edited the metrics"


# ---------------------------------------------------------------- the history listed


@pytest.fixture
def made(tmp_path):
    return create_doc(tmp_path / "artifacts", title="Launch plan", by="agent", turn="reply-1",
                      tabs=[{"title": "Overview", "markdown": "Lead."}, {"title": "Budget", "markdown": PAIRS}])


def _type(folder, text):
    lead = load_doc(folder)["tabs"][0]["blocks"][0]
    return apply(folder, [{"op": "update", "id": lead["id"], "set": {"text": text}}], by="user")[0]


def _budget_cell(folder, value, *, by="user", turn=None):
    doc = load_doc(folder)
    tab = doc["tabs"][1]
    table = next(b for b in tab["blocks"] if b["type"] == "table")
    op = {"op": "update_cell", "table": table["id"], "row": table["rows"][0]["id"], "col": "Budget ($K)", "value": value}
    return apply(folder, [op], tab=tab["id"], by=by, turn=turn)[0]


def test_a_new_doc_is_its_first_draft_with_nothing_before_it(made):
    listed = history.list_versions(made.folder)
    assert listed["versions"] == []
    assert listed["current"]["summary"] == "First draft · 2 tabs"
    assert listed["current"]["by"] == "agent" and listed["current"]["rev"] == 1


def test_each_version_says_who_made_it_when_and_what_changed_newest_first(made, monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(store.time, "time", lambda: now[0])
    _type(made.folder, "New lead.")                       # ends the agent's draft
    now[0] += 6 * 60
    _budget_cell(made.folder, "80")                       # a pause: a new session
    _budget_cell(made.folder, "90", by="agent", turn="reply-2")
    listed = history.list_versions(made.folder)
    assert [(v["by"], v["summary"]) for v in listed["versions"]] == [
        ("user", "Budget: Paid search 72 → 80"),
        ("user", "Overview: edited “New lead.”"),
        ("agent", "First draft · 2 tabs"),
    ]
    assert listed["current"] == {"rev": 4, "at": listed["current"]["at"], "by": "agent",
                                 "summary": "Budget: Paid search 80 → 90"}
    newest, older = listed["versions"][0], listed["versions"][1]
    assert store._seconds_since(older["at"], now[0]) == pytest.approx(6 * 60, abs=1)
    assert newest["at"] >= older["at"]


def test_a_limit_shows_the_newest_and_still_describes_them(made):
    for n in range(4):
        _budget_cell(made.folder, str(n), by="agent", turn=f"reply-{n + 2}")
    listed = history.list_versions(made.folder, limit=2)
    assert [v["summary"] for v in listed["versions"]] == ["Budget: Paid search 1 → 2", "Budget: Paid search 0 → 1"]


def test_what_changed_between_two_kept_versions_is_worked_out_once(made, monkeypatch):
    _type(made.folder, "One.")
    _budget_cell(made.folder, "80", by="agent", turn="reply-2")
    calls = []
    real = history.describe
    monkeypatch.setattr(history, "describe", lambda a, b: calls.append(1) or real(a, b))
    history.list_versions(made.folder)
    assert len(calls) == 2  # the kept pair, and the doc now against the newest version
    history.list_versions(made.folder)
    assert len(calls) == 2  # a rev never changes, so neither does what changed between two
    _budget_cell(made.folder, "90", by="agent", turn="reply-2")
    history.list_versions(made.folder)
    assert len(calls) == 3  # only the doc now, at its new rev


def test_a_listing_reads_only_the_versions_a_new_summary_needs(made, monkeypatch):
    for n in range(4):
        _budget_cell(made.folder, str(n), by="agent", turn=f"reply-{n + 2}")
    read = []
    real = history._load_version
    monkeypatch.setattr(history, "_load_version", lambda folder, rev: read.append(rev) or real(folder, rev))
    history.list_versions(made.folder)
    assert sorted(read) == [1, 2, 3, 4]  # every kept version once; the doc now is already in hand
    read.clear()
    history.list_versions(made.folder)
    assert read == []  # all of it worked out already


def test_the_oldest_version_left_after_older_ones_went_has_nothing_to_compare(made, monkeypatch):
    monkeypatch.setattr(store, "HISTORY_KEEP", 2)
    for n in range(4):
        _budget_cell(made.folder, str(n), by="agent", turn=f"reply-{n + 2}")
    versions = history.list_versions(made.folder)["versions"]
    assert [v["summary"] for v in versions] == ["Budget: Paid search 1 → 2", None]


def test_reading_a_version_gives_the_doc_as_it_was(made):
    _type(made.folder, "Typed.")
    (kept,) = list_history(made.folder)
    assert history.read_version(made.folder, kept["rev"])["tabs"][0]["blocks"][0]["text"] == "Lead."


@pytest.mark.parametrize("rev", [0, 2, 99, -1])
def test_a_version_that_is_not_kept_is_refused(made, rev):
    _type(made.folder, "Typed.")
    with pytest.raises(DocError) as err:
        history.read_version(made.folder, rev)
    assert err.value.code == "no_version"


def test_a_version_file_the_index_does_not_list_is_not_read(made):
    """The index is what is kept: a file left over from a version let go is not offered back."""
    _type(made.folder, "Typed.")
    hist = made.folder.joinpath(*store.HISTORY_DIR)
    (hist / "0000099.json").write_text((hist / "0000001.json").read_text())
    with pytest.raises(DocError):
        history.read_version(made.folder, 99)


# ---------------------------------------------------------------- restoring


def test_restoring_brings_the_version_back_as_a_new_write(made):
    _type(made.folder, "Typed.")
    before = load_doc(made.folder)
    doc, replaced = history.restore(made.folder, 1)
    assert doc["tabs"][0]["blocks"][0]["text"] == "Lead."
    assert doc["id"] == before["id"] and doc["rev"] == before["rev"] + 1 == 3
    assert {b["v"] for t in doc["tabs"] for b in t["blocks"]} == {3}
    assert replaced == before["rev"]
    assert load_doc(made.folder) == doc
    assert "Lead." in (made.folder / DOC_HTML).read_text()


def test_nothing_is_lost_and_restoring_the_replaced_version_undoes_it(made):
    typed = _type(made.folder, "Typed.")
    _doc, replaced = history.restore(made.folder, 1)
    assert history.read_version(made.folder, replaced)["tabs"] == typed["tabs"]
    undone, _ = history.restore(made.folder, replaced)
    assert undone["tabs"][0]["blocks"][0]["text"] == "Typed."


def test_a_restore_is_a_version_of_its_own_and_says_where_it_came_from(made):
    _type(made.folder, "Typed.")
    history.restore(made.folder, 1)
    listed = history.list_versions(made.folder)
    assert listed["current"]["restoredFrom"]["rev"] == 1
    _type(made.folder, "Typed straight after.")  # no pause, the same person: still its own version
    listed = history.list_versions(made.folder)
    assert listed["versions"][0]["restoredFrom"]["rev"] == 1
    assert listed["current"]["summary"] == "Overview: edited “Typed straight after.”"
    assert "restoredFrom" not in listed["current"]


def test_an_edit_made_against_the_doc_before_a_restore_is_refused(made):
    typed = _type(made.folder, "Typed.")
    history.restore(made.folder, 1)
    lead = typed["tabs"][0]["blocks"][0]
    with pytest.raises(DocError) as err:
        apply(made.folder, [{"op": "update", "id": lead["id"], "set": {"text": "Agent, mid-reply."}}],
              by="agent", if_rev=typed["rev"], turn="reply-2")
    assert err.value.code == "stale"


def test_restoring_a_version_that_is_not_kept_writes_nothing(made):
    before = (made.folder / DOC_FILENAME).read_bytes()
    with pytest.raises(DocError):
        history.restore(made.folder, 7)
    assert (made.folder / DOC_FILENAME).read_bytes() == before


# ---------------------------------------------------------------- over HTTP


@pytest.fixture
def client(monkeypatch, granted_product_permissions):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


@pytest.fixture
def doc(client):
    res = client.post("/api/v1/docs/", json={"projectId": str(GENERAL_PROJECT_ID), "title": "Plan", "markdown": "Lead."})
    assert res.status_code == 200, res.text
    return res.json()


def _url(doc, suffix=""):
    return f"/api/v1/docs/{GENERAL_PROJECT_ID}/{doc['artifactId']}{suffix}"


@pytest.fixture
def later(monkeypatch):
    """Move the store's clock past a pause, so the next write starts a new version."""
    now = [time.time()]
    monkeypatch.setattr(store.time, "time", lambda: now[0])

    def pause():
        now[0] += store.SESSION_PAUSE_S + 1
    return pause


def test_list_read_restore_and_undo_over_http(client, doc, later):
    later()
    lead = doc["doc"]["tabs"][0]["blocks"][0]
    client.post(_url(doc, "/ops"), json={"ops": [{"op": "update", "id": lead["id"], "set": {"text": "Typed."}}]})
    listed = client.get(_url(doc, "/history"))
    assert listed.status_code == 200, listed.text
    body = listed.json()
    (first,) = body["versions"]
    assert first["by"] == "user" and first["summary"] == "First draft"  # a doc the person started
    assert body["current"]["summary"] == "Edited “Typed.”"

    version = client.get(_url(doc, f"/history/{first['rev']}"))
    assert version.status_code == 200 and version.json()["doc"]["tabs"][0]["blocks"][0]["text"] == "Lead."

    restored = client.post(_url(doc, f"/history/{first['rev']}/restore"), json={"clientId": "tab-1"})
    assert restored.status_code == 200, restored.text
    assert restored.json()["doc"]["tabs"][0]["blocks"][0]["text"] == "Lead."
    undo = client.post(_url(doc, f"/history/{restored.json()['replacedRev']}/restore"))
    assert undo.status_code == 200, undo.text
    assert client.get(_url(doc)).json()["doc"]["tabs"][0]["blocks"][0]["text"] == "Typed."


def test_a_restore_is_announced_to_the_open_editors(client, doc, monkeypatch, later):
    from cowork.api.v1.endpoints import docs

    later()
    heard = []
    monkeypatch.setattr(docs, "after_write", lambda folder, d, result, **kw: heard.append((d["rev"], kw["by"],
                                                                                         kw["client_id"])))
    lead = doc["doc"]["tabs"][0]["blocks"][0]
    client.post(_url(doc, "/ops"), json={"ops": [{"op": "update", "id": lead["id"], "set": {"text": "Typed."}}]})
    client.post(_url(doc, "/history/1/restore"), json={"clientId": "tab-1"})
    assert heard[-1] == (3, "user", "tab-1")


@pytest.mark.parametrize("path,code", [
    ("/history/9", 404), ("/history/abc", 422), ("/history/-1", 404), ("/history/9/restore", 404),
])
def test_a_version_that_does_not_exist_is_a_404_and_a_bad_rev_a_422(client, doc, path, code):
    method = client.post if path.endswith("restore") else client.get
    res = method(_url(doc, path))
    assert res.status_code == code, res.text


@pytest.mark.parametrize("limit", [0, 101])
def test_the_list_limit_is_bounded(client, doc, limit):
    assert client.get(_url(doc, f"/history?limit={limit}")).status_code == 422


@pytest.mark.parametrize("route", ["doc_history", "doc_version", "restore_version"])
def test_only_the_owner_sees_or_restores_versions(monkeypatch, tmp_path, route):
    """A version can hold what was later taken out: a reviewer who reads the doc gets none of it."""
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from cowork.api.v1.endpoints import docs

    folder = create_doc(tmp_path, title="Plan", markdown="Secret.", by="user").folder
    _type(folder, "Public.")
    before = (folder / DOC_FILENAME).read_bytes()

    async def allowed(scope, permission):
        return None

    monkeypatch.setattr(docs, "require_product_permission", allowed)
    monkeypatch.setattr(docs, "review_artifact_for_request", lambda *a: (SimpleNamespace(), folder, {}, False))
    session = SimpleNamespace(scope=None)
    call = {"doc_history": lambda: docs.doc_history("local", "0" * 32, session, limit=30),
            "doc_version": lambda: docs.doc_version("local", "0" * 32, 1, session),
            "restore_version": lambda: docs.restore_version("local", "0" * 32, 1, session, None)}[route]
    with pytest.raises(HTTPException) as err:
        asyncio.run(call())
    assert err.value.status_code == 403
    assert (folder / DOC_FILENAME).read_bytes() == before


def test_a_restore_needs_permission_to_manage_artifacts(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from cowork.api.v1.endpoints import docs

    folder = create_doc(tmp_path, title="Plan", markdown="Lead.", by="user").folder
    _type(folder, "Typed.")

    async def refused(scope, permission):
        assert permission == "artifact.manage"
        raise HTTPException(status_code=403, detail="no")

    monkeypatch.setattr(docs, "require_product_permission", refused)
    with pytest.raises(HTTPException):
        asyncio.run(docs.restore_version("local", "0" * 32, 1, SimpleNamespace(scope=None), None))
    assert json.loads((folder / DOC_FILENAME).read_text())["tabs"][0]["blocks"][0]["text"] == "Typed."
