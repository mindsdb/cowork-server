"""The agent's doc tools, end to end: create, fill section by section, edit, read."""
from __future__ import annotations

import asyncio
import json
import re
import threading
from types import SimpleNamespace

import pytest

from cowork.docs.hub import conv_topic, doc_topic, hub
from cowork.docs.model import MAX_CHART_SERIES, MAX_METRICS
from cowork.docs.prompt import DOC_PROMPT
from cowork.docs.store import DOC_FILENAME, doc_lock, find_doc_folder, list_doc_folders, list_history, load_doc, open_session
from cowork.harnesses.anton_harness.doc_tools import DOC_TOOL_NAMES, DocToolContext, build_doc_tools


def _text(outcome) -> str:
    return getattr(outcome, "content", outcome)


def _ok(outcome) -> bool:
    return getattr(outcome, "ok", True) is not False


@pytest.fixture
def tools(tmp_path):
    artifacts = tmp_path / ".anton" / "artifacts"
    ctx = DocToolContext(artifacts_base=artifacts, conversation_id="c0ffee00-0000-4000-8000-000000000001")
    defs = {t.name: t for t in build_doc_tools(ctx)}
    session = SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=artifacts), _artifacts_touched=set(),
                              _session_id="", _turn_count=0)
    return defs, session, artifacts


async def _drain(gen):
    """A doc tool streams one or more progress steps, then its result."""
    from anton.core.tools.progress import ToolProgress

    steps, result = [], None
    async for item in gen:
        if isinstance(item, ToolProgress):
            steps.append(item.text)
        else:
            result = item
    return steps, result


def _call(defs, session, name, **args):
    return asyncio.run(_drain(defs[name].handler(session, args)))[1]


def _steps(defs, session, name, **args):
    return asyncio.run(_drain(defs[name].handler(session, args)))[0]


def test_the_four_tools_are_built_and_create_doc_carries_the_prompt(tools):
    defs, _session, _ = tools
    assert tuple(defs) == DOC_TOOL_NAMES
    assert "## Docs" in defs["create_doc"].prompt
    for name in DOC_TOOL_NAMES:
        assert defs[name].input_schema["type"] == "object"


def test_outline_then_fill_one_section_per_call(tools):
    defs, session, artifacts = tools
    out = _call(defs, session, "create_doc", title="Vendor shortlist",
                markdown="Acme is the pick: lowest cost at full coverage.\n\n"
                         "{{pending: Comparison — price, coverage, contract terms}}\n\n"
                         "{{pending: Risks — what could slip}}")
    text = _text(out)
    assert _ok(out), text
    assert "Created doc “Vendor shortlist”" in text
    pending = re.findall(r"(b_[a-z0-9]{8})  pending", text)
    assert len(pending) == 2

    folder = next(artifacts.iterdir())
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["type"] == "document" and meta["primary"] == DOC_FILENAME
    assert folder.name in session._artifacts_touched

    out = _call(defs, session, "edit_doc", doc="Vendor shortlist", if_rev=1, ops=[{
        "op": "replace", "target": pending[0],
        "markdown": "## Comparison\n\n| Vendor | Price ($/yr) | Coverage |\n| --- | --- | --- |\n"
                    "| Acme | 12,000 | Full |\n| Beta | 15,500 | Partial |",
    }])
    assert _ok(out), _text(out)
    assert "now rev 2" in _text(out)
    assert "table" in _text(out)
    assert pending[1] in _text(out)  # the section still to write is named

    doc = load_doc(folder)
    table = next(b for b in doc["tabs"][0]["blocks"] if b["type"] == "table")
    assert [c["kind"] for c in table["columns"]] == ["text", "number", "text"]
    # One reply is one version: kept whole once the next session starts, not one per call.
    assert list_history(folder) == []
    assert open_session(folder)["by"] == "agent" and open_session(folder)["turn"]


def test_read_doc_returns_editable_ids_and_the_rev(tools):
    defs, session, _ = tools
    _call(defs, session, "create_doc", title="Notes", markdown="## Agenda\n\n- [ ] Budget\n- [ ] Hiring")
    full = _text(_call(defs, session, "read_doc", doc="Notes"))
    assert "rev 1" in full
    task_id = re.search(r"\[(b_[a-z0-9]{8})\] task\n- \[ \] Budget", full).group(1)
    out = _call(defs, session, "edit_doc", doc="Notes", ops=[{"op": "update", "id": task_id, "set": {"done": True}}])
    assert _ok(out)
    assert "- [x] Budget" in _text(_call(defs, session, "read_doc", doc="Notes"))
    outline = _text(_call(defs, session, "read_doc", doc="Notes", view="outline"))
    assert "task [x]" in outline


def test_an_edit_that_rewrites_a_block_names_the_words_it_took_out(tools):
    defs, session, _ = tools
    _call(defs, session, "create_doc", title="Hiring", markdown=(
        "## Risk\n\n> [!WARNING] Referral pipeline dried up.\n"
        "> Track referral-sourced candidates against the role targets.\n\nOwners meet on **Fridays at ten**."))
    full = _text(_call(defs, session, "read_doc", doc="Hiring"))
    callout = re.search(r"\[(b_[a-z0-9]{8})\] callout", full).group(1)

    out = _text(_call(defs, session, "edit_doc", doc="Hiring", ops=[{
        "op": "replace", "target": callout,
        "markdown": "> [!WARNING] Referral pipeline dried up.\n> Mitigation owner: Priya.",
    }]))
    assert "Words this edit took out: “Track referral-sourced candidates against the role targets.”" in out

    # Kept words, words moved to another block in the same call, and emphasis
    # dropped from words still there are not taken out.
    full = _text(_call(defs, session, "read_doc", doc="Hiring"))
    callout = re.search(r"\[(b_[a-z0-9]{8})\] callout", full).group(1)
    para = re.search(r"\[(b_[a-z0-9]{8})\] paragraph", full).group(1)
    out = _text(_call(defs, session, "edit_doc", doc="Hiring", ops=[
        {"op": "replace", "target": callout,
         "markdown": "> [!WARNING] Referral pipeline dried up.\n> Mitigation owner: Priya. Recovery: a bonus push."},
        {"op": "replace", "target": para, "markdown": "Mitigation owner: Priya. Owners meet on Fridays at ten."},
    ]))
    assert "took out" not in out


def test_filling_a_pending_section_takes_nothing_out(tools):
    defs, session, _ = tools
    text = _text(_call(defs, session, "create_doc", title="Plan", markdown="{{pending: Goals — the three outcomes}}"))
    pending = re.search(r"(b_[a-z0-9]{8})  pending", text).group(1)
    out = _text(_call(defs, session, "edit_doc", doc="Plan", ops=[
        {"op": "replace", "target": pending, "markdown": "## Goals\n\nShip it."}]))
    assert "took out" not in out


def test_a_filled_in_rev_of_zero_names_no_read_and_a_word_is_refused(tools):
    defs, session, _ = tools
    _call(defs, session, "create_doc", title="Plan", markdown="Lead.")
    lead = re.search(r"(b_[a-z0-9]{8})  p", _text(_call(defs, session, "read_doc", doc="Plan", view="outline"))).group(1)
    # Docs start at rev 1, so 0 is a model's filled-in default, not a read to guard against.
    out = _call(defs, session, "edit_doc", doc="Plan", if_rev=0,
                ops=[{"op": "update", "id": lead, "set": {"text": "New lead."}}])
    assert _ok(out), _text(out)
    # A rev that isn't a number used to be dropped silently, turning the stale guard off.
    out = _call(defs, session, "edit_doc", doc="Plan", if_rev="latest",
                ops=[{"op": "update", "id": lead, "set": {"text": "Newer."}}])
    assert not _ok(out) and "if_rev" in _text(out)


def test_refusals_say_what_to_do(tools):
    defs, session, _ = tools
    _call(defs, session, "create_doc", title="Plan", markdown="Lead.")
    out = _call(defs, session, "edit_doc", doc="Plan", ops=[{"op": "edit_text", "find": "nowhere", "replace": "x"}])
    assert not _ok(out)
    assert "Refused (find_none)" in _text(out)
    out = _call(defs, session, "read_doc", doc="Missing doc")
    assert not _ok(out)
    assert "list_docs" in _text(out)


def test_list_docs_names_every_doc(tools):
    defs, session, _ = tools
    assert "No docs" in _text(_call(defs, session, "list_docs"))
    _call(defs, session, "create_doc", title="One", markdown="a")
    _call(defs, session, "create_doc", title="Two", tabs=[{"title": "Mon", "markdown": "x"}, {"title": "Tue"}])
    listing = _text(_call(defs, session, "list_docs"))
    assert "“One”" in listing and "“Two”" in listing and "“Tue”" in listing


def test_an_edit_waiting_for_the_doc_leaves_the_event_loop_free(tools):
    defs, session, artifacts = tools
    _call(defs, session, "create_doc", title="Plan", markdown="Lead.")
    doc_id = list_doc_folders(artifacts)[0]["id"]
    held, release = threading.Event(), threading.Event()

    def typing():  # the person's edit, on an HTTP thread, holding the doc's lock
        with doc_lock(find_doc_folder(artifacts, doc_id)):
            held.set()
            release.wait(2)

    threading.Thread(target=typing).start()
    held.wait(1)

    async def scenario():
        edit = asyncio.create_task(_drain(defs["edit_doc"].handler(
            session, {"doc": doc_id, "ops": [{"op": "edit_text", "find": "Lead", "replace": "Led"}]})))
        await asyncio.sleep(0.05)  # other requests and streams still run meanwhile
        waiting = not edit.done()
        release.set()
        return waiting, (await asyncio.wait_for(edit, 5))[1]

    waiting, outcome = asyncio.run(scenario())
    assert waiting and _ok(outcome)


def test_the_conversation_hears_about_a_new_doc_as_it_is_made(tools):
    defs, session, _ = tools
    conv = "c0ffee00-0000-4000-8000-000000000001"

    async def scenario():
        conv_q = hub.subscribe(conv_topic(conv))
        try:
            await _drain(defs["create_doc"].handler(session, {"title": "Live", "markdown": "{{pending: Body}}"}))
            created = await asyncio.wait_for(conv_q.get(), 1)
            doc_q = hub.subscribe(doc_topic(created["artifactId"]))
            try:
                await _drain(defs["edit_doc"].handler(session, {"doc": "Live", "ops": [
                    {"op": "replace", "target": {"find": "Body"}, "markdown": "## Body\n\nDone."}]}))
                updated = await asyncio.wait_for(doc_q.get(), 1)
            finally:
                hub.unsubscribe(doc_topic(created["artifactId"]), doc_q)
            return created, updated
        finally:
            hub.unsubscribe(conv_topic(conv), conv_q)

    created, updated = asyncio.run(scenario())
    assert created["type"] == "doc.created" and created["title"] == "Live"
    assert created["card"]["kind"] == "Doc"
    assert updated["type"] == "doc.updated" and updated["by"] == "agent" and updated["rev"] == 2
    assert [b["type"] for b in updated["doc"]["tabs"][0]["blocks"]] == ["heading", "paragraph"]


def test_a_card_that_fails_is_logged_as_a_warning_and_the_doc_still_opens(tools, monkeypatch, caplog):
    """The card is extra: the doc is already written. But a failure must be seen,
    not dropped at debug, where a bug left every doc.created without its card."""
    from cowork.harnesses.anton_harness import doc_tools

    defs, session, _ = tools
    conv = "c0ffee00-0000-4000-8000-000000000001"

    def broken_card(*_args, **_kwargs):
        raise NameError("name 'oops' is not defined")

    monkeypatch.setattr(doc_tools, "card_for_folder", broken_card)

    async def scenario():
        conv_q = hub.subscribe(conv_topic(conv))
        try:
            outcome = (await _drain(defs["create_doc"].handler(session, {"title": "Card", "markdown": "x"})))[1]
            return outcome, await asyncio.wait_for(conv_q.get(), 1)
        finally:
            hub.unsubscribe(conv_topic(conv), conv_q)

    with caplog.at_level("DEBUG", logger=doc_tools.logger.name):
        outcome, created = asyncio.run(scenario())
    assert _ok(outcome) and created["type"] == "doc.created" and created["card"] is None
    (record,) = [r for r in caplog.records if "doc card unavailable" in r.getMessage()]
    assert record.levelname == "WARNING" and record.exc_info


def test_each_call_shows_in_the_chat_as_a_labelled_step(tools):
    defs, session, _ = tools
    assert _steps(defs, session, "create_doc", title="Board memo", markdown="{{pending: Ask}}") == [
        "Creating doc “Board memo”"]
    assert _steps(defs, session, "edit_doc", doc="Board memo", ops=[
        {"op": "replace", "target": {"find": "Ask"}, "markdown": "## The ask\n\n$2M."}]) == ["Writing “The ask”"]
    assert _steps(defs, session, "edit_doc", doc="Board memo", ops=[
        {"op": "edit_text", "find": "$2M", "replace": "$2.5M"}]) == ["Editing “Board memo” · 1 change"]


def test_the_registry_dispatches_a_doc_tool_like_any_streaming_tool(tools):
    """Through anton's own dispatcher: progress relays, the outcome comes back."""
    from anton.core.tools.registry import ToolRegistry

    defs, session, _ = tools
    registry = ToolRegistry()
    for tool in defs.values():
        registry.register_tool(tool)
    outcome = asyncio.run(registry.dispatch_tool(session, "create_doc", {"title": "Via registry", "markdown": "x"}))
    assert outcome.ok is True
    assert "Created doc “Via registry”" in outcome.content


def test_a_chat_that_subscribes_late_still_hears_about_the_new_doc(tools):
    """A new conversation learns its id from the turn's first event; the doc may already exist by then."""
    defs, session, _ = tools
    conv = "c0ffee00-0000-4000-8000-000000000001"

    async def scenario():
        await _drain(defs["create_doc"].handler(session, {"title": "Early", "markdown": "x"}))
        late = hub.subscribe(conv_topic(conv), replay=True)
        try:
            seen = []
            while not late.empty():
                seen.append(late.get_nowait())
            return seen
        finally:
            hub.unsubscribe(conv_topic(conv), late)

    seen = asyncio.run(scenario())
    assert any(e["type"] == "doc.created" and e["title"] == "Early" and e["replayed"] for e in seen)


def _declared(value, schema):
    """Keep only what `schema` declares, as a model that sends nothing else would."""
    for branch in schema.get("anyOf", []):
        kind = branch.get("type")
        if (kind == "object" and isinstance(value, dict)) or (kind == "array" and isinstance(value, list)) \
                or (kind == "string" and isinstance(value, str)) or (kind == "integer" and isinstance(value, int)):
            return _declared(value, branch)
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        return {k: _declared(v, props[k]) for k, v in value.items() if k in props}
    if isinstance(value, list) and "items" in schema:
        return [_declared(v, schema["items"]) for v in value]
    return value


def test_every_field_an_op_takes_is_in_the_schema(tools):
    # A router-picked model sent bare {"op": "replace"} on every call while the
    # schema declared only `op`, so no section was ever filled.
    defs, session, _ = tools
    schema = defs["edit_doc"].input_schema
    out = _call(defs, session, "create_doc", title="Launch plan",
                markdown="Launch is Nov 18.\n\n{{pending: Owners — who does what}}\n\n- [ ] Book the review")
    ids = dict((t, i) for i, t in re.findall(r"(b_[a-z0-9]{8})  (\w+)", _text(out)))
    rev = int(re.search(r"rev (\d+)", _text(out)).group(1))

    def edit(*ops):
        args = _declared({"doc": "Launch plan", "if_rev": rev, "ops": list(ops)}, schema)
        res = _call(defs, session, "edit_doc", **args)
        assert _ok(res), _text(res)
        return int(re.search(r"rev (\d+)", _text(res)).group(1))

    rev = edit({"op": "replace", "target": ids["pending"],
                "markdown": "## Owners\n\n| Task | Owner | Status |\n|---|---|---|\n| Copy | Dana | Open |"})
    table = next(b["id"] for b in load_doc(next(session._workspace.artifacts_dir.iterdir()))["tabs"][0]["blocks"]
                 if b["type"] == "table")
    rev = edit({"op": "update", "id": ids["task"], "set": {"done": True}},
               {"op": "insert", "position": {"after": table}, "markdown": "> [!WARNING]\n> Review queue is 2 weeks."},
               {"op": "edit_text", "find": "Nov 18", "replace": "Nov 20"},
               {"op": "update_cell", "table": table, "row": 0, "col": "Owner", "value": "Priya"},
               {"op": "add_rows", "table": table, "rows": [["Checkout", "Marco", "Open"]]},
               {"op": "replace", "target": {"find": "Review queue"}, "markdown": "> [!WARNING]\n> Queue is 3 weeks."})

    doc = load_doc(next(session._workspace.artifacts_dir.iterdir()))
    blocks = doc["tabs"][0]["blocks"]
    kinds = [b["type"] for b in blocks]
    assert "pending" not in kinds and "heading" in kinds and "callout" in kinds
    assert "Nov 20" in blocks[0]["text"]
    assert next(b for b in blocks if b["type"] == "task")["done"] is True
    t = next(b for b in blocks if b["type"] == "table")
    owner = next(c["id"] for c in t["columns"] if c["name"] == "Owner")
    assert [r["cells"][owner] for r in t["rows"]] == ["Priya", "Marco"]
    assert "3 weeks" in next(b for b in blocks if b["type"] == "callout")["text"]


def test_the_prompt_gives_the_limits_the_format_enforces():
    assert f"(up to {MAX_METRICS} tiles)" in DOC_PROMPT
    assert f"up to {MAX_CHART_SERIES} series" in DOC_PROMPT
