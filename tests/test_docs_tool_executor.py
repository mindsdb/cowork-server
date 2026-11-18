"""cowork.docs.tools.run_doc_tool: the doc tools' work, shared by the in-process
ToolDefs and a remote turn's host tools."""
from __future__ import annotations

import json
import re
import threading

import pytest

from cowork.docs import tools as doc_tools
from cowork.docs.model import DocError
from cowork.docs.prompt import DOC_PROMPT, REMOTE_RETRY_RULE
from cowork.docs.store import DOC_FILENAME, find_doc_folder
from cowork.docs.tools import DOC_TOOL_NAMES, TOO_BIG_TEXT, DocToolContext, remote_doc_tool_specs, run_doc_tool
from cowork.harnesses.anton_harness.doc_tools import build_doc_tools


@pytest.fixture
def ctx(tmp_path):
    return DocToolContext(
        artifacts_base=tmp_path / ".anton" / "artifacts", conversation_id="c0ffee00-0000-4000-8000-000000000003",
    )


async def _run(ctx, name, on_written=None, **tc_input):
    steps: list[str] = []

    async def emit(text: str) -> None:
        steps.append(text)

    result = await run_doc_tool(ctx, name, tc_input, emit, on_written=on_written)
    return steps, result


async def test_a_call_emits_its_step_then_answers_and_names_the_doc_it_wrote(ctx):
    steps, made = await _run(ctx, "create_doc", title="Board memo", markdown="{{pending: Ask}}")
    assert steps == ["Creating doc “Board memo”"]
    assert made.ok and "Created doc “Board memo”" in made.text
    folder = find_doc_folder(ctx.artifacts_base, "Board memo")
    assert made.touched_slug == folder.name

    steps, edited = await _run(ctx, "edit_doc", doc="Board memo", ops=[
        {"op": "replace", "target": {"find": "Ask"}, "markdown": "## The ask\n\n$2M."}])
    assert steps == ["Writing “The ask”"]
    assert edited.ok and edited.touched_slug == folder.name

    _steps, read = await _run(ctx, "read_doc", doc="Board memo")
    assert read.ok and read.touched_slug is None and "$2M." in read.text


async def test_the_write_is_reported_from_the_worker_thread_before_the_card_is_built(ctx, monkeypatch):
    order: list[tuple[str, bool]] = []
    loop_thread = threading.get_ident()

    def written(slug, tool):
        order.append((f"written:{tool}", threading.get_ident() != loop_thread))

    def card(*_args, **_kwargs):
        order.append(("card", threading.get_ident() != loop_thread))
        return None

    monkeypatch.setattr(doc_tools, "card_for_folder", card)
    await _run(ctx, "create_doc", on_written=written, title="Plan", markdown="Lead.")
    assert order == [("written:create_doc", True), ("card", True)]


async def test_an_unknown_tool_is_refused(ctx):
    _steps, out = await _run(ctx, "delete_doc", doc="x")
    assert not out.ok and "delete_doc" in out.text


async def test_the_owner_check_refuses_an_edit_and_writes_nothing(ctx):
    await _run(ctx, "create_doc", title="Plan", markdown="Lead.")
    folder = find_doc_folder(ctx.artifacts_base, "Plan")
    before = (folder / DOC_FILENAME).read_bytes()
    written: list[str] = []

    def refuse(_folder):
        raise DocError("not_owner", "only the doc's owner can edit it.")

    ctx.owner_check = refuse
    _steps, out = await _run(ctx, "edit_doc", on_written=lambda slug, tool: written.append(slug),
                             doc="Plan", ops=[{"op": "edit_text", "find": "Lead", "replace": "Led"}])
    assert not out.ok and out.text.startswith("Refused (not_owner)")
    assert (folder / DOC_FILENAME).read_bytes() == before and written == []
    _steps, read = await _run(ctx, "read_doc", doc="Plan")
    assert read.ok


async def test_a_read_over_the_cap_is_refused_before_it_renders(ctx, monkeypatch):
    await _run(ctx, "create_doc", title="Long", markdown="word " * 400)
    ctx.max_result_bytes = 1000

    def no_render(*_args, **_kwargs):
        raise AssertionError("rendered a read the cap refuses")

    monkeypatch.setattr(doc_tools.render, "full", no_render)
    _steps, out = await _run(ctx, "read_doc", doc="Long")
    assert not out.ok and out.text == TOO_BIG_TEXT
    monkeypatch.undo()
    _steps, outline = await _run(ctx, "read_doc", doc="Long", view="outline")
    assert outline.ok


def test_the_remote_specs_are_the_doc_tools_with_the_retry_rule_only_in_their_prompt(ctx):
    specs = remote_doc_tool_specs()
    assert [s["name"] for s in specs] == list(DOC_TOOL_NAMES)
    for spec in specs:
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,63}", spec["name"])
        assert spec["description"] and spec["input_schema"]["type"] == "object"
        assert spec["timeout_s"] == 60
    json.dumps(specs)
    assert {s["name"]: s.get("prompt") for s in specs} == {
        "create_doc": DOC_PROMPT + REMOTE_RETRY_RULE, "read_doc": None, "edit_doc": None, "list_docs": None,
    }
    assert "did not answer" in REMOTE_RETRY_RULE and "did not answer" not in DOC_PROMPT
    assert {t.name: t.prompt for t in build_doc_tools(ctx)}["create_doc"] == DOC_PROMPT
