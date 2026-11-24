"""cowork.docs.tools.run_doc_tool: the doc tools' work, shared by the in-process
ToolDefs and a remote turn's host tools."""
from __future__ import annotations

import json
import re
import threading

import pytest

from cowork.docs import tools as doc_tools
from cowork.docs.model import DocError
from cowork.docs.prompt import DOC_GUIDE, DOC_PROMPT, DOC_SKILL_LABEL, REMOTE_DOC_PROMPT, REMOTE_RETRY_RULE
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


async def test_a_tab_over_the_read_limit_comes_back_as_its_outline(ctx, monkeypatch):
    monkeypatch.setattr(doc_tools, "READ_LIMIT_BYTES", 2048)
    rows = "\n".join(f"| r{i} | {'x' * 40} |" for i in range(80))
    await _run(ctx, "create_doc", title="Big", markdown=f"Lead.\n\n| Name | Note |\n| --- | --- |\n{rows}")

    _steps, out = await _run(ctx, "read_doc", doc="Big")
    assert out.ok and "over the 2 KB a read returns" in out.text and "x" * 40 not in out.text
    table_id = re.search(r"^(b_\w+)\s+table", out.text, re.M).group(1)

    _steps, block = await _run(ctx, "read_doc", doc="Big", ids=[table_id])
    assert block.ok and "x" * 40 in block.text


def test_the_remote_specs_are_the_doc_tools_with_the_retry_rule_only_in_their_prompt(ctx):
    specs = remote_doc_tool_specs()
    assert [s["name"] for s in specs] == list(DOC_TOOL_NAMES)
    for spec in specs:
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,63}", spec["name"])
        assert spec["description"] and spec["input_schema"]["type"] == "object"
        assert spec["timeout_s"] == 60
    json.dumps(specs)
    # Shaped as on the desktop: list_docs carries the short prompt, the rest wait for the skill.
    assert {s["name"]: (s.get("prompt"), s.get("unlock_skill")) for s in specs} == {
        "create_doc": (None, DOC_SKILL_LABEL), "read_doc": (None, DOC_SKILL_LABEL),
        "edit_doc": (None, DOC_SKILL_LABEL), "list_docs": (REMOTE_DOC_PROMPT, None),
    }
    assert REMOTE_DOC_PROMPT == DOC_PROMPT + REMOTE_RETRY_RULE and "did not answer" not in DOC_PROMPT
    assert DOC_GUIDE not in REMOTE_DOC_PROMPT


def test_a_remote_turn_with_doc_tools_brings_the_docs_skill():
    from cowork.docs.remote import docs_skill
    from cowork.schemas.turn_skills import TurnSkillEntry

    [(label, entry)] = docs_skill().items()
    assert label == DOC_SKILL_LABEL and set(entry.files) == {"SKILL.md"} and not entry.always_on
    assert DOC_GUIDE in entry.files["SKILL.md"]
    # A person's own skill of that name gives way, as a host skill wins on the desktop.
    mine = {"docs": TurnSkillEntry(files={"SKILL.md": "mine"}), "notes": TurnSkillEntry(files={"SKILL.md": "n"})}
    from cowork.docs.remote import RemoteDocTools

    merged = RemoteDocTools.with_docs_skill(None, mine)
    assert merged["docs"] == entry and merged["notes"] == mine["notes"]


def test_in_process_only_list_docs_is_offered_up_front_and_the_rest_wait_for_the_docs_skill(ctx):
    tools = {t.name: t for t in build_doc_tools(ctx)}
    assert tools["list_docs"].prompt == DOC_PROMPT and tools["list_docs"].unlock_skill is None
    for name in ("create_doc", "read_doc", "edit_doc"):
        assert tools[name].unlock_skill == DOC_SKILL_LABEL and tools[name].prompt is None
    # The always-on prompt names the skill; the writing rules stay in it.
    assert f"`recall_skill` with the label `{DOC_SKILL_LABEL}`" in DOC_PROMPT
    assert "Doc Markdown" in DOC_GUIDE and "Doc Markdown" not in DOC_PROMPT


async def test_a_stale_edit_is_refused_even_when_the_doc_cannot_be_read_again(ctx, monkeypatch):
    await _run(ctx, "create_doc", title="Board memo", markdown="Intro text here.")
    real_load = doc_tools.load_doc
    calls = []

    def load(folder):
        calls.append(folder)
        if len(calls) > 1:  # the reread for the refusal: the doc went away meanwhile
            raise DocError("not_found", "gone")
        return real_load(folder)

    def stale(*_args, **_kwargs):
        raise DocError("stale", "the doc changed", ids=[])

    monkeypatch.setattr(doc_tools, "load_doc", load)
    monkeypatch.setattr(doc_tools, "apply", stale)
    _steps, result = await _run(ctx, "edit_doc", doc="Board memo", ops=[
        {"op": "replace", "target": {"find": "Intro"}, "markdown": "New intro."}])
    assert not result.ok and result.text.startswith("Refused (stale): the doc changed")
