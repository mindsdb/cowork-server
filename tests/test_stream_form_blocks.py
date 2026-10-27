"""A connection form the model left out of its answer is added to it.

`request_credentials` returns the form as a fenced block and asks the model
to paste it; the side panel opens only a block in the answer. A model that
ranks "summarize tool output" above that request leaves it out. With
`form_blocks` on, the formatter adds the turn's form to the answer, and the
saved answer keeps it. The tool's result reaches the formatter only in the
turn's history rows, so every test builds them as the harness does.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta

from cowork.harnesses.anton_harness.harness import _captured_turn_history
from cowork.harnesses.anton_harness.stream_formatter import _ANTON_FAILURE_REPLY, format_responses_stream
from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
from cowork.streaming.answer_text import accumulate_answer_text

# cowork's extractFormSpec (parseFormSpec.js): the side panel opens the last match.
_RENDERER_FORM = re.compile(r"```data-vault-form[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```")
_SUMMARY = "The connection form is ready in the side panel."


async def _form_result(title: str = "Connect to Postgres") -> str:
    """What the real tool returns for a built-in connector's form."""
    return await _cowork_request_credentials(
        session=None,
        tc_input={"engine": "postgres", "_connector_id": "postgres", "title": title},
    )


def _block(result: str) -> str:
    """The fenced form block in a tool result."""
    return _RENDERER_FORM.search(result).group(0)


def _form_id(block: str) -> str:
    """The form id inside a fenced form block."""
    return json.loads(_RENDERER_FORM.search(block).group(1))["form_id"]


def _history(results: list[tuple[str, str, bool]], *, compacted: bool = False):
    """The TurnHistory the harness builds for a turn that made these tool
    calls, each (tool name, result text, is_error), then answered."""
    turn: list[dict] = []
    for index, (name, content, is_error) in enumerate(results):
        call_id = f"call_{index}"
        turn.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        ]})
        result = {"type": "tool_result", "tool_use_id": call_id, "content": content}
        if is_error:
            result["is_error"] = True
        turn.append({"role": "user", "content": [result]})
    turn.append({"role": "assistant", "content": _SUMMARY})
    return _captured_turn_history(
        session=SimpleNamespace(turn_messages=lambda: turn), seed_len=None, compacted=compacted,
    )


async def _run(*events, form_blocks: bool = True):
    """Run the formatter; return (completed text, text the handlers save)."""

    async def _gen():
        for event in events:
            yield event

    saved: list[str] = []
    completed = ""
    async for sse in format_responses_stream(
        _gen(), "m", lambda event_type, data: accumulate_answer_text(saved, event_type, data),
        form_blocks=form_blocks,
    ):
        payload = json.loads(sse.split("data: ", 1)[1])
        if payload["type"] == "response.completed":
            completed = payload["response"]["output"][0]["content"][0]["text"]
    return completed, "".join(saved)


def _opened_form_id(answer: str) -> str:
    """The form id of the block the side panel opens for ``answer``."""
    return json.loads(_RENDERER_FORM.findall(answer)[-1])["form_id"]


async def test_an_answer_that_summarizes_gets_the_form():
    result = await _form_result()
    history = _history([("request_credentials", result, False)])

    completed, saved = await _run(StreamTextDelta(text=_SUMMARY), history)

    assert completed == saved
    assert saved.startswith(_SUMMARY + "\n\n")
    assert saved.endswith(_block(result) + "\n")
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_a_compacted_turn_gets_the_form():
    result = await _form_result()
    history = _history([("request_credentials", result, False)], compacted=True)
    assert history.rows == [] and history.read

    _, saved = await _run(StreamTextDelta(text=_SUMMARY), history)

    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_a_pasted_form_shows_once():
    result = await _form_result()
    answer = f"Fill this in:\n\n{_block(result)}\n"

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved == answer


async def test_a_pasted_form_the_renderer_reads_loosely_shows_once():
    result = await _form_result()
    loose = _block(result).replace("\n}\n```", ",\n}\n```")
    answer = f"Fill this in:\n\n{loose}\n"

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved == answer


async def test_a_pasted_form_with_smart_quotes_shows_once():
    result = await _form_result()
    smart = _block(result).replace('"form_id": "', "“form_id”: “", 1)
    smart = smart[:smart.index("“form_id”: “")] + smart[smart.index("“form_id”: “"):].replace('"', "”", 1)
    answer = f"Fill this in:\n\n{smart}\n"
    assert "“form_id”" in answer

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved == answer


async def test_a_refused_call_adds_nothing():
    refused = await _cowork_request_credentials(session=None, tc_input={"title": "No engine"})
    assert "```" not in refused

    _, saved = await _run(StreamTextDelta(text=_SUMMARY), _history([("request_credentials", refused, False)]))

    assert saved == _SUMMARY


async def test_a_call_anton_marked_is_error_adds_nothing():
    result = await _form_result()

    _, saved = await _run(StreamTextDelta(text=_SUMMARY), _history([("request_credentials", result, True)]))

    assert saved == _SUMMARY


async def test_a_note_after_the_block_still_yields_the_form():
    result = await _form_result() + "\n\nStop retrying this approach and try another one."

    _, saved = await _run(StreamTextDelta(text=_SUMMARY), _history([("request_credentials", result, False)]))

    assert _opened_form_id(saved) == _form_id(_block(result))
    assert "Stop retrying" not in saved


async def test_only_the_last_form_counts():
    first, second = await _form_result("First"), await _form_result("Second")
    history = _history([("request_credentials", first, False), ("request_credentials", second, False)])

    pasted_answer = f"Use this one:\n\n{_block(second)}\n"
    _, pasted = await _run(StreamTextDelta(text=pasted_answer), history)
    _, summarized = await _run(StreamTextDelta(text=_SUMMARY), history)

    assert pasted == pasted_answer
    assert _block(first) not in summarized
    assert _opened_form_id(summarized) == _form_id(_block(second))


async def test_a_later_different_block_does_not_count_as_the_form():
    result, other = await _form_result(), await _form_result("Other")
    answer = f"{_block(result)}\n\nOr this:\n\n{_block(other)}\n"

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved.startswith(answer)
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_a_paste_cut_off_inside_its_fence_is_closed_first():
    result = await _form_result()
    answer = 'Here it is:\n\n```data-vault-form\n{\n  "title": "Conn'

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved.startswith(answer + "\n```")
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_an_answer_cut_off_inside_a_code_block_is_closed_first():
    result = await _form_result()
    answer = "Checking the driver:\n\n```python\nimport psycopg"

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved.startswith(answer + "\n```\n\n")
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_a_cut_off_form_fence_glued_to_a_sentence_is_closed_first():
    result = await _form_result()
    answer = 'Fill in the form.```data-vault-form\n{\n  "title": "Co'

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved.startswith(answer + "\n```\n\n")
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_an_answer_cut_off_inside_a_tilde_block_is_closed_with_tildes():
    result = await _form_result()
    answer = "~~~python\nimport psycopg"

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved.startswith(answer + "\n~~~\n\n")


@pytest.mark.parametrize("answer", [
    "```psql``` is the client you need.",
    "````md\n```sql\nselect 1\n````",
    "~~~text\n```\n~~~",
], ids=["inline-code-line", "four-backtick-block", "tilde-block"])
async def test_an_answer_outside_every_code_block_gets_no_closing_fence(answer):
    result = await _form_result()

    _, saved = await _run(StreamTextDelta(text=answer), _history([("request_credentials", result, False)]))

    assert saved == f"{answer}\n\n{_block(result)}\n"


async def test_a_replaced_answer_still_gets_the_form():
    result = await _form_result()

    _, saved = await _run(
        StreamTextDelta(text="Draft."),
        StreamTaskProgress(phase="continuation", message="Task incomplete — continuing (1/3)..."),
        StreamTextDelta(text=_SUMMARY),
        _history([("request_credentials", result, False)]),
    )

    assert saved.startswith(_SUMMARY)
    assert "Draft." not in saved
    assert _opened_form_id(saved) == _form_id(_block(result))


async def test_a_failure_reply_gets_no_form():
    result = await _form_result()
    failure = _ANTON_FAILURE_REPLY + "the provider refused the request."

    _, saved = await _run(StreamTextDelta(text=failure), _history([("request_credentials", result, False)]))

    assert saved == failure


async def test_off_by_default_adds_nothing():
    result = await _form_result()

    _, saved = await _run(
        StreamTextDelta(text=_SUMMARY), _history([("request_credentials", result, False)]), form_blocks=False,
    )

    assert saved == _SUMMARY
