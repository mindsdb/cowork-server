from __future__ import annotations

import json

from anton.core.llm.provider import StreamToolResult, StreamToolUseEnd, StreamToolUseStart

from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream


async def _events(*items):
    for item in items:
        yield item


def _parse_sse(chunks: list[str]) -> list[dict]:
    parsed = []
    for chunk in chunks:
        for frame in chunk.strip("\n").split("\n\n"):
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    parsed.append(json.loads(line[len("data:"):].strip()))
    return parsed


async def _format(*items) -> list[dict]:
    chunks = [c async for c in format_responses_stream(_events(*items), model="claude-sonnet-4-6")]
    return _parse_sse(chunks)


async def test_a_tool_message_gets_its_own_role():
    events = await _format(
        StreamToolUseStart(id="tc_1", name="generate_artifact"),
        StreamToolUseEnd(id="tc_1"),
        StreamToolResult(name="generate_artifact", action="message", content="## Brief", id="tc_1"),
    )
    messages = [e for e in events if e.get("thought_role") == "thought.tool_call.message"]
    assert len(messages) == 1
    assert messages[0]["content"] == "## Brief"
    assert messages[0]["tool_use_id"] == "tc_1"
    assert messages[0]["tool_name"] == "generate_artifact"
    assert "cell_status" not in messages[0]
    assert not [e for e in events if e.get("thought_role") == "thought.scratchpad.result"]
    # Not answer text: it must not reach the persisted assistant message.
    assert not [e for e in events if e.get("type") == "response.output_text.delta"]


async def test_a_scratchpad_dump_keeps_the_result_role():
    events = await _format(
        StreamToolResult(name="scratchpad", action="dump", content="cells", id="t1"),
    )
    roles = [e.get("thought_role") for e in events if e.get("type") == "response.in_progress"]
    assert "thought.scratchpad.result" in roles
    assert "thought.tool_call.message" not in roles


async def test_a_scratchpad_result_with_a_message_action_stays_a_cell_result():
    """A discarded scratchpad call echoes the model's own `action`; it must
    not be shown to the user as an agent message."""
    events = await _format(
        StreamToolResult(name="scratchpad", action="message", content="[error] exec failed", id="t1"),
    )
    roles = [e.get("thought_role") for e in events if e.get("type") == "response.in_progress"]
    assert "thought.scratchpad.result" in roles
    assert "thought.tool_call.message" not in roles
