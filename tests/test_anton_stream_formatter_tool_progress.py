from __future__ import annotations

import json

from anton.core.llm.provider import StreamTaskProgress, StreamToolUseEnd, StreamToolUseStart

from cowork.harnesses.anton_harness import stream_formatter as sf
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream


async def _events(*items):
    for item in items:
        yield item


async def _timed(clock: dict, *pairs):
    """Like `_events`, but sets `clock["t"]` right before yielding each
    item — for tests that monkeypatch `time.time` to `lambda: clock["t"]`
    and need specific events processed at specific fake timestamps. A
    plain `iter([...])` fed to `time.time` doesn't work here: `_event()`
    stamps `at_ms` via its own `time.time()` call on every emitted event,
    not just the throttle check, so a fixed-length iterator runs out
    before the events under test are even reached."""
    for t, item in pairs:
        clock["t"] = t
        yield item


def _parse_sse(chunks: list[str]) -> list[dict]:
    parsed = []
    for chunk in chunks:
        for frame in chunk.strip("\n").split("\n\n"):
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    parsed.append(json.loads(line[len("data:"):].strip()))
    return parsed


class TestToolProgressRoleMapping:
    async def test_progress_and_done_map_to_tool_call_roles_in_real_prod_order(self):
        # Real order: the LLM stream fully proliferates start/end for the
        # tool_use block BEFORE anton dispatches it — tool_progress/tool_done
        # only exist after that (ENG-763 stage 2 design doc, fact #5). Do NOT
        # insert an artificial "start -> progress -> end" sequence — that
        # would hide exactly the bug an earlier design revision had.
        chunks = [
            c async for c in format_responses_stream(
                _events(
                    StreamToolUseStart(id="tc_1", name="streaming_probe"),
                    StreamToolUseEnd(id="tc_1"),
                    StreamTaskProgress(phase="tool_progress", message="step 1", id="tc_1"),
                    StreamTaskProgress(
                        phase="tool_done", message="streaming_probe",
                        eta_seconds=1.5, id="tc_1",
                    ),
                ),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)

        progress = [e for e in events if e.get("thought_role") == "thought.tool_call.progress"]
        assert len(progress) == 1
        assert progress[0]["tool_use_id"] == "tc_1"
        # tool_name must resolve — this only works if tool_names isn't
        # popped on StreamToolUseEnd, since tool_progress/tool_done arrive
        # strictly after it in the real event order.
        assert progress[0]["tool_name"] == "streaming_probe"
        assert progress[0]["content"] == "step 1"

        done = [e for e in events if e.get("thought_role") == "thought.tool_call.end"]
        assert len(done) == 1
        assert done[0]["tool_use_id"] == "tc_1"
        assert done[0]["eta_seconds"] == 1.5
        assert done[0]["ok"] is None

    async def test_tool_done_forwards_the_ok_verdict(self):
        # Without this, a failed tool renders as a success everywhere
        # downstream — tool_done fires unconditionally by design (even on a
        # handler exception), and with no verdict attached every consumer
        # has no choice but to treat "it fired" as "it succeeded"
        # (anton PR #304 review).
        #
        # Set post-construction, not via the constructor: the anton-agent
        # version this repo's uv.lock resolves today predates the `ok`
        # field (it's added on the anton side of this same fix and hasn't
        # been promoted past `main` yet — see the design doc's rollout
        # notes). `getattr(event, "ok", None)` in the formatter already
        # handles an anton without the field at all; this only needs to
        # simulate an anton new enough to *set* it.
        tool_done = StreamTaskProgress(
            phase="tool_done", message="streaming_probe",
            eta_seconds=0.5, id="tc_1",
        )
        tool_done.ok = False
        chunks = [
            c async for c in format_responses_stream(
                _events(
                    StreamToolUseStart(id="tc_1", name="streaming_probe"),
                    StreamToolUseEnd(id="tc_1"),
                    StreamTaskProgress(phase="tool_progress", message="step 1", id="tc_1"),
                    tool_done,
                ),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)

        done = [e for e in events if e.get("thought_role") == "thought.tool_call.end"]
        assert len(done) == 1
        assert done[0]["ok"] is False

    async def test_only_the_first_progress_per_id_is_throttle_exempt_but_tool_done_always_is(self, monkeypatch):
        # Mirrors production's throttle window (PROGRESS_THROTTLE = 0.25s).
        #
        # A plain `iter([...])` fed to `time.time` (one value per call)
        # is NOT enough here: `_event()` stamps `at_ms` via its own
        # `time.time()` call on every emitted event (not just the
        # throttle check), and `response.created`/StreamToolUseStart/
        # StreamToolUseEnd all emit before the first tool_progress is even
        # reached — the iterator runs out mid-stream and the generator
        # raises "async generator raised StopIteration" before any assert
        # runs, on both old and new code. Use a clock that holds its value
        # until explicitly advanced instead.
        clock = {"t": 100.0}
        monkeypatch.setattr(sf.time, "time", lambda: clock["t"])

        chunks = [
            c async for c in format_responses_stream(
                _timed(
                    clock,
                    (100.0, StreamToolUseStart(id="tc_1", name="streaming_probe")),
                    (100.0, StreamToolUseEnd(id="tc_1")),
                    (100.0, StreamTaskProgress(phase="tool_progress", message="step 1", id="tc_1")),
                    (100.05, StreamTaskProgress(phase="tool_progress", message="step 2", id="tc_1")),
                    (100.1, StreamTaskProgress(phase="tool_progress", message="step 3", id="tc_1")),
                    (100.1, StreamTaskProgress(
                        phase="tool_done", message="streaming_probe",
                        eta_seconds=0.1, id="tc_1",
                    )),
                ),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)

        progress = [e for e in events if e.get("thought_role") == "thought.tool_call.progress"]
        # "step 1" (first-for-id, exempt) emits at t=100.0. "step 2" is not
        # exempt, but the throttle anchor (last_progress) is still its
        # initial 0.0 because step 1 was exempt and never updated it, so
        # 100.05 - 0.0 clears the window — it emits and becomes the new
        # anchor. "step 3" arrives only 0.05s after that anchor — inside
        # the 0.25s window — and is dropped.
        assert [e["content"] for e in progress] == ["step 1", "step 2"]

        done = [e for e in events if e.get("thought_role") == "thought.tool_call.end"]
        assert len(done) == 1  # tool_done is never throttled

    async def test_tool_done_without_any_progress_keeps_the_old_fallback_role(self):
        # Regression: the 8 generic tools that never stream ToolProgress
        # must not start showing a step just because tool_done now has a
        # closing role for progress-emitting tools.
        chunks = [
            c async for c in format_responses_stream(
                _events(
                    StreamToolUseStart(id="tc_2", name="web_search"),
                    StreamToolUseEnd(id="tc_2"),
                    StreamTaskProgress(phase="tool_done", message="web_search", id="tc_2"),
                ),
                model="claude-sonnet-4-6",
            )
        ]
        events = _parse_sse(chunks)

        done = [e for e in events if e.get("phase") == "tool_done"]
        assert len(done) == 1
        assert done[0]["thought_role"] == "thought.progress"
        assert done[0]["content"] == "tool_done: web_search"
        assert not [e for e in events if e.get("thought_role") == "thought.tool_call.end"]
