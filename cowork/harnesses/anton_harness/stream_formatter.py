"""SSE event formatter — turns ChatSession.turn_stream() events into
OpenAI Responses API SSE strings.

Emits typed events:
    response.created            (with conversation_id)
    response.in_progress        (thought/tool activity, carries thought_role)
    response.output_text.delta  (assistant text deltas)
    response.completed          (final response object)
    response.failed             (error)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from cowork.schemas.responses import (
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    Role,
)


@dataclass
class ArtifactCreated:
    """Synthetic post-turn event: an artifact folder appeared during the
    turn (detected by the harness via the artifacts-dir diff, not by any
    agent tool call). Rides the same stream as Anton's `Stream*` events and
    is mapped to a `response.artifact_created` SSE event below, so the
    renderer shows an inline card for every artifact type, identically and
    deterministically — live and on reload."""

    artifact: dict


@dataclass
class TurnHistory:
    """Synthetic post-turn event carrying the turn's tool block-rows
    (assistant `tool_use` / user `tool_result`) so the responses route can
    persist them into the `messages` table. Recorded via `event_sink` only —
    never emitted as SSE: the client rebuilds all tool activity from the
    existing thought events, so these rows are for LLM-history replay only."""

    rows: list


@dataclass
class SkillCreated:
    """Synthetic post-turn event: a skill the agent built for the user appeared
    in the drafts dir during the turn (detected by the harness via a dir diff,
    not by any agent tool call). Mapped to a `response.skill_created` SSE event
    below. The payload is self-contained (full SKILL.md + sibling files), so the
    renderer shows the draft card identically live and on reload — the skill is
    NOT saved until the user explicitly does so from the card."""

    skill: dict


PHASE_LABELS = {
    "planning": "Planning",
    "analyzing": "Analyzing",
    "executing": "Executing",
    "scratchpad": "Running code",
    "scratchpad_start": "Running code",
    "scratchpad_done": "Code complete",
    "connect_datasource": "Connecting",
    "interactive": "Interactive",
    "context": "Context",
}

PROGRESS_THROTTLE = 0.25  # seconds


def classify_cell_status(content: str) -> str:
    """Classify a scratchpad tool-result as ok / timeout / error.

    The renderer uses this to show a killed cell as distinctly dead rather
    than indistinguishable from a slow-but-running one.

    An exec result arrives as ``json.dumps(asdict(cell))`` (see
    ``ChatSession.turn_stream`` → ``StreamToolResult``), so we inspect the
    structured ``error`` field rather than sniffing the rendered text. That
    matters: a *successful* cell whose own stdout contains "[error]" or
    "Cell timed out" (e.g. a log-analysis cell) must NOT be misclassified —
    only the cell's error field decides. Non-exec results (e.g. a `dump`
    notebook string, or other tools) aren't JSON; for those we fall back to
    a best-effort text sniff. Timeout-kill text in the error → "timeout";
    any other non-empty error → "error".
    """
    if not content:
        return "ok"
    try:
        cell = json.loads(content)
    except (ValueError, TypeError):
        cell = None
    if isinstance(cell, dict) and "error" in cell:
        err = (cell.get("error") or "").strip()
        if not err:
            return "ok"
        low = err.lower()
        if "timed out" in low or "of inactivity" in low or "cell killed" in low:
            return "timeout"
        return "error"
    # Fallback for non-JSON results (dump notebook, non-scratchpad tools).
    low = content.lower()
    if "cell timed out" in low or "of inactivity" in low or "cell killed" in low:
        return "timeout"
    if "[error]" in low or "exec failed" in low:
        return "error"
    return "ok"


async def format_responses_stream(
    event_stream: AsyncIterator,
    model: str,
    event_sink: Optional[Callable[[str, dict], None]] = None,
) -> AsyncIterator[str]:
    """Yield Responses-API SSE strings derived from ChatSession events.

    `event_sink` (optional) is called with `(event_type, payload_dict)` for
    every event before it's serialised to SSE. Used by the responses
    route to capture a per-turn event log to disk so the client can
    rebuild the Thinking block + scratchpad cells when the conversation
    is reopened (without keeping localStorage state).
    """
    from anton.core.llm.provider import (
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamTextDelta,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    # StreamReasoningDelta is newer than the rest — it arrived with the
    # reasoning-subtype channel (ENG-1109) and does not exist in anton-agent
    # builds before it. Desktop staging installs resolve anton from PyPI, which
    # lags the staging branch (staging cowork-server can be paired with an
    # older published anton until anton is promoted), so importing it in the
    # same unconditional block would make THIS formatter — run on every single
    # turn — raise ImportError, which the responses route can only redact to a
    # generic "An unexpected error occurred", breaking all chat (ENG-1167).
    # Guard it: an older anton simply never emits reasoning deltas, so falling
    # back to a sentinel class no event can be an instance of leaves the branch
    # below dead and the turn degrades gracefully instead of failing.
    try:
        from anton.core.llm.provider import StreamReasoningDelta
    except ImportError:
        class StreamReasoningDelta:  # type: ignore[no-redef]
            """Placeholder for pre-ENG-1109 anton; never instantiated."""

    resp_id = f"resp-{uuid.uuid4().hex[:12]}"
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    seq = 0
    last_progress = 0.0
    collected_text: list[str] = []
    # Round-boundary paragraphing (ENG-887): armed when an LLM round
    # completes, consumed by the next text delta. `text_tail` keeps the
    # last two streamed chars so a round that already ends with a blank
    # line doesn't get a second one; empty until the first visible text,
    # which also stops a break from landing before anything was said.
    # `round_had_text` scopes the max_tokens exception below to the round
    # that actually streamed the truncated text.
    round_break = False
    round_had_text = False
    text_tail = ""

    def _event(event_type: str, data: dict) -> str:
        # Wall-clock millisecond stamp on every event. The renderer
        # uses this (over `Date.now()` at the moment of replay) so
        # historical conversations rebuild correct reasoning /
        # execution durations: synchronous replay through the stream
        # reducer would otherwise see every `now()` collapse to the
        # same JS-tick value, producing 0ms across the board.
        # Persisted into the turns sidecar via `event_sink`, so the
        # field is also there for future replays.
        if "at_ms" not in data:
            data["at_ms"] = int(time.time() * 1000)
        if event_sink is not None:
            try:
                event_sink(event_type, data)
            except Exception:
                # Recording is best-effort — never break the live stream.
                pass
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    tool_json_parts: dict[str, list[str]] = {}
    tool_names: dict[str, str] = {}
    # Tool-use ids that have emitted at least one tool_progress phase —
    # only these get closed via thought.tool_call.end when their tool_done
    # arrives. Every other generic tool keeps falling through to the plain
    # thought.progress role, unchanged.
    progress_tool_ids: set[str] = set()

    resp = Response(id=resp_id, model=model, status=ResponseStatus.created)
    seq += 1
    created_data = {
        "type": "response.created",
        "sequence_number": seq,
        "response": resp.model_dump(),
    }
    yield _event("response.created", created_data)

    async for event in event_stream:
        if isinstance(event, StreamTextDelta):
            text = event.text
            if round_break:
                if text_tail and not text_tail.endswith("\n\n"):
                    text = "\n\n" + text
                round_break = False
            if event.text:
                round_had_text = True
            text_tail = (text_tail + text)[-2:]
            collected_text.append(text)
            seq += 1
            yield _event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "sequence_number": seq,
                "item_id": msg_id,
                "delta": text,
            })

        elif isinstance(event, StreamToolUseStart):
            tool_names[event.id] = event.name
            tool_json_parts[event.id] = []
            if "scratchpad" in event.name:
                role = Role.thought_scratchpad_start.value
            elif "memorize" in event.name:
                role = Role.thought_memorize_start.value
            elif "recall" in event.name:
                role = Role.thought_recall_start.value
            else:
                role = Role.thought_progress.value
            # `tool_use_id` rides along on start/end/result/progress
            # events so the renderer can correlate them. Without it,
            # multi-tool turns (LLM emits start/end for cells A, B, C
            # upfront, then anton dispatches them sequentially) end up
            # patching the wrong step when results arrive — the
            # frontend's "patch the last scratchpad step" heuristic
            # silently misattributes A's output to C.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": event.name,
                "tool_use_id": event.id,
            })

        elif isinstance(event, StreamToolUseDelta):
            if event.id in tool_json_parts:
                tool_json_parts[event.id].append(event.json_delta)

        elif isinstance(event, StreamToolUseEnd):
            # NOT popped here (was: tool_names.pop(...)) — tool_progress and
            # tool_done for this id arrive strictly AFTER StreamToolUseEnd in
            # the real event order (the LLM stream fully proliferates
            # start/end before anton dispatches the tool), so the name must
            # stay available until tool_done finally pops it below.
            name = tool_names.get(event.id, "")
            parts = tool_json_parts.pop(event.id, [])
            accumulated = "".join(parts)
            if "scratchpad" in name:
                role = Role.thought_scratchpad_end.value
            elif "memorize" in name:
                role = Role.thought_memorize_end.value
            elif "recall" in name:
                role = Role.thought_recall_end.value
            else:
                role = Role.thought_progress.value
            # 64 KB cap — old 2 KB cap routinely chopped scratchpad
            # JSON mid-`code` field, leaving the desktop renderer with
            # an unparseable string and the inspector showing "No code
            # captured for this cell." 64 KB covers every cell we've
            # seen in practice without bloating the SSE stream or the
            # persisted turns log.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": accumulated[:65536],
                "tool_use_id": event.id,
            })

        elif isinstance(event, StreamToolResult):
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_scratchpad_result.value,
                "content": event.content[:65536],
                "tool_name": getattr(event, "name", "") or "",
                "tool_action": getattr(event, "action", "") or "",
                "tool_use_id": getattr(event, "id", None) or "",
                # ok / timeout / error — lets the renderer show a killed cell
                # as dead instead of stuck "running". Additive: older clients
                # ignore the field.
                "cell_status": classify_cell_status(event.content),
            })

        elif isinstance(event, StreamTaskProgress):
            # scratchpad_start / scratchpad_done phases now carry the
            # source tool_use_id so the renderer correlates them to
            # the right step instead of the last scratchpad step.
            # We DO NOT throttle scratchpad-phase events even when
            # under PROGRESS_THROTTLE — dropping a scratchpad_done
            # would leave the cell stuck in_progress in the UI.
            phase_str = event.phase or ""
            is_scratchpad_phase = phase_str in ("scratchpad_start", "scratchpad_done")
            is_tool_progress = phase_str == "tool_progress"
            is_tool_done = phase_str == "tool_done" and event.id in progress_tool_ids

            if is_tool_progress and not event.id:
                # Can't correlate to a step at all — drop it before it
                # touches progress_tool_ids or spends the throttle window
                # on nothing.
                pass
            else:
                # First tool_progress for a given id must never be dropped
                # by throttling — losing it would mean the lazily-created
                # step in the renderer is never created, and a later
                # tool_done for the same id would then have nothing to
                # close (thought.tool_call.end on the frontend is a no-op
                # when there's no matching step).
                is_first_progress_for_id = is_tool_progress and event.id not in progress_tool_ids
                if is_tool_progress:
                    progress_tool_ids.add(event.id)

                # tool_done is never throttled for the same reason
                # scratchpad_done isn't — it's the ONLY event that closes a
                # tool-progress step, and it's yielded right after the last
                # tool_progress (well within one PROGRESS_THROTTLE window).
                never_throttle = is_scratchpad_phase or is_tool_done or is_first_progress_for_id
                now = time.time()
                should_emit = never_throttle or (now - last_progress >= PROGRESS_THROTTLE)

                if should_emit:
                    if not never_throttle:
                        last_progress = now
                    seq += 1
                    if is_tool_progress:
                        yield _event("response.in_progress", {
                            "type": "response.in_progress",
                            "sequence_number": seq,
                            "thought_role": Role.thought_tool_call_progress.value,
                            "content": event.message or "",
                            "tool_name": tool_names.get(event.id, ""),
                            "tool_use_id": event.id or "",
                        })
                    elif is_tool_done:
                        yield _event("response.in_progress", {
                            "type": "response.in_progress",
                            "sequence_number": seq,
                            "thought_role": Role.thought_tool_call_end.value,
                            "tool_use_id": event.id,
                            # Real measured duration from anton — without
                            # it, the frontend can only measure from the
                            # first tool_progress (the step is created
                            # lazily), undercounting time spent before the
                            # first progress line. Same field scratchpad
                            # already relies on for executionDurationMs.
                            "eta_seconds": getattr(event, "eta_seconds", None),
                            # Tool's own verdict (anton ToolOutcome.ok,
                            # ENG-1276) — None/True render as success on the
                            # frontend, only an explicit False marks the step
                            # failed. Without this, tool_done firing (which is
                            # unconditional by design, even on a handler
                            # exception) rendered as an unconditional success —
                            # the exact gap anton PR #304's review caught.
                            "ok": getattr(event, "ok", None),
                        })
                        progress_tool_ids.discard(event.id)
                        tool_names.pop(event.id, None)
                    else:
                        label = PHASE_LABELS.get(event.phase, event.phase)
                        msg = f"{label}: {event.message}" if event.message else label
                        yield _event("response.in_progress", {
                            "type": "response.in_progress",
                            "sequence_number": seq,
                            "thought_role": Role.thought_progress.value,
                            "content": msg,
                            "phase": event.phase,
                            "message": event.message,
                            "eta_seconds": getattr(event, "eta_seconds", None),
                            "tool_use_id": getattr(event, "id", None) or "",
                        })

        elif isinstance(event, StreamReasoningDelta):
            # The model's own reasoning text — NOT part of the final answer.
            # Shape matches hermes_harness's existing thought.progress +
            # subtype convention exactly, so the frontend's ephemeral
            # "current thought" handling (responseStreamAdapter.js) picks
            # this up identically without any client-side change.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_progress.value,
                "content": event.text,
                "subtype": "reasoning",
            })

        elif isinstance(event, StreamContextCompacted):
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_context_compacted.value,
                "content": event.message,
            })

        elif isinstance(event, ArtifactCreated):
            seq += 1
            yield _event("response.artifact_created", {
                "type": "response.artifact_created",
                "sequence_number": seq,
                "artifact": event.artifact,
            })

        elif isinstance(event, TurnHistory):
            # Recorded for LLM-history replay only — not surfaced to the client
            # (tool activity is rendered from the thought events above), so it
            # goes straight to the sink without an SSE frame or sequence number.
            if event_sink is not None:
                try:
                    event_sink("response.turn_history", {
                        "type": "response.turn_history",
                        "rows": event.rows,
                    })
                except Exception:
                    pass

        elif isinstance(event, SkillCreated):
            seq += 1
            yield _event("response.skill_created", {
                "type": "response.skill_created",
                "sequence_number": seq,
                "skill": event.skill,
            })

        elif isinstance(event, StreamComplete):
            # One LLM round is over. anton's agent loop streams the next
            # round's narration into the same output item, and a round ends
            # without trailing whitespace — so without a paragraph break the
            # rounds fuse mid-sentence ("…the Google Sheet.My regex…") in
            # the live stream, the persisted message, and every replay.
            # Exception: a round cut off at max_tokens with no tool calls is
            # resumed mid-sentence by a continuation round (see anton's
            # ChatSession._stream_and_handle_tools), so that boundary must
            # stay seamless.
            llm = event.response
            truncated = (
                llm.stop_reason in ("max_tokens", "length")
                and not llm.tool_calls
            )
            if not truncated:
                round_break = True
            elif round_had_text:
                round_break = False
            # A truncated round that streamed nothing (all thinking tokens)
            # leaves any armed break in place — the truncation belongs to
            # text the user never saw, so it can't glue the visible rounds.
            round_had_text = False

    full_text = "".join(collected_text)
    resp_completed = Response(
        id=resp_id,
        model=model,
        status=ResponseStatus.completed,
        output=[ResponseOutput(
            id=msg_id,
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=full_text)],
        )],
    )
    seq += 1
    yield _event("response.completed", {
        "type": "response.completed",
        "sequence_number": seq,
        "response": resp_completed.model_dump(),
    })
