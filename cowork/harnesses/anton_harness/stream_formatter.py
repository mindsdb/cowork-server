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
    "browser_action": "Browsing",
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


def classify_browser_status(content: str) -> str:
    """Classify a `browser_control` tool-result envelope into a UI status.

    The browser tool returns a JSON envelope whose `status` is one of the
    canonical `BrowserErrorKind`s (`ok` + five error kinds); an `ok` maps to
    the "ok" UI status and every error kind to "error". A hard failure is
    prefixed `ERROR: ` (for anton's circuit-breaker) with the JSON following;
    we strip that prefix before parsing. Anything unparseable is treated as
    an error so the renderer never shows an unobserved action as success.
    """
    if not content:
        return "error"
    text = content.strip()
    if text.startswith("ERROR:"):
        text = text[len("ERROR:"):].strip()
    try:
        env = json.loads(text)
    except (ValueError, TypeError):
        return "error"
    if isinstance(env, dict) and env.get("status") == "ok":
        return "ok"
    return "error"


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

    resp_id = f"resp-{uuid.uuid4().hex[:12]}"
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    seq = 0
    last_progress = 0.0
    collected_text: list[str] = []

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

    # Install the per-turn browser-span sink so the browser_control handler
    # can emit its content-free `browser_tool_span` through the same
    # event_sink that records the turn. Runs in the same asyncio task as the
    # tool dispatch, so the ContextVar is visible to the handler. Reset in
    # the generator's finally so it never leaks across turns.
    from .browser_telemetry import reset_span_sink, set_span_sink

    _span_token = set_span_sink(
        (lambda et, data: _event(et, data)) if event_sink is not None else None
    )

    resp = Response(id=resp_id, model=model, status=ResponseStatus.created)
    seq += 1
    created_data = {
        "type": "response.created",
        "sequence_number": seq,
        "response": resp.model_dump(),
    }
    yield _event("response.created", created_data)

    try:
        async for event in event_stream:
            if isinstance(event, StreamTextDelta):
                collected_text.append(event.text)
                seq += 1
                yield _event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "sequence_number": seq,
                    "item_id": msg_id,
                    "delta": event.text,
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
                name = tool_names.pop(event.id, "")
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
                # The browser tool's envelope carries a canonical `status`, so its
                # cell_status comes from a browser-aware classifier (ok/error)
                # rather than the scratchpad exec-cell classifier.
                _tool_name = getattr(event, "name", "") or ""
                if _tool_name == "browser_control":
                    _cell_status = classify_browser_status(event.content)
                else:
                    _cell_status = classify_cell_status(event.content)
                yield _event("response.in_progress", {
                    "type": "response.in_progress",
                    "sequence_number": seq,
                    "thought_role": Role.thought_scratchpad_result.value,
                    "content": event.content[:65536],
                    "tool_name": _tool_name,
                    "tool_action": getattr(event, "action", "") or "",
                    "tool_use_id": getattr(event, "id", None) or "",
                    # ok / timeout / error — lets the renderer show a killed cell
                    # as dead instead of stuck "running". Additive: older clients
                    # ignore the field.
                    "cell_status": _cell_status,
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
                now = time.time()
                should_emit = is_scratchpad_phase or (now - last_progress >= PROGRESS_THROTTLE)
                if should_emit:
                    if not is_scratchpad_phase:
                        last_progress = now
                    label = PHASE_LABELS.get(event.phase, event.phase)
                    msg = f"{label}: {event.message}" if event.message else label
                    seq += 1
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

            elif isinstance(event, SkillCreated):
                seq += 1
                yield _event("response.skill_created", {
                    "type": "response.skill_created",
                    "sequence_number": seq,
                    "skill": event.skill,
                })

            elif isinstance(event, StreamComplete):
                pass

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
    # Reset the per-turn browser-span sink in a finally so it is always
    # restored — even if the stream errors or the producer is cancelled.
    # A subsequent turn installs its own sink at the top of the generator.
    finally:
        reset_span_sink(_span_token)
