"""SSE event formatter — turns ChatSession.turn_stream() events into
OpenAI Responses API SSE strings.

Emits typed events:
    response.created            (with conversation_id)
    response.in_progress        (thought/tool activity, carries thought_role)
    response.output_text.delta  (assistant text deltas)
    response.answer_reset       (drop the answer so far; the next delta replaces it)
    response.answer_restore     (a hand-back gives back the answer a reset dropped)
    response.figure_sources     (each figure in the answer and the steps it came from)
    response.completed          (final response object)
    response.failed             (error)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, AsyncIterator, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.figure_sources import (
    CellResult,
    FigureSources,
    ToolArguments,
    ToolInput,
    TurnStep,
    find_figure_sources,
    has_figures,
    is_evidence_call,
    is_evidence_tool,
    mark_repeated_figures,
    ran_cell,
)
from cowork.schemas.figure_sources import EarlierAnswer, FigureOrigin
from cowork.schemas.responses import (
    IncompleteDetails,
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    Role,
)
from cowork.streaming.liveness import MODEL_WAIT_PHASE, model_wait_sse

logger = logging.getLogger(__name__)


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
    existing thought events, so these rows are for LLM-history replay. The
    formatter also reads the tool results in them: anton streams only the
    scratchpad's, and a figure can come from a web tool's, read_text_file's
    or a connector tool's (figure_sources.is_evidence_tool). Both lists
    stay as the producer sent them; the formatter parses them (ToolRows)
    only when the answer has figures."""

    rows: list[dict[str, Any]]
    # The rows as the agent read them, when the producer stores fewer: the
    # remote paths cap each result, and a turn that compacted its context
    # mid-way stores none (its rows are near or over the context window, so
    # the next turn replays it as text only). The formatter reads these
    # instead; None means `rows` are what the agent read. The pod's rows
    # arrive unchecked, so an entry may not be a row at all.
    read: list[object] | None = None


@dataclass
class TurnContext:
    """Synthetic pre-turn event: what the formatter needs to know about the
    turn that its other events don't carry. Never emitted as SSE. Every
    producer of a turn's stream yields it first: AntonHarness.stream_response
    for a turn in this process (from the web UI or a channel),
    ResponsesHandler._run_remote_turn for a web turn on the remote worker,
    and remote_turn_events for a channel turn there.

    ``question`` is the text the turn was sent with. The formatter counts as
    the question only the part the user typed (_typed_text), so that a figure
    the answer repeats from it is marked as the question's rather than as
    having no source. ``files`` finds the project file a figure's step read
    (None when the producer can't know the project). ``history`` is the
    conversation's earlier messages as the model is seeded with them. Its
    cells pass the names, functions and data they bound on to this turn's
    cells, so a figure that a cell of this turn computed from data an earlier
    turn's cell read names that cell too. The formatter reads it only when
    the answer has figures and this turn ran a cell, and then only its last
    figure_sources.MAX_EARLIER_CELLS cells; no figure's origin depends on it.
    The formatter reads only its tool_use and tool_result blocks (ToolRows)
    and leaves every other message out.

    ``earlier_answers`` are the figures the conversation's last
    conversations.MAX_EARLIER_ANSWERS answers list in their saved events,
    which the producer reads when the turn starts
    (conversations.read_earlier_answers). An earlier turn's step never holds
    a figure itself. A figure no step of this turn holds and the question
    doesn't write is marked as an earlier answer's when one of those answers
    wrote it in the same form with an origin other than none
    (figure_sources.mark_repeated_figures); nothing of that answer is
    copied."""

    question: str
    files: AgentFiles | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    earlier_answers: tuple[EarlierAnswer, ...] = ()


@dataclass(frozen=True)
class StreamedResult:
    """A tool result the stream carried, with the call's arguments as the
    model streamed them. Kept raw while the turn streams; the arguments are
    parsed (ToolInput.from_text) only when the answer has figures, in the
    worker thread that finds their sources."""

    tool_use_id: str
    tool_name: str
    action: str
    tool_input: str
    content: str

    def step(self) -> TurnStep:
        """The result as a step the figure matcher reads."""
        return TurnStep(
            tool_use_id=self.tool_use_id,
            tool_name=self.tool_name,
            action=self.action,
            tool_input=ToolInput.from_text(text=self.tool_input),
            content=self.content,
        )


@dataclass(frozen=True)
class EvidenceCall:
    """One of the turn's calls of a tool figure_sources.is_evidence_tool
    counts, with its arguments as the model streamed them (empty when the
    stream never ended the call). Kept raw while the turn streams; the
    arguments are parsed only when the turn compacted, no tool rows
    arrived for it, and the call's tool_done said ok=True."""

    tool_use_id: str
    tool_name: str
    tool_input: str

    def counts_as_step(self, *, successful_tool_calls: frozenset[str]) -> bool:
        """Whether the matcher would count the call's result as a step, by
        the tests it applies to that result: the call's tool_done said
        ok=True (TurnEvidence.successful_tool_calls), and it is not a read
        of a memory file (figure_sources.is_evidence_call)."""
        return self.tool_use_id in successful_tool_calls and is_evidence_call(
            tool_name=self.tool_name, arguments=ToolArguments.from_text(text=self.tool_input),
        )


@dataclass(frozen=True)
class TurnEvidence:
    """What an answer's figures are matched against."""

    answer: str
    # The text the turn was sent with (TurnContext.question).
    question: str
    # What the user typed answering the agent's questions during the turn.
    answers: tuple[str, ...] = ()
    # The tool results the stream carried, which anton sends only for the
    # scratchpad.
    steps: tuple[StreamedResult, ...] = ()
    # The turn's tool rows, which hold every tool's result (TurnHistory), and
    # the conversation's earlier messages (TurnContext.history), both as
    # their producers sent them.
    tool_rows: Sequence[object] = ()
    history: Sequence[object] = ()
    files: AgentFiles | None = None
    # The calls, of any tool, whose tool_done said ok=True. A history row
    # of a web tool's, read_text_file's or a connector tool's result omits
    # the call's verdict, and a result whose verdict is False or None can
    # be error text the tool or anton wrote, which can quote figures, such
    # as an API error's status code. So only an explicit True lets such a
    # result be a step. A scratchpad cell's printed output stays usable
    # when the cell failed.
    successful_tool_calls: frozenset[str] = frozenset()
    # This turn's calls of the tools figure_sources.is_evidence_tool counts,
    # as the stream announced them, when the turn compacted its context
    # mid-way (the stream carried a StreamContextCompacted) and no tool rows
    # arrived for it (no TurnHistory): their results never reached the
    # formatter. A producer whose anton predates ChatSession.turn_messages
    # (an older cloud pod, or a PyPI-channel desktop install on the anton
    # floor) sends no rows after a compaction. Empty otherwise.
    calls_without_rows: tuple[EvidenceCall, ...] = ()
    # The figures the conversation's last answers list (TurnContext
    # .earlier_answers), which a figure no step holds may repeat.
    earlier_answers: tuple[EarlierAnswer, ...] = ()


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
    # ENG-1537: anton is deliberately idle, waiting out a velocity rate-limit
    # before resuming the same step. Without a label the renderer would show the
    # raw phase ("rate_limited: waiting 30s…"), which reads like a leaked
    # constant. The turn is still alive — this is not a failure state.
    "rate_limited": "Rate limited",
    # A skill draft the pod reported was rejected, or trimmed a sibling file —
    # see `remote_skill_draft_result` (cowork/services/task_objects.py).
    "skill_draft_dropped": "Skill draft",
    # The completion verifier forced a continuation; the text after this
    # replaces the answer streamed before it.
    "continuation": "Continuing",
    # The turn is handing back with an explanation instead of continuing, so
    # the text after it adds to the answer rather than replacing it.
    "handback": "Wrapping up",
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


# anton's reply when every attempt at the turn failed: its error, not an answer
# (the fallback reply in anton core.session's turn loop). Final anton releases
# through 2.26.10.14.1 (the anton-agent floor in pyproject.toml) and older pod
# images still send it; from mindsdb/anton#569 on, the turn raises instead. The
# check can go once the floor and the pod image both carry #569.
_ANTON_FAILURE_REPLY = "An unexpected error occurred: "

# The header over each block the client appends, for the agent only, to the
# text the user typed: a connect form's state, picked Drive files, attached
# Notion pages.
_APPENDED_CONTEXT = re.compile(r"^\[[^\n]*Anton-only context[^\n]*\]$", re.MULTILINE)

# anton hands the model a cell's result in labeled parts (format_cell_result):
# what the cell printed under [output], then [logs], [stderr] and [error].
# anton's session appends [artifact lint] after a cell that changed an
# artifact.
_CELL_RESULT_PART = re.compile(r"^\[(?:output|logs|stderr|error|artifact lint)\]\n", re.MULTILINE)
# The note format_cell_result puts after printed output it cut short, with
# the line breaks before the next part.
_TRUNCATION_NOTE = re.compile(r"\n\n\.\.\. \(truncated, \d+ chars total\)\n*\Z")


class _HistoryRow(BaseModel):
    """A history row: its text, or its blocks, each read on its own."""

    model_config = ConfigDict(extra="ignore")

    content: str | list[Any] = ""


class _ToolUseBlock(BaseModel):
    """An assistant row's tool call, as far as the figure matcher reads it."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class _ResultPart(BaseModel):
    """A part of a tool result: text, or another kind (an image) that has
    none."""

    model_config = ConfigDict(extra="ignore")

    type: str
    text: str = ""


class _ToolResultBlock(BaseModel):
    """A user row's tool result, as far as the figure matcher reads it."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[_ResultPart] = ""
    # anton marks a result is_error only when it answered the call itself
    # without running it: arguments it refused, or a call an error cut off.
    # None of such a result is the tool's output.
    is_error: bool = False

    @property
    def text(self) -> str:
        """The result's text: the string, or the text parts of the list."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content if part.type == "text")


class _CellArguments(BaseModel):
    """A scratchpad call's arguments, as far as a replayed cell reads them."""

    model_config = ConfigDict(extra="ignore")

    action: str = ""
    code: str = ""
    one_line_description: str = ""


_ROW_BLOCK: TypeAdapter[_ToolUseBlock | _ToolResultBlock] = TypeAdapter(
    Annotated[_ToolUseBlock | _ToolResultBlock, Field(discriminator="type")],
)


@dataclass(frozen=True)
class ToolRows:
    """The tool calls and results in history rows (an assistant row's
    tool_use blocks, a user row's tool_result blocks), each keyed by its
    tool_use_id, in row order. Read once (ToolRows.read) and then asked for
    its steps and the results it holds."""

    calls: Mapping[str, _ToolUseBlock]
    results: Mapping[str, _ToolResultBlock]

    @classmethod
    def read(cls, *, rows: Sequence[object]) -> ToolRows:
        """The tool_use and tool_result blocks of ``rows``. Any other block,
        and a row or block that doesn't read as one, is left out. A later
        block with the same tool_use_id replaces an earlier one."""
        calls: dict[str, _ToolUseBlock] = {}
        results: dict[str, _ToolResultBlock] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                content = _HistoryRow.model_validate(row).content
            except ValidationError:
                continue
            for block in content if isinstance(content, list) else ():
                try:
                    parsed = _ROW_BLOCK.validate_python(block)
                except ValidationError:
                    continue
                if isinstance(parsed, _ToolUseBlock):
                    calls[parsed.id] = parsed
                else:
                    results[parsed.tool_use_id] = parsed
        return cls(calls=calls, results=results)

    def steps(self) -> list[TurnStep]:
        """Each call that has its result here, in call order. A call whose
        result anton marked is_error never ran, so it is left out. The
        matcher decides which tools' calls are steps
        (figure_sources.is_evidence_tool)."""
        steps = []
        for tool_use_id, call in self.calls.items():
            result = self.results.get(tool_use_id)
            if result is None or result.is_error:
                continue
            cell = None
            action = ""
            if call.name == "scratchpad":
                arguments = _cell_arguments(call.input)
                action = arguments.action
                cell = _replayed_cell(result.text, arguments=arguments)
            steps.append(TurnStep(
                tool_use_id=tool_use_id,
                tool_name=call.name,
                action=action,
                tool_input=ToolInput.from_call(arguments=call.input),
                content=result.text,
                cell=cell,
            ))
        return steps


async def find_answer_sources(evidence: TurnEvidence) -> FigureSources:
    """The figures in an answer and the steps they came from.

    Found in a worker thread: finding a step's file reads the disk, a shared
    mount in org mode, and every other request waits on the loop. Never
    raises: the sources annotate an answer that is already whole, so failing
    to find them must not fail the answer.
    """
    try:
        return await asyncio.to_thread(_answer_sources, evidence)
    except Exception:
        logger.exception("[stream_formatter] could not find the answer's figure sources")
        return FigureSources()


def _answer_sources(evidence: TurnEvidence) -> FigureSources:
    """find_answer_sources's work, run in a worker thread. An answer with no
    figures reads none of the turn's rows or history."""
    if not has_figures(evidence.answer):
        return FigureSources()
    streamed = [result.step() for result in evidence.steps]
    streamed_ids = {step.tool_use_id for step in streamed}
    rows = ToolRows.read(rows=evidence.tool_rows)
    from_rows = [step for step in rows.steps() if step.tool_use_id not in streamed_ids]
    this_turn = [
        step for step in [*streamed, *from_rows]
        if not is_evidence_tool(name=step.tool_name) or step.tool_use_id in evidence.successful_tool_calls
    ]
    # An earlier turn's cells only pass names, functions and data on to this
    # turn's cells, so a turn that ran no cell reads none of them. Oldest
    # first, so what they bound reaches this turn's cells. An earlier turn's
    # step never holds a figure.
    earlier = (
        [replace(step, earlier_turn=True) for step in ToolRows.read(rows=evidence.history).steps()]
        if any(ran_cell(step) for step in this_turn) else []
    )
    question = "\n".join([_typed_text(evidence.question), *evidence.answers])
    sources = find_figure_sources(
        answer=evidence.answer, question=question, steps=[*earlier, *this_turn], files=evidence.files,
    )
    try:
        sources = mark_repeated_figures(sources=sources, earlier_answers=evidence.earlier_answers)
    except Exception:
        # find_answer_sources would drop every source this turn found, so a
        # failure here keeps them and marks no figure as repeated.
        logger.exception("[stream_formatter] could not mark the figures an earlier answer wrote")
    if any(
        call.counts_as_step(successful_tool_calls=evidence.successful_tool_calls)
        for call in evidence.calls_without_rows
    ):
        # The turn compacted mid-way and no tool rows arrived for it, so a web
        # page, a file or a connector's records it read are missing, and a
        # figure no step holds may be one it read: such a figure gets no mark
        # rather than "no source". A figure an earlier answer gave keeps its
        # mark. A call the matcher would not count as a step hides no source:
        # one whose tool_done did not say ok=True (it failed, gave no
        # verdict, or sat in a round anton threw away), a read of a memory
        # file, or a call of a connector tool that returns no records. Tool
        # rows that do arrive hold every call that ran, through a compaction
        # too; a call missing there never ran, as in a round anton threw away
        # and retried. This can go once production pods run an anton with
        # ChatSession.turn_messages, which sends the rows after a compaction,
        # and the anton-agent floor in pyproject.toml requires one.
        return sources.model_copy(update={
            "figures": [figure for figure in sources.figures if figure.origin != FigureOrigin.none],
        })
    return sources


def _typed_text(text: str) -> str:
    """``text`` up to the first block the client appended to it."""
    appended = _APPENDED_CONTEXT.search(text)
    return text[:appended.start()].rstrip() if appended else text


def _cell_arguments(arguments: dict[str, Any]) -> _CellArguments:
    """A scratchpad call's arguments, or none when they don't read as one."""
    try:
        return _CellArguments.model_validate(arguments)
    except ValidationError:
        return _CellArguments()


def _replayed_cell(result: str, *, arguments: _CellArguments) -> CellResult:
    """A cell's result as history rows hold it: its code and description
    from the call's arguments, and what it printed, the [output] part,
    without the logs, stderr, error and artifact lint the model saw beside
    it, or the note on how long output cut short was."""
    stdout = ""
    if result.startswith("[output]\n"):
        printed = result[len("[output]\n"):]
        next_part = _CELL_RESULT_PART.search(printed)
        stdout = _TRUNCATION_NOTE.sub("", printed[:next_part.start()] if next_part else printed)
    return CellResult(code=arguments.code, stdout=stdout, description=arguments.one_line_description)


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
        StreamAskUser,
        StreamAskUserAnswered,
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamTextDelta,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
        StreamTurnEnded,
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
    # anton's completion verifier can force a continuation, whose text replaces
    # the answer streamed before it instead of continuing it. Latched when that
    # boundary arrives and spent on the next text delta — never at the boundary
    # itself, because a continuation that never speaks (stopped, failed, or all
    # tool calls) must leave the answer the user already read standing.
    superseded_pending = False
    # What the last spent boundary set aside, returned if the turn hands back
    # instead of delivering the replacement it promised.
    superseded_stash = ""
    # Why a turn that handed back stopped (anton's `ended_by`); None if it finished.
    ended_by: str | None = None

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
    # Each tool call's arguments and each result, kept to find where the
    # answer's figures came from once the answer is final.
    tool_inputs: dict[str, str] = {}
    turn_results: list[StreamedResult] = []
    turn_context = TurnContext(question="")
    user_answers: list[str] = []
    tool_rows: Sequence[object] = ()
    # Whether a TurnHistory arrived, even one with no rows.
    tool_rows_arrived = False
    # The tool name of each call of a tool figure_sources.is_evidence_tool
    # counts, by tool_use_id.
    evidence_calls: dict[str, str] = {}
    successful_tool_calls: set[str] = set()
    compacted = False
    # Whether the turn's last text is anton's failure reply, which carries no
    # figures. Final anton releases through 2.26.10.14.1 (the anton-agent
    # floor) and older pod images stream that reply as one delta and end the
    # turn on it, so each delta with visible text sets or clears this. From
    # mindsdb/anton#569 on, the turn raises instead, so this can go once the
    # floor and the pod image both carry #569.
    failure_reply = False

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
            if event.text.strip():
                failure_reply = event.text.startswith(_ANTON_FAILURE_REPLY)
            # `.strip()`, not truthiness: a lone newline is text enough to
            # satisfy the latch and not enough to be an answer, so spending it
            # there leaves a blank message where the user had read a real one.
            if superseded_pending and event.text.strip():
                superseded_pending = False
                # Held, not dropped. A continuation normally narrates before its
                # first tool call, so this text is not always the standalone
                # answer it promised; if it hands back instead, the answer that
                # was already read has to come back rather than be lost from the
                # transcript and from the history the next turn is rebuilt from.
                superseded_stash = "".join(collected_text)
                collected_text.clear()
                # The armed paragraph break belongs between two rounds of the
                # answer being discarded, not in front of its replacement.
                round_break = False
                round_had_text = False
                text_tail = ""
                seq += 1
                yield _event("response.answer_reset", {
                    "type": "response.answer_reset",
                    "sequence_number": seq,
                    "item_id": msg_id,
                })
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
            if is_evidence_tool(name=event.name):
                evidence_calls[event.id] = event.name
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
            tool_inputs[event.id] = accumulated
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

        elif (
            isinstance(event, StreamToolResult)
            and getattr(event, "action", None) == "message"
            # A scratchpad result carries the model's own `action`.
            and getattr(event, "name", "") != "scratchpad"
        ):
            # A tool's message to the user (generate_artifact's brief when the
            # agent acts first). Its own role rather than the scratchpad
            # result's: a client that predates it ignores it instead of
            # patching it into the last scratchpad cell. Not answer text, so
            # it stays out of the persisted assistant message and the history.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_tool_call_message.value,
                "content": event.content[:65536],
                "tool_name": getattr(event, "name", "") or "",
                "tool_use_id": getattr(event, "id", None) or "",
            })

        elif isinstance(event, StreamToolResult):
            tool_use_id = getattr(event, "id", None) or ""
            turn_results.append(StreamedResult(
                tool_use_id=tool_use_id,
                tool_name=getattr(event, "name", "") or "",
                action=getattr(event, "action", "") or "",
                tool_input=tool_inputs.get(tool_use_id, ""),
                content=event.content,
            ))
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
            if phase_str == "tool_done" and event.id:
                # ChatSession.turn_stream emits this even when the tool has
                # no tool_progress. Capture it before UI throttling or name
                # removal; evidence filtering uses the result's tool name.
                if event.ok is True:
                    successful_tool_calls.add(event.id)
                else:
                    successful_tool_calls.discard(event.id)
            # A keep-alive from the cloud pod while a model call is quiet.
            # Never throttled: dropping it lets the idle bounds end a live
            # turn, and it cannot flood the stream: the pod writes one from
            # its heartbeat only after MODEL_WAIT_TICK_S (20 s,
            # anton.core.llm.liveness) of otherwise quiet wire, see
            # anton.cloud_turn.__main__._model_wait_line. It also stays out
            # of event_sink and off the throttle anchor: it says the turn is
            # alive, and a reload has nothing to show for it.
            if phase_str == MODEL_WAIT_PHASE:
                seq += 1
                yield model_wait_sse(
                    message=event.message or "",
                    waited_s=getattr(event, "eta_seconds", None),
                    sequence_number=seq,
                )
                continue
            # Latched ahead of the throttling below: this notice is
            # rate-limited like any other, but the reset it implies is not.
            if phase_str == "continuation":
                superseded_pending = True
            # anton gave up and is explaining instead of continuing. Its
            # diagnosis is an extra message, not a replacement, so an unspent
            # boundary is dropped and a spent one gives its text back — the
            # answer the continuation failed to improve on is still the only
            # real content the turn produced, and the next turn's history is
            # rebuilt from it.
            elif phase_str == "handback":
                superseded_pending = False
                if superseded_stash:
                    collected_text.insert(0, superseded_stash)
                    seq += 1
                    yield _event("response.answer_restore", {
                        "type": "response.answer_restore",
                        "sequence_number": seq,
                        "item_id": msg_id,
                        "text": superseded_stash,
                    })
                    superseded_stash = ""
            is_scratchpad_phase = phase_str in ("scratchpad_start", "scratchpad_done")
            # ENG-1537: the rate-limit notice must never be throttled away. It
            # fires once per wait and is the ONLY thing distinguishing a
            # deliberate 90s pause from a hang — drop it and a correct fix gets
            # reported as a freeze. One event per wait, so exempting it can't
            # flood the stream.
            is_rate_limited_notice = phase_str == "rate_limited"
            # A lost/trimmed skill draft is one-shot per draft, same shape as
            # the rate-limit notice — throttling it away would silently
            # recreate the exact bug it exists to surface.
            is_skill_draft_notice = phase_str == "skill_draft_dropped"
            is_tool_progress = phase_str == "tool_progress"
            is_tool_done = phase_str == "tool_done" and event.id in progress_tool_ids

            if is_tool_progress and not event.id:
                # Can't correlate to a step at all — drop it before it
                # touches progress_tool_ids or spends the throttle window
                # on nothing.
                pass
            else:
                # Every tool_progress line is a step announcement and is
                # never throttled (ENG-2981): reasoning_start from a tool's
                # own LLM calls lands in the same window, and dropping the
                # line loses a pipeline step from the chat. Losing the FIRST
                # line would also leave the renderer's lazily-created tool
                # step missing, so a later tool_done would close nothing.
                #
                # Invariant: a tool emits a handful of these per run (today
                # only anton's generate_artifact, one line per pipeline
                # step). A tool that streams many lines must not use
                # tool_progress — the live tail has its own tool_peek phase,
                # relayed only to hosts that set live_tool_peek (this server
                # does not). If that ever changes, revisit this exemption,
                # e.g. drop only a repeat of the same line for the same id.
                if is_tool_progress:
                    progress_tool_ids.add(event.id)

                # tool_done is never throttled for the same reason
                # scratchpad_done isn't — it's the ONLY event that closes a
                # tool-progress step, and it's yielded right after the last
                # tool_progress (well within one PROGRESS_THROTTLE window).
                never_throttle = (
                    is_scratchpad_phase
                    or is_tool_done
                    or is_tool_progress
                    or is_rate_limited_notice
                    or is_skill_draft_notice
                )
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
            # Shape follows the thought.progress + subtype convention the
            # frontend's ephemeral "current thought" handling
            # (responseStreamAdapter.js) already understands, so no
            # client-side change is needed.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_progress.value,
                "content": event.text,
                "subtype": "reasoning",
            })

        elif isinstance(event, StreamContextCompacted):
            compacted = True
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

        elif isinstance(event, StreamAskUser):
            seq += 1
            request = event.request
            yield _event("response.ask_user", {
                "type": "response.ask_user",
                "sequence_number": seq,
                "question_id": event.id,
                "prompt": request.prompt,
                "select": request.select,
                "allow_custom": request.allow_custom,
                "timeout_s": request.timeout_s,
                "options": [
                    {"value": o.value, "label": o.label, "detail": o.detail}
                    for o in request.options
                ],
            })

        elif isinstance(event, StreamAskUserAnswered):
            seq += 1
            answer = event.answer
            # Typed by the user, so a figure in it is the question's. A picked
            # option is the agent's own wording and stays out.
            if answer.status == "answered" and answer.text:
                user_answers.append(answer.text)
            yield _event("response.ask_user_answered", {
                "type": "response.ask_user_answered",
                "sequence_number": seq,
                "question_id": event.id,
                "status": answer.status,
                "values": list(answer.values),
                "text": answer.text,
            })

        elif isinstance(event, TurnHistory):
            tool_rows = event.read if event.read is not None else event.rows
            tool_rows_arrived = True
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

        elif isinstance(event, TurnContext):
            turn_context = event

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

        elif isinstance(event, StreamTurnEnded):
            ended_by = event.ended_by

    full_text = "".join(collected_text)
    # A turn the user stops, or one that fails or stalls, never reaches this
    # line, so its partial answer carries no figure sources. A turn that
    # handed back (ended_by) does: its answer is whole, so its figures get
    # sources as a finished turn's do, before the final event.
    sources = FigureSources() if failure_reply else await find_answer_sources(TurnEvidence(
        answer=full_text,
        question=turn_context.question,
        answers=tuple(user_answers),
        steps=tuple(turn_results),
        tool_rows=tool_rows,
        history=turn_context.history,
        files=turn_context.files,
        successful_tool_calls=frozenset(successful_tool_calls),
        earlier_answers=turn_context.earlier_answers,
        calls_without_rows=tuple(
            EvidenceCall(tool_use_id=call_id, tool_name=name, tool_input=tool_inputs.get(call_id, ""))
            for call_id, name in evidence_calls.items()
        ) if compacted and not tool_rows_arrived else (),
    ))
    if sources.figures:
        seq += 1
        yield _event("response.figure_sources", {
            "type": "response.figure_sources",
            "sequence_number": seq,
            "item_id": msg_id,
            **sources.model_dump(mode="json"),
        })
    # Still response.completed, so every client sees the answer end; the
    # status says it stopped early. The message itself is whole either way.
    resp_completed = Response(
        id=resp_id,
        model=model,
        status=ResponseStatus.incomplete if ended_by else ResponseStatus.completed,
        incomplete_details=IncompleteDetails(reason=ended_by) if ended_by else None,
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
