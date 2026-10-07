"""A silent model call must not end an in-process turn; a hung turn still must.

anton reports a model call waiting on the provider through
``session.model_calls``. While it does, and the turn's buffer has been quiet
for ``MODEL_WAIT_TICK_SECONDS``, the producer's ticker appends a ``model_wait``
frame, which resets the server watchdog, the Redis tail and the UI's idle cut.
When anton reports nothing (a hung tool, an open question, an anton without
the tracker), no frame goes out and the watchdog reaps the turn as before.

Timings are shrunk: the watchdog reaps after 0.15 s of no records, and the
ticker writes after 0.05 s of quiet, so a 0.5 s silent call outlives the
watchdog three times over.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace
from uuid import uuid4

import fakeredis.aioredis
import pytest
from anton.core.llm.provider import StreamTextDelta

import cowork.handlers.responses as responses_mod
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, INTERRUPTED_TURN_MESSAGE
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.streaming import buffer as buffer_mod
from cowork.streaming.buffer import FileStreamBuffer, RedisStreamBuffer, read_records, turn_buffer_path
from cowork.streaming.registry import TurnLifecycle, registry

# Imported softly so that, on a build without the ticker, the turn-level tests
# below fail on what the turn does rather than on this import.
try:
    from cowork.streaming import liveness
except ImportError:
    liveness = None

CID = "conv-model-wait"
SILENT_CALL_SECONDS = 0.5


@pytest.fixture(autouse=True)
def _shrunk_bounds(monkeypatch):
    registry_module = sys.modules["cowork.streaming.registry"]
    monkeypatch.setattr(registry_module, "_IDLE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(registry_module, "_MAX_TURN_IDLE_SECONDS", 0.15)
    if liveness is not None:
        monkeypatch.setattr(liveness, "MODEL_WAIT_TICK_SECONDS", 0.05)
        monkeypatch.setattr(liveness, "MODEL_WAIT_POLL_SECONDS", 0.01)
    registry.reset()
    yield
    registry.reset()


class _Tracker:
    """anton's ModelCallTracker as the ticker sees it."""

    def __init__(self, waiting: bool) -> None:
        self.waiting = waiting

    def snapshot(self):
        if not self.waiting:
            return None
        return SimpleNamespace(
            role="planning", open_for_s=61.0, quiet_for_s=61.0,
            message="Waiting for the model (1m 1s)",
        )


class _Harness:
    """An anton-like harness whose only model call is silent.

    ``session`` is what it attaches to the ticker: one with a tracker, or one
    without the attribute, as an anton that predates it. Real anton builds the
    session with ``model_calls = None`` and arms the turn's tracker inside
    ``turn_stream``, after the harness has attached the session, so this
    harness holds the tracker back until after ``attach`` the same way. A
    ticker that read the tracker once at attach would then see None and fail
    these tests. ``hang`` makes the call never return, the shape of a hung
    tool once the tracker reports nothing. ``partial`` is text streamed before
    the call goes quiet, and ``raises`` is what the call ends with instead of
    its answer. ``then_hang`` answers the silent call, stops reporting a wait,
    and then hangs, the shape of a tool the model called that never returns.
    """

    formatter = staticmethod(format_responses_stream)

    def __init__(
        self, *, session: object, hang: bool = False, partial: str | None = None,
        raises: Exception | None = None, then_hang: bool = False,
    ) -> None:
        self.session = session
        self._tracker = getattr(session, "model_calls", None)
        if self._tracker is not None:
            session.model_calls = None
        self.hang = hang
        self.partial = partial
        self.raises = raises
        self.then_hang = then_hang

    async def stream_response(
        self, *, conversation, input, model=None, reasoning_effort=None,
        disabled_connections=None, trace_tags=None, trace_metadata=None,
        channel_context=None, tool_messages=False, model_wait=None,
    ):
        if model_wait is not None:
            model_wait.attach(session=self.session)
        if self._tracker is not None:
            self.session.model_calls = self._tracker
        try:
            if self.partial is not None:
                yield StreamTextDelta(text=self.partial)
            if self.hang:
                await asyncio.Event().wait()
            await asyncio.sleep(SILENT_CALL_SECONDS)
            if self.raises is not None:
                raise self.raises
            if self.then_hang:
                self._tracker.waiting = False
            yield StreamTextDelta(text="done")
            if self.then_hang:
                await asyncio.Event().wait()
        finally:
            if model_wait is not None:
                model_wait.detach()


def _handler(monkeypatch, saved: dict, harness: _Harness):
    handler = object.__new__(responses_mod.ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            return SimpleNamespace(id=uuid4())

        def finalize_pending(self, conv_id, message_id=None):
            pass

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events

    class FakeSession:
        def close(self):
            pass

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness)
    return handler


async def _run(monkeypatch, buffer, harness: _Harness, saved: dict):
    handler = _handler(monkeypatch, saved, harness)
    lifecycle = TurnLifecycle()
    handle = await registry.start(
        conversation_id=CID, turn_id=0, buffer=buffer, lifecycle=lifecycle,
        producer_coro=handler._run_turn(
            conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
            disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
            lifecycle=lifecycle,
        ),
    )
    return handle


def _model_wait_frames(sse_frames: list[str]) -> list[dict]:
    payloads = [json.loads(f.split("data: ", 1)[1]) for f in sse_frames if "data: " in f]
    return [p for p in payloads if p.get("phase") == "model_wait"]


async def test_a_silent_model_call_outlives_the_watchdog(monkeypatch, tmp_path):
    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    harness = _Harness(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))

    handle = await _run(monkeypatch, buffer, harness, saved)
    await asyncio.wait_for(handle.task, timeout=5)

    records = list(read_records(buffer.path))
    assert handle.lifecycle.timed_out is False
    assert records[-1].type == "Done"
    assert saved["assistant"] == "done"
    waits = _model_wait_frames([r.data.get("sse", "") for r in records])
    assert len(waits) >= 3
    assert waits[0]["type"] == "response.in_progress"
    assert waits[0]["thought_role"] == "thought.progress"
    assert waits[0]["content"] == "Still working: Waiting for the model (1m 1s)"
    assert waits[0]["eta_seconds"] == 61.0
    # A live signal only: a reload replays the saved events, which never hold it.
    assert not [e for e in saved["events"] if e.get("phase") == "model_wait"]


async def _collect(reader):
    return [rec async for rec in reader.tail(0)]


async def test_a_silent_model_call_outlives_the_redis_tail(monkeypatch, caplog):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(buffer_mod, "get_redis", lambda: client)
    monkeypatch.setattr(buffer_mod, "REDIS_TAIL_IDLE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(RedisStreamBuffer, "_BLOCK_MS", 20)
    caplog.set_level(logging.WARNING)
    saved: dict = {}
    buffer = RedisStreamBuffer(conversation_id=CID, turn_id=0)
    harness = _Harness(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))

    handle = await _run(monkeypatch, buffer, harness, saved)
    # A reader on another replica, as the web tier tails a turn.
    reader = RedisStreamBuffer(conversation_id=CID, turn_id=0)
    records = await asyncio.wait_for(_collect(reader), timeout=5)
    await asyncio.wait_for(handle.task, timeout=5)

    assert records[-1].type == "Done"
    # No warning from the Redis tail, and none anywhere about going quiet.
    assert not [
        r for r in caplog.records
        if (r.name == "cowork.streaming.buffer" and r.levelno >= logging.WARNING)
        or "went quiet" in r.getMessage()
    ]
    assert not [r for r in records if r.type == "Interrupted"]
    assert _model_wait_frames([r.data.get("sse", "") for r in records])
    assert not [e for e in saved["events"] if e.get("phase") == "model_wait"]


@pytest.mark.parametrize(
    "session",
    [
        pytest.param(SimpleNamespace(model_calls=_Tracker(waiting=False)), id="nothing-waiting"),
        pytest.param(SimpleNamespace(), id="anton-without-tracker"),
    ],
)
async def test_a_hung_turn_with_no_waiting_call_is_still_reaped(monkeypatch, tmp_path, session):
    # A tool that never returns, or an open question: anton reports no call
    # waiting on the provider, so no frame goes out and the watchdog ends the
    # turn exactly as it would without the ticker.
    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    harness = _Harness(session=session, hang=True)

    handle = await _run(monkeypatch, buffer, harness, saved)
    await asyncio.wait_for(handle.task, timeout=5)

    records = list(read_records(buffer.path))
    assert handle.lifecycle.timed_out is True
    assert records[-1].type == "Interrupted"
    assert not _model_wait_frames([r.data.get("sse", "") for r in records])
    assert saved["events"][-1]["code"] == GENERIC_TURN_ERROR_CODE
    assert saved["events"][-1]["error"] == INTERRUPTED_TURN_MESSAGE


async def test_a_tool_that_hangs_after_a_silent_call_is_still_reaped(monkeypatch, tmp_path):
    # The model is what calls the tool, so a hung tool always follows a call
    # that waited. Frames must stop when that call ends; a ticker that kept
    # its first waiting snapshot would keep the hung turn alive forever.
    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    harness = _Harness(
        session=SimpleNamespace(model_calls=_Tracker(waiting=True)), then_hang=True,
    )

    handle = await _run(monkeypatch, buffer, harness, saved)
    await asyncio.wait_for(handle.task, timeout=5)

    records = list(read_records(buffer.path))
    sse = [r.data.get("sse", "") for r in records]
    waits_at = [i for i, s in enumerate(sse) if _model_wait_frames([s])]
    assert waits_at, "the silent call got no frame, so the hang below proves nothing"
    answer_at = next(i for i, s in enumerate(sse) if "response.output_text.delta" in s)
    assert max(waits_at) < answer_at, "a frame went out after the call ended"
    assert handle.lifecycle.timed_out is True
    assert records[-1].type == "Interrupted"


_TERMINAL_TYPES = {"Done", "Cancelled", "Error", "Interrupted"}


@pytest.mark.parametrize(
    "reason", [None, "something-newer", "stalled"], ids=["user-stop", "unknown-reason", "ui-stall"],
)
async def test_a_cancel_during_a_silent_call_ends_with_the_terminal_record(
    monkeypatch, tmp_path, reason,
):
    # The ticker is writing frames when the cancel lands. It must stop before
    # the terminal record, and a Stop must still save as a Stop.
    from cowork.api.v1.endpoints.responses import CancelRequest, cancel_response
    from cowork.db.scoped import LOCAL_SCOPE

    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    harness = _Harness(
        session=SimpleNamespace(model_calls=_Tracker(waiting=True)), hang=True, partial="partial",
    )

    handle = await _run(monkeypatch, buffer, harness, saved)

    async def _first_wait_frame():
        while not _model_wait_frames([r.data.get("sse", "") for r in read_records(buffer.path)]):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_first_wait_frame(), timeout=5)
    await cancel_response(CancelRequest(conversation_id=CID, reason=reason), LOCAL_SCOPE)
    await asyncio.wait_for(handle.task, timeout=5)

    records = list(read_records(buffer.path))
    terminals = [i for i, r in enumerate(records) if r.type in _TERMINAL_TYPES]
    assert terminals == [len(records) - 1]
    assert handle.lifecycle.timed_out is False
    assert saved["assistant"] == "partial"
    assert not [e for e in saved["events"] if e.get("phase") == "model_wait"]
    if reason != "stalled":
        # Any reason but "stalled" is a Stop, so a newer client's reason never
        # turns the user's own Stop into a stall card.
        assert records[-1].type == "Cancelled"
        assert not [e for e in saved["events"] if e.get("type") == "response.failed"]
    else:
        assert records[-1].type == "Interrupted"
        assert saved["events"][-1]["code"] == "stalled"


async def test_a_deadline_after_a_silent_call_saves_model_timeout(monkeypatch, tmp_path):
    # anton's deadline ends the call. The turn fails with the deadline's own
    # code and copy, not the generic error.
    message = "The model sent no output for 10 minutes, so the call was stopped."
    deadline = RuntimeError(message)
    deadline.code = "model_timeout"
    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    harness = _Harness(
        session=SimpleNamespace(model_calls=_Tracker(waiting=True)), raises=deadline,
    )

    handle = await _run(monkeypatch, buffer, harness, saved)
    await asyncio.wait_for(handle.task, timeout=5)

    records = list(read_records(buffer.path))
    assert records[-1].type == "Error"
    failed = saved["events"][-1]
    assert failed["type"] == "response.failed"
    assert failed["code"] == "model_timeout"
    assert failed["error"] == message
    assert failed["request_id"]
    sse_failed = [
        json.loads(r.data["sse"].split("data: ", 1)[1]) for r in records
        if r.data.get("sse", "").startswith("event: response.failed\n")
    ]
    assert [f["code"] for f in sse_failed] == ["model_timeout"]



@pytest.mark.parametrize("fails", [False, True])
async def test_the_anton_harness_attaches_its_session_for_the_turn_only(monkeypatch, fails):
    import cowork.services.artifact_autopublish as autopublish
    import cowork.services.task_objects as task_objects
    from cowork.harnesses.anton_harness import harness as harness_mod

    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    seen_during_turn: list[object] = []

    class _Recorder:
        def __init__(self) -> None:
            self.session: object | None = None

        def attach(self, *, session):
            self.session = session

        def detach(self):
            self.session = None

    recorder = _Recorder()

    class _Session:
        async def turn_stream(self, user_input, **_kwargs):
            seen_during_turn.append(recorder.session)
            if fails:
                raise RuntimeError("the turn failed")
            yield StreamTextDelta(text="done")

    session = _Session()

    async def _fake_build(self, conversation, **kwargs):
        return session, None, None

    monkeypatch.setattr(harness_mod.AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(
        id="conv-1", project_id="proj-1", project=SimpleNamespace(path="/tmp", name="tmp")
    )

    async def _drain():
        return [
            event
            async for event in harness_mod.AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}],
                model_wait=recorder,
            )
        ]

    if fails:
        with pytest.raises(RuntimeError):
            await _drain()
    else:
        await _drain()

    assert seen_during_turn == [session]
    assert recorder.session is None


# ── The contract with the real anton ─────────────────────────────────
# Everything above uses stand-ins, and cowork-server reads anton's tracker and
# error by attribute and by class name. These run against the installed
# anton, so they skip until uv.lock moves to an anton release that has the
# tracker, then catch a rename the stand-ins would hide.


def _real_anton_liveness():
    liveness_mod = pytest.importorskip("anton.core.llm.liveness")
    if getattr(liveness_mod, "ModelCallTracker", None) is None:
        pytest.skip("installed anton predates the model-call tracker (ENG-3281)")
    return liveness_mod


async def test_a_real_anton_tracker_drives_the_ticker():
    real = _real_anton_liveness()
    tracker = real.ModelCallTracker()
    tracker.open(role="planning", idle_timeout_s=600.0).awaiting = True
    buffer = _Buffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=tracker))

    async with ticker.running(buffer=buffer):
        await asyncio.sleep(0.2)

    waits = _waits(buffer)
    assert waits, "no frame from a real tracker reporting a waiting call"
    assert waits[0]["message"].startswith("Waiting for the model (")
    assert isinstance(waits[0]["eta_seconds"], float)


def test_a_real_anton_deadline_maps_to_model_timeout_in_process_and_remote():
    _real_anton_liveness()
    from anton.core.llm.provider import ModelCallTimeoutError

    from cowork.handlers.turn_errors import MODEL_TIMEOUT_CODE, friendly_turn_error, remote_turn_error

    exc = ModelCallTimeoutError(role="planning", model="m", idle_timeout_s=600.0)
    assert friendly_turn_error(exc)[0] == MODEL_TIMEOUT_CODE
    # The pod's turn_failed carries "TypeName: message" (anton's _scrub).
    assert remote_turn_error(f"{type(exc).__name__}: {exc}")[0] == MODEL_TIMEOUT_CODE


# ── The ticker on its own ─────────────────────────────────────────────


class _Buffer:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []
        self.is_closed = False

    @property
    def latest_seq(self) -> int:
        return len(self.records)

    async def append(self, type_, data):
        self.records.append((type_, data))
        return len(self.records) - 1


def _waits(buffer: _Buffer) -> list[dict]:
    return _model_wait_frames([data.get("sse", "") for _, data in buffer.records])


class _BusyBuffer(_Buffer):
    """A record lands between every two polls for as long as ``busy`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.busy = True
        self._landed = 0

    @property
    def latest_seq(self) -> int:
        if self.busy:
            self._landed += 1
        return len(self.records) + self._landed


async def test_no_frame_goes_out_while_other_records_keep_arriving():
    # The wire is not quiet, so the turn needs no help; a frame here would
    # also keep a hung turn alive if the tracker were ever wrong. No race with
    # the clock: the buffer moves between every two polls while it is busy.
    buffer = _BusyBuffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))

    async with ticker.running(buffer=buffer):
        await asyncio.sleep(0.3)
        assert _waits(buffer) == []
        buffer.busy = False
        await asyncio.sleep(0.3)

    assert _waits(buffer)


async def test_frames_come_no_faster_than_the_quiet_window():
    # Each frame restarts the window, so a long silent call gets one frame per
    # window, not one per poll: the documented 20 to 25 s cadence, not 5 s.
    loop = asyncio.get_running_loop()
    stamps: list[float] = []

    class _TimedBuffer(_Buffer):
        async def append(self, type_, data):
            stamps.append(loop.time())
            return await super().append(type_, data)

    buffer = _TimedBuffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))

    async with ticker.running(buffer=buffer):
        await asyncio.sleep(0.4)

    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    assert len(stamps) >= 3, f"{len(stamps)} frames in 0.4 s"
    assert min(gaps) >= liveness.MODEL_WAIT_TICK_SECONDS * 0.9, f"frames {gaps} s apart"


async def test_no_frame_after_detach_or_once_the_buffer_closes():
    buffer = _Buffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))
    ticker.detach()

    async with ticker.running(buffer=buffer):
        await asyncio.sleep(0.2)
        assert _waits(buffer) == []
        ticker.attach(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))
        buffer.is_closed = True
        await asyncio.sleep(0.2)

    assert _waits(buffer) == []


async def test_the_loop_is_stopped_when_the_turn_leaves_it():
    buffer = _Buffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=_Tracker(waiting=True)))

    with pytest.raises(RuntimeError):
        async with ticker.running(buffer=buffer):
            raise RuntimeError("the turn failed")
    await asyncio.sleep(0.2)

    assert _waits(buffer) == []
    tickers = [t for t in asyncio.all_tasks() if "_tick" in repr(t.get_coro())]
    assert tickers == []


async def test_a_tracker_that_raises_stops_the_ticks_not_the_turn(caplog):
    class _Broken:
        def snapshot(self):
            raise ValueError("tracker bug")

    buffer = _Buffer()
    ticker = liveness.ModelWaitTicker()
    ticker.attach(session=SimpleNamespace(model_calls=_Broken()))

    caplog.set_level(logging.ERROR, logger="cowork.streaming.liveness")
    async with ticker.running(buffer=buffer):
        await asyncio.sleep(0.2)

    assert _waits(buffer) == []
    logged = [
        r for r in caplog.records
        if r.name == "cowork.streaming.liveness" and r.levelno == logging.ERROR
    ]
    assert len(logged) == 1
    assert "model-wait ticker failed" in logged[0].getMessage()
    assert logged[0].exc_info is not None, "the traceback was not logged"


def test_the_frame_omits_a_sequence_number_it_does_not_have():
    frame = liveness.model_wait_sse(message="Waiting for the model (20s)", waited_s=20.0)
    payload = json.loads(frame.split("data: ", 1)[1])
    assert frame.startswith("event: response.in_progress\n")
    assert "sequence_number" not in payload
    assert payload["phase"] == "model_wait"
    assert payload["message"] == "Waiting for the model (20s)"
    assert payload["content"] == "Still working: Waiting for the model (20s)"
    assert payload["eta_seconds"] == 20.0
    assert isinstance(payload["at_ms"], int)
