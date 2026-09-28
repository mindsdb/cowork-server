"""A finished turn must not leave its scratchpad processes running.

cowork-server builds a fresh anton ChatSession for every turn, and every
scratchpad that session starts is a child process. Nothing used to stop those
processes, so each turn that ran Python left one alive, holding memory and file
handles, until the server exited.

The turn has to close its pads on every exit: success, error and Stop. It must
not call `session.close()`, which also reaps the full-stack backends the turn
launched, because those keep serving after the turn ends. And it must use the
pads' `close()`, never `cleanup()` or `reset()`, which delete the namespace
snapshot the next turn restores its variables from.

Each stub pad here is a real child process blocked on stdin, the way anton's
scratchpad_boot waits between cells, so "closed" means the OS process exited,
not that a method was called.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from anton.core.backends.base import ScratchpadRuntime
from anton.core.backends.manager import ScratchpadManager

import cowork.services.artifact_autopublish as autopublish
import cowork.services.connectors.probe as probe_module
import cowork.services.task_objects as task_objects
from cowork.common.chat_session import close_session_scratchpads, drain_scratchpad_closes
from cowork.harnesses.anton_harness.harness import AntonHarness

#: How long a turn's pad may outlive the turn. Generous so a loaded CI runner
#: does not flake; a leaked pad never exits, so the bound only has to be finite.
EXIT_BOUND_S = 5.0

_BLOCK_ON_STDIN = "import sys; sys.stdin.read()"


async def _spawn_blocked_child() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", _BLOCK_ON_STDIN, stdin=asyncio.subprocess.PIPE
    )


class _ChildPad(ScratchpadRuntime):
    """A scratchpad whose runtime is a real child process blocked on stdin."""

    def __init__(self, name: str, **base_kwargs) -> None:
        super().__init__(name, **base_kwargs)
        self.proc: asyncio.subprocess.Process | None = None
        self.discarded_snapshot = False

    async def start(self) -> None:
        self.proc = await _spawn_blocked_child()

    async def close(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()

    async def cleanup(self) -> None:
        # The real runtime deletes the namespace snapshot here.
        self.discarded_snapshot = True
        await self.close()

    async def reset(self) -> None:
        self.discarded_snapshot = True
        await self.close()
        await self.start()

    async def cancel(self) -> None:
        await self.reset()

    async def install_packages(self, packages: list[str]) -> str:
        return ""

    async def execute_streaming(self, code: str, **_kwargs):
        return
        yield


class _Session:
    """ChatSession stand-in that owns a REAL anton ScratchpadManager.

    Its turn provisions the "main" pad, which is what anton's scratchpad tool
    does on the first cell, and launches one tracked backend, then runs `body`.
    `close()` mirrors anton's: reap tracked backends, then close every pad.
    """

    def __init__(self, pads: list[_ChildPad], body, project_path) -> None:
        def factory(*, name, **kwargs):
            base = {
                key: kwargs[key]
                for key in ("coding_provider", "coding_model", "coding_api_key",
                            "coding_base_url", "cells", "workspace_path")
            }
            pad = _ChildPad(name, **base)
            pads.append(pad)
            return pad

        self._scratchpads = ScratchpadManager(
            runtime_factory=factory,
            coding_provider="stub",
            coding_model="stub",
            coding_api_key="",
            coding_base_url="",
            workspace_path=project_path,
        )
        self._tracked_backends: dict[str, dict] = {}
        self._body = body

    async def turn_stream(self, *_args, **_kwargs):
        await self._scratchpads.get_or_create("main")
        self._tracked_backends["dash"] = {"proc": await _spawn_blocked_child()}
        async for event in self._body():
            yield event

    async def close(self) -> None:
        for info in self._tracked_backends.values():
            info["proc"].kill()
            await info["proc"].wait()
        await self._scratchpads.close_all()


async def _completes():
    return
    yield


async def _raises():
    raise RuntimeError("model call failed")
    yield


def _stub_artifact_steps(monkeypatch) -> None:
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", lambda *_a, **_k: ([], set(), None))
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)


async def _kill_survivors(procs) -> None:
    for proc in procs:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.fixture
async def turns(monkeypatch, tmp_path):
    """Drive AntonHarness.stream_response against a fresh `_Session` per turn,
    the way production builds a fresh ChatSession per turn. Kills every child
    at teardown, so a failing run does not leak processes into the runner."""
    _stub_artifact_steps(monkeypatch)
    pads: list[_ChildPad] = []
    sessions: list[_Session] = []
    bodies: list = []

    async def _fake_build(self, conversation, **_kwargs):
        session = _Session(pads, bodies.pop(0), tmp_path)
        sessions.append(session)
        return session, None, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(
        id="conv-1", project_id="proj-1", project=SimpleNamespace(path=str(tmp_path), name="p")
    )

    def run(body):
        bodies.append(body)
        return AntonHarness().stream_response(
            conversation=conversation, input=[{"type": "text", "text": "hi"}]
        )

    yield SimpleNamespace(run=run, pads=pads, sessions=sessions)

    await _kill_survivors(
        [p.proc for p in pads]
        + [info["proc"] for s in sessions for info in s._tracked_backends.values()]
    )


async def _drain(stream) -> list:
    return [event async for event in stream]


async def _assert_exits(proc: asyncio.subprocess.Process, what: str) -> None:
    try:
        await asyncio.wait_for(proc.wait(), EXIT_BOUND_S)
    except TimeoutError:
        pytest.fail(f"{what} (pid {proc.pid}) was still running {EXIT_BOUND_S}s after its turn ended")


def _assert_backend_and_snapshot_survive(turns) -> None:
    backend = turns.sessions[0]._tracked_backends["dash"]["proc"]
    assert backend.returncode is None, "a turn's backend must outlive the turn: no session.close()"
    assert not turns.pads[0].discarded_snapshot, "close(), not cleanup()/reset(): keep the namespace snapshot"


async def test_completed_turn_closes_its_pad_and_the_next_turn_gets_a_fresh_one(turns):
    await _drain(turns.run(_completes))
    await _assert_exits(turns.pads[0].proc, "the first turn's scratchpad")
    _assert_backend_and_snapshot_survive(turns)

    await _drain(turns.run(_completes))
    assert len(turns.pads) == 2, "the second turn provisions its own pad"
    assert turns.pads[1].proc.pid != turns.pads[0].proc.pid
    await _assert_exits(turns.pads[1].proc, "the second turn's scratchpad")


async def test_failed_turn_closes_its_pad(turns):
    with pytest.raises(RuntimeError, match="model call failed"):
        await _drain(turns.run(_raises))
    await _assert_exits(turns.pads[0].proc, "the failed turn's scratchpad")
    _assert_backend_and_snapshot_survive(turns)


async def test_stopped_turn_closes_its_pad(turns):
    """Stop cancels the producer task (RunHandle.cancel in
    cowork/streaming/registry.py), so the CancelledError lands on whatever the
    turn is awaiting."""
    parked = asyncio.Event()

    async def _parks_until_stopped():
        parked.set()
        await asyncio.Event().wait()
        yield

    task = asyncio.create_task(_drain(turns.run(_parks_until_stopped)))
    await parked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_exits(turns.pads[0].proc, "the stopped turn's scratchpad")
    _assert_backend_and_snapshot_survive(turns)


async def test_pad_closes_even_when_artifact_indexing_raises(turns, monkeypatch):
    """The close runs before the finally's other steps, so a step that raises
    cannot skip it."""

    def _indexing_fails(*_a, **_k):
        raise OSError("artifacts dir unreadable")

    monkeypatch.setattr(task_objects, "index_turn_artifacts", _indexing_fails)
    with pytest.raises(OSError, match="artifacts dir unreadable"):
        await _drain(turns.run(_completes))
    await _assert_exits(turns.pads[0].proc, "the turn's scratchpad")


async def test_every_pad_the_turn_started_closes(turns):
    """`launch_backend` starts a slug pad in the same manager as "main", so a
    turn can own several pads. A close that stopped only one leaks the rest."""

    async def _starts_a_slug_pad():
        await turns.sessions[-1]._scratchpads.get_or_create("dash")
        return
        yield

    await _drain(turns.run(_starts_a_slug_pad))
    assert [pad.name for pad in turns.pads] == ["main", "dash"]
    for pad in turns.pads:
        await _assert_exits(pad.proc, f"the turn's {pad.name!r} scratchpad")


async def test_shutdown_drain_waits_for_a_scheduled_close(tmp_path):
    pads: list[_ChildPad] = []
    session = _Session(pads, _completes, tmp_path)
    await session._scratchpads.get_or_create("main")
    try:
        close_session_scratchpads(session, owner="test")
        await drain_scratchpad_closes()
        assert pads[0].proc.returncode is not None, "drain returned before the close finished"
    finally:
        await _kill_survivors([p.proc for p in pads])


async def test_shutdown_drain_also_waits_for_a_close_queued_while_it_waits():
    """A turn that finishes unwinding during the drain queues its close then."""
    first_may_finish = asyncio.Event()
    closed: list[str] = []

    class _Manager:
        def __init__(self, name: str, *, gate: asyncio.Event | None = None, delay: float = 0.0):
            self.name, self.gate, self.delay = name, gate, delay

        async def close_all(self):
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
            closed.append(self.name)

    close_session_scratchpads(SimpleNamespace(_scratchpads=_Manager("first", gate=first_may_finish)), owner="first")
    drain = asyncio.create_task(drain_scratchpad_closes())
    await asyncio.sleep(0)
    close_session_scratchpads(SimpleNamespace(_scratchpads=_Manager("late", delay=0.2)), owner="late")
    first_may_finish.set()
    await drain
    assert closed == ["first", "late"], "drain returned before the late close finished"


async def test_shutdown_drain_warns_and_leaves_a_slow_close_running(caplog):
    """Shutdown goes on to reap backends after this, so a slow close must not
    make the drain raise, and the drain must not cancel it either."""
    may_finish = asyncio.Event()
    cancelled: list[bool] = []

    class _SlowManager:
        async def close_all(self):
            try:
                await may_finish.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    with caplog.at_level(logging.WARNING, logger="cowork.common.chat_session"):
        close_session_scratchpads(SimpleNamespace(_scratchpads=_SlowManager()), owner="slow")
        await drain_scratchpad_closes(timeout=0.1)
        assert not cancelled, "the drain cancelled a close it timed out on"
        may_finish.set()
        await drain_scratchpad_closes()

    records = [
        r for r in caplog.records
        if r.name == "cowork.common.chat_session" and r.levelno == logging.WARNING
    ]
    assert [r.getMessage() for r in records] == ["1 scratchpad close(s) did not finish within 0.1s"]


def test_the_lifespan_drains_closes_after_turns_unwind_and_before_backends_are_reaped():
    """The turn drains queue each turn's close as it unwinds, so the close
    drain has to follow them; backend and pool teardown follow it."""
    from cowork import server

    source = inspect.getsource(server.lifespan)
    calls = [
        "await registry.shutdown()",
        "await drain_background_tasks()",
        "await drain_scratchpad_closes()",
        "shutdown_launched_backends()",
    ]
    positions = [source.find(call) for call in calls]
    assert -1 not in positions, dict(zip(calls, positions))
    assert positions == sorted(positions), dict(zip(calls, positions))


async def test_a_failed_close_is_logged_not_raised(caplog):
    class _BrokenManager:
        async def close_all(self):
            raise RuntimeError("pad close exploded")

    session = SimpleNamespace(_scratchpads=_BrokenManager())
    with caplog.at_level(logging.WARNING, logger="cowork.common.chat_session"):
        close_session_scratchpads(session, owner="conversation c-1")
        await drain_scratchpad_closes()

    records = [
        r for r in caplog.records
        if r.name == "cowork.common.chat_session" and r.levelno == logging.WARNING
    ]
    assert len(records) == 1
    assert records[0].getMessage() == "Closing the scratchpads of conversation c-1 failed"
    assert isinstance(records[0].exc_info[1], RuntimeError)


async def test_a_session_without_scratchpads_is_logged(caplog):
    """anton's ChatSession always sets `_scratchpads`, so a session without it
    means an anton release renamed it. The desktop wheel installs anton outside
    uv.lock, where the pin below never runs, so the skipped close has to show."""
    with caplog.at_level(logging.WARNING, logger="cowork.common.chat_session"):
        close_session_scratchpads(SimpleNamespace(), owner="conversation c-2")

    records = [
        r for r in caplog.records
        if r.name == "cowork.common.chat_session" and r.levelno == logging.WARNING
    ]
    assert [r.getMessage() for r in records] == [
        "Cannot close the scratchpads of conversation c-2: the session has no _scratchpads"
    ]


def test_the_close_reaches_the_manager_anton_itself_closes():
    """anton has no public pads-only close, so close_session_scratchpads reads
    the private `_scratchpads`. Pin that anton's own close() still closes pads
    through it, so a locked anton bump that renames it fails here. The desktop
    wheel installs anton outside uv.lock, where this never runs; there the
    missing attribute logs a warning instead
    (test_a_session_without_scratchpads_is_logged)."""
    from anton.core.session import ChatSession

    assert "self._scratchpads.close_all()" in inspect.getsource(ChatSession.close)


# --- Connector probe: same leak, same close ---------------------------------


@pytest.fixture
async def probe_turns(monkeypatch, tmp_path):
    """Run CredentialProbe.run() against a `_Session` from the patched
    build_chat_session, with a stub pad in place of anton's real one."""
    pads: list[_ChildPad] = []
    sessions: list[_Session] = []

    def run(body, *, timeout_seconds: float = 30.0):
        def _fake_build(_config):
            session = _Session(pads, body, tmp_path)
            sessions.append(session)
            return session

        monkeypatch.setattr(probe_module, "build_chat_session", _fake_build)
        probe = probe_module.CredentialProbe(
            engine="postgres",
            credentials={"password": "hunter2"},
            llm_client=None,
            workspace=None,
            timeout_seconds=timeout_seconds,
        )
        return probe.run()

    yield SimpleNamespace(run=run, pads=pads, sessions=sessions)

    await _kill_survivors(
        [p.proc for p in pads]
        + [info["proc"] for s in sessions for info in s._tracked_backends.values()]
    )


async def test_probe_closes_its_pad_after_its_verdict(probe_turns):
    events = await _drain(probe_turns.run(_completes))
    assert events[-1][0] == "verdict"
    await _assert_exits(probe_turns.pads[0].proc, "the probe's scratchpad")
    assert not probe_turns.pads[0].discarded_snapshot


async def test_probe_closes_its_pad_after_a_timeout(probe_turns):
    async def _hangs():
        await asyncio.Event().wait()
        yield

    events = await _drain(probe_turns.run(_hangs, timeout_seconds=0.2))
    verdict = events[-1][1]
    assert verdict.status == "failure" and "timed out" in verdict.error
    await _assert_exits(probe_turns.pads[0].proc, "the timed-out probe's scratchpad")


async def test_probe_closes_its_pad_after_a_crash(probe_turns):
    events = await _drain(probe_turns.run(_raises))
    verdict = events[-1][1]
    assert verdict.status == "failure" and "crashed" in verdict.error
    await _assert_exits(probe_turns.pads[0].proc, "the crashed probe's scratchpad")


# --- Against anton's real scratchpad runtime --------------------------------


def _open_fd_count() -> int:
    return len(os.listdir("/dev/fd"))


async def _settled_fd_count() -> int:
    """Open descriptors once the closing pipes stop changing the count.

    A pad's pipes close in callbacks shortly after the process exits, so read
    until two samples agree."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + EXIT_BOUND_S
    count = _open_fd_count()
    while loop.time() < deadline:
        await asyncio.sleep(0.1)
        now = _open_fd_count()
        if now == count:
            break
        count = now
    return count


@pytest.mark.skipif(sys.platform == "win32", reason="counts descriptors through /dev/fd")
async def test_real_pad_exits_its_descriptors_close_and_the_next_turn_restores_its_namespace(
    monkeypatch, tmp_path
):
    """The same contract against anton's real LocalScratchpadRuntime. The
    turn's pad process exits, the server gets back the pipes it held, and the
    next turn's fresh pad reloads the namespace snapshot. `_build_chat_session`
    sets the persistence env var in production."""
    from anton.core.backends.local import local_scratchpad_runtime_factory

    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
    _stub_artifact_steps(monkeypatch)
    conv_id = str(uuid.uuid4())
    cells = iter(["x = 41", "print(x + 1)"])
    seen: list = []

    class _RealPadSession:
        def __init__(self):
            self._scratchpads = ScratchpadManager(
                runtime_factory=local_scratchpad_runtime_factory,
                coding_provider="", coding_model="", coding_api_key="", coding_base_url="",
                workspace_path=tmp_path, session_id=conv_id,
            )

        async def turn_stream(self, *_a, **_k):
            pad = await self._scratchpads.get_or_create("main")
            cell = await pad.execute(next(cells))
            seen.append((pad._proc, cell))
            return
            yield

    async def _fake_build(self, conversation, **_kwargs):
        return _RealPadSession(), None, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    conversation = SimpleNamespace(
        id=conv_id, project_id="proj-1", project=SimpleNamespace(path=str(tmp_path), name="p")
    )

    def turn():
        return AntonHarness().stream_response(
            conversation=conversation, input=[{"type": "text", "text": "hi"}]
        )

    try:
        await _drain(turn())
        first_proc, first_cell = seen[0]
        assert first_cell.error is None, first_cell.error
        await _assert_exits(first_proc, "the first turn's scratchpad_boot")
        # Measured after the first turn so the venv build it did is not counted.
        fds_between_turns = await _settled_fd_count()

        await _drain(turn())
        second_proc, second_cell = seen[1]
        assert second_proc.pid != first_proc.pid
        assert second_cell.error is None, second_cell.error
        assert second_cell.stdout.strip() == "42", "the next turn must restore the snapshot"
        await _assert_exits(second_proc, "the second turn's scratchpad_boot")
        assert await _settled_fd_count() <= fds_between_turns, "the turn's pad pipes stayed open"
    finally:
        await _kill_survivors([proc for proc, _cell in seen])
