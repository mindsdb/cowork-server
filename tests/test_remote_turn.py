"""remote_turn_events tests: verify its own dispatch logic in isolation.
ResponsesHandler is replaced wholesale per test; its helpers have coverage elsewhere."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest

import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory
from cowork.schemas.figure_sources import EarlierAnswer, Figure, FigureOrigin
from cowork.turnqueue.producer import RemoteWorkspace
from cowork.turnqueue.remote_turn import RemoteTurnFailed, remote_turn_events


class _FakeScope:
    org_id = "org-123"
    user_id = None


class _FakeSession:
    scope = _FakeScope()


def _fake_handler(
    monkeypatch, *, persist_turn_memory=None, remote_artifacts_context=None, remote_seed_history=None,
    remote_agent_files=None, remote_workspace=None, earlier_answers=None,
):
    """Replace ResponsesHandler with a stand-in exposing only the methods
    remote_turn_events calls, resolved at call time like every other name here.
    The earlier answers' read is replaced too, by ``earlier_answers`` or by
    one that finds none."""

    async def no_earlier_answers(*, conversation_id, scope):
        return ()

    monkeypatch.setattr(remote_turn_mod, "read_earlier_answers", earlier_answers or no_earlier_answers)
    monkeypatch.setattr(remote_turn_mod, "ResponsesHandler", type(
        "FakeResponsesHandler", (), {
            "_remote_artifacts_context": staticmethod(remote_artifacts_context or (lambda s, c: None)),
            "_remote_seed_history": staticmethod(remote_seed_history or (lambda s, c: ([], None))),
            "_persist_remote_compaction": staticmethod(lambda s, *, conv_id, data, seed_info: None),
            "_remote_workspace": staticmethod(remote_workspace or (lambda s, c: None)),
            "_remote_agent_files": staticmethod(remote_agent_files or (lambda s, c, *, workspace: None)),
            "_remote_started_at": staticmethod(lambda s, c: None),
            "_persist_turn_memory": staticmethod(persist_turn_memory or (lambda s, c, e: None)),
        },
    ))


async def _drain(gen):
    """The turn's events after the context every turn opens with, which
    test_the_turn_opens_with_its_context pins."""
    events = []
    try:
        async for event in gen:
            if not isinstance(event, TurnContext):
                events.append(event)
    except RemoteTurnFailed as exc:
        return events, exc
    return events, None


@pytest.mark.asyncio
async def test_the_turn_opens_with_its_context(monkeypatch, tmp_path):
    conv_id = uuid4()
    files = AgentFiles.in_cloud_pod(tmp_path, conversation_id=conv_id, workspace_rel_path="projects/general")
    earlier = EarlierAnswer(figures=[Figure(start=0, end=5, text="4,600", origin=FigureOrigin.step)])
    session = _FakeSession()
    reads = []

    async def read_earlier_answers(*, conversation_id, scope):
        reads.append((conversation_id, scope))
        return (earlier,)

    _fake_handler(
        monkeypatch, remote_agent_files=lambda s, c, *, workspace: files if c == conv_id else None,
        earlier_answers=read_earlier_answers,
    )

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "There are 250."}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events = [event async for event in remote_turn_events(
        session=session, conv_id=conv_id, org_id="org-123", user_id=None,
        input_text="How many at S1?", model="anton", turn_rows=[],
    )]

    assert events[0] == TurnContext(question="How many at S1?", files=files, earlier_answers=(earlier,))
    assert not any(isinstance(event, TurnContext) for event in events[1:])
    # Read in a unit of its own under the channel's scope, never on the
    # channel's shared session.
    assert reads == [(conv_id, session.scope)]


@pytest.mark.asyncio
async def test_the_turn_looks_up_its_workspace_once(monkeypatch):
    """The job and the pod's files are placed by the same project path."""
    workspace = RemoteWorkspace(project_id="proj-1", workspace_rel_path="projects/proj")
    looked_up: list[object] = []
    placed: list[RemoteWorkspace | None] = []
    sent: list[RemoteWorkspace | None] = []

    def remote_workspace(session, conv_id):
        looked_up.append(conv_id)
        return workspace

    def remote_agent_files(session, conv_id, *, workspace):
        placed.append(workspace)

    _fake_handler(monkeypatch, remote_workspace=remote_workspace, remote_agent_files=remote_agent_files)

    async def fake_replies(**kwargs):
        sent.append(kwargs["workspace"])
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert len(looked_up) == 1
    assert placed == [workspace]
    assert sent == [workspace]


@pytest.mark.asyncio
async def test_turn_delta_becomes_a_stream_text_delta(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "hi"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=turn_rows,
    ))

    assert failure is None
    assert len(events) == 1
    assert events[0].text == "hi"


@pytest.mark.asyncio
async def test_turn_failed_raises_with_the_workers_code_and_message(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_failed", {"code": "anton_error", "message": "boom"}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is not None
    assert failure.code == "anton_error"
    assert failure.message == "boom"


@pytest.mark.asyncio
async def test_turn_failed_falls_back_to_the_generic_message_and_code(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_failed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE

    _, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure.code == GENERIC_TURN_ERROR_CODE
    assert failure.message == GENERIC_TURN_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_turn_history_mutates_the_callers_turn_rows_list(monkeypatch):
    _fake_handler(monkeypatch)

    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": "1"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "1"}]},
    ]

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_history", {"rows": rows}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=turn_rows,
    ))

    assert turn_rows == rows


@pytest.mark.asyncio
async def test_turn_history_reaches_the_formatter_as_the_pod_sent_it(monkeypatch):
    """The formatter finds the answer's figures in every tool's result, so it
    reads the rows before the replay caps trim them; the caller saves the
    trimmed ones."""
    _fake_handler(monkeypatch)
    page = "Overtime rate: $42 per hour.\n" + "filler " * 4_000
    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "w1", "name": "web_fetch", "input": {"url": "https://example.com/rates"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w1", "content": page}]},
    ]

    async def fake_replies(**kwargs):
        yield "turn_history", {"rows": rows}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    events, _ = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="What does overtime cost?", model="anton", turn_rows=turn_rows,
    ))

    [history] = [event for event in events if isinstance(event, TurnHistory)]
    assert history.read == rows
    assert history.rows == turn_rows
    assert turn_rows[1]["content"][0]["content"].startswith("[tool output omitted")


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["compaction-reply", "compacted-step"])
async def test_a_compacted_turns_history_reaches_the_formatter_and_none_is_kept(monkeypatch, signal):
    """A turn that compacted ran near or over the context window, so its rows
    are read for the answer's figures and not kept for replay. A summary
    sends both signals; a hard truncation sends only the step. The pod goes
    on streaming the turn's rounds after the compaction."""
    _fake_handler(monkeypatch)
    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "w1", "name": "web_fetch", "input": {"url": "https://example.com/rates"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w1", "content": "Overtime: $42"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "w4", "name": "web_fetch", "input": {"url": "https://example.com/lines"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w4", "content": "9 lines"}]},
    ]

    async def fake_replies(**kwargs):
        if signal == "compacted-step":
            yield "turn_step", {"step": "compacted", "message": "Context compacted"}
        yield "turn_step", {"step": "tool_start", "id": "w4", "name": "web_fetch"}
        yield "turn_step", {"step": "tool_end", "id": "w4", "args": json.dumps({"url": "https://example.com/lines"})}
        yield "turn_step", {"step": "progress", "phase": "tool_done", "id": "w4", "ok": True}
        if signal == "compaction-reply":
            yield "turn_compaction", {"summary": "EARLIER: …", "covered_through": 1}
        yield "turn_history", {"rows": rows}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    events, _ = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="What does overtime cost?", model="anton", turn_rows=turn_rows,
    ))

    [history] = [event for event in events if isinstance(event, TurnHistory)]
    assert history.read == rows
    assert history.rows == []
    assert turn_rows == []


@pytest.mark.asyncio
async def test_a_compacted_turns_history_frame_without_a_list_of_rows_reaches_the_formatter_as_no_rows(monkeypatch):
    """A compacted turn's rows reach the formatter only as the pod sent
    them, so a frame whose rows are not a list carries none: the formatter
    hears of no rows, as from a pod that sends no frame after a compaction."""
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "turn_step", {"step": "compacted", "message": "Context compacted"}
        yield "turn_history", {"rows": "not a list"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    events, _ = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="What does overtime cost?", model="anton", turn_rows=turn_rows,
    ))

    assert [event for event in events if isinstance(event, TurnHistory)] == []
    assert turn_rows == []


@pytest.mark.asyncio
async def test_the_turns_context_carries_the_history_the_pod_is_seeded_with(monkeypatch):
    seeded = [
        {"role": "user", "content": "What is at S1?"},
        {"role": "assistant", "content": "S1 holds 250 units."},
    ]
    _fake_handler(monkeypatch, remote_seed_history=lambda s, c: (seeded, None))

    async def fake_replies(**kwargs):
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events = [event async for event in remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="And at S2?", model="anton", turn_rows=[],
    )]

    assert isinstance(events[0], TurnContext)
    assert events[0].history == seeded


@pytest.mark.asyncio
async def test_turn_memory_calls_persist_turn_memory(monkeypatch):
    calls = []
    _fake_handler(monkeypatch, persist_turn_memory=lambda s, c, e: calls.append(e))

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_memory", {"entries": [{"key": "likes", "value": "pizza"}]}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert calls == [[{"key": "likes", "value": "pizza"}]]


@pytest.mark.asyncio
async def test_no_artifacts_context_skips_indexing_without_error(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is None
    assert events == []


@pytest.mark.asyncio
async def test_vault_secrets_are_registered_before_the_seed_history_is_built(monkeypatch):
    """A channel turn skips handle(), so this module registers the vault's
    secrets itself, and must do it before the seed history is scrubbed."""
    calls = []

    def fake_seed_history(_session, _conv_id):
        calls.append(("seed_history", None))
        return [], None

    async def fake_register(scope):
        calls.append(("register", scope))

    _fake_handler(monkeypatch, remote_seed_history=fake_seed_history)
    monkeypatch.setattr(remote_turn_mod, "register_vault_secrets", fake_register)

    async def fake_replies(**kwargs):
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    session = _FakeSession()
    _, failure = await _drain(remote_turn_events(
        session=session, conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is None
    assert calls == [("register", session.scope), ("seed_history", None)]


@pytest.mark.parametrize("workspace_mode", [None, "ephemeral", "persistent"])
@pytest.mark.parametrize("failed", [False, True])
async def test_only_authorized_persistent_turns_touch_saved_artifacts(monkeypatch, workspace_mode, failed):
    from unittest.mock import AsyncMock, Mock
    from cowork.services import task_objects

    context = (object(), object(), "project", "Project")
    _fake_handler(monkeypatch, remote_artifacts_context=lambda *_: context)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    index = Mock(return_value=([], set(), None))
    publish = AsyncMock(return_value=[])
    monkeypatch.setattr(task_objects, "index_turn_artifacts", index)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", publish)

    async def replies(**kwargs):
        if workspace_mode is not None:
            yield "progress", {"phase": "workspace_authorized", "workspace_mode": workspace_mode}
        yield "turn_delta", {"text": "model still runs"}
        yield ("turn_failed" if failed else "turn_completed"), {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    events, _ = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))
    assert events[0].text == "model still runs"
    assert index.call_count == int(workspace_mode == "persistent")
    assert publish.await_count == int(workspace_mode == "persistent" and not failed)


@pytest.mark.parametrize("failed", [False, True])
async def test_only_an_unfinished_turn_claims_unattributed_artifacts(monkeypatch, failed):
    """ENG-2961 R1: a turn cut short between anton's metadata write and its
    provenance append leaves a folder with no provenance; only a non-clean exit
    may claim it. The producer tells `index_turn_artifacts` how the turn ended
    and leaves the provenance filter to it."""
    from unittest.mock import AsyncMock, Mock
    from cowork.services import task_objects

    context = (object(), object(), "project", "Project")
    _fake_handler(monkeypatch, remote_artifacts_context=lambda *_: context)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    index = Mock(return_value=([], set(), None))
    monkeypatch.setattr(task_objects, "index_turn_artifacts", index)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield ("turn_failed" if failed else "turn_completed"), {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))
    assert index.call_args.kwargs["attribute_by_provenance"] is True
    assert index.call_args.kwargs["completed_cleanly"] is (not failed)
