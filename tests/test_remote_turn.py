"""remote_turn_events tests: verify its own dispatch logic in isolation.
ResponsesHandler is replaced wholesale per test; its helpers have coverage elsewhere."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest

import cowork.handlers.responses as responses_mod
import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory
from cowork.schemas.figure_sources import EarlierAnswer, Figure, FigureOrigin
from cowork.turnqueue.producer import RemoteWorkspace
from cowork.turnqueue.redis_client import cancel_flag_key
from cowork.turnqueue.remote_turn import RemoteTurnFailed, remote_turn_events
from tests._fakes import inline_run_db
from tests._logging_assertions import assert_log_fields


class _FakeScope:
    org_mode = True
    org_id = "org-123"
    # An organization's scope, which cowork.db.units.run_efs reads for the share.
    partitioned = True
    user_id = None


class _FakeSession:
    scope = _FakeScope()


def _no_memory(session, conv_id, entries, principal):
    """ResponsesHandler._persist_turn_memory's own signature, so a call that
    leaves an argument out fails as the real one does."""


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
    # Each unit runs inline, on a session that reaches no database.
    monkeypatch.setattr(remote_turn_mod, "run_db", inline_run_db(_FakeSession()))
    monkeypatch.setattr(remote_turn_mod, "ResponsesHandler", type(
        "FakeResponsesHandler", (), {
            "_remote_artifacts_context": staticmethod(remote_artifacts_context or (lambda s, c: None)),
            "_remote_seed_history": staticmethod(remote_seed_history or (lambda s, c: ([], None))),
            "_remote_workspace": staticmethod(remote_workspace or (lambda s, c: None)),
            "_remote_agent_files": staticmethod(remote_agent_files or (lambda s, c, *, workspace: None)),
            "_remote_started_at": staticmethod(lambda s, c: None),
        },
    ))
    # The pod's memory and compaction go through the helpers the browser
    # producer shares (save_turn_memory, save_turn_compaction), which run
    # their units with cowork.handlers.responses's own run_db.
    monkeypatch.setattr(responses_mod, "run_db", inline_run_db(_FakeSession()))
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_persist_turn_memory", staticmethod(persist_turn_memory or _no_memory),
    )
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_persist_remote_compaction",
        staticmethod(lambda s, *, conv_id, data, seed_info: None),
    )


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
        scope=session.scope, conv_id=conv_id, org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="And at S2?", model="anton", turn_rows=[],
    )]

    assert isinstance(events[0], TurnContext)
    assert events[0].history == seeded


@pytest.mark.asyncio
async def test_turn_memory_calls_persist_turn_memory(monkeypatch):
    """Outside org tenancy a channel turn writes memory with no principal.
    The fake keeps the real signature, so a call that drops the principal
    fails."""
    from cowork.db.scoped import LOCAL_SCOPE

    calls = []

    def persist(session, conv_id, entries, principal):
        calls.append((entries, principal))

    _fake_handler(monkeypatch, persist_turn_memory=persist)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_memory", {"entries": [{"key": "likes", "value": "pizza"}]}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    await _drain(remote_turn_events(
        scope=LOCAL_SCOPE, conv_id=uuid4(), org_id=None, user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert calls == [([{"key": "likes", "value": "pizza"}], None)]


async def test_an_org_channel_turn_saves_no_memory_and_says_so_at_info(monkeypatch, caplog):
    """In org tenancy a channel turn has no member, and neither memory tier
    takes an entry without one. The turn skips the write and logs one INFO
    line naming the conversation, rather than an ERROR with a stack for a
    refusal that is by design."""
    import logging

    from cowork.db.scoped import TenantScope

    calls = []

    def persist(session, conv_id, entries, principal):
        calls.append(entries)

    _fake_handler(monkeypatch, persist_turn_memory=persist)

    async def fake_replies(**kwargs):
        yield "turn_memory", {"entries": [{"key": "likes", "value": "pizza"}]}
        yield "turn_delta", {"text": "noted"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)
    conv_id = uuid4()

    with caplog.at_level(logging.INFO):
        events, failure = await _drain(remote_turn_events(
            scope=TenantScope(org_mode=True, org_id="org-123"), conv_id=conv_id, org_id="org-123",
            user_id=None, input_text="hi", model="anton", turn_rows=[], correlation_id="corr-1",
        ))

    assert failure is None
    assert [event.text for event in events] == ["noted"]
    assert calls == []
    logged = [
        r for r in caplog.records
        if r.name in (remote_turn_mod.logger.name, responses_mod.logger.name) and r.levelno >= logging.INFO
    ]
    assert [(r.name, r.levelname, r.getMessage()) for r in logged] == [(
        remote_turn_mod.logger.name, "INFO",
        "channel turn memory not saved: an organization's channel turn has no member",
    )]
    assert (logged[0].conversation_id, logged[0].request_id) == (str(conv_id), "corr-1")


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("turn_memory", {"entries": [{"key": "likes", "value": "pizza"}]}),
        ("turn_compaction", {"summary": "EARLIER: …", "covered_through": 1}),
    ],
    ids=["memory", "compaction"],
)
async def test_a_busy_pool_on_a_memory_or_compaction_frame_keeps_the_channel_turn(
    monkeypatch, controller_flags, one_connection_pool, caplog, kind, data,
):
    """The pod sends a memory or a compaction frame while the pool frees no
    connection within POOL_TIMEOUT. The write is dropped with one WARNING,
    and the turn goes on to its reply: it does not fail, and the controller
    is not told to stop the pod's turn."""
    import logging

    from cowork.db import units
    from cowork.db.scoped import LOCAL_SCOPE

    _fake_handler(monkeypatch)
    # The write's unit waits for the app's pool, which the test holds.
    monkeypatch.setattr(responses_mod, "run_db", units.run_db)

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "before "}
        yield kind, data
        yield "turn_delta", {"text": "after"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)
    conv_id = uuid4()

    with caplog.at_level(logging.WARNING), one_connection_pool.connect():
        events, failure = await _drain(remote_turn_events(
            scope=LOCAL_SCOPE, conv_id=conv_id, org_id=None, user_id=None,
            input_text="hi", model="anton", turn_rows=[],
        ))

    assert failure is None
    assert [event.text for event in events] == ["before ", "after"]
    assert await controller_flags.keys("*") == []
    warned = [r for r in caplog.records if r.name == responses_mod.logger.name and r.levelno >= logging.WARNING]
    assert [r.levelname for r in warned] == ["WARNING"]
    assert warned[0].getMessage().startswith("[responses] no database connection freed in time")
    assert warned[0].conversation_id == str(conv_id)


@pytest.mark.asyncio
async def test_no_artifacts_context_skips_indexing_without_error(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events, failure = await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
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
        scope=session.scope, conv_id=uuid4(), org_id="org-123", user_id=None,
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
    diff = Mock(return_value=task_objects.ArtifactChanges(created=[], touched=set()))
    publish = AsyncMock(return_value=[])
    monkeypatch.setattr(task_objects, "turn_artifact_changes", diff)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", publish)

    async def replies(**kwargs):
        if workspace_mode is not None:
            yield "progress", {"phase": "workspace_authorized", "workspace_mode": workspace_mode}
        yield "turn_delta", {"text": "model still runs"}
        yield ("turn_failed" if failed else "turn_completed"), {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    events, _ = await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))
    assert events[0].text == "model still runs"
    assert diff.call_count == int(workspace_mode == "persistent")
    assert publish.await_count == int(workspace_mode == "persistent" and not failed)


@pytest.mark.parametrize("failed", [False, True])
async def test_only_an_unfinished_turn_claims_unattributed_artifacts(monkeypatch, failed):
    """ENG-2961 R1: a turn cut short between anton's metadata write and its
    provenance append leaves a folder with no provenance; only a non-clean exit
    may claim it. The producer tells `turn_artifact_changes` how the turn ended
    and leaves the provenance filter to it."""
    from unittest.mock import AsyncMock, Mock
    from cowork.services import task_objects

    context = (object(), object(), "project", "Project")
    _fake_handler(monkeypatch, remote_artifacts_context=lambda *_: context)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    diff = Mock(return_value=task_objects.ArtifactChanges(created=[], touched=set()))
    monkeypatch.setattr(task_objects, "turn_artifact_changes", diff)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield ("turn_failed" if failed else "turn_completed"), {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))
    assert diff.call_args.kwargs["attribute_by_provenance"] is True
    assert diff.call_args.kwargs["completed_cleanly"] is (not failed)


async def test_a_channel_turn_records_the_ids_of_the_folders_it_made(monkeypatch, tmp_path):
    """The turn end passes each new folder's id, read with the turn's diff,
    to the record, which marks the id conflicted when its row names another
    folder."""
    from unittest.mock import AsyncMock, Mock
    from cowork.services import task_objects

    context = (SimpleNamespace(created_by="user"), tmp_path, str(uuid4()), "Project")
    _fake_handler(monkeypatch, remote_artifacts_context=lambda *_: context)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    artifact_id = uuid4().hex
    monkeypatch.setattr(task_objects, "turn_artifact_changes", Mock(return_value=task_objects.ArtifactChanges(
        created=["made"], touched={"made"}, created_ids={"made": artifact_id},
    )))
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))
    recorded = {}
    monkeypatch.setattr(task_objects, "record_new_artifacts", lambda session, **kwargs: recorded.update(kwargs))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))

    assert recorded["slugs"] == ["made"]
    assert recorded["artifact_ids"] == {"made": artifact_id}


@pytest.mark.parametrize(
    ("completed", "tracked"),
    [({"artifacts_touched": ["mine"]}, {"mine"}), ({}, None)],
    ids=["pod-list", "older-pod"],
)
async def test_the_pods_list_bounds_a_channel_turns_diff(monkeypatch, completed, tracked):
    from unittest.mock import AsyncMock, Mock
    from cowork.services import task_objects

    context = (object(), object(), "project", "Project")
    _fake_handler(monkeypatch, remote_artifacts_context=lambda *_: context)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    diff = Mock(return_value=task_objects.ArtifactChanges(created=[], touched=set()))
    publish = AsyncMock(return_value=[])
    monkeypatch.setattr(task_objects, "turn_artifact_changes", diff)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", publish)

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_completed", completed

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org", user_id="user",
        input_text="run", model="m", turn_rows=[],
    ))
    assert diff.call_args.kwargs["tracked_new"] == tracked
    assert diff.call_args.kwargs["tracked_edits"] == tracked
    # With the list the publish opens only the turn's folders too.
    assert publish.call_args.kwargs["only_touched"] is (tracked is not None)


@pytest.fixture
def controller_flags(monkeypatch):
    """The Redis the controller's cancel flags are written to."""
    import fakeredis.aioredis

    from cowork.streaming import cancellation

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cancellation, "get_redis", lambda: client)
    return client


def _replies_that_wait(sent: list[dict]):
    """A controller that answers with one delta, then runs on with no terminal."""
    started = asyncio.Event()

    async def replies(**kwargs):
        sent.append(kwargs)
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "partial"}
        started.set()
        await asyncio.Event().wait()

    return replies, started


@pytest.mark.parametrize("correlation_id", [None, "corr-given"], ids=["minted", "given"])
async def test_a_cancelled_channel_turn_stops_its_controller_turn(monkeypatch, controller_flags, correlation_id):
    """Shutdown's drain of channel tasks cancels a channel turn still running.
    No registry handle covers it, so the turn itself writes the controller's
    cancel flag, under the correlation id the job went out with."""
    _fake_handler(monkeypatch)
    sent: list[dict] = []
    replies, started = _replies_that_wait(sent)
    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)

    turn = asyncio.create_task(_drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[], correlation_id=correlation_id,
    )))
    await asyncio.wait_for(started.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    sent_id = sent[0]["correlation_id"]
    assert sent_id, "the job went out with no correlation id"
    if correlation_id is not None:
        assert sent_id == correlation_id
    assert await controller_flags.keys("*") == [cancel_flag_key(sent_id)]


async def test_a_channel_turn_its_reader_abandons_stops_its_controller_turn(monkeypatch, controller_flags):
    _fake_handler(monkeypatch)
    sent: list[dict] = []
    replies, _ = _replies_that_wait(sent)
    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)

    events = remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    )
    async for event in events:
        if getattr(event, "text", None) == "partial":
            break
    await events.aclose()

    assert await controller_flags.keys("*") == [cancel_flag_key(sent[0]["correlation_id"])]


@pytest.mark.parametrize("terminal", ["turn_completed", "turn_failed"])
async def test_a_channel_turn_that_reached_its_terminal_writes_no_cancel_flag(
    monkeypatch, controller_flags, terminal,
):
    _fake_handler(monkeypatch)

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "done"}
        yield terminal, {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)

    await _drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert await controller_flags.keys("*") == []


async def test_a_cancelled_channel_turn_stays_cancelled_when_the_flag_cannot_be_written(
    monkeypatch, caplog, owned_logger,
):
    """Redis down at shutdown: the controller cannot be told, which is logged,
    and the channel task still ends cancelled rather than with the Redis
    error, which drain_background_tasks would get in place of the cancel. The
    hosted log line names the turn in fields, since its formatter drops a
    message's arguments."""
    import logging

    from cowork.streaming import cancellation

    def unreachable():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(cancellation, "get_redis", unreachable)
    logged = owned_logger(cancellation.logger.name, level=logging.WARNING)
    _fake_handler(monkeypatch)
    sent: list[dict] = []
    replies, started = _replies_that_wait(sent)
    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)

    turn = asyncio.create_task(_drain(remote_turn_events(
        scope=_FakeScope(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    )))
    await asyncio.wait_for(started.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    records = [
        r for r in caplog.records
        if r.name == cancellation.logger.name and r.levelno == logging.WARNING
    ]
    correlation_id = sent[0]["correlation_id"]
    fields = {
        "event": "controller_turn_stop_failed",
        "request_id": correlation_id,
        "turn_id": correlation_id,
        "reason": "channel_turn_ended",
    }
    assert [{name: getattr(r, name) for name in fields} for r in records] == [fields]
    assert isinstance(records[0].exc_info[1], ConnectionError)
    assert_log_fields(logged.output(), **fields)


class _Recorded(NamedTuple):
    """One call of record_new_artifacts, as its spy saw it."""

    slugs: list[str]
    project_id: UUID
    creator: str | None
    scope: object
    on_efs_thread: bool


# The project of _turn_with_a_new_artifact, as its artifacts context names it.
_PROJECT_ID = str(uuid4())


def _real_units_after_the_inputs_read(monkeypatch):
    """The turn's opening read of its inputs runs inline on a fake session, as
    _fake_handler sets up; every later unit runs as a real unit on the test
    database."""
    from cowork.db import units

    inline = inline_run_db(_FakeSession())

    async def run_db(fn, *, scope):
        if fn.func is remote_turn_mod._read_channel_turn_inputs:
            return await inline(fn, scope=scope)
        return await units.run_db(fn, scope=scope)

    monkeypatch.setattr(remote_turn_mod, "run_db", run_db)


def _turn_with_a_new_artifact(monkeypatch, *, diff=None, snapshot=None):
    """A channel turn in a project where the turn's end finds one new
    artifact, "report". The record runs as a real unit on the test database,
    in the scope the turn is given, with a spy in place of
    record_new_artifacts; ``diff`` and ``snapshot`` replace the file store
    halves. Returns the spy's calls."""
    from unittest.mock import AsyncMock

    from cowork.db.units import on_efs_thread
    from cowork.services import task_objects

    conversation = SimpleNamespace(created_by="user-1")
    _fake_handler(
        monkeypatch, remote_artifacts_context=lambda *_: (conversation, "/project/artifacts", _PROJECT_ID, "Project"),
    )
    _real_units_after_the_inputs_read(monkeypatch)
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", snapshot or (lambda *_: (set(), {})))
    monkeypatch.setattr(
        task_objects, "turn_artifact_changes",
        diff or (lambda **_: task_objects.ArtifactChanges(created=["report"], touched={"report"})),
    )
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))
    recorded: list[_Recorded] = []

    def record_new_artifacts(session, *, conversation_id, project_id, slugs, creator, artifact_ids=None):
        recorded.append(_Recorded(slugs, project_id, creator, session.scope, on_efs_thread()))

    monkeypatch.setattr(task_objects, "record_new_artifacts", record_new_artifacts)
    return recorded


def _completed_persistent_turn(monkeypatch, *, before_completed=None):
    """A controller that authorizes a persistent workspace and completes the
    turn, after awaiting ``before_completed`` when one is given."""

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        if before_completed is not None:
            await before_completed()
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)


def _channel_turn(*, scope, conv_id=None):
    return remote_turn_events(
        scope=scope, conv_id=conv_id or uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    )


async def test_a_channel_turn_diffs_on_a_file_store_thread_and_records_in_a_unit(monkeypatch):
    """The diff reads the shared file store, so it runs on a file store
    thread. The record writes rows, so it runs as a database unit off that
    thread, in the scope the turn is given: a stalled pool then holds no file
    store slot."""
    from cowork.db.scoped import TenantScope
    from cowork.db.units import on_efs_thread
    from cowork.services import task_objects

    org = TenantScope(org_mode=True, org_id="org-123")
    diffs: list[bool] = []

    def diff(**kwargs):
        diffs.append(on_efs_thread())
        return task_objects.ArtifactChanges(created=["report"], touched={"report"})

    recorded = _turn_with_a_new_artifact(monkeypatch, diff=diff)
    _completed_persistent_turn(monkeypatch)

    _, failure = await _drain(_channel_turn(scope=org))

    assert failure is None
    assert diffs == [True]
    assert recorded == [_Recorded(["report"], UUID(_PROJECT_ID), "user-1", org, False)]


def _artifact_record_log(caplog):
    """The one record the turn's artifact bookkeeping logged."""
    [record] = [r for r in caplog.records if r.name == remote_turn_mod.logger.name]
    return record


async def test_a_channel_turn_whose_artifact_diff_fails_still_finishes(monkeypatch, caplog, owned_logger):
    """The diff runs in the turn's `finally`, where an exception would
    replace the turn's real outcome. It is logged as a WARNING with the
    conversation and the project instead, and nothing is recorded or
    carded."""
    import logging

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import task_objects

    def failing_diff(**kwargs):
        raise RuntimeError("the artifacts folder could not be read")

    recorded = _turn_with_a_new_artifact(monkeypatch, diff=failing_diff)
    _completed_persistent_turn(monkeypatch)
    logged = owned_logger(remote_turn_mod.logger.name, level=logging.WARNING)
    conv_id = uuid4()

    _, failure = await _drain(_channel_turn(scope=LOCAL_SCOPE, conv_id=conv_id))

    assert failure is None
    assert recorded == []
    publish = task_objects.publish_and_card_turn_artifacts
    assert (publish.await_args.kwargs["new_slugs"], publish.await_args.kwargs["touched_slugs"]) == ([], set())
    record = _artifact_record_log(caplog)
    assert (record.levelname, record.getMessage()) == ("WARNING", "Could not index artifacts created this turn")
    assert isinstance(record.exc_info[1], RuntimeError)
    assert (record.conversation_id, record.project_id) == (str(conv_id), _PROJECT_ID)
    assert_log_fields(logged.output(), log_level="WARNING", conversation_id=str(conv_id), project_id=_PROJECT_ID)


async def test_a_channel_turn_whose_artifact_record_fails_still_finishes_and_cards(
    monkeypatch, caplog, owned_logger,
):
    """The record runs in the turn's `finally` too. Its failure is logged as
    an ERROR with the conversation, the project and the slugs, as the web
    path logs it: in org mode nothing records those folders later. The turn
    keeps its own outcome, and the folder it created still gets its card."""
    import logging

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import task_objects

    _turn_with_a_new_artifact(monkeypatch)

    def failing_record(session, **kwargs):
        raise RuntimeError("the index row could not be written")

    monkeypatch.setattr(task_objects, "record_new_artifacts", failing_record)
    _completed_persistent_turn(monkeypatch)
    logged = owned_logger(remote_turn_mod.logger.name, level=logging.WARNING)
    conv_id = uuid4()

    _, failure = await _drain(_channel_turn(scope=LOCAL_SCOPE, conv_id=conv_id))

    assert failure is None
    assert task_objects.publish_and_card_turn_artifacts.await_args.kwargs["new_slugs"] == ["report"]
    record = _artifact_record_log(caplog)
    assert (record.levelname, record.getMessage()) == (
        "ERROR", "Could not record the artifacts a channel turn created",
    )
    assert isinstance(record.exc_info[1], RuntimeError)
    assert (record.conversation_id, record.project_id, record.artifact_slugs) == (
        str(conv_id), _PROJECT_ID, ("report",),
    )
    assert_log_fields(
        logged.output(), log_level="ERROR",
        conversation_id=str(conv_id), project_id=_PROJECT_ID, artifact_slugs=["report"],
    )


async def test_a_refused_artifact_record_names_the_conversation_project_and_slugs(
    monkeypatch, caplog, owned_logger,
):
    """The pool refuses the record's unit. The database filter keeps only
    the call site of a database error, so the ids ride the record."""
    import logging

    from cowork.db import units
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.units import DatabaseBusy
    from cowork.services import task_objects

    slugs = ["chart", "sales-report"]
    recorded = _turn_with_a_new_artifact(
        monkeypatch,
        diff=lambda **_: task_objects.ArtifactChanges(created=slugs, touched=set(slugs)),
    )

    def refuse(*, scope):
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(units, "unit_session", refuse)
    _completed_persistent_turn(monkeypatch)
    logged = owned_logger(remote_turn_mod.logger.name, level=logging.WARNING)
    conv_id = uuid4()

    _, failure = await _drain(_channel_turn(scope=LOCAL_SCOPE, conv_id=conv_id))

    assert failure is None
    assert recorded == []
    record = _artifact_record_log(caplog)
    assert record.levelname == "ERROR"
    assert record.getMessage() == (
        "Database operation failed: error_type=DatabaseBusy sqlstate=unknown "
        f"site=remote_turn._record_turn_artifacts:{record.lineno}"
    )
    assert (record.conversation_id, record.project_id, record.artifact_slugs) == (
        str(conv_id), _PROJECT_ID, tuple(slugs),
    )
    assert_log_fields(
        logged.output(), log_level="ERROR",
        conversation_id=str(conv_id), project_id=_PROJECT_ID, artifact_slugs=slugs,
    )


@pytest.mark.usefixtures("cleanup_tmp_projects")
async def test_a_channel_turn_writes_the_index_row_and_the_creators_owner_row(monkeypatch, tmp_path):
    """No spy: the turn runs in an organization's scope, its inputs are read
    in a real unit, and the turn's end writes the new folder's task_objects
    row and its creator's owner row on the test database. Without the owner
    row its creator could no longer edit it."""
    from unittest.mock import AsyncMock

    from cowork.db import units
    from cowork.db.scoped import TenantScope
    from cowork.handlers.responses import ResponsesHandler
    from cowork.models.task_object import TaskObject
    from cowork.services import artifact_ownership as ownership
    from cowork.services import task_objects
    from test_artifact_ownership import make_conversation, make_project, project_root, write_artifact

    org_id, creator = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, creator)
    source = project_root(project)
    slug = "sales-report"
    _fake_handler(monkeypatch, remote_artifacts_context=ResponsesHandler._remote_artifacts_context)
    monkeypatch.setattr(remote_turn_mod, "run_db", units.run_db)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        write_artifact(source.base, slug, conversation=conversation_id)
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", replies)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=creator)
    _, failure = await _drain(remote_turn_events(
        scope=scope, conv_id=conversation_id, org_id=org_id, user_id=creator,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is None
    with units.unit_session(scope=scope) as session:
        indexed = session.exec(
            session.select(TaskObject).where(TaskObject.conversation_id == conversation_id, TaskObject.ref == slug)
        ).all()
        owner = ownership.resolve_artifact_owner(session, source, slug)
    assert (len(indexed), owner) == (1, ownership.OwnerResolution(creator, "recorded"))


async def test_a_cancel_while_the_turn_end_waits_for_a_file_store_slot_still_records_its_artifacts(monkeypatch):
    """Shutdown's drain of channel tasks cancels a channel turn whose end
    waits for a file store slot while the mount stalls. The cancel waits for
    the slot, the diff and the record, then ends the task cancelled: in org
    mode nothing records the turn's artifacts later."""
    from cowork.db import units
    from cowork.db.scoped import LOCAL_SCOPE

    recorded = _turn_with_a_new_artifact(monkeypatch)
    slots = units._efs_slots()
    holders = [object() for _ in range(units.EFS_SLOTS)]

    async def take_every_slot():
        for holder in holders:
            await slots.acquire_on_behalf_of(holder)

    _completed_persistent_turn(monkeypatch, before_completed=take_every_slot)
    turn = asyncio.create_task(_drain(_channel_turn(scope=LOCAL_SCOPE)))
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while slots.statistics().tasks_waiting == 0:
            assert loop.time() < deadline, "the turn's end never waited for a slot"
            await asyncio.sleep(0.01)
        turn.cancel()
        await asyncio.sleep(0.2)
        assert not turn.done(), "the cancel did not wait for the turn's artifact record"
        assert recorded == []
    finally:
        for holder in holders:
            if holder in slots.statistics().borrowers:
                slots.release_on_behalf_of(holder)
    await asyncio.wait_for(asyncio.gather(turn, return_exceptions=True), timeout=5)

    assert turn.cancelled()
    assert [call.slugs for call in recorded] == [["report"]]


async def test_a_channel_turns_artifact_diff_waits_past_pool_timeout_for_a_file_store_slot(monkeypatch, caplog):
    """The controller completed the turn while every file store slot stays
    taken past POOL_TIMEOUT. The turn's diff waits for a slot rather than
    fail, and records the artifact once the slots free: in org mode nothing
    records it later."""
    import logging

    from cowork.db import units
    from cowork.db.scoped import LOCAL_SCOPE
    from test_responses_remote_backend import _FileStoreSlots

    monkeypatch.setattr(units.db_session.settings.database, "pool_timeout", 1)
    slots = _FileStoreSlots()
    recorded = _turn_with_a_new_artifact(monkeypatch)
    _completed_persistent_turn(monkeypatch, before_completed=slots.take)
    with caplog.at_level(logging.WARNING, logger=remote_turn_mod.logger.name):
        turn = asyncio.create_task(_drain(_channel_turn(scope=LOCAL_SCOPE)))
        try:
            await slots.until_someone_waits()
            await asyncio.sleep(1.5)
            assert not turn.done(), "the diff gave up waiting for a slot"
            assert recorded == []
        finally:
            slots.free()
        _, failure = await asyncio.wait_for(turn, timeout=5)

    assert failure is None
    assert [call.slugs for call in recorded] == [["report"]]
    assert not [
        r for r in caplog.records
        if r.name == remote_turn_mod.logger.name and r.getMessage() == "Could not index artifacts created this turn"
    ]


async def test_a_channel_turns_artifact_bookkeeping_leaves_the_event_loop_free(monkeypatch):
    """The snapshot at turn start and the diff at turn end read the shared
    file store. Each is made slow here, and the loop keeps ticking through
    both."""
    import time

    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.services import task_objects

    slow = 1.0
    steps = []

    def slow_snapshot(base):
        steps.append("snapshot")
        time.sleep(slow)
        return set(), {}

    def slow_diff(**kwargs):
        steps.append("diff")
        time.sleep(slow)
        return task_objects.ArtifactChanges(created=["report"], touched={"report"})

    recorded = _turn_with_a_new_artifact(monkeypatch, diff=slow_diff, snapshot=slow_snapshot)
    _completed_persistent_turn(monkeypatch)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    loop = asyncio.get_running_loop()
    beat = asyncio.create_task(heartbeat())
    started = loop.time()
    try:
        _, failure = await _drain(_channel_turn(scope=LOCAL_SCOPE))
    finally:
        beat.cancel()
    elapsed = loop.time() - started

    assert failure is None
    assert steps == ["snapshot", "diff"]
    assert [call.slugs for call in recorded] == [["report"]]
    assert elapsed >= 2 * slow
    # A free loop ticks about 100 times a second here; a blocked one stops.
    assert ticks >= 50 * elapsed * 0.5, f"{ticks} ticks in {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_a_busy_file_store_fails_the_channel_turn_before_its_seed_history_is_built(monkeypatch):
    """With no file store slot free, the vault's secrets cannot be registered,
    and a seed history scrubbed by key shape alone would carry a saved
    password to the pod. The turn fails instead, as it does on a busy pool."""
    import cowork.db.session as db_session
    from cowork.db import units
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.units import FilesystemBusy
    from test_run_efs import holding

    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    seeded = []
    _fake_handler(monkeypatch, remote_seed_history=lambda s, c: seeded.append(c) or ([], None))

    async def fake_replies(**kwargs):
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    session = _FakeSession()
    async with holding(units.EFS_SLOTS, LOCAL_SCOPE):
        with pytest.raises(FilesystemBusy):
            await _drain(remote_turn_events(
                scope=session.scope, conv_id=uuid4(), org_id="org-123", user_id=None,
                input_text="hi", model="anton", turn_rows=[],
            ))

    assert seeded == []
