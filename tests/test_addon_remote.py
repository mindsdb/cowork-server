"""POST /api/v1/responses/addon and /addon/withdraw on the remote backend: the
turn runs in a pod (ENG-4194). Jev decides here; the pod's verdict is the
status. The Redis side is tests/turnqueue/test_remote_addons.py."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import cowork.api.v1.endpoints.responses as endpoints
from cowork.api.v1.endpoints.responses import SharedTurn
from cowork.common.settings.app_settings import get_app_settings
from cowork.handlers import addon_route, jev_route
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.server import create_app
from cowork.streaming.addons import TurnAddon
from cowork.streaming.turn_index import TurnIndexEntry
from cowork.turnqueue.addons import RemoteAddonResult, RemoteWithdrawResult
from cowork.turnqueue.producer import step_stream_events
from anton.core.llm.provider import StreamTextDelta
from tests.test_addon_route import TARGET, _Client, _Response, _verdict

CID = "00000000-0000-0000-0000-0000000000a1"
ADDON_ID = "a" * 32


@pytest.fixture(autouse=True)
def _remote_backend(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    monkeypatch.setenv("COWORK_TURN_ADDON_ROUTING_ENABLED", "true")
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture()
def turn(monkeypatch):
    state = {"found": SharedTurn(index=TurnIndexEntry(turn_id=1, run_id=None, correlation_id="corr-1"),
                                 buffer=None, in_flight=True)}

    async def fake_shared_turn(conversation_id, scope):
        return state["found"] if conversation_id == CID else None

    monkeypatch.setattr(endpoints, "_shared_turn", fake_shared_turn)
    return state


@pytest.fixture()
def inputs(monkeypatch):
    """What Jev reads about the pod's turn: its question, recent tool calls
    and the messages it already took."""
    state = {"task": "Build the 2025 sales dashboard.", "progress": ["[ran tool: scratchpad]"], "added": []}

    async def fake_run_db(fn, *, scope):
        return state["task"]

    async def fake_progress(*, conversation_id, correlation_id):
        assert (conversation_id, correlation_id) == (CID, "corr-1")
        return state["progress"]

    async def fake_added(correlation_id):
        assert correlation_id == "corr-1"
        return state["added"]

    monkeypatch.setattr("cowork.db.units.run_db", fake_run_db)
    monkeypatch.setattr(addon_route, "remote_progress", fake_progress)
    monkeypatch.setattr(addon_route, "remote_added_texts", fake_added)
    return state


@pytest.fixture()
def pod(monkeypatch):
    calls: list[dict] = []
    state = {"result": RemoteAddonResult.ADDED, "withdraw": RemoteWithdrawResult.WITHDRAWN, "calls": calls}

    async def fake_submit(**kwargs):
        calls.append(kwargs)
        return state["result"], ADDON_ID

    async def fake_withdraw(**kwargs):
        calls.append(kwargs)
        return state["withdraw"]

    monkeypatch.setattr(addon_route, "submit_remote_addon", fake_submit)
    monkeypatch.setattr(endpoints, "withdraw_remote_addon", fake_withdraw)
    return state


@pytest.fixture()
def jev(monkeypatch):
    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: (TARGET, "jev"))

    def script(client):
        monkeypatch.setattr(jev_route, "_http_client", lambda: client)
        return client

    return script


def _post(client, conversation_id=CID, text="actually use the sales DB", **body):
    return client.post("/api/v1/responses/addon",
                       json={"conversation_id": conversation_id, "text": text, **body})


def _withdraw(client, conversation_id=CID, addon_id=ADDON_ID):
    return client.post("/api/v1/responses/addon/withdraw",
                       json={"conversation_id": conversation_id, "addon_id": addon_id})


# ── POST /responses/addon ──────────────────────────────────────────────────


def test_jev_adds_and_the_pod_takes_it(client, turn, inputs, pod, jev):
    captured: dict = {}
    jev(_Client(response=_Response(200, _verdict(p_add=0.9)), captured=captured))
    inputs["added"] = ["use the sales DB"]
    resp = _post(client, text="and the same for EMEA")
    assert resp.status_code == 200
    assert resp.json() == {"status": "added", "addon_id": ADDON_ID, "decided_by": "jev"}
    assert pod["calls"] == [{"conversation_id": CID, "correlation_id": "corr-1",
                             "text": "and the same for EMEA", "decided_by": "jev"}]
    assert captured["json"]["state"] == {
        "running_task": "Build the 2025 sales dashboard.",
        "recent_progress": ["[ran tool: scratchpad]"],
        "already_added_to_task": ["use the sales DB"],
        "new_user_message": "and the same for EMEA",
    }


def test_force_adds_without_asking_jev(client, turn, inputs, pod, jev, monkeypatch):
    jev(_Client(exception=AssertionError("Jev must not be asked")))
    asked = []

    async def never(**kwargs):
        asked.append(kwargs)
        raise AssertionError("Jev must not be asked")

    monkeypatch.setattr(addon_route, "_jev_refusal", never)
    monkeypatch.setattr(addon_route, "_remote_jev_inputs", never)
    resp = _post(client, force=True)
    assert resp.json() == {"status": "added", "addon_id": ADDON_ID, "decided_by": "user"}
    assert asked == []
    assert pod["calls"][0]["decided_by"] == "user"


def test_jev_queueing_it_never_reaches_the_pod(client, turn, inputs, pod, jev):
    jev(_Client(response=_Response(200, _verdict(choice="queue_as_next_task", p_add=0.1))))
    assert _post(client).json() == {"status": "queued", "reason": "next_task"}
    assert pod["calls"] == []


@pytest.mark.parametrize(("result", "reason"), [
    (RemoteAddonResult.REFUSED, "no_turn"),
    (RemoteAddonResult.NOT_FOUND, "no_turn"),
    (RemoteAddonResult.TIMEOUT, "pod_timeout"),
    (RemoteAddonResult.TOO_LARGE, "too_large"),
])
def test_pod_verdicts_that_queue(client, turn, inputs, pod, result, reason):
    pod["result"] = result
    assert _post(client, force=True).json() == {"status": "queued", "reason": reason}


def test_a_failure_reading_what_was_added_queues(client, turn, inputs, pod, jev, monkeypatch):
    async def boom(correlation_id):
        raise RuntimeError("redis down")

    monkeypatch.setattr(addon_route, "remote_added_texts", boom)
    assert _post(client).json() == {"status": "queued", "reason": "jev_error"}
    assert pod["calls"] == []


def test_unreadable_task_and_progress_still_ask_jev(client, turn, inputs, pod, jev, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr("cowork.db.units.run_db", boom)
    monkeypatch.setattr(addon_route, "remote_progress", boom)
    captured: dict = {}
    jev(_Client(response=_Response(200, _verdict()), captured=captured))
    assert _post(client).json()["status"] == "added"
    assert captured["json"]["state"]["running_task"] == ""
    assert captured["json"]["state"]["recent_progress"] == []


def test_a_hosted_org_without_a_stored_key_asks_on_a_minted_key(client, turn, inputs, pod, monkeypatch):
    # Hosted orgs keep no stored MindsHub key: the check mints one, asks, and
    # revokes it, like the route gate does for the turn itself.
    from cowork.turnqueue import producer

    minted, revoked, captured = [], [], {}

    async def fake_mint(**kwargs):
        minted.append(kwargs)
        return {"base_url": "https://minds.example/v1", "api_key": "turn-key"}

    async def fake_revoke(correlation_id):
        revoked.append(correlation_id)

    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: addon_route._MintKey(workspace_id="ws1", model="jev"))
    monkeypatch.setattr(producer, "_mint_llm_block", fake_mint)
    monkeypatch.setattr(producer, "_revoke_turn_key_quietly", fake_revoke)
    monkeypatch.setattr(jev_route, "_http_client",
                        lambda: _Client(response=_Response(200, _verdict(p_add=0.9)), captured=captured))

    assert _post(client).json() == {"status": "added", "addon_id": ADDON_ID, "decided_by": "jev"}
    assert len(minted) == 1 and minted[0]["workspace_id"] == "ws1"
    assert revoked == [minted[0]["correlation_id"]]


def test_a_failed_mint_queues_as_unchecked(client, turn, inputs, pod, monkeypatch):
    from cowork.turnqueue import producer

    async def boom(**kwargs):
        raise RuntimeError("auth down")

    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: addon_route._MintKey(workspace_id=None, model="jev"))
    monkeypatch.setattr(producer, "_mint_llm_block", boom)
    assert _post(client).json() == {"status": "queued", "reason": "jev_error"}
    assert pod["calls"] == []


def test_finished_turn_queues_as_no_turn(client, turn, pod):
    turn["found"] = turn["found"]._replace(in_flight=False)
    assert _post(client, force=True).json() == {"status": "queued", "reason": "no_turn"}
    assert pod["calls"] == []


def test_unknown_conversation_is_not_found(client, turn, pod):
    resp = _post(client, conversation_id="someone-elses", force=True)
    assert resp.status_code == 404
    assert pod["calls"] == []


def test_disabled_never_reaches_the_pod(client, turn, pod, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_ADDON_ROUTING_ENABLED", "false")
    assert _post(client, force=True).json() == {"status": "queued", "reason": "disabled"}
    assert pod["calls"] == []


# ── POST /responses/addon/withdraw ─────────────────────────────────────────


@pytest.mark.parametrize(("result", "status", "body"), [
    (RemoteWithdrawResult.WITHDRAWN, 200, {"status": "withdrawn"}),
    (RemoteWithdrawResult.ALREADY_READ, 409, {"status": "already_read"}),
    (RemoteWithdrawResult.NOT_FOUND, 404, {"status": "not_found"}),
    (RemoteWithdrawResult.UNAVAILABLE, 503, {"status": "unavailable"}),
])
def test_withdraw_verdicts_map_to_statuses(client, turn, pod, result, status, body):
    pod["withdraw"] = result
    resp = _withdraw(client)
    assert (resp.status_code, resp.json()) == (status, body)
    assert pod["calls"] == [{"conversation_id": CID, "correlation_id": "corr-1", "addon_id": ADDON_ID}]


@pytest.mark.parametrize("in_flight,conversation_id", [(False, CID), (True, "someone-elses")])
def test_withdraw_without_a_running_turn_is_not_found(client, turn, pod, in_flight, conversation_id):
    turn["found"] = turn["found"]._replace(in_flight=in_flight)
    assert _withdraw(client, conversation_id=conversation_id).status_code == 404
    assert pod["calls"] == []


# ── The pod's addon steps on the app's stream ──────────────────────────────


def _step(state, **extra):
    return {"step": "addon", "type": "response.addon", "addon_id": ADDON_ID, "state": state, **extra}


@pytest.mark.parametrize("state", ["added", "read", "returned"])
def test_the_states_the_app_renders_are_relayed(state):
    events = step_stream_events(_step(state, text="use the sales DB", decided_by="jev"))
    assert events == [TurnAddon(ADDON_ID, state, "use the sales DB", "jev")]


@pytest.mark.parametrize("state", ["refused", "withdrawn", "already_read", "something_new"])
def test_verdicts_on_one_request_are_not_relayed(state):
    assert step_stream_events(_step(state, request_id="r" * 32)) == []


def test_a_returned_message_the_pod_no_longer_knows_leaves_out_what_it_lacks():
    (event,) = step_stream_events(_step("returned", text="and EMEA", decided_by=None))
    assert event.payload() == {"type": "response.addon", "addon_id": ADDON_ID,
                               "state": "returned", "text": "and EMEA"}


async def test_a_relayed_addon_is_streamed_and_saved_as_response_addon():
    async def _events():
        for event in step_stream_events(_step("read", text="use the sales DB", decided_by="user")):
            yield event

    recorded: list[tuple[str, dict]] = []
    chunks = [c async for c in format_responses_stream(
        _events(), model="m", event_sink=lambda t, d: recorded.append((t, d)),
    )]
    frames = [json.loads(line[len("data:"):]) for chunk in chunks
              for line in chunk.splitlines() if line.startswith("data:")]
    addon = next(f for f in frames if f.get("type") == "response.addon")
    assert {k: addon[k] for k in ("addon_id", "state", "text", "decided_by")} == {
        "addon_id": ADDON_ID, "state": "read", "text": "use the sales DB", "decided_by": "user",
    }
    assert any(t == "response.addon" for t, _ in recorded)


# A hosted turn the person added a message to, as the pod wrote it (ENG-4194):
# captured from anton.cloud_turn's stream_turn driving a real ChatSession with
# a scripted model. Text deltas and step lines interleave on one wire.
_OPENING = "I'll pull the daily prices and build a compact dashboard."
_ACK = "Got it, six months instead. I'll rebuild the dashboard."
_FINAL = "Updated the dashboard to cover the past six months."


def _mid_turn_pod_lines() -> list[dict]:
    def _tool(call_id):
        return [
            {"kind": "tool_start", "id": call_id, "name": "scratchpad"},
            {"kind": "tool_end", "id": call_id, "args": '{"action": "exec", "name": "main", "code": "print(1)"}'},
        ]

    def _ran(call_id):
        return [
            {"kind": "round_end", "stop_reason": "tool_use", "had_tool_calls": True},
            {"kind": "progress", "phase": "scratchpad_start", "message": "Fetch prices", "id": call_id},
            {"kind": "progress", "phase": "scratchpad_done", "message": "Fetch prices", "eta_seconds": 0.1, "id": call_id},
            {"kind": "tool_result", "id": call_id, "name": "scratchpad", "action": "exec", "content": "{}"},
        ]

    addon = {"kind": "addon", "addon_id": ADDON_ID, "text": "actually past 6 months", "decided_by": "jev"}
    return [
        {"kind": "delta", "text": _OPENING},
        *_tool("t1"),
        {**addon, "state": "added"},
        *_ran("t1"),
        {**addon, "state": "read"},
        {"kind": "progress", "phase": "reasoning_start", "message": "Thinking..."},
        {"kind": "delta", "text": _ACK},
        *_tool("t2"),
        *_ran("t2"),
        {"kind": "delta", "text": _FINAL},
        {"kind": "round_end", "stop_reason": "end_turn", "had_tool_calls": False},
    ]


async def test_a_message_read_mid_turn_leaves_the_narration_before_each_tool():
    """The app files text that a tool start follows as a step and keeps only
    the last round as the answer (ENG-3554). A message read at a step boundary
    must not change that: no answer reset or restore, and every narration
    reaches the app before the tool call that ends its round."""
    async def _events():
        for line in _mid_turn_pod_lines():
            if line["kind"] == "delta":
                yield StreamTextDelta(text=line["text"])
            else:
                for event in step_stream_events({**line, "step": line["kind"]}):
                    yield event

    frames = []
    async for chunk in format_responses_stream(_events(), model="m"):
        frames += [json.loads(line[len("data:"):]) for line in chunk.splitlines() if line.startswith("data:")]
    types = [f["type"] for f in frames]
    assert "response.answer_reset" not in types
    assert "response.answer_restore" not in types
    texts = [(i, f["delta"]) for i, f in enumerate(frames) if f["type"] == "response.output_text.delta"]
    starts = [i for i, f in enumerate(frames) if f.get("thought_role") == "thought.scratchpad.start"]
    opening = next(i for i, d in texts if _OPENING in d)
    ack = next(i for i, d in texts if _ACK in d)
    final = next(i for i, d in texts if _FINAL in d)
    assert opening < starts[0] < ack < starts[1] < final
    read = next(i for i, f in enumerate(frames) if f["type"] == "response.addon" and f["state"] == "read")
    assert starts[0] < read < ack


# ── The running task, from the database ────────────────────────────────────


def test_the_running_task_is_the_pending_question(tmp_path):
    from sqlmodel import Session, SQLModel, create_engine

    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.models.conversation import Conversation
    from cowork.models.project import Project
    from cowork.services.conversations import ConversationService

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as raw:
        session = ScopedSession(raw, LOCAL_SCOPE)
        project = Project(name="P", path=str(tmp_path / "p"))
        session.add(project)
        session.commit()
        session.refresh(project)
        conv = Conversation(project_id=project.id, topic="t")
        session.add(conv)
        session.commit()
        session.refresh(conv)
        service = ConversationService(session)
        assert addon_route._running_task(session, conversation_id=conv.id) == ""
        service.save_user_message(conv.id, "an earlier question")
        service.save_user_message(
            conv.id, [{"type": "input_text", "text": "Build the 2025 sales dashboard."}], pending=True,
        )
        assert addon_route._running_task(session, conversation_id=conv.id) == "Build the 2025 sales dashboard."
