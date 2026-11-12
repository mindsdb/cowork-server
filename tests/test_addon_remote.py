"""POST /api/v1/responses/addon and /addon/withdraw on the remote backend: the
turn runs in a pod (ENG-4194). Jev decides here; the pod's verdict is the
status. The Redis side is tests/turnqueue/test_remote_addons.py."""

from __future__ import annotations

import asyncio
import json

import pytest
from redis.exceptions import RedisError
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
from cowork.turnqueue import addons as remote_addons
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


@pytest.fixture(autouse=True)
def redis_side(monkeypatch):
    """The Redis reads the route makes before the pod, scripted: the pod said
    "ready", nobody is over the rate limit, and the turn took nothing yet."""
    state = {"ready": True, "rate_limited": False, "totals": (0, 0), "rate_calls": 0}

    async def ready(correlation_id):
        if isinstance(state["ready"], Exception):
            raise state["ready"]
        return state["ready"]

    async def rate_limited(user_key, *, limit, window_s):
        state["rate_calls"] += 1
        if isinstance(state["rate_limited"], Exception):
            raise state["rate_limited"]
        return state["rate_limited"]

    async def totals(correlation_id):
        return state["totals"]

    monkeypatch.setattr(endpoints, "remote_addons_ready", ready)
    monkeypatch.setattr(addon_route, "remote_rate_limited", rate_limited)
    monkeypatch.setattr(addon_route, "remote_addon_totals", totals)
    return state


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
        kwargs.pop("deadline")
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
                             "text": "and the same for EMEA", "decided_by": "jev", "local_key": None}]
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


@pytest.mark.parametrize(("error", "reason"), [
    (RedisError("redis down"), "redis_error"),
    (RuntimeError("bug"), "internal"),
])
def test_a_failure_reading_what_was_added_queues(client, turn, inputs, pod, jev, monkeypatch, error, reason):
    async def boom(correlation_id):
        raise error

    monkeypatch.setattr(addon_route, "remote_added_texts", boom)
    assert _post(client).json() == {"status": "queued", "reason": reason}
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
    monkeypatch.setattr(producer, "mint_llm_block", fake_mint)
    monkeypatch.setattr(producer, "revoke_turn_key_quietly", fake_revoke)
    monkeypatch.setattr(jev_route, "_http_client",
                        lambda: _Client(response=_Response(200, _verdict(p_add=0.9)), captured=captured))

    assert _post(client).json() == {"status": "added", "addon_id": ADDON_ID, "decided_by": "jev"}
    assert len(minted) == 1 and minted[0]["workspace_id"] == "ws1"
    # Lives just past the Jev call, not a whole turn.
    assert minted[0]["ttl_seconds"] == 30
    assert revoked == [minted[0]["correlation_id"]]


@pytest.mark.parametrize("error", [RuntimeError("auth down"), asyncio.CancelledError()], ids=["failed", "cancelled"])
async def test_a_failed_or_cancelled_mint_still_revokes(monkeypatch, error):
    """The mint may have issued a key before it failed or was cancelled:
    revoked by its correlation id either way."""
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.turnqueue import producer

    minted, revoked = [], []

    async def mint(**kwargs):
        minted.append(kwargs)
        raise error

    async def revoke(correlation_id):
        revoked.append(correlation_id)

    async def target(fn):
        return addon_route._MintKey(workspace_id=None, model="jev")

    monkeypatch.setattr("cowork.db.units.run_on_unit_threads", target)
    monkeypatch.setattr(producer, "mint_llm_block", mint)
    monkeypatch.setattr(producer, "revoke_turn_key_quietly", revoke)
    call = addon_route._jev_refusal(conversation_id=CID, task="t", progress=[], added=[], text="x", scope=LOCAL_SCOPE)
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        assert await call == {"status": "queued", "reason": "internal"}
    assert revoked == [minted[0]["correlation_id"]]


def test_minted_key_ttl_tracks_the_jev_timeout(monkeypatch):
    from cowork.common.settings.app_settings import TurnQueueSettings

    assert addon_route.minted_key_ttl(TurnQueueSettings()) == 30
    monkeypatch.setenv("COWORK_TURN_JEV_ROUTE_TIMEOUT_SECONDS", "25.5")
    assert addon_route.minted_key_ttl(TurnQueueSettings()) == 36


def test_a_jev_target_never_shows_its_key():
    assert "turn-key" not in repr(TARGET)


def test_a_failed_mint_queues_as_internal(client, turn, inputs, pod, monkeypatch):
    from cowork.turnqueue import producer

    async def boom(**kwargs):
        raise RuntimeError("auth down")

    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: addon_route._MintKey(workspace_id=None, model="jev"))
    monkeypatch.setattr(producer, "mint_llm_block", boom)
    monkeypatch.setattr(producer, "revoke_turn_key_quietly", lambda correlation_id: asyncio.sleep(0))
    assert _post(client).json() == {"status": "queued", "reason": "internal"}
    assert pod["calls"] == []


# ── Support, limits and the app's ids ──────────────────────────────────────


def test_a_pod_that_never_said_ready_is_not_supported(client, turn, inputs, pod, monkeypatch, redis_side):
    """Checked before Jev and before any key is minted."""
    from cowork.turnqueue import producer

    async def never(**kwargs):
        raise AssertionError("must not be called")

    redis_side["ready"] = False
    monkeypatch.setattr(addon_route, "_jev_refusal", never)
    monkeypatch.setattr(producer, "mint_llm_block", never)
    assert _post(client).json() == {"status": "queued", "reason": "not_supported"}
    assert pod["calls"] == []


def test_a_redis_error_reading_ready_queues_as_redis_error(client, turn, pod, redis_side):
    redis_side["ready"] = RedisError("down")
    assert _post(client, force=True).json() == {"status": "queued", "reason": "redis_error"}


@pytest.mark.parametrize(("over", "reason"), [(True, "rate_limited"), (RedisError("down"), "redis_error")])
def test_the_shared_rate_limit(client, turn, pod, redis_side, over, reason):
    redis_side["rate_limited"] = over
    assert _post(client, force=True).json() == {"status": "queued", "reason": reason}
    assert pod["calls"] == []


@pytest.mark.parametrize(("totals", "reason"), [
    ((20, 100), "too_many"),
    ((3, 64 * 1024 - 5), "too_large"),
])
def test_the_per_turn_limits(client, turn, pod, redis_side, totals, reason):
    redis_side["totals"] = totals
    assert _post(client, force=True, text="hello world").json() == {"status": "queued", "reason": reason}
    assert pod["calls"] == []


def test_a_message_over_the_character_limit_is_too_large(client, turn, pod, redis_side):
    assert _post(client, force=True, text="a" * 4001).json() == {"status": "queued", "reason": "too_large"}
    assert redis_side["rate_calls"] == 0
    assert _post(client, force=True, text="a" * 4000).json()["status"] == "added"


@pytest.mark.parametrize(("body", "status"), [
    # A JSON escape for a lone surrogate: refused by request validation, as
    # any malformed field is.
    (b'{"conversation_id":"%s","text":"a\\ud800"}' % CID.encode(), 422),
    # Bytes that are not UTF-8: refused when the body is parsed.
    (b'{"conversation_id":"%s","text":"a\xff"}' % CID.encode(), 400),
], ids=["lone-surrogate", "invalid-utf8"])
def test_text_that_is_not_valid_unicode_is_refused(client, turn, pod, body, status):
    resp = client.post("/api/v1/responses/addon", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == status
    assert pod["calls"] == []


def test_the_handler_guards_against_a_lone_surrogate_too():
    """Behind request validation, in case a parser ever lets one through."""
    assert addon_route.valid_text("usá la base") is True
    assert addon_route.valid_text("a\ud800") is False


@pytest.mark.parametrize(("local_key", "kept"), [
    ("draft-42_A", "draft-42_A"),
    ("k" * 64, "k" * 64),
    ("k" * 65, None),
    ("has space", None),
    ("", None),
])
def test_local_key_is_passed_on_only_when_valid(client, turn, pod, local_key, kept):
    assert _post(client, force=True, local_key=local_key).json()["status"] == "added"
    assert pod["calls"][0]["local_key"] == kept


def test_a_response_id_for_another_turn_is_no_turn(client, turn, pod, monkeypatch):
    async def ids(buffer):
        return {"resp-current"}

    monkeypatch.setattr(endpoints, "turn_response_ids", ids)
    assert _post(client, force=True, response_id="resp-old").json() == {"status": "queued", "reason": "no_turn"}
    assert pod["calls"] == []
    assert _post(client, force=True, response_id="resp-current").json()["status"] == "added"


async def test_the_decision_lock_times_out_as_busy(monkeypatch):
    from cowork.db.scoped import LOCAL_SCOPE

    monkeypatch.setattr(addon_route, "DECIDE_LOCK_TIMEOUT_S", 0.05)
    async with addon_route._deciding("corr-1", addon_route.budget_deadline()):
        result = await addon_route.route_remote_addon(
            conversation_id=CID, correlation_id="corr-1", text="x", force=True, scope=LOCAL_SCOPE,
        )
    assert result == {"status": "queued", "reason": "busy"}
    assert addon_route._remote_deciding == {}


async def test_the_lock_entry_outlives_a_waiter():
    """A waiter keeps the entry: a third caller queues on the same lock
    instead of a fresh one, and the entry goes with the last of them."""
    order = []

    async def hold(name, delay):
        async with addon_route._deciding("corr-1", addon_route.budget_deadline()):
            order.append(name)
            await asyncio.sleep(delay)

    first = asyncio.create_task(hold("first", 0.05))
    await asyncio.sleep(0)
    second = asyncio.create_task(hold("second", 0))
    await asyncio.sleep(0)
    assert addon_route._remote_deciding["corr-1"].users == 2
    await first
    third = asyncio.create_task(hold("third", 0))
    await asyncio.gather(second, third)
    assert order == ["first", "second", "third"]
    assert addon_route._remote_deciding == {}


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


@pytest.mark.parametrize("state", ["added", "read", "returned", "withdrawn", "ready"])
def test_the_sync_step_mapping_renders_no_addon_step(state):
    """Addon steps go through relay_addon_step (tests/turnqueue/test_remote_addons.py),
    which checks the addon is this turn's; a channel turn renders none."""
    assert step_stream_events(_step(state, text="use the sales DB", decided_by="jev")) == []


def test_an_event_without_what_it_lacks_leaves_it_out():
    event = TurnAddon(ADDON_ID, "returned", "and EMEA", None)
    assert event.payload() == {"type": "response.addon", "addon_id": ADDON_ID,
                               "state": "returned", "text": "and EMEA"}
    assert TurnAddon(ADDON_ID, "withdrawn", "x", "jev", "k-1").payload()["local_key"] == "k-1"


async def test_a_relayed_addon_is_streamed_and_saved_as_response_addon():
    async def _events():
        yield TurnAddon(ADDON_ID, "read", "use the sales DB", "user", "k-1")

    recorded: list[tuple[str, dict]] = []
    chunks = [c async for c in format_responses_stream(
        _events(), model="m", event_sink=lambda t, d: recorded.append((t, d)),
    )]
    frames = [json.loads(line[len("data:"):]) for chunk in chunks
              for line in chunk.splitlines() if line.startswith("data:")]
    addon = next(f for f in frames if f.get("type") == "response.addon")
    assert {k: addon[k] for k in ("addon_id", "state", "text", "decided_by", "local_key")} == {
        "addon_id": ADDON_ID, "state": "read", "text": "use the sales DB", "decided_by": "user", "local_key": "k-1",
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
            elif line["kind"] == "addon":
                # What relay_addon_step yields for an addon this turn holds.
                yield TurnAddon(line["addon_id"], line["state"], line["text"], line["decided_by"])
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


# ── The web turn's relay: only addons this turn holds reach the app ────────


async def test_the_web_relay_streams_and_saves_only_this_turns_addons(monkeypatch):
    import fakeredis.aioredis

    from cowork.handlers import responses as responses_mod
    from cowork.turnqueue.redis_client import addon_index_key
    from tests.test_responses_remote_backend import _FakeBuffer, _remote_handler

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(remote_addons, "get_redis", lambda: fake)
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)
    seen: dict = {}

    async def fake_replies(**kwargs):
        corr = seen["corr"] = kwargs["correlation_id"]
        await fake.hset(addon_index_key(corr), ADDON_ID, json.dumps(
            {"text": "use the sales DB", "decided_by": "jev", "at": 1, "status": "added", "local_key": "k-1"}))
        await fake.hset(addon_index_key(corr), "b" * 32, json.dumps(
            {"text": "gave up on", "decided_by": "jev", "at": 2, "status": "abandoned"}))
        yield "turn_step", {"step": "addon", "type": "response.addon", "state": "ready"}
        yield "turn_step", _step("added", text="pod's own words", decided_by="user")
        yield "turn_step", {**_step("added"), "addon_id": "b" * 32}
        yield "turn_step", {**_step("added"), "addon_id": "f" * 32, "text": "injected"}
        yield "turn_step", _step("withdrawn", request_id="r" * 32)
        yield "turn_delta", {"text": "done"}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)
    await handler._produce_remote(
        conv_id=CID, input_text="hi", original_content="hi", model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    addon_events = [e for e in saved["events"] if e.get("type") == "response.addon"]
    assert [(e["addon_id"], e["state"], e["text"], e["decided_by"], e["local_key"]) for e in addon_events] == [
        (ADDON_ID, "added", "use the sales DB", "jev", "k-1"),
        (ADDON_ID, "withdrawn", "use the sales DB", "jev", "k-1"),
    ]
    # "ready" is recorded for /addon's support check, never streamed.
    assert await remote_addons.remote_addons_ready(seen["corr"], r=fake) is True


# ── One budget for the whole request (8 s; the app waits 10 s) ─────────────


async def test_the_lock_wait_never_passes_the_budget(monkeypatch):
    from cowork.db.scoped import LOCAL_SCOPE

    async with addon_route._deciding("corr-1", addon_route.budget_deadline()):
        started = asyncio.get_running_loop().time()
        result = await addon_route.route_remote_addon(
            conversation_id=CID, correlation_id="corr-1", text="x", force=True, scope=LOCAL_SCOPE,
            deadline=addon_route.time.monotonic() + 0.05,
        )
    assert result == {"status": "queued", "reason": "busy"}
    assert asyncio.get_running_loop().time() - started < 1


def test_the_budget_fits_inside_the_apps_wait():
    assert addon_route.ADDON_BUDGET_S == 8
    assert addon_route.DECIDE_LOCK_TIMEOUT_S < addon_route.ADDON_BUDGET_S


async def test_a_slow_jev_stage_is_jev_timeout_and_reaches_no_pod(monkeypatch, redis_side):
    """Reading Jev's inputs, minting and asking Jev share what is left."""
    from cowork.db.scoped import LOCAL_SCOPE

    submitted = []

    async def slow_inputs(**kwargs):
        await asyncio.sleep(5)

    async def submit(**kwargs):
        submitted.append(kwargs)
        return RemoteAddonResult.ADDED, ADDON_ID

    monkeypatch.setattr(addon_route, "_remote_jev_inputs", slow_inputs)
    monkeypatch.setattr(addon_route, "submit_remote_addon", submit)
    result = await addon_route.route_remote_addon(
        conversation_id=CID, correlation_id="corr-1", text="x", force=False, scope=LOCAL_SCOPE,
        deadline=addon_route.time.monotonic() + 0.1,
    )
    assert result == {"status": "queued", "reason": "jev_timeout"}
    assert submitted == []


async def test_jevs_own_timeout_is_cut_to_the_budget(monkeypatch):
    from cowork.db.scoped import LOCAL_SCOPE

    seen = {}

    async def decide(**kwargs):
        seen["timeout"] = kwargs["timeout_seconds"]
        return jev_route.AddonRoute(error="timeout")

    async def target(fn):
        return TARGET, "jev"

    monkeypatch.setattr("cowork.db.units.run_on_unit_threads", target)
    monkeypatch.setattr(jev_route, "decide_addon", decide)
    result = await addon_route._jev_refusal(
        conversation_id=CID, task="t", progress=[], added=[], text="x", scope=LOCAL_SCOPE,
        deadline=addon_route.time.monotonic() + 1.0,
    )
    assert result == {"status": "queued", "reason": "jev_timeout"}
    assert 0 < seen["timeout"] <= 1.0


async def test_the_pod_wait_ends_with_the_budget_and_withdraws(monkeypatch):
    import fakeredis.aioredis

    from cowork.turnqueue.addons import submit_remote_addon

    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    started = asyncio.get_running_loop().time()
    result, addon_id = await submit_remote_addon(
        conversation_id="conv-1", correlation_id="corr-1", text="x", decided_by="user", r=r,
        deadline=addon_route.time.monotonic() + 0.2,
    )
    assert result is RemoteAddonResult.TIMEOUT
    assert asyncio.get_running_loop().time() - started < 2
    kinds = [json.loads(raw)["kind"] for raw in await r.lrange("cowork:answer:corr-1", 0, -1)]
    assert kinds == ["addon", "addon_withdraw"]
    assert json.loads(await r.hget("cowork:addons:corr-1", addon_id))["status"] == "abandoned"


async def test_no_budget_left_before_the_push_pushes_nothing():
    import fakeredis.aioredis

    from cowork.turnqueue.addons import submit_remote_addon

    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    result, addon_id = await submit_remote_addon(
        conversation_id="conv-1", correlation_id="corr-1", text="x", decided_by="user", r=r,
        deadline=addon_route.time.monotonic() - 1,
    )
    assert result is RemoteAddonResult.TIMEOUT
    assert await r.llen("cowork:answer:corr-1") == 0
    assert json.loads(await r.hget("cowork:addons:corr-1", addon_id))["status"] == "abandoned"
