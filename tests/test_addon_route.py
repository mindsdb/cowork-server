"""Messages sent while a turn runs: Jev adds them to the turn or they queue
(ENG-4194).

POST /api/v1/responses/addon and /addon/withdraw, the `addon_route` Jev
question, the per-turn channel (cowork.streaming.addons) and the harness's
`response.addon` events. Authorization is /answer's: tested there and in the
scope guard tests, so only the 404 shape is repeated here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.handlers import addon_route, jev_route
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.server import create_app
from cowork.streaming.addons import TurnAddon, addons
from cowork.streaming.registry import RunHandle, registry

CID = "conv-addon-test"
TARGET = jev_route.JevTarget.mindshub("https://minds.example/v1/", "turn-key")


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _Client:
    """Stands in for jev_route's shared httpx client."""

    def __init__(self, *, response=None, exception=None, delay=0.0, captured=None):
        self._response = response
        self._exception = exception
        self._delay = delay
        self.captured = captured if captured is not None else {}

    async def post(self, url, *, headers, json, timeout):
        self.captured.update(url=url, json=json, timeout=timeout)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exception is not None:
            raise self._exception
        return self._response


def _verdict(choice="add_to_current_task", p_add=0.9):
    return {
        "model": "jev-1.13.0",
        "answers": {"addon_route": {
            "type": "choice", "choice": choice,
            "probabilities": {"add_to_current_task": p_add, "queue_as_next_task": 1 - p_add},
        }},
    }


class _Session:
    """The anton contract (mindsdb/anton add_user_input): a pending list the
    loop drains at step boundaries."""

    def __init__(self, *, running=True):
        self.running = running
        self.inputs: list[dict] = []
        self.on_user_input_read = None
        self.history = [{"role": "user", "content": "seed"}]

    def add_user_input(self, addon_id, text):
        if not self.running:
            return False
        self.inputs.append({"id": addon_id, "text": text})
        return True

    def withdraw_user_input(self, addon_id):
        for i, entry in enumerate(self.inputs):
            if entry["id"] == addon_id:
                del self.inputs[i]
                return True
        return False

    def take_unread_user_inputs(self):
        unread, self.inputs = self.inputs, []
        return unread

    def step_boundary(self):
        """What anton does between rounds: show every unread input, then say so."""
        shown, self.inputs = self.inputs, []
        if shown and self.on_user_input_read is not None:
            self.on_user_input_read([e["id"] for e in shown])


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_ADDON_ROUTING_ENABLED", "true")
    yield
    addons.reset()
    registry.reset()
    get_app_settings.cache_clear()


async def _forever():
    await asyncio.sleep(3600)


@pytest.fixture()
async def run():
    """An in-flight turn for CID."""
    handle = RunHandle(
        conversation_id=CID, turn_id=1,
        buffer=SimpleNamespace(is_closed=False, is_finishing=False),
        task=asyncio.create_task(_forever()),
    )
    registry._by_cid[CID] = handle
    yield handle
    handle.task.cancel()


@pytest.fixture()
def session(run):
    """The running turn's anton session, with its channel open."""
    session = _Session()
    addons.open(CID, session, task="Build the 2025 sales dashboard.", progress=lambda: ["[ran tool: scratchpad]"])
    return session


@pytest.fixture()
def jev(monkeypatch):
    """Point the decision at a scripted client; returns a setter."""
    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: (TARGET, "jev"))

    def script(client):
        monkeypatch.setattr(jev_route, "_http_client", lambda: client)
        return client

    return script


@pytest.fixture()
def client():
    return TestClient(create_app())


def _post(client, text="actually use the sales DB", **body):
    return client.post("/api/v1/responses/addon", json={"conversation_id": CID, "text": text, **body})


def _withdraw(client, addon_id):
    return client.post("/api/v1/responses/addon/withdraw", json={"conversation_id": CID, "addon_id": addon_id})


# ── POST /responses/addon ──────────────────────────────────────────────────


async def test_unknown_conversation_is_404(client):
    resp = _post(client)
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


async def test_disabled_queues_without_asking_jev(client, session, jev, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_ADDON_ROUTING_ENABLED", "false")
    jev(_Client(exception=AssertionError("Jev must not be asked")))
    resp = _post(client)
    assert resp.status_code == 200
    assert resp.json() == {"status": "queued", "reason": "disabled"}
    assert session.inputs == []


async def test_blank_text_is_queued_as_empty(client, session):
    assert _post(client, text="   ").json() == {"status": "queued", "reason": "empty"}


async def test_a_finished_turn_queues_as_no_turn(client, run, session):
    run.buffer.is_closed = True
    assert _post(client).json() == {"status": "queued", "reason": "no_turn"}


async def test_an_old_anton_queues_as_not_supported(client, run):
    """An anton without add_user_input gets no channel."""
    assert addons.open(CID, SimpleNamespace(history=[]), task="t", progress=list) is None
    assert _post(client).json() == {"status": "queued", "reason": "not_supported"}


@pytest.mark.parametrize(("found", "status", "body"), [
    (SimpleNamespace(in_flight=True), 200, {"status": "queued", "reason": "not_supported"}),
    (SimpleNamespace(in_flight=False), 200, {"status": "queued", "reason": "no_turn"}),
    (None, 404, {"status": "not_found"}),
], ids=["running", "finished", "foreign"])
async def test_remote_backend_never_adds(client, monkeypatch, found, status, body):
    """A pod can't take a message mid-turn yet: it is always queued."""
    from cowork.api.v1.endpoints import responses as endpoint

    async def _found(conversation_id, scope):
        return found

    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    monkeypatch.setattr(endpoint, "_shared_turn", _found)
    resp = _post(client)
    assert (resp.status_code, resp.json()) == (status, body)


async def test_jev_adds_a_correction(client, session, jev):
    captured: dict = {}
    jev(_Client(response=_Response(200, _verdict(p_add=0.9)), captured=captured))
    body = _post(client).json()

    assert body["status"] == "added"
    assert body["decided_by"] == "jev"
    assert session.inputs == [{"id": body["addon_id"], "text": "actually use the sales DB"}]
    assert captured["url"] == "https://minds.example/v1/decisions"
    assert captured["json"] == {
        "model": "jev",
        "state": {
            "running_task": "Build the 2025 sales dashboard.",
            "recent_progress": ["[ran tool: scratchpad]"],
            "already_added_to_task": [],
            "new_user_message": "actually use the sales DB",
        },
        "questions": {"addon_route": jev_route.ADDON_ROUTE_QUESTION},
    }


async def test_jev_sees_the_messages_already_added(client, session, jev):
    captured: dict = {}
    jev(_Client(response=_Response(200, _verdict()), captured=captured))
    _post(client, text="use the sales DB")
    _post(client, text="and the same for EMEA")
    assert captured["json"]["state"]["already_added_to_task"] == ["use the sales DB"]


async def test_a_new_request_is_queued_as_next_task(client, session, jev):
    jev(_Client(response=_Response(200, _verdict(choice="queue_as_next_task", p_add=0.1))))
    assert _post(client, text="now email this to the team").json() == {"status": "queued", "reason": "next_task"}
    assert session.inputs == []


@pytest.mark.parametrize("p_add", [0.59, 0.3])
async def test_an_unsure_add_is_queued_as_low_confidence(client, session, jev, p_add):
    jev(_Client(response=_Response(200, _verdict(p_add=p_add))))
    assert _post(client).json() == {"status": "queued", "reason": "low_confidence"}
    assert session.inputs == []


async def test_the_add_threshold_is_a_setting(client, session, jev, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_ADDON_MIN_P_ADD", "0.95")
    jev(_Client(response=_Response(200, _verdict(p_add=0.9))))
    assert _post(client).json()["reason"] == "low_confidence"


async def test_a_jev_timeout_queues(client, session, jev, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_JEV_ROUTE_TIMEOUT_SECONDS", "0.05")
    jev(_Client(response=_Response(200, _verdict()), delay=1.0))
    assert _post(client).json() == {"status": "queued", "reason": "jev_timeout"}
    assert session.inputs == []


@pytest.mark.parametrize("scripted", [
    {"response": _Response(500, {})},
    {"response": _Response(200, {"answers": {"addon_route": {"choice": "maybe"}}})},
    {"exception": RuntimeError("boom")},
], ids=["http-500", "malformed", "raised"])
async def test_a_jev_error_queues(client, session, jev, scripted):
    jev(_Client(**scripted))
    assert _post(client).json() == {"status": "queued", "reason": "jev_error"}
    assert session.inputs == []


async def test_no_decision_model_queues_as_jev_off(client, session, monkeypatch):
    monkeypatch.setattr(addon_route, "_jev_target", lambda scope: None)
    assert _post(client).json() == {"status": "queued", "reason": "jev_off"}


async def test_force_adds_without_asking_jev(client, session, jev):
    jev(_Client(exception=AssertionError("Jev must not be asked")))
    body = _post(client, force=True).json()
    assert body["status"] == "added"
    assert body["decided_by"] == "user"
    assert [e["id"] for e in session.inputs] == [body["addon_id"]]


async def test_force_still_needs_the_flag(client, session, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_ADDON_ROUTING_ENABLED", "false")
    assert _post(client, force=True).json() == {"status": "queued", "reason": "disabled"}


async def test_a_turn_that_stops_taking_messages_queues_as_no_turn(client, session):
    """anton refuses once its turn is ending, after the server's own checks."""
    session.running = False
    assert _post(client, force=True).json() == {"status": "queued", "reason": "no_turn"}


# ── POST /responses/addon/withdraw ─────────────────────────────────────────


async def test_withdraw_before_the_model_reads_it(client, session):
    addon_id = _post(client, force=True).json()["addon_id"]
    resp = _withdraw(client, addon_id)
    assert resp.status_code == 200
    assert resp.json() == {"status": "withdrawn"}
    assert session.inputs == []


async def test_withdraw_after_the_model_read_it_is_409(client, session):
    addon_id = _post(client, force=True).json()["addon_id"]
    session.step_boundary()
    resp = _withdraw(client, addon_id)
    assert resp.status_code == 409
    assert resp.json() == {"status": "already_read"}


async def test_withdraw_racing_the_step_boundary_is_409(client, session):
    """anton took it but has not reported it read yet: the model has it."""
    addon_id = _post(client, force=True).json()["addon_id"]
    session.inputs.clear()
    assert _withdraw(client, addon_id).status_code == 409


@pytest.mark.parametrize("registered", [True, False])
async def test_withdraw_unknown_is_404(client, run, session, registered):
    if not registered:
        registry.reset()
    resp = _withdraw(client, "nope")
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


# ── the channel's events, the harness and the formatter ────────────────────


def test_channel_events_follow_each_message(run):
    session = _Session()
    channel = addons.open(CID, session, task="t", progress=list)
    first = channel.add("use the sales DB", decided_by="jev")
    second = channel.add("and EMEA", decided_by="user")
    assert [(e.addon_id, e.state) for e in channel.drain()] == [(first, "added"), (second, "added")]

    session.step_boundary()
    assert [(e.addon_id, e.state) for e in channel.drain()] == [(first, "read"), (second, "read")]
    assert channel.drain() == []


def test_unread_messages_are_returned_at_turn_end(run):
    session = _Session()
    channel = addons.open(CID, session, task="t", progress=list)
    read = channel.add("use the sales DB", decided_by="jev")
    session.step_boundary()
    unread = channel.add("and EMEA", decided_by="user")

    events = channel.take_returned()
    assert [(e.addon_id, e.state, e.decided_by) for e in events] == [
        (read, "added", "jev"), (read, "read", "jev"), (unread, "added", "user"), (unread, "returned", "user"),
    ]
    assert events[-1].text == "and EMEA"


def test_closing_a_channel_unhooks_the_session(run):
    session = _Session()
    channel = addons.open(CID, session, task="t", progress=list)
    assert session.on_user_input_read is not None
    addons.close(CID, channel)
    assert addons.get(CID) is None
    assert session.on_user_input_read is None


class _Step:
    """Any anton event; the harness passes it through."""


class _TurnSession(_Session):
    """A turn that takes two messages, reads the first at its step boundary,
    and ends with the second unread."""

    def __init__(self):
        super().__init__(running=False)
        self.ids: list[str] = []

    async def turn_stream(self, *_args, **_kwargs):
        self.running = True
        channel = addons.get("c1")
        self.ids = [channel.add("use the sales DB", decided_by="jev"), channel.add("and EMEA", decided_by="user")]
        yield _Step()
        first = self.inputs.pop(0)
        self.on_user_input_read([first["id"]])
        yield _Step()
        self.running = False


def test_the_harness_streams_added_read_and_returned(monkeypatch):
    from tests.test_anton_turn_capture import _drive

    session = _TurnSession()
    events = _drive(monkeypatch, session)
    addon_events = [(e.addon_id, e.state) for e in events if isinstance(e, TurnAddon)]
    first, second = session.ids
    assert addon_events == [(first, "added"), (second, "added"), (first, "read"), (second, "returned")]
    # The channel is gone with the turn: a late message queues.
    assert addons.get("c1") is None


def test_an_old_anton_turn_streams_no_addon_events(monkeypatch):
    from tests.test_anton_turn_capture import _drive, _FakeSession

    events = _drive(monkeypatch, _FakeSession(compact=False))
    assert not any(isinstance(e, TurnAddon) for e in events)


async def test_the_formatter_maps_an_addon_to_response_addon():
    async def _events():
        yield TurnAddon("a1", "added", "use the sales DB", "jev")

    recorded: list[tuple[str, dict]] = []
    chunks = [
        c async for c in format_responses_stream(
            _events(), model="m", event_sink=lambda t, d: recorded.append((t, d)),
        )
    ]
    frames = [
        json.loads(line[len("data:"):]) for chunk in chunks for line in chunk.splitlines() if line.startswith("data:")
    ]
    addon = next(f for f in frames if f.get("type") == "response.addon")
    assert {k: addon[k] for k in ("addon_id", "state", "text", "decided_by")} == {
        "addon_id": "a1", "state": "added", "text": "use the sales DB", "decided_by": "jev",
    }
    # Saved with the answer's events, so a reload replays it.
    assert any(t == "response.addon" for t, _ in recorded)


# ── the Jev question ───────────────────────────────────────────────────────


async def _decide_addon(monkeypatch, client, *, min_p_add=0.6):
    monkeypatch.setattr(jev_route, "_http_client", lambda: client)
    return await jev_route.decide_addon(
        target=TARGET,
        state=jev_route.addon_state(task="t", progress=[], added=[], message="m"),
        model="jev", timeout_seconds=1.0, min_p_add=min_p_add,
    )


@pytest.mark.parametrize(("choice", "p_add", "add", "chose_add"), [
    ("add_to_current_task", 0.6, True, True),
    ("add_to_current_task", 0.59, False, True),
    ("queue_as_next_task", 0.2, False, False),
    # A probability that disagrees with the choice never adds.
    ("queue_as_next_task", 0.7, False, False),
])
async def test_decide_addon_adds_only_a_confident_add(monkeypatch, choice, p_add, add, chose_add):
    result = await _decide_addon(monkeypatch, _Client(response=_Response(200, _verdict(choice, p_add))))
    assert (result.add, result.chose_add, result.p_add, result.error) == (add, chose_add, p_add, "")
    assert result.model == "jev-1.13.0"


@pytest.mark.parametrize(("client", "error"), [
    (_Client(response=_Response(429, {})), "http_429"),
    (_Client(response=_Response(200, {"answers": {}})), "malformed_response"),
    (_Client(response=_Response(200, _verdict(p_add=1.5))), "malformed_response"),
    (_Client(exception=RuntimeError("boom")), "RuntimeError"),
], ids=["http", "missing", "out-of-range", "raised"])
async def test_decide_addon_never_raises(monkeypatch, client, error):
    result = await _decide_addon(monkeypatch, client)
    assert result.add is False
    assert result.error == error


def test_addon_question_offers_both_choices():
    assert set(jev_route.ADDON_ROUTE_QUESTION["criteria"]) == {"add_to_current_task", "queue_as_next_task"}


@pytest.mark.parametrize(("mode", "key", "expected"), [
    ("own", None, ("https://own.example/decide", "own-model")),
    ("jev", "mk", ("https://minds.example/v1/decisions", "jev")),
    ("jev", None, None),
    ("off", "mk", None),
    ("llm", "mk", None),
])
def test_jev_target_follows_the_gate_mode(monkeypatch, mode, key, expected):
    from cowork.common.settings import user_settings
    from cowork.services import providers

    settings = SimpleNamespace(decision_model="own-model", minds_url="https://minds.example")
    monkeypatch.setattr(user_settings, "get_user_settings", lambda scope: settings)
    monkeypatch.setattr(user_settings, "provider_api_key_str", lambda s, p: key)
    monkeypatch.setattr(providers, "minds_chat_base_url", lambda url: "https://minds.example/v1")
    monkeypatch.setattr(jev_route, "gate_mode", lambda s: mode)
    monkeypatch.setattr(
        jev_route, "own_decision_target", lambda s: jev_route.JevTarget(url="https://own.example/decide", api_key="k"),
    )
    found = addon_route._jev_target(None)
    if expected is None:
        assert found is None
    else:
        assert (found[0].url, found[1]) == expected
