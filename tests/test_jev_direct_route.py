"""The Jev route gate and the direct answer it routes to.

Jev decides (jev_route), the user's own model answers (direct_answer), and the
streamed producer (ResponsesHandler._produce_answer) saves and frames the turn
or hands it to the agent. The decision's wording and threshold are measured by
scripts/eval_jev_route; these tests pin the wiring around them.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from cowork.common.settings.user_settings import Provider
from cowork.handlers import direct_answer, jev_route
from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    RouteDecision,
    RouterBinding,
    decide_route,
)
from cowork.handlers.responses import ResponsesHandler

_TOOL_ROWS = [
    {"role": "user", "content": "Build me a dashboard of our 2025 sales."},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "query_datasource", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "revenue=4812330"}]},
    {"role": "assistant", "content": "Your dashboard is ready. Total revenue was $4.81M."},
]
_RECALL = [*_TOOL_ROWS, {"role": "user", "content": "what were those headline numbers again?"}]


def _route_kwargs(history, **overrides):
    return {
        "history": history,
        "has_non_text_input": False,
        "has_attachments": False,
        "has_disabled_connections": False,
        **overrides,
    }


def _jev(result: jev_route.JevRoute, seen: list | None = None):
    async def ask(messages):
        if seen is not None:
            seen.append(messages)
        return result
    return ask


# ── decide_route with Jev ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jev_routes_a_recall_after_tool_work_direct():
    """The target case: the previous turn ran tools, but the numbers are in
    its visible reply. The structural rule that sent every such follow-up to
    the agent is the LLM gate's; with Jev, Jev decides from the markers."""
    seen = []
    decision = await decide_route(**_route_kwargs(
        _RECALL, jev=_jev(jev_route.JevRoute(direct=True, p_direct=0.99, model="jev-1.13.0"), seen),
    ))

    assert decision == RouteDecision(
        route=DIRECT_CONTEXT, reason="jev_direct", provider="jev", model="jev-1.13.0", p_direct=0.99,
    )
    # No answer text: the user's model writes it afterwards.
    assert decision.text == ""
    # Jev saw the tool markers, not the tool output.
    assert "[ran tool: query_datasource]" in seen[0][1]["content"]
    assert "revenue=4812330" not in json.dumps(seen[0])


@pytest.mark.asyncio
async def test_jev_needs_agent_delegates_without_a_fallback_flag():
    decision = await decide_route(**_route_kwargs(
        _RECALL, jev=_jev(jev_route.JevRoute(direct=False, p_direct=0.02, model="jev-1.13.0")),
    ))

    assert (decision.route, decision.reason, decision.fallback, decision.p_direct) == (
        DELEGATED_AGENTIC, "jev_needs_agent", False, 0.02,
    )


@pytest.mark.asyncio
async def test_a_jev_failure_delegates_as_a_fallback():
    decision = await decide_route(**_route_kwargs(_RECALL, jev=_jev(jev_route.JevRoute(error="timeout"))))

    assert (decision.route, decision.reason, decision.fallback) == (DELEGATED_AGENTIC, "jev_timeout", True)


@pytest.mark.asyncio
async def test_code_checks_still_delegate_before_jev_is_asked():
    seen = []
    ask = _jev(jev_route.JevRoute(direct=True, p_direct=1.0), seen)

    file_turn = await decide_route(**_route_kwargs([{"role": "user", "content": "what's in q3.xlsx?"}], jev=ask))
    attachment_turn = await decide_route(**_route_kwargs(_RECALL, jev=ask, has_attachments=True))

    assert file_turn.reason == "explicit_file_context"
    assert attachment_turn.reason == "attachments_present"
    assert seen == []


@pytest.mark.asyncio
async def test_the_llm_gate_keeps_the_prior_tools_rule():
    """Without Jev (providers other than MindsHub), a follow-up to tool work
    still delegates before any model is asked."""
    decision = await decide_route(**_route_kwargs(_RECALL, binding_factory=lambda: pytest.fail("gate asked")))

    assert decision.reason == "prior_turn_ran_tools"


# ── ResponsesHandler._jev_gate ─────────────────────────────────────────────


def _settings(*, planning=Provider.MINDS_CLOUD, enabled=True, model="opus-5-5", own=None):
    from pydantic import SecretStr

    own = own or {}
    return SimpleNamespace(
        resolved_planning_provider=planning,
        jev_decisions_enabled=enabled,
        minds_url="https://api.mindshub.ai",
        resolved_planning_model=model,
        decision_model_url=own.get("url", ""),
        decision_model_api_key=SecretStr(own["key"]) if own.get("key") else None,
        decision_model=own.get("model", "jev"),
    )


def _gate(monkeypatch, settings, *, stored_key=None):
    import cowork.handlers.responses as responses

    monkeypatch.setattr(responses, "get_user_settings", lambda *_a, **_k: settings)
    monkeypatch.setattr(responses, "provider_api_key_str", lambda *_a, **_k: stored_key or "")
    return ResponsesHandler()


async def _asked(monkeypatch, handler, turn_llm=None):
    """The decide() arguments the gate's callable passes, or None when there is no callable."""
    seen = {}

    async def fake_decide(**kwargs):
        seen.update(kwargs)
        return jev_route.JevRoute(direct=True, p_direct=0.9)

    monkeypatch.setattr(jev_route, "decide", fake_decide)
    ask = handler._jev_gate(turn_llm)
    if not callable(ask):
        return ask
    await ask([{"role": "user", "content": "hi"}])
    return seen


def test_jev_gate_leaves_other_providers_on_the_llm_gate(monkeypatch):
    handler = _gate(monkeypatch, _settings(planning=Provider.ANTHROPIC), stored_key="k")

    assert handler._jev_gate(None) is None


def test_jev_gate_is_off_when_the_organization_opted_out(monkeypatch):
    handler = _gate(monkeypatch, _settings(enabled=False), stored_key="k")

    assert handler._jev_gate({"llm": {"base_url": "https://x/v1", "api_key": "turn"}}) == "off"


def test_jev_gate_without_a_key_falls_back_to_the_llm_gate(monkeypatch):
    handler = _gate(monkeypatch, _settings(), stored_key=None)

    assert handler._jev_gate(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("turn_llm", "stored_key", "target"), [
    # A remote turn asks on the key the gate minted for it.
    ({"llm": {"base_url": "https://minds.example/v1", "api_key": "turn"}}, "stored",
     jev_route.JevTarget(url="https://minds.example/v1/decisions", api_key="turn")),
    # The desktop asks on the stored key, at the chat base URL.
    (None, "stored", jev_route.JevTarget(url="https://api.mindshub.ai/v1/decisions", api_key="stored")),
])
async def test_jev_gate_asks_on_the_turns_key(monkeypatch, turn_llm, stored_key, target):
    seen = await _asked(monkeypatch, _gate(monkeypatch, _settings(), stored_key=stored_key), turn_llm)

    assert seen["target"] == target
    assert (seen["model"], seen["timeout_seconds"], seen["min_p_direct"]) == ("jev", 3.0, 0.5)


@pytest.mark.asyncio
async def test_route_request_with_jev_off_runs_the_agent_without_a_gate(monkeypatch):
    import cowork.handlers.responses as responses

    handler = ResponsesHandler()
    handler.harness_name = "anton"
    handler._jev_gate = lambda _turn_llm: "off"

    async def no_gate(**_kwargs):
        pytest.fail("the gate must not run when Jev is off")

    monkeypatch.setattr(responses, "decide_route", no_gate)
    decision, _turn_llm = await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "hi"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert (decision.route, decision.reason) == (DELEGATED_AGENTIC, "jev_disabled")


@pytest.mark.asyncio
async def test_route_request_hands_the_jev_gate_to_decide_route(monkeypatch):
    import cowork.handlers.responses as responses

    handler = ResponsesHandler()
    handler.harness_name = "anton"
    ask = _jev(jev_route.JevRoute(direct=True, p_direct=0.9))
    handler._jev_gate = lambda _turn_llm: ask
    seen = {}

    async def fake_decide_route(**kwargs):
        seen.update(kwargs)
        return RouteDecision(route=DIRECT_CONTEXT, reason="jev_direct", provider="jev", p_direct=0.9)

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)
    decision, _turn_llm = await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "hi"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert seen["jev"] is ask
    assert decision.reason == "jev_direct"


# ── direct_answer ──────────────────────────────────────────────────────────


class _Provider:
    """A streaming provider scripted with anton stream events, or a delay."""

    def __init__(self, events=(), *, delay=0.0, raise_after=None):
        self.events = list(events)
        self.delay = delay
        self.raise_after = raise_after
        self.calls = []
        self.closed = False

    def native_web_tools(self):
        return {"web_search", "web_fetch"}

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._events()

    async def _events(self):
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for i, event in enumerate(self.events):
                if self.raise_after is not None and i == self.raise_after:
                    raise RuntimeError("stream broke")
                yield event
        finally:
            self.closed = True


def _text(text):
    from anton.core.llm.provider import StreamTextDelta

    return StreamTextDelta(text=text)


def _complete(tool_calls=()):
    from anton.core.llm.provider import StreamComplete

    return StreamComplete(response=SimpleNamespace(content="", tool_calls=list(tool_calls), stop_reason="end_turn"))


def _binding(provider, model="opus-5-5"):
    return RouterBinding(provider=provider, model=model, label="minds_cloud")


@pytest.mark.asyncio
async def test_start_answer_streams_once_text_starts():
    provider = _Provider([_text("Revenue was "), _text("$4.81M."), _complete()])

    answer = await direct_answer.start_answer(_binding(provider), history=_RECALL)

    assert answer.first_text == "Revenue was "
    assert [t async for t in answer.rest()] == ["$4.81M."]
    call = provider.calls[0]
    assert call["model"] == "opus-5-5"
    assert [t["name"] for t in call["tools"]] == ["delegate"]
    # Web search runs on the gateway, so the model can answer current facts.
    assert call["native_web_tools"] == {"web_search", "web_fetch"}
    assert "opus-5-5" in call["system"]


@pytest.mark.asyncio
async def test_start_answer_delegates_on_a_tool_call():
    from anton.core.llm.provider import StreamToolUseStart

    provider = _Provider([StreamToolUseStart(id="t", name="delegate"), _complete()])

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None
    assert provider.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [
    [_complete(tool_calls=[SimpleNamespace(name="delegate")])],
    [_text("   "), _complete()],
    [],
])
async def test_start_answer_delegates_when_no_text_comes(events):
    assert await direct_answer.start_answer(_binding(_Provider(events)), history=_RECALL) is None


@pytest.mark.asyncio
async def test_start_answer_delegates_on_a_failure_before_text():
    provider = _Provider([_text("never")], raise_after=0)

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None


@pytest.mark.asyncio
async def test_start_answer_delegates_when_the_first_event_is_late(monkeypatch):
    monkeypatch.setattr(direct_answer, "_FIRST_EVENT_SECONDS", 0.01)
    provider = _Provider([_text("late")], delay=0.5)

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None
    assert provider.closed


@pytest.mark.asyncio
async def test_the_answer_sees_more_history_than_the_gate():
    """A long reply the gate would cut at 1,500 chars reaches the answer whole:
    recalling it is what the direct path is for."""
    long_reply = "Headline numbers. " + "detail " * 400 + "Churn was 3.1%."
    history = [
        {"role": "user", "content": "analyse it"},
        {"role": "assistant", "content": long_reply},
        {"role": "user", "content": "what was churn again?"},
    ]
    provider = _Provider([_text("3.1%"), _complete()])

    await direct_answer.start_answer(_binding(provider), history=history)

    assert provider.calls[0]["messages"][1]["content"] == long_reply.strip()


@pytest.mark.asyncio
async def test_complete_answer_joins_the_text_or_delegates():
    assert await direct_answer.complete_answer(
        _binding(_Provider([_text("Hi "), _text("there."), _complete()])), history=_RECALL,
    ) == "Hi there."
    # A failure part way, with nothing saved yet, still delegates.
    assert await direct_answer.complete_answer(
        _binding(_Provider([_text("Hi "), _text("there.")], raise_after=1)), history=_RECALL,
    ) is None


def test_system_prompt_carries_the_product_facts_without_recall_skill():
    """The direct model has no skills, so the product facts the agent would
    recall come inline, and the line telling it to recall them does not."""
    prompt = direct_answer.system_prompt(model="opus-5-5")

    assert "COWORK — PRODUCT FACTS" in prompt
    assert "recall_skill" not in prompt
    assert "Serving model: opus-5-5" in prompt


# ── ResponsesHandler._answer_binding ───────────────────────────────────────


def test_answer_binding_uses_the_composer_pick_on_the_minted_key(monkeypatch):
    from anton.core.llm.openai import OpenAIProvider

    handler = _gate(monkeypatch, _settings(model="sonnet-5-5"))
    binding = handler._answer_binding(
        model="opus-5-5", turn_llm={"llm": {"base_url": "https://minds.example/v1", "api_key": "turn"}},
    )

    assert binding.model == "opus-5-5"
    assert isinstance(binding.provider, OpenAIProvider)
    assert binding.provider.native_web_tools() == {"web_search", "web_fetch"}


def test_answer_binding_falls_back_to_the_planning_model_on_stored_settings(monkeypatch):
    import cowork.handlers.responses as responses

    handler = _gate(monkeypatch, _settings(model="sonnet-5-5"))
    planning = object()
    seen = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(planning_provider=planning)

    monkeypatch.setattr(responses, "build_llm_client", fake_client)
    binding = handler._answer_binding(model=None, turn_llm=None)

    assert (binding.model, binding.provider, seen["model_override"]) == ("sonnet-5-5", planning, None)


# ── ResponsesHandler._produce_answer, against the test database ────────────


class _Buffer:
    def __init__(self):
        self.frames: list[str] = []
        self.status = None

    @property
    def is_closed(self):
        return self.status is not None

    async def append(self, _kind, data):
        self.frames.append(data["sse"])

    async def close(self, status):
        if self.status is None:
            self.status = status


def _events(buffer):
    return [frame.split("\n", 1)[0].removeprefix("event: ") for frame in buffer.frames]


def _payload(frame):
    return json.loads(frame.split("data: ", 1)[1].strip())


async def _conversation():
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.units import run_db
    from cowork.services.conversations import ConversationService
    from cowork.services.projects import GENERAL_PROJECT_ID

    def seed(session):
        return ConversationService(session).create_conversation("topic", project_id=GENERAL_PROJECT_ID).id

    return await run_db(seed, scope=LOCAL_SCOPE)


async def _rows(conversation_id):
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.units import run_db
    from cowork.services.conversations import ConversationService

    def read(session):
        messages = ConversationService(session).get_ordered_messages(conversation_id, include_pending=True)
        return [(m.role, m.content, m.pending, m.harness) for m in messages]

    return await run_db(read, scope=LOCAL_SCOPE)


async def _answer(chunks, *, raise_after=None, delay=0.0):
    async def events():
        for i, chunk in enumerate(chunks):
            if delay:
                await asyncio.sleep(delay)
            if raise_after is not None and i == raise_after:
                raise RuntimeError("stream broke")
            yield _text(chunk)
        yield _complete()

    stream = events()
    first = await anext(stream)
    return direct_answer.DirectAnswer(first_text=first.text, _events=stream)


async def _produce(monkeypatch, conversation_id, answer, *, delegate=None, lifecycle=None):
    from cowork.streaming.registry import TurnLifecycle

    async def fake_start(_binding, *, history):
        return answer

    monkeypatch.setattr(direct_answer, "start_answer", fake_start)
    handler = ResponsesHandler()
    handler.harness_name = "anton"
    buffer = _Buffer()
    coro = handler._produce_answer(
        lifecycle=lifecycle or TurnLifecycle(),
        conv_id=conversation_id,
        original_content="what were those headline numbers again?",
        question="what were those headline numbers again?",
        route=RouteDecision(route=DIRECT_CONTEXT, reason="jev_direct", provider="jev", p_direct=0.99),
        binding=_binding(_Provider()),
        history=_RECALL,
        trace_metadata={},
        buffer=buffer,
        delegate=delegate or (lambda: pytest.fail("must not delegate")),
    )
    return handler, buffer, coro


@pytest.mark.asyncio
async def test_produce_answer_streams_then_saves_the_turn(monkeypatch):
    conversation_id = await _conversation()
    answer = await _answer(["Revenue was ", "$4.81M, ", "up 14.2%."])
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer)

    await coro

    assert _events(buffer) == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.memory_read",
        "response.figure_sources",
        "response.completed",
    ]
    assert buffer.status == "completed"
    created = _payload(buffer.frames[0])
    assert (created["harness"], created["response"]["model"]) == ("cowork-direct", "opus-5-5")
    deltas = [_payload(f) for f in buffer.frames if f.startswith("event: response.output_text.delta")]
    assert "".join(d["delta"] for d in deltas) == "Revenue was $4.81M, up 14.2%."
    assert {d["response_route_reason"] for d in deltas} == {"jev_direct"}
    completed = _payload(buffer.frames[-1])
    assert completed["assistant_message_id"]
    # The question and the answer land together, the question no longer pending.
    assert await _rows(conversation_id) == [
        ("user", "what were those headline numbers again?", False, None),
        ("assistant", "Revenue was $4.81M, up 14.2%.", False, "cowork-direct"),
    ]
    assert created["user_message_id"]


@pytest.mark.asyncio
async def test_produce_answer_hands_a_delegated_turn_to_the_agent(monkeypatch):
    conversation_id = await _conversation()
    delegated = []

    async def delegate():
        delegated.append(True)

    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, None, delegate=delegate)

    await coro

    assert delegated == [True]
    # The agent saves the question itself; the direct path wrote nothing.
    assert buffer.frames == []
    assert await _rows(conversation_id) == []


@pytest.mark.asyncio
async def test_produce_answer_saves_the_partial_answer_on_stop(monkeypatch):
    conversation_id = await _conversation()
    answer = await _answer(["Revenue was ", "$4.81M", " and more"], delay=0.05)
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer)

    task = asyncio.create_task(coro)
    while not any(f.startswith("event: response.output_text.delta") for f in buffer.frames):
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert buffer.status == "cancelled"
    rows = await _rows(conversation_id)
    assert rows[0][2] is False  # the question rejoined history
    assert rows[1][0] == "assistant" and rows[1][1].startswith("Revenue was")


@pytest.mark.asyncio
async def test_produce_answer_reports_a_failure_part_way_and_keeps_the_text(monkeypatch):
    conversation_id = await _conversation()
    answer = await _answer(["Revenue was ", "$4.81M", " broken"], raise_after=2)
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer)

    await coro

    assert buffer.status == "error"
    assert _events(buffer)[-1] == "response.failed"
    # The turn's correlation id is the reference a user can quote.
    assert _payload(buffer.frames[-1])["request_id"]
    rows = await _rows(conversation_id)
    assert [r[2] for r in rows] == [False, False]
    assert rows[1][1] == "Revenue was $4.81M"


# ── the opt-out reaches anton's completion check ───────────────────────────


@pytest.mark.parametrize(("enabled", "verifier_jev"), [(False, "off"), (True, "on")])
def test_jev_opt_out_turns_off_antons_completion_check(enabled, verifier_jev):
    from anton.config.settings import AntonSettings

    from cowork.common.settings.user_settings import UserSettings
    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    anton = AntonSettings(_env_file=None)
    applied = _overlay_user_settings(anton, UserSettings.model_validate({"jev_decisions_enabled": enabled}))

    assert anton.verifier_jev == verifier_jev
    assert ("verifier_jev" in applied) is (not enabled)


def test_route_state_puts_the_latest_message_apart():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]

    assert jev_route.route_state(messages) == {"latest_user_message": "c", "earlier_conversation": messages[:2]}



@pytest.mark.asyncio
async def test_jev_gate_unwraps_the_real_stored_minds_key(monkeypatch):
    """Through the real UserSettings and key helper: the key is a SecretStr,
    and the header must carry its value, never its masked form."""
    import cowork.handlers.responses as responses
    from cowork.common.settings.user_settings import UserSettings

    settings = UserSettings.model_validate({"planning_provider": "minds_cloud", "minds_api_key": "stored-key-123"})
    monkeypatch.setattr(responses, "get_user_settings", lambda *_a, **_k: settings)

    seen = await _asked(monkeypatch, ResponsesHandler())

    assert seen["target"].api_key == "stored-key-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(("planning", "enabled"), [
    (Provider.ANTHROPIC, True),  # brought their own inference keys
    (Provider.MINDS_CLOUD, False),  # MindsHub under ZDR, with Jev there turned off
])
async def test_an_own_decision_model_routes_any_provider(monkeypatch, planning, enabled):
    own = {"url": " https://api.typesafe.ai/v1/systemone ", "key": "own-key", "model": "jev-latest"}
    handler = _gate(monkeypatch, _settings(planning=planning, enabled=enabled, own=own), stored_key="minds")

    seen = await _asked(monkeypatch, handler, {"llm": {"base_url": "https://minds/v1", "api_key": "turn"}})

    # Posted to as given: hosts serve the same body at different paths.
    assert seen["target"] == jev_route.JevTarget(url="https://api.typesafe.ai/v1/systemone", api_key="own-key")
    assert seen["model"] == "jev-latest"


@pytest.mark.asyncio
async def test_an_own_decision_model_needs_both_url_and_key(monkeypatch):
    handler = _gate(monkeypatch, _settings(planning=Provider.ANTHROPIC, own={"url": "https://d.example/v1"}))

    assert await _asked(monkeypatch, handler) is None


def test_answer_binding_runs_on_the_users_own_provider(monkeypatch):
    """A turn routed direct on someone's own keys answers on those keys, never
    on a MindsHub key the gate happened to mint."""
    import cowork.handlers.responses as responses

    handler = _gate(monkeypatch, _settings(planning=Provider.ANTHROPIC, model="claude-opus-5-5"))
    planning = object()
    monkeypatch.setattr(responses, "build_llm_client", lambda **_k: SimpleNamespace(planning_provider=planning))

    binding = handler._answer_binding(
        model=None, turn_llm={"llm": {"base_url": "https://minds.example/v1", "api_key": "turn"}},
    )

    assert (binding.provider, binding.model, binding.label) == (planning, "claude-opus-5-5", "anthropic")


def test_decision_model_key_is_an_admin_owned_secret():
    from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

    assert UserSettings.field_is_sensitive("decision_model_api_key")
    for key in ("decision_model_api_key", "decision_model_url", "decision_model", "jev_decisions_enabled"):
        assert setting_is_org_scoped(key), key


@pytest.mark.asyncio
async def test_produce_answer_ended_by_a_shutdown_reads_as_interrupted(monkeypatch):
    """A turn the server ended is not the user's Stop: it says so, carries its
    request id, and keeps what was written."""
    from cowork.streaming.registry import RunIdentity, TurnLifecycle

    conversation_id = await _conversation()
    answer = await _answer(["Revenue was ", "$4.81M", " and more"], delay=0.05)
    lifecycle = TurnLifecycle(identity=RunIdentity(correlation_id="corr-direct"))
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer, lifecycle=lifecycle)

    task = asyncio.create_task(coro)
    while not any(f.startswith("event: response.output_text.delta") for f in buffer.frames):
        await asyncio.sleep(0.01)
    lifecycle.shutting_down = True
    lifecycle.shutdown_frame_deadline = asyncio.get_running_loop().time() + 5
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert buffer.status == "interrupted"
    failed = _payload(buffer.frames[-1])
    assert _events(buffer)[-1] == "response.failed"
    assert failed["request_id"] == "corr-direct"
    rows = await _rows(conversation_id)
    assert rows[1][1].startswith("Revenue was")
    assert failed.get("assistant_message_id")
