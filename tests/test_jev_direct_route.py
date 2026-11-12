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
from cowork.streaming.registry import TurnLifecycle

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
async def test_a_jev_timeout_delegates_without_asking_the_llm_gate():
    """The gate's time is spent: asking the LLM gate would only start the agent later."""
    decision = await decide_route(**_route_kwargs(
        _GREETING, jev=_jev(jev_route.JevRoute(error="timeout")),
        binding_factory=lambda: pytest.fail("LLM gate asked"),
    ))

    assert (decision.route, decision.reason, decision.fallback) == (DELEGATED_AGENTIC, "jev_timeout", True)


_GREETING = [{"role": "user", "content": "hi there"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["http_403", "http_402", "http_500", "transport_error", "malformed_response"])
async def test_a_refused_or_failed_jev_falls_back_to_the_llm_gate(monkeypatch, error):
    """Jev bills like any MindsHub model: a spent allowance or a restriction on
    the organization is a refusal, and the LLM gate decides instead, so the
    turn keeps its direct answer rather than running the agent."""
    from cowork.handlers import response_routing

    asked = []

    async def fake_gate(binding, *, history, **_kwargs):
        asked.append((binding.model, history))
        return "Hello! How can I help?"

    monkeypatch.setattr(response_routing, "_gate", fake_gate)
    binding = RouterBinding(provider=object(), model="mindshub_air", label="minds-cloud")

    decision = await decide_route(**_route_kwargs(
        _GREETING, jev=_jev(jev_route.JevRoute(error=error)), binding=binding,
    ))

    assert (decision.route, decision.reason, decision.text) == (
        DIRECT_CONTEXT, "router_direct_response", "Hello! How can I help?",
    )
    assert decision.model == "mindshub_air"
    assert [model for model, _history in asked] == ["mindshub_air"]


@pytest.mark.asyncio
async def test_the_llm_gate_fallback_keeps_the_prior_tools_rule():
    """The LLM gate was measured with the structural rule, so its fallback
    keeps it: a follow-up to tool work runs the agent."""
    decision = await decide_route(**_route_kwargs(
        _RECALL, jev=_jev(jev_route.JevRoute(error="http_403")),
        binding_factory=lambda: pytest.fail("LLM gate asked"),
    ))

    assert decision.reason == "prior_turn_ran_tools"


@pytest.mark.asyncio
async def test_jev_passes_on_whether_the_answer_needs_a_web_search():
    decision = await decide_route(**_route_kwargs(
        _GREETING, jev=_jev(jev_route.JevRoute(direct=True, p_direct=0.9, needs_search=False)),
    ))

    assert decision.needs_search is False


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
     jev_route.JevTarget(url="https://minds.example/v1/decisions", api_key="turn", mindshub_gateway=True)),
    # The desktop asks on the stored key, at the chat base URL.
    (None, "stored", jev_route.JevTarget(url="https://api.mindshub.ai/v1/decisions", api_key="stored", mindshub_gateway=True)),
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


def _complete(tool_calls=(), stop_reason="end_turn"):
    from anton.core.llm.provider import StreamComplete

    return StreamComplete(response=SimpleNamespace(content="", tool_calls=list(tool_calls), stop_reason=stop_reason))


def _tool_start():
    from anton.core.llm.provider import StreamToolUseStart

    return StreamToolUseStart(id="t1", name="delegate")


def _binding(provider, model="opus-5-5"):
    return RouterBinding(provider=provider, model=model, label="minds_cloud")


@pytest.mark.asyncio
async def test_start_answer_streams_once_text_starts():
    provider = _Provider([_text("Revenue was "), _text("$4.81M."), _complete()])

    answer = await direct_answer.start_answer(_binding(provider), history=_RECALL)

    # Shorter than the held opening: the whole answer, checked, at once.
    assert answer.first_text == "Revenue was $4.81M."
    assert answer.complete is True
    assert [t async for t in answer.rest()] == []
    call = provider.calls[0]
    assert call["model"] == "opus-5-5"
    assert [t["name"] for t in call["tools"]] == ["delegate"]
    # Web search runs on the gateway, so the model can answer current facts.
    assert call["native_web_tools"] == {"web_search", "web_fetch"}
    assert "opus-5-5" in call["system"]
    # No effort picked: the provider's default.
    assert call["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_start_answer_runs_at_the_composers_reasoning_effort():
    """The effort the user picked reaches their model, which also keeps
    MindsHub from classifying an Air pick a second time."""
    provider = _Provider([_text("Hi."), _complete()])
    binding = RouterBinding(provider=provider, model="mindshub_air", label="minds_cloud", reasoning_effort="low")

    await direct_answer.start_answer(binding, history=_RECALL)

    assert provider.calls[0]["reasoning_effort"] == "low"


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
async def test_start_answer_waits_less_when_jev_expects_no_web_search(monkeypatch):
    """A wrong direct route reaches the agent sooner when no search runs
    before the first text; web search stays available either way."""
    monkeypatch.setattr(direct_answer, "_FIRST_EVENT_SECONDS", 5.0)
    monkeypatch.setattr(direct_answer, "_FIRST_EVENT_NO_SEARCH_SECONDS", 0.01)
    late = _Provider([_text("late")], delay=0.5)

    assert await direct_answer.start_answer(_binding(late), history=_RECALL, needs_search=False) is None
    assert late.calls[0]["native_web_tools"] == {"web_search", "web_fetch"}

    searching = _Provider([_text("Sunny, 21C."), _complete()], delay=0.05)
    answer = await direct_answer.start_answer(_binding(searching), history=_RECALL, needs_search=True)
    assert answer.first_text == "Sunny, 21C."


def test_first_event_budget_is_shorter_without_a_search():
    assert direct_answer.first_event_seconds(needs_search=False) < direct_answer.first_event_seconds(needs_search=True)


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
        model="opus-5-5", reasoning_effort="high",
        turn_llm={"llm": {"base_url": "https://minds.example/v1", "api_key": "turn"}},
    )

    assert binding.model == "opus-5-5"
    assert binding.reasoning_effort == "high"
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


async def _produce(
    monkeypatch, conversation_id, answer, *, delegate=None, lifecycle=None, started=None, first_take=None,
    gate_key=None,
):
    from cowork.streaming.registry import TurnLifecycle

    lifecycle = lifecycle or TurnLifecycle()

    async def fake_start(_binding, *, history, needs_search=True, ui_locale=None):
        if started is not None:
            # An answer that has not started yet: wait to be stopped.
            started.set()
            await asyncio.Event().wait()
        return answer

    monkeypatch.setattr(direct_answer, "start_answer", fake_start)
    handler = ResponsesHandler()
    handler.harness_name = "anton"
    buffer = _Buffer()
    coro = handler._produce_answer(
        lifecycle=lifecycle,
        conv_id=conversation_id,
        original_content="what were those headline numbers again?",
        question="what were those headline numbers again?",
        route=RouteDecision(route=DIRECT_CONTEXT, reason="jev_direct", provider="jev", p_direct=0.99),
        binding=_binding(_Provider()),
        history=_RECALL,
        trace_metadata={},
        buffer=buffer,
        delegate=delegate or (lambda: pytest.fail("must not delegate")),
        first_take=first_take,
        gate_key=gate_key,
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
@pytest.mark.parametrize("delegates", [False, True, "fails"])
async def test_produce_answer_revokes_the_gate_key_unless_the_agent_takes_the_turn(monkeypatch, delegates):
    """An answered turn runs no agent turn to revoke the gate's key; a
    delegated one runs on that key, and the agent's producer revokes it. A
    hand-off that fails before the agent's turn ends runs nothing on it."""
    import cowork.handlers.responses as responses

    conversation_id = await _conversation()
    revoked = []

    async def revoke(correlation_id):
        revoked.append(correlation_id)

    monkeypatch.setattr(responses, "_revoke_turn_key_quietly", revoke)
    buffers = []

    async def delegate():
        if delegates == "fails":
            raise RuntimeError("harness no longer resolves")
        await buffers[0].close("completed")

    answer = None if delegates else await _answer(["Revenue was ", "$4.81M."])
    _handler, buffer, coro = await _produce(
        monkeypatch, conversation_id, answer, delegate=delegate if delegates else None,
        gate_key={"correlation_id": "corr-gate"},
    )
    buffers.append(buffer)

    await coro

    assert revoked == ([] if delegates is True else ["corr-gate"])


@pytest.mark.asyncio
async def test_produce_answer_hands_a_delegated_turn_to_the_agent(monkeypatch):
    conversation_id = await _conversation()
    delegated = []
    buffers = []

    async def delegate():
        # The agent's producer ends the turn and closes the buffer.
        delegated.append(True)
        await buffers[0].close("completed")

    lifecycle = TurnLifecycle()
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, None, delegate=delegate, lifecycle=lifecycle)
    buffers.append(buffer)

    await coro

    assert delegated == [True]
    # Nothing shown by the direct answer; the question was saved pending
    # before the answer started, and the agent's producer is handed that row.
    assert buffer.frames == []
    rows = await _rows(conversation_id)
    assert rows == [("user", "what were those headline numbers again?", True, None)]
    assert lifecycle.saved_question is True
    assert lifecycle.question_id is not None


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
    (Provider.MINDS_CLOUD, False),  # MindsHub, with Jev there turned off
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


@pytest.mark.asyncio
async def test_start_answer_closes_the_model_call_on_stop():
    provider = _Provider([_text("late")], delay=10)

    task = asyncio.create_task(direct_answer.start_answer(_binding(provider), history=_RECALL))
    while not provider.calls:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
    assert provider.closed


async def _stopped_before_text(monkeypatch, lifecycle=None):
    """Stop a direct turn while its answer has not started."""
    conversation_id = await _conversation()
    started = asyncio.Event()
    _handler, buffer, coro = await _produce(
        monkeypatch, conversation_id, None, lifecycle=lifecycle, started=started,
    )
    task = asyncio.create_task(coro)
    await started.wait()
    return conversation_id, buffer, task


@pytest.mark.asyncio
async def test_produce_answer_keeps_the_question_on_a_stop_before_text(monkeypatch):
    conversation_id, buffer, task = await _stopped_before_text(monkeypatch)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # The stream ends, and the question stays in the conversation, with no
    # answer, as an agent turn stopped before its first output does.
    assert buffer.status == "cancelled"
    assert await _rows(conversation_id) == [("user", "what were those headline numbers again?", False, None)]


@pytest.mark.asyncio
async def test_produce_answer_ended_by_a_shutdown_before_text_reads_as_interrupted(monkeypatch):
    from cowork.streaming.registry import RunIdentity, TurnLifecycle

    lifecycle = TurnLifecycle(identity=RunIdentity(correlation_id="corr-direct"))
    conversation_id, buffer, task = await _stopped_before_text(monkeypatch, lifecycle)

    lifecycle.shutting_down = True
    lifecycle.shutdown_frame_deadline = asyncio.get_running_loop().time() + 5
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert buffer.status == "interrupted"
    assert _events(buffer) == ["response.failed"]
    assert _payload(buffer.frames[-1])["request_id"] == "corr-direct"
    assert await _rows(conversation_id) == [("user", "what were those headline numbers again?", False, None)]


@pytest.mark.asyncio
async def test_produce_answer_leaves_a_stop_after_delegation_to_the_agent(monkeypatch):
    """Once delegated, the agent's producer owns the turn: a Stop there must
    not save the question a second time."""
    conversation_id = await _conversation()
    running = asyncio.Event()
    buffers = []

    async def delegate():
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await buffers[0].close("cancelled")

    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, None, delegate=delegate)
    buffers.append(buffer)
    task = asyncio.create_task(coro)
    await running.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert buffer.status == "cancelled"
    assert buffer.frames == []
    # Saved once, before delegation; the agent's producer finalizes it.
    assert len(await _rows(conversation_id)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("history", "after_router_answer", "reason"), [
    ([{"role": "user", "content": "Create two records and confirm on a card first"}], False,
     "write_or_approval_request"),
    ([{"role": "user", "content": "yes"}], False, "write_or_approval_request"),
    (_GREETING, True, "follow_up_to_router_answer"),
])
async def test_write_approval_and_direct_follow_ups_reach_the_agent_before_jev(history, after_router_answer, reason):
    """ENG-3896's checks hold with Jev too: only the agent shows the real
    approval card, so Jev is never asked about these turns."""
    seen = []
    decision = await decide_route(**_route_kwargs(
        history, after_router_answer=after_router_answer,
        jev=_jev(jev_route.JevRoute(direct=True, p_direct=1.0), seen),
    ))

    assert (decision.route, decision.reason) == (DELEGATED_AGENTIC, reason)
    assert seen == []


@pytest.mark.asyncio
async def test_the_question_is_saved_before_the_answer_starts(monkeypatch):
    """A crash while the answer waits for its first text cannot lose the
    question: it is saved pending first, as on the agent's producers."""
    conversation_id = await _conversation()
    started = asyncio.Event()
    _handler, _buffer, coro = await _produce(monkeypatch, conversation_id, None, started=started)
    task = asyncio.create_task(coro)
    await started.wait()

    assert await _rows(conversation_id) == [("user", "what were those headline numbers again?", True, None)]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_delegated_turns_producers_reuse_the_saved_question():
    """Both agent producers' first units take the row the direct producer
    saved, so a delegated turn has its question once."""
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.units import run_db
    from cowork.handlers.responses import _save_remote_question, _start_turn
    from cowork.streaming.registry import TurnLifecycle
    from datetime import datetime, timezone
    from functools import partial

    conversation_id = await _conversation()
    lifecycle = TurnLifecycle()
    first = await run_db(partial(
        _start_turn, lifecycle=lifecycle, conversation_id=conversation_id,
        content="what were those headline numbers again?", sent_at=datetime.now(timezone.utc),
    ), scope=LOCAL_SCOPE)
    again = await run_db(partial(
        _start_turn, lifecycle=lifecycle, conversation_id=conversation_id,
        content="what were those headline numbers again?", sent_at=datetime.now(timezone.utc),
    ), scope=LOCAL_SCOPE)
    remote = await run_db(partial(
        _save_remote_question, lifecycle=lifecycle, conversation_id=conversation_id,
        content="what were those headline numbers again?",
    ), scope=LOCAL_SCOPE)

    assert first.question_id == again.question_id == remote == lifecycle.question_id
    assert len(await _rows(conversation_id)) == 1


# ── the held opening ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_long_answer_is_held_until_its_opening_then_streams(monkeypatch):
    monkeypatch.setattr(direct_answer, "_HOLD_CHARS", 20)
    provider = _Provider([_text("Revenue was $4.81M "), _text("and churn 2.1%. "), _text("Margin 31%."), _complete()])

    answer = await direct_answer.start_answer(_binding(provider), history=_RECALL)

    assert answer.first_text == "Revenue was $4.81M and churn 2.1%. "
    assert answer.complete is False
    assert [t async for t in answer.rest()] == ["Margin 31%."]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [[_tool_start()], [_complete(tool_calls=[{"name": "delegate"}])]])
async def test_a_preamble_then_delegate_never_reaches_the_user(ending):
    """The model writes a preamble, then hands the turn to the agent: inside
    the held opening, so nothing was shown and the agent runs."""
    provider = _Provider([_text("Let me pull that from your data."), *ending])

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None
    assert provider.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "Done! I've created both records for you.",
    "Shall I proceed? Please answer yes or no.",
    "You can install the ChatGPT desktop app from openai.com.",
])
async def test_an_opening_the_gate_would_reject_delegates(text):
    """The LLM gate's own checks (`answer_rejection`) hold on the direct answer."""
    provider = _Provider([_text(text), _complete()])

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None


@pytest.mark.asyncio
async def test_a_cut_off_answer_delegates():
    provider = _Provider([_text("Revenue was"), _complete(stop_reason="max_tokens")])

    assert await direct_answer.start_answer(_binding(provider), history=_RECALL) is None


@pytest.mark.asyncio
async def test_a_tool_call_after_the_opening_ends_the_answer_where_it_is(monkeypatch, caplog):
    monkeypatch.setattr(direct_answer, "_HOLD_CHARS", 10)
    provider = _Provider([_text("Revenue was $4.81M. "), _text("Also "), _tool_start()])

    answer = await direct_answer.start_answer(_binding(provider), history=_RECALL)
    with caplog.at_level("WARNING"):
        rest = [t async for t in answer.rest()]

    assert rest == ["Also "]
    assert any(getattr(r, "event", None) == "direct_answer_ended_after_release" for r in caplog.records)


@pytest.mark.asyncio
async def test_thinking_before_the_reply_earns_the_longer_wait(monkeypatch):
    """A picked effort thinks before it writes: that is work, not a stall."""
    from anton.core.llm.provider import StreamReasoningDelta

    monkeypatch.setattr(direct_answer, "_FIRST_EVENT_NO_SEARCH_SECONDS", 0.05)
    monkeypatch.setattr(direct_answer, "_FIRST_EVENT_SECONDS", 5.0)

    class _Thinking(_Provider):
        async def _events(self):
            try:
                yield StreamReasoningDelta(text="thinking")
                await asyncio.sleep(0.1)
                yield _text("Hi.")
                yield _complete()
            finally:
                self.closed = True

    answer = await direct_answer.start_answer(_binding(_Thinking()), history=_RECALL, needs_search=False)

    assert answer.first_text == "Hi."


@pytest.mark.asyncio
async def test_the_answer_gets_room_to_think_at_a_picked_effort():
    provider = _Provider([_text("Hi."), _complete()])

    await direct_answer.start_answer(_binding(provider), history=_RECALL)

    assert provider.calls[0]["max_tokens"] >= 16_384


@pytest.mark.asyncio
async def test_switched_off_web_tools_never_reach_the_direct_answer():
    provider = _Provider([_text("Hi."), _complete()])
    binding = RouterBinding(provider=provider, model="opus-5-5", label="minds_cloud", web_tools=frozenset())

    await direct_answer.start_answer(binding, history=_RECALL)

    call = provider.calls[0]
    assert call["native_web_tools"] is None
    assert "cannot search the web" in call["system"]


@pytest.mark.parametrize(("env", "allowed"), [
    ({}, {"web_search", "web_fetch"}),
    ({"ANTON_WEB_SEARCH_ENABLED": "false"}, {"web_fetch"}),
    ({"ANTON_WEB_FETCH_ENABLED": "0"}, {"web_search"}),
    ({"ANTON_INTERNET_RESEARCH_ENABLED": "off"}, set()),
])
def test_the_direct_answer_honors_the_servers_web_tool_switches(monkeypatch, env, allowed):
    from cowork.handlers.responses import _allowed_web_tools

    for name in ("ANTON_WEB_SEARCH_ENABLED", "ANTON_WEB_FETCH_ENABLED", "ANTON_INTERNET_RESEARCH_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert _allowed_web_tools(Provider.MINDS_CLOUD) == allowed


@pytest.mark.asyncio
async def test_jev_reads_long_replies_whole_within_its_budget():
    """The target case: headline numbers stated past the gate's 1,500 chars
    reach Jev, so it can see they are in visible text."""
    from cowork.handlers import response_routing

    long_reply = "Your dashboard is ready. " + "Detail. " * 300 + "Total revenue was $4.81M."
    history = [
        {"role": "user", "content": "Build me a dashboard."},
        {"role": "assistant", "content": long_reply},
        {"role": "user", "content": "what were those headline numbers again?"},
    ]
    seen = []
    await decide_route(**_route_kwargs(history, jev=_jev(jev_route.JevRoute(direct=True, p_direct=0.99), seen)))

    assert "$4.81M" in seen[0][1]["content"]

    # A conversation past the budget drops its oldest messages, never the latest.
    many = [{"role": r, "content": "x" * 9_000} for r in ("user", "assistant") * 10]
    many.append({"role": "user", "content": "and now?"})
    seen.clear()
    await decide_route(**_route_kwargs(many, jev=_jev(jev_route.JevRoute(direct=True, p_direct=0.99), seen)))

    assert sum(len(m["content"]) for m in seen[0]) <= response_routing.JEV_HISTORY_CHARS
    assert seen[0][-1]["content"] == "and now?"
    assert seen[0][0]["role"] == "user"


class _Take:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


@pytest.mark.asyncio
async def test_a_failed_hand_off_still_ends_the_turn(monkeypatch):
    """Delegation that raises before the agent's producer runs (a harness
    that no longer resolves) must not leave the client's stream open, or
    the question pending."""
    conversation_id = await _conversation()
    take = _Take()

    async def delegate():
        raise RuntimeError("no such harness")

    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, None, delegate=delegate, first_take=take)

    await coro

    assert _events(buffer) == ["response.failed"]
    assert buffer.status == "error"
    assert await _rows(conversation_id) == [("user", "what were those headline numbers again?", False, None)]
    assert take.cancelled


@pytest.mark.asyncio
async def test_the_first_take_goes_to_the_agent_on_delegation(monkeypatch):
    conversation_id = await _conversation()
    take = _Take()
    buffers = []

    async def delegate():
        await buffers[0].close("completed")

    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, None, delegate=delegate, first_take=take)
    buffers.append(buffer)
    await coro

    # Left running: the agent's producer shows it, then ends it.
    assert take.cancelled is False


@pytest.mark.asyncio
async def test_the_first_take_ends_once_the_answer_is_shown(monkeypatch):
    conversation_id = await _conversation()
    take = _Take()
    answer = await _answer(["Revenue was ", "$4.81M."])
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer, first_take=take)

    await coro

    assert take.cancelled
    assert buffer.status == "completed"


@pytest.mark.asyncio
async def test_a_streamed_answers_sequence_numbers_have_no_gap(monkeypatch):
    conversation_id = await _conversation()
    answer = await _answer(["Revenue was ", "$4.81M."])
    _handler, buffer, coro = await _produce(monkeypatch, conversation_id, answer)

    await coro

    numbers = [_payload(frame)["sequence_number"] for frame in buffer.frames]
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "reply"), [
    ("which model are you?", "I'm Cowork, running on Claude Opus 5.5 by Anthropic."),
    ("what's the difference between ChatGPT and Gemini?", "ChatGPT is OpenAI's assistant; Gemini is Google's."),
])
async def test_the_direct_answer_may_name_its_own_model_and_products_the_user_named(question, reply):
    provider = _Provider([_text(reply), _complete()])

    answer = await direct_answer.start_answer(
        _binding(provider), history=[{"role": "user", "content": question}],
    )

    assert answer is not None and answer.first_text == reply


@pytest.mark.parametrize(("model", "allowed"), [
    ("opus-5-5", {"claude", "anthropic"}),
    ("claude-sonnet-5-5", {"claude", "anthropic"}),
    ("gpt-5.1", {"openai", "chatgpt"}),
    ("openai/o3", {"openai", "chatgpt"}),
    # MindsHub's own aliases name no vendor.
    ("gpt-terra", set()),
    ("muse-spark", set()),
    ("gemini-3-pro", {"gemini", "bard"}),
])
def test_allowed_products_follow_the_serving_models_vendor(model, allowed):
    assert direct_answer.allowed_products(model=model, history=[]) == allowed


def test_allowed_products_read_only_the_users_messages():
    history = [
        {"role": "assistant", "content": "Unlike ChatGPT, I can run code."},
        {"role": "user", "content": [{"type": "text", "text": "and Perplexity?"}]},
    ]

    assert direct_answer.allowed_products(model="muse-spark", history=history) == {"perplexity"}


@pytest.mark.asyncio
async def test_the_held_opening_ends_at_a_word_boundary(monkeypatch):
    """A name split across the hold point is checked whole, not as "Cla"."""
    monkeypatch.setattr(direct_answer, "_HOLD_CHARS", 20)
    provider = _Provider([_text("I can help. I am Cla"), _text("ude, by Anthropic."), _complete()])

    answer = await direct_answer.start_answer(_binding(provider, model="muse-spark"), history=_RECALL)

    assert answer is None


@pytest.mark.asyncio
async def test_the_shown_opening_never_ends_inside_a_word(monkeypatch):
    monkeypatch.setattr(direct_answer, "_HOLD_CHARS", 10)
    provider = _Provider([_text("Revenue was $4.81M, and marg"), _text("in 31%."), _complete()])

    answer = await direct_answer.start_answer(_binding(provider), history=_RECALL)

    assert answer.first_text == "Revenue was $4.81M, and "
    assert "".join([t async for t in answer.rest()]) == "margin 31%."


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [
    [_text("More."), _tool_start()],
    [_text("More."), _complete(stop_reason="max_tokens")],
    [_text(" Install the ChatGPT app."), _complete()],
])
async def test_complete_answer_delegates_what_a_streamed_answer_could_only_log(monkeypatch, ending):
    """Nothing is shown before the whole answer is in hand, so a late tool
    call, a cut-off or a rejected later passage still reaches the agent."""
    monkeypatch.setattr(direct_answer, "_HOLD_CHARS", 10)
    provider = _Provider([_text("Revenue was $4.81M. "), *ending])

    assert await direct_answer.complete_answer(_binding(provider), history=_RECALL) is None
