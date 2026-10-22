"""The instant first take: a short reply on the gate's model that fills the
wait on a delegated turn and hands over to Anton as soon as Anton speaks."""

from __future__ import annotations

import asyncio
import json

import pytest

from cowork.handlers import first_take as ft
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision, RouterBinding
from cowork.handlers.responses import ResponsesHandler


class _Buffer:
    def __init__(self) -> None:
        self.frames: list[str] = []
        self.is_closed = False

    async def append(self, type_: str, data: dict) -> int:
        assert type_ == "sse"
        self.frames.append(data["sse"])
        return len(self.frames)

    def events(self) -> list[tuple[str, dict]]:
        out = []
        for frame in self.frames:
            head, data = frame.split("\ndata: ", 1)
            out.append((head.removeprefix("event: "), json.loads(data)))
        return out


class _Provider:
    """Streams the given pieces, waiting on `gate` (if any) before each one."""

    def __init__(self, pieces, *, gate: asyncio.Event | None = None, fail: Exception | None = None) -> None:
        self.pieces = pieces
        self.gate = gate
        self.fail = fail
        self.calls: list[dict] = []
        self.closed = False

    async def stream(self, **kwargs):
        from anton.core.llm.provider import StreamTextDelta

        self.calls.append(kwargs)
        try:
            if self.fail:
                raise self.fail
            for piece in self.pieces:
                if self.gate is not None:
                    await self.gate.wait()
                    self.gate.clear()
                yield StreamTextDelta(text=piece)
        finally:
            self.closed = True


def _take(provider, history=None) -> ft.FirstTake:
    history = history or [{"role": "user", "content": "How did revenue do last quarter?"}]
    return ft.FirstTake(RouterBinding(provider=provider, model="fast", label="minds_cloud"), history)


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


# ── what the model sees ────────────────────────────────────────────────────


def test_history_is_the_latest_message_and_three_before_it():
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "scratchpad"}, {"type": "text", "text": "x" * 900}]},
        {"role": "user", "content": "the question"},
    ]
    taken = ft.take_history(history)
    assert [m["role"] for m in taken] == ["user", "assistant", "user"]
    assert taken[0]["content"] == "three"
    assert taken[1]["content"].startswith("[ran tool: scratchpad]") and len(taken[1]["content"]) == 400
    assert taken[-1] == {"role": "user", "content": "the question"}


def test_no_take_without_a_user_message_last():
    assert ft.take_history([{"role": "assistant", "content": "hi"}]) == []
    assert ft.take_history([]) == []


# ── what it writes ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frames_wait_for_the_turn_then_stream_into_its_buffer():
    provider = _Provider(["  I'll pull ", "last quarter's revenue."])
    take = _take(provider).start()
    buffer = _Buffer()
    await _settle()
    assert buffer.frames == []  # nothing before the turn's response.created

    take.attach(buffer)
    await _settle()
    events = buffer.events()
    assert [e for e, _ in events] == [ft.DELTA, ft.DELTA, ft.DONE]
    assert "".join(d["delta"] for e, d in events if e == ft.DELTA) == "I'll pull last quarter's revenue."
    assert all("at_ms" in d for _, d in events)

    request = provider.calls[0]
    assert request["model"] == "fast" and request["max_tokens"] == ft._MAX_TOKENS
    assert "tools" not in request
    assert "never state a fact" in " ".join(request["system"].split())


@pytest.mark.asyncio
async def test_clear_takes_it_off_screen_and_stops_it():
    gate = asyncio.Event()
    provider = _Provider(["Looking ", "into ", "it."], gate=gate)
    take = _take(provider).start()
    buffer = _Buffer()
    take.attach(buffer)
    gate.set()
    await _settle()
    assert [e for e, _ in buffer.events()] == [ft.DELTA]

    await take.clear(buffer)
    gate.set()
    await _settle()
    assert [e for e, _ in buffer.events()] == [ft.DELTA, ft.CLEARED]
    assert provider.closed

    await take.clear(buffer)  # every way out of the turn calls it again
    assert len(buffer.frames) == 2


@pytest.mark.asyncio
async def test_a_take_never_shown_needs_no_clearing():
    gate = asyncio.Event()
    take = _take(_Provider(["late"], gate=gate)).start()
    buffer = _Buffer()
    take.attach(buffer)
    await take.clear(buffer)
    gate.set()
    await _settle()
    assert buffer.frames == []


@pytest.mark.asyncio
async def test_the_gate_guards_withdraw_it():
    buffer = _Buffer()
    take = _take(_Provider(["I'll check that, ", "as ChatGPT would."])).start()
    take.attach(buffer)
    await _settle()
    assert [e for e, _ in buffer.events()] == [ft.DELTA, ft.CLEARED]


@pytest.mark.asyncio
async def test_it_stops_at_the_word_cap():
    buffer = _Buffer()
    take = _take(_Provider(["word " * 10] * 10)).start()
    take.attach(buffer)
    await _settle()
    words = "".join(d.get("delta", "") for _, d in buffer.events()).split()
    assert len(words) <= ft._MAX_WORDS


@pytest.mark.asyncio
async def test_a_failing_model_means_no_take_and_no_error():
    buffer = _Buffer()
    take = _take(_Provider([], fail=RuntimeError("402"))).start()
    take.attach(buffer)
    await _settle()
    await take.clear(buffer)
    assert buffer.frames == []


@pytest.mark.asyncio
async def test_a_slow_model_is_dropped(monkeypatch):
    monkeypatch.setattr(ft, "_FIRST_WORD_SECONDS", 0.01)
    buffer = _Buffer()
    take = _take(_Provider(["late"], gate=asyncio.Event())).start()
    take.attach(buffer)
    await asyncio.sleep(0.05)
    assert buffer.frames == []


async def _take_text(pieces) -> tuple[str, list[str]]:
    buffer = _Buffer()
    take = _take(_Provider(pieces)).start()
    take.attach(buffer)
    await _settle()
    events = buffer.events()
    return "".join(d.get("delta", "") for _, d in events), [e for e, _ in events]


@pytest.mark.asyncio
async def test_a_repeated_sentence_is_cut_after_the_first():
    # As streamed live: the same sentence twice, no space between.
    pieces = ["I", "’ll", " write", " and", " run", " it", ".", "I", "’ll", " write", " and", " run", " it", "."]
    text, kinds = await _take_text(pieces)
    assert text == "I’ll write and run it."
    assert kinds[-1] == ft.DONE


@pytest.mark.asyncio
async def test_only_the_first_of_two_sentences_is_shown():
    text, _ = await _take_text(["I'll pull your sales. ", "Then I'll chart them."])
    assert text.strip() == "I'll pull your sales."
    text, _ = await _take_text(["Voy a escribir el script", "».", "Voy a escribir"])
    assert text == "Voy a escribir el script»."


@pytest.mark.asyncio
async def test_a_decimal_or_abbreviation_does_not_end_the_sentence():
    text, _ = await _take_text(["I'll check Python 3", ".", "12 notes, e.g. the", " changelog."])
    assert text == "I'll check Python 3.12 notes, e.g. the changelog."


def test_where_the_first_sentence_ends():
    assert ft._first_sentence_end("I'll pull it.") is None
    assert ft._first_sentence_end("I'll pull it.I'll") == len("I'll pull it.")
    assert ft._first_sentence_end("Done! ¿Qué") == len("Done!")
    assert ft._first_sentence_end("Version 3.12 is") is None


@pytest.mark.asyncio
async def test_it_logs_its_outcome_and_time_to_first_word(caplog):
    buffer = _Buffer()
    take = _take(_Provider(["On it."])).start()
    take.attach(buffer)
    with caplog.at_level("WARNING", logger=ft.__name__):
        await _settle()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("[first-take]"))
    assert "model=fast outcome=shown first_word_ms=" in line and "first_word_ms=None" not in line


class _ReasoningProvider:
    """Streams reasoning events forever, then (never) text."""

    async def stream(self, **_kwargs):
        from anton.core.llm.provider import StreamReasoningDelta

        while True:
            await asyncio.sleep(0.005)
            yield StreamReasoningDelta(text="thinking")


@pytest.mark.asyncio
async def test_reasoning_events_do_not_push_the_first_word_back(monkeypatch, caplog):
    monkeypatch.setattr(ft, "_FIRST_WORD_SECONDS", 0.03)
    buffer = _Buffer()
    take = _take(_ReasoningProvider()).start()
    take.attach(buffer)
    with caplog.at_level("WARNING", logger=ft.__name__):
        await asyncio.sleep(0.1)
    assert buffer.frames == []
    assert any("outcome=timeout" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_take_cut_off_mid_stream_comes_off_screen(monkeypatch):
    monkeypatch.setattr(ft, "_IDLE_SECONDS", 0.02)
    gate = asyncio.Event()
    provider = _Provider(["I'll pull ", "your sales."], gate=gate)
    take = _take(provider).start()
    buffer = _Buffer()
    take.attach(buffer)
    gate.set()  # the first piece only; the second never comes
    await asyncio.sleep(0.1)
    assert [e for e, _ in buffer.events()] == [ft.DELTA, ft.CLEARED]


@pytest.mark.asyncio
async def test_a_model_error_after_words_takes_them_off_screen():
    class _Failing:
        async def stream(self, **_kwargs):
            from anton.core.llm.provider import StreamTextDelta

            yield StreamTextDelta(text="I'll pull ")
            raise RuntimeError("upstream reset")

    buffer = _Buffer()
    take = _take(_Failing()).start()
    take.attach(buffer)
    await _settle()
    assert [e for e, _ in buffer.events()] == [ft.DELTA, ft.CLEARED]


@pytest.mark.asyncio
async def test_a_cancelled_take_is_logged(caplog):
    take = _take(_Provider(["late"], gate=asyncio.Event())).start()
    await _settle()
    with caplog.at_level("WARNING", logger=ft.__name__):
        take.cancel()
        await _settle()
    assert any("outcome=cancelled" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_cancel_before_any_stream_writes_nothing():
    buffer = _Buffer()
    take = _take(_Provider(["hi"])).start()
    take.cancel()
    take.attach(buffer)
    await _settle()
    assert buffer.frames == []


def test_what_takes_its_place():
    def frame(kind, **data):
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"

    # Anton's text is narration until the renderer says otherwise: it no longer
    # ends the take here, so the take finishes its sentence (ENG-3554).
    assert not ft.clears_take("response.output_text.delta", frame("response.output_text.delta", delta="Let me check."))
    assert not ft.clears_take("response.output_text.delta", frame("response.output_text.delta", delta="\n\n"))
    for kind in ("response.ask_user", "response.completed", "response.failed"):
        assert ft.clears_take(kind, frame(kind))
    for kind in ("response.created", "response.in_progress", "response.artifact_created"):
        assert not ft.clears_take(kind, frame(kind))


# ── where it starts ────────────────────────────────────────────────────────


def _handler():
    handler = ResponsesHandler()
    handler.harness_name = "anton"
    return handler


async def _route(handler, monkeypatch, *, first_take: bool, enabled: bool = True):
    import cowork.handlers.responses as responses

    provider = _Provider(["On it."])

    async def delegated(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", delegated)
    monkeypatch.setattr(
        responses, "_first_take_binding", lambda minted, **kwargs: RouterBinding(provider=provider, model="m", label="x"),
    )
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: enabled)
    await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "Chart my sales"}],
        has_attachments=False,
        has_disabled_connections=False,
        first_take=first_take,
    )
    return provider


@pytest.mark.asyncio
async def test_route_request_starts_it_beside_the_gate(monkeypatch):
    handler = _handler()
    provider = await _route(handler, monkeypatch, first_take=True)
    assert isinstance(handler.first_take, ft.FirstTake)
    await _settle()
    assert provider.calls[0]["messages"][-1] == {"role": "user", "content": "Chart my sales"}
    handler.first_take.cancel()


@pytest.mark.asyncio
async def test_switched_off_or_not_asked_for_means_none(monkeypatch):
    handler = _handler()
    await _route(handler, monkeypatch, first_take=False)
    assert handler.first_take is None
    await _route(handler, monkeypatch, first_take=True, enabled=False)
    assert handler.first_take is None


@pytest.mark.asyncio
async def test_attachments_never_start_one(monkeypatch):
    handler = _handler()
    import cowork.handlers.responses as responses

    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "what's in this?"}],
        has_attachments=True,
        has_disabled_connections=False,
        first_take=True,
    )
    assert handler.first_take is None


@pytest.mark.asyncio
async def test_first_take_and_gate_build_one_provider_per_request(monkeypatch):
    from types import SimpleNamespace

    import cowork.handlers.response_routing as routing
    import cowork.handlers.responses as responses

    providers = []
    gated = []
    settings = SimpleNamespace(
        resolved_gate_model="first-model",
        resolved_router_provider=SimpleNamespace(value="openai_compatible"),
    )

    def build_client(**kwargs):
        assert kwargs["settings"] is settings
        provider = _Provider(["On it."])
        providers.append(provider)
        return SimpleNamespace(router_provider=provider)

    async def gate(binding, *, history):
        gated.append(binding)
        await _settle()
        return None

    monkeypatch.setattr(routing, "get_user_settings", lambda: settings)
    monkeypatch.setattr(routing, "build_llm_client", build_client)
    monkeypatch.setattr(routing, "_gate", gate)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    for model in ("first-model", "second-model"):
        settings.resolved_gate_model = model
        handler = _handler()
        decision, _ = await handler._route_request(
            conversation_id=None, gate_rows=[], harness_input=[{"type": "text", "text": "hi"}],
            has_attachments=False, has_disabled_connections=False, first_take=True,
        )
        assert decision.reason == "router_declined_direct_response"
        assert handler.first_take is not None
        await _settle()
        handler.first_take.cancel()

    assert len(providers) == 2
    assert [binding.provider for binding in gated] == providers
    assert [binding.model for binding in gated] == ["first-model", "second-model"]
    assert [provider.calls[0]["model"] for provider in providers] == ["first-model", "second-model"]


@pytest.mark.asyncio
async def test_no_gate_model_is_resolved_once_for_take_and_gate(monkeypatch):
    from unittest.mock import Mock

    import cowork.handlers.response_routing as routing
    import cowork.handlers.responses as responses

    resolve = Mock(return_value=None)
    monkeypatch.setattr(routing, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    handler = _handler()
    decision, _ = await handler._route_request(
        conversation_id=None, gate_rows=[], harness_input=[{"type": "text", "text": "hi"}],
        has_attachments=False, has_disabled_connections=False, first_take=True,
    )
    assert handler.first_take is None
    assert decision.reason == "router_model_unavailable"
    resolve.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,enabled", [(False, True), (True, False)])
async def test_skipped_gate_and_disabled_take_do_not_resolve_a_provider(monkeypatch, requested, enabled):
    from unittest.mock import Mock

    import cowork.handlers.response_routing as routing
    import cowork.handlers.responses as responses

    resolve = Mock(side_effect=AssertionError("no provider is needed"))
    monkeypatch.setattr(routing, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: enabled)
    monkeypatch.setattr(responses, "scrubbed_openai_dump", lambda row: row)
    handler = _handler()
    decision, _ = await handler._route_request(
        conversation_id=None,
        gate_rows=[{"role": "assistant", "content": [{"type": "tool_use", "name": "scratchpad"}]}],
        harness_input=[{"type": "text", "text": "summarize that"}],
        has_attachments=False, has_disabled_connections=False, first_take=requested,
    )
    assert decision.reason == "prior_turn_ran_tools"
    assert handler.first_take is None
    resolve.assert_not_called()


@pytest.mark.asyncio
async def test_first_take_resolution_failure_does_not_hide_gate_pool_timeout(monkeypatch):
    from unittest.mock import Mock

    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    import cowork.handlers.response_routing as routing
    import cowork.handlers.responses as responses

    resolve = Mock(side_effect=PoolTimeoutError("pool full"))
    monkeypatch.setattr(routing, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_settings_binding", resolve)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    handler = _handler()
    with pytest.raises(PoolTimeoutError):
        await handler._route_request(
            conversation_id=None, gate_rows=[], harness_input=[{"type": "text", "text": "hi"}],
            has_attachments=False, has_disabled_connections=False, first_take=True,
        )
    assert handler.first_take is None
    # Failed resolutions stay retryable, preserving the gate's existing boundary.
    assert resolve.call_count == 2


@pytest.mark.asyncio
async def test_shared_openai_provider_keeps_streams_and_trace_contexts_separate(monkeypatch):
    from types import SimpleNamespace

    import httpx
    import openai
    from anton.core.llm.openai import OpenAIProvider
    from anton.core.llm.tracing import get_trace_context

    import cowork.handlers.response_routing as routing
    import cowork.handlers.responses as responses

    provider = OpenAIProvider(
        api_key="test-key", base_url="https://example.invalid/v1",
        flavor=OpenAIProvider.FLAVOR_OPENAI, reasoning_effort="low",
    )
    initial_calls = []
    calls = []
    closed = []
    both_started = asyncio.Event()
    take_wait = asyncio.Event()

    class Stream:
        def __init__(self, tag):
            self.tag = tag

        async def __aiter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="Hello.")
            if self.tag == "cowork-first-take":
                await take_wait.wait()
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed"))

        async def close(self):
            closed.append(self.tag)

    async def create(**kwargs):
        tag = get_trace_context().tags[0]
        assert kwargs["extra_headers"]["Langfuse-Tags"] == f"anton,{tag}"
        calls.append((tag, kwargs))
        if "summary" in kwargs.get("reasoning", {}):
            initial_calls.append(tag)
            if len(initial_calls) == 2:
                both_started.set()
            await both_started.wait()
            raise openai.BadRequestError(
                "reasoning summary unavailable",
                response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
                body={"param": "reasoning.summary"},
            )
        return Stream(tag)

    monkeypatch.setattr(provider, "_emit_trace_headers", True)
    monkeypatch.setattr(provider._client.responses, "create", create)
    monkeypatch.setattr(routing, "_settings_binding", lambda: RouterBinding(provider, "gpt-5", "openai"))
    monkeypatch.setattr(responses, "_settings_binding", routing._settings_binding)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    handler = _handler()
    try:
        decision, _ = await handler._route_request(
            conversation_id=None, gate_rows=[], harness_input=[{"type": "text", "text": "hi"}],
            has_attachments=False, has_disabled_connections=False, first_take=True,
        )
        assert decision.text == "Hello."
        assert sorted(initial_calls) == ["cowork-first-take", "cowork-gate"]
        assert len(calls) == 4
        assert closed == ["cowork-gate"]
        handler.first_take.cancel()
        await _settle()
        assert sorted(closed) == ["cowork-first-take", "cowork-gate"]
        assert provider._reasoning_summary is False
    finally:
        if handler.first_take:
            handler.first_take.cancel()
            await _settle()
        await provider.aclose()


# ── which model it speaks with ─────────────────────────────────────────────


def test_hosted_uses_the_gates_minted_binding():
    from cowork.handlers.responses import _first_take_binding

    minted = RouterBinding(provider="minted-key", model="mindshub_air", label="minds_cloud")
    assert _first_take_binding(minted) is minted


def test_elsewhere_uses_the_gates_stored_binding(monkeypatch):
    import cowork.handlers.responses as responses
    from cowork.handlers.responses import _first_take_binding

    stored = RouterBinding(provider="router-provider", model="gate-model", label="anthropic")
    monkeypatch.setattr(responses, "_settings_binding", lambda: stored)
    assert _first_take_binding(None) is stored


def test_no_gate_model_means_no_take(monkeypatch):
    import cowork.handlers.responses as responses
    from cowork.handlers.responses import _first_take_binding

    monkeypatch.setattr(responses, "_settings_binding", lambda: None)
    assert _first_take_binding(None) is None

    def broken():
        raise ValueError("Anthropic API key is not configured")

    monkeypatch.setattr(responses, "_settings_binding", broken)
    assert _first_take_binding(None) is None


@pytest.mark.asyncio
async def test_route_request_hands_the_gates_binding_over(monkeypatch):
    import cowork.handlers.responses as responses

    minted = RouterBinding(provider="minted-key", model="mindshub_air", label="minds_cloud")
    seen = {}

    async def router_binding():
        return minted, {"correlation_id": "c"}

    async def delegated(**kwargs):
        seen["gate"] = kwargs["binding"]
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    def binding_for(binding, **kwargs):
        seen["take"] = binding
        return None

    monkeypatch.setattr(responses, "decide_route", delegated)
    monkeypatch.setattr(responses, "_first_take_binding", binding_for)
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    handler = _handler()
    handler._router_binding = router_binding
    await handler._route_request(
        conversation_id=None, gate_rows=[], harness_input=[{"type": "text", "text": "hi"}],
        has_attachments=False, has_disabled_connections=False, first_take=True,
    )
    assert seen["take"] is minted and seen["gate"] is minted
    assert handler.first_take is None


@pytest.mark.asyncio
async def test_a_routing_error_cancels_the_take(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4

    import cowork.handlers.responses as responses
    from cowork.schemas.responses import ResponsesRequest
    from tests.test_response_routing import _routing_handler

    handler = _routing_handler(monkeypatch)
    conv_id = uuid4()
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(
            get_conversation=lambda _cid: SimpleNamespace(id=conv_id),
            message_count=lambda _conversation: 0,
            get_recent_messages=lambda *_args, **_kwargs: [],
        ),
    )

    async def no_secrets(_scope):
        return None

    monkeypatch.setattr(responses, "register_vault_secrets", no_secrets)
    take = _take(_Provider(["late"], gate=asyncio.Event())).start()

    async def refused(**_kwargs):
        handler.first_take = take
        raise TimeoutError("pool")

    handler._route_request = refused
    with pytest.raises(TimeoutError):
        await handler.handle(ResponsesRequest(input="hi", conversation=str(conv_id), stream=True))
    await _settle()
    assert handler.first_take is None
    assert take._writer.done()


def test_the_setting_defaults_on():
    from cowork.common.settings.user_settings import UserSettings

    assert UserSettings.model_fields["first_take_enabled"].default is True


# ── in a real producer ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anton_speaking_does_not_clear_it_the_turns_end_does():
    """In-process producer, end to end: created, then the take's words, then
    Anton's words (narration or answer: the renderer decides), then the take's
    clear right before the turn ends (ENG-3554). Nothing of it reaches the
    saved turn."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace
    from uuid import uuid4

    from tests.test_completion_message_id_inprocess import _handler_with_formatter

    take = _take(_Provider(["Pulling your sales now."])).start()

    async def formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        await asyncio.sleep(0.05)  # Anton's first model call, still silent
        yield 'event: response.in_progress\ndata: {"thought_role": "thought.progress"}\n\n'
        yield 'event: response.output_text.delta\ndata: {"delta": "Here is the chart."}\n\n'
        yield ('event: response.completed\ndata: {"type": "response.completed", '
               '"response": {"output": [{"content": [{"type": "output_text", "text": "Here is the chart."}]}]}}\n\n')

    buffer = _Buffer()
    buffer.close = lambda _status, extra=None: asyncio.sleep(0)
    handler = _handler_with_formatter(formatter)
    with (
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        conv_svc.return_value.save_user_message.return_value = SimpleNamespace(id=uuid4())
        saved = SimpleNamespace(id=uuid4())
        conv_svc.return_value.save_assistant_turn.return_value = saved
        await handler._produce(
            conv_id=uuid4(), harness_input=[{"type": "text", "text": "chart my sales"}],
            original_content="chart my sales", model="anton", disabled=None,
            harness_name="anton", harness_id="anton", buffer=buffer, first_take=take,
        )

    kinds = [frame.split("\n", 1)[0].removeprefix("event: ") for frame in buffer.frames]
    assert kinds[0] == "response.created"
    assert kinds.index(ft.DELTA) < kinds.index("response.output_text.delta") < kinds.index(ft.CLEARED)
    assert kinds.index(ft.CLEARED) < kinds.index("response.completed")
    assert kinds.count(ft.CLEARED) == 1
    assert all(k not in kinds[kinds.index(ft.CLEARED) + 1:] for k in (ft.DELTA, ft.DONE))
    saved_text = json.dumps([str(c) for c in conv_svc.return_value.save_assistant_turn.call_args_list])
    assert "Pulling your sales" not in saved_text
