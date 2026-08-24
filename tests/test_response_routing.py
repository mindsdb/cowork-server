from types import SimpleNamespace

import pytest

from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    RouteDecision,
    RouterBinding,
    decide_route,
)
from cowork.handlers.responses import ResponsesHandler


class _Provider:
    value = "minds_cloud"


class _Client:
    """Stands in for both the LLMClient and its router provider."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.router_provider = self
        self.router_model = "router-model"

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(*, content="", tool_calls=None, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
    )


def test_handler_defers_anton_harness_initialization(monkeypatch):
    import cowork.handlers.responses as responses

    monkeypatch.setattr(
        responses,
        "get_user_settings",
        lambda scope: SimpleNamespace(harness="anton"),
    )
    monkeypatch.setattr(
        responses,
        "get_harness",
        lambda name: (_ for _ in ()).throw(AssertionError("Anton must be lazy")),
    )

    handler = ResponsesHandler(session=object())

    assert handler.harness is None


@pytest.mark.asyncio
async def test_text_context_routes_direct_with_resolved_router(monkeypatch):
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="The result was 42."))

    async def fake_gate(binding, *, history):
        assert binding.provider is client
        return "The result was 42."

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(routing, "_gate", fake_gate)

    decision = await decide_route(
        history=[
            {"role": "user", "content": "What was the result?"},
            {"role": "assistant", "content": "The result was 42."},
            {"role": "user", "content": "Repeat it."},
        ],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "The result was 42."
    assert decision.provider == "minds_cloud"
    assert decision.model == "router-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("has_non_text_input", "has_attachments", "has_disabled_connections", "reason"),
    [
        (True, False, False, "non_text_input"),
        (False, True, False, "attachments_present"),
        (False, False, True, "connection_context_present"),
    ],
)
async def test_ineligible_request_shapes_delegate_without_router_call(
    monkeypatch, has_non_text_input, has_attachments, has_disabled_connections, reason
):
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(routing, "build_llm_client", lambda: (_ for _ in ()).throw(AssertionError()))

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=has_non_text_input,
        has_attachments=has_attachments,
        has_disabled_connections=has_disabled_connections,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == reason


@pytest.mark.asyncio
async def test_model_override_wins_over_settings_router_model(monkeypatch):
    """ENG-1656 follow-up: the composer's per-conversation model pick must
    drive the router gate too, not just cosmetically label the response."""
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="The result was 42."))
    seen_models = []

    async def fake_gate(binding, *, history):
        seen_models.append(binding.model)
        return "The result was 42."

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(routing, "_gate", fake_gate)

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        model_override="picked-model",
    )

    assert seen_models == ["picked-model"]
    assert decision.model == "picked-model"


@pytest.mark.asyncio
async def test_router_decline_delegates(monkeypatch):
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="DELEGATE"))
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(
        routing,
        "_gate",
        lambda *args, **kwargs: __import__("asyncio").sleep(0, result=None),
    )

    decision = await decide_route(
        history=[{"role": "user", "content": "Search the web for today's news."}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_declined_direct_response"


@pytest.mark.asyncio
async def test_router_error_fails_open_to_anton(monkeypatch):
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True


@pytest.mark.asyncio
async def test_slow_gate_times_out_and_fails_open(monkeypatch):
    import asyncio

    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: _Client(_response(content="unused")))

    async def hung_gate(binding, *, history):
        await asyncio.sleep(30)

    monkeypatch.setattr(routing, "_gate", hung_gate)
    monkeypatch.setattr(routing, "_GATE_TIMEOUT_SECONDS", 0.01)

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_timeout"
    assert decision.fallback is True


def test_sse_frames_use_real_newlines():
    from cowork.streaming import sse_frame

    frame = sse_frame("response.created", {"type": "response.created"})

    assert frame.startswith("event: response.created\n")
    assert "\ndata: " in frame
    assert frame.endswith("\n\n")
    assert "\\n" not in frame  # regression: escaped newlines broke SSE parsing


def _routing_handler(monkeypatch):
    import cowork.handlers.responses as responses

    monkeypatch.setattr(
        responses,
        "get_user_settings",
        lambda scope: SimpleNamespace(harness="anton"),
    )
    return ResponsesHandler(session=object())


@pytest.mark.asyncio
async def test_route_request_runs_gate_under_org_scope(monkeypatch):
    import cowork.handlers.responses as responses
    from cowork.common.settings.user_settings import _current_scope

    handler = _routing_handler(monkeypatch)
    sentinel_scope = object()
    handler.scope = sentinel_scope

    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen["scope"] = _current_scope.get()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    decision, turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert seen["scope"] is sentinel_scope
    assert turn_llm is None


@pytest.mark.asyncio
async def test_route_request_forwards_model_to_decide_route(monkeypatch):
    """ENG-1656 follow-up: the composer's model pick must reach the gate."""
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen["model_override"] = kwargs.get("model_override")
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
        model="picked-model",
    )

    assert seen["model_override"] == "picked-model"


@pytest.mark.asyncio
async def test_ineligible_route_skips_the_history_query(monkeypatch):
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: (_ for _ in ()).throw(AssertionError("must not touch the DB")),
    )

    decision, turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=True,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "attachments_present"
    assert turn_llm is None


@pytest.mark.asyncio
async def test_history_query_failure_fails_open(monkeypatch):
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(
            get_ordered_messages=lambda _cid: (_ for _ in ()).throw(RuntimeError("db down")),
        ),
    )

    decision, turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True
    assert turn_llm is None


@pytest.mark.asyncio
async def test_pre_minted_binding_skips_settings_and_is_used(monkeypatch):
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(
        routing, "get_user_settings",
        lambda: (_ for _ in ()).throw(AssertionError("settings must not load")),
    )
    monkeypatch.setattr(
        routing, "build_llm_client",
        lambda: (_ for _ in ()).throw(AssertionError("client must not build")),
    )
    provider = _Client(_response(content="Hi there."))

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        binding=RouterBinding(provider=provider, model="minds-free", label="minds_cloud"),
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.provider == "minds_cloud"
    assert decision.model == "minds-free"
    assert provider.calls and provider.calls[0]["model"] == "minds-free"


@pytest.mark.asyncio
async def test_router_binding_mints_per_turn_key_in_hosted_org_mode(monkeypatch):
    import cowork.handlers.responses as responses
    import cowork.turnqueue.producer as producer
    from cowork.common.settings.user_settings import Provider

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses, "TurnQueueSettings",
        lambda: SimpleNamespace(backend="remote", is_remote=True, turn_key_ttl_seconds=1200),
    )
    monkeypatch.setattr(
        responses, "get_user_settings",
        lambda scope: SimpleNamespace(
            resolved_router_provider=Provider.MINDS_CLOUD,
            resolved_router_model="kimi",
        ),
    )
    block = {"provider": "minds-cloud", "api_key": "mdb_test", "base_url": "http://gw/v1"}
    minted = {}

    async def fake_mint(*, org_id, user_id, correlation_id, settings):
        minted["corr"] = correlation_id
        return block

    monkeypatch.setattr(producer, "_mint_llm_block", fake_mint)

    binding, turn_llm = await handler._router_binding()

    assert binding is not None
    assert binding.label == "minds_cloud"
    assert binding.model == "kimi"
    assert type(binding.provider).__name__ == "OpenAIProvider"
    assert turn_llm == {"correlation_id": minted["corr"], "llm": block}


@pytest.mark.asyncio
async def test_router_binding_absent_outside_remote_backend(monkeypatch):
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses, "TurnQueueSettings", lambda: SimpleNamespace(backend="inprocess", is_remote=False)
    )

    assert await handler._router_binding() == (None, None)


@pytest.mark.asyncio
async def test_produce_direct_persists_before_emitting_and_roots_metadata(monkeypatch):
    import json
    from uuid import uuid4 as _uuid4

    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    calls = []
    monkeypatch.setattr(responses, "get_open_session", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(responses, "ScopedSession", lambda session, scope: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(
            save_user_message=lambda cid, content, pending=False: calls.append("save_user") or SimpleNamespace(id=_uuid4()),
            save_assistant_turn=lambda cid, text, events, harness=None: calls.append("save_assistant"),
        ),
    )

    class _Buffer:
        def __init__(self):
            self.frames = []
            self.closed = None

        async def append(self, kind, record):
            calls.append("emit")
            self.frames.append(record["sse"])

        async def close(self, reason):
            self.closed = reason

    buffer = _Buffer()
    conv_id = _uuid4()
    route = RouteDecision(
        route=DIRECT_CONTEXT, reason="router_direct_response", model="m", text="Hi.",
    )

    await handler._produce_direct(
        lifecycle=SimpleNamespace(discarded=False),
        conv_id=conv_id,
        original_content="Hello",
        route=route,
        buffer=buffer,
    )

    assert buffer.closed == "completed"
    assert calls.index("save_assistant") < calls.index("emit")
    created = json.loads(buffer.frames[0].split("data: ", 1)[1])
    assert created["conversation_id"] == str(conv_id)
    assert created["harness"] == "cowork-direct"
    for frame in buffer.frames:
        assert frame.endswith("\n\n") and "\\n" not in frame
