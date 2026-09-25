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

    async def stream(self, **kwargs):
        """The scripted response as events — what anton's base `LLMProvider.stream`
        does for a provider that only implements `complete`."""
        from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

        response = await self.complete(**kwargs)
        for call in response.tool_calls:
            yield StreamToolUseStart(id=getattr(call, "id", "t"), name=getattr(call, "name", "delegate"))
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)


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
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
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
async def test_gate_runs_on_resolved_gate_model_not_the_router_pick(monkeypatch):
    """ENG-1851: the gate's model is `resolved_gate_model`, not the user's
    router pick and not the client's router model. Both are chosen for chat
    or summarization; a model picked for those is routinely too slow to gate
    a turn inside the budget. (Reverses the ENG-1656 follow-up's routing half.)"""
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="The result was 42."))
    client.router_model = "opus"  # the user's router pick, as the client sees it
    seen_models = []

    async def fake_gate(binding, *, history):
        seen_models.append(binding.model)
        return "The result was 42."

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(
            resolved_router_provider=_Provider(),
            resolved_router_model="opus",
            resolved_gate_model="mindshub_air",
        ),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(routing, "_gate", fake_gate)

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert seen_models == ["mindshub_air"]
    assert decision.model == "mindshub_air"


@pytest.mark.asyncio
async def test_router_decline_delegates(monkeypatch):
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="DELEGATE"))
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
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
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
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
    """The budget bounds the time to the gate's first streamed event. Miss it
    and the turn delegates, attributed, with the stream closed behind it."""
    import cowork.handlers.response_routing as routing

    provider = _StreamProvider([_text("Hello!"), _complete()], first_delay=0.5)
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: SimpleNamespace(router_provider=provider))
    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 0.02)

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_timeout"
    assert decision.fallback is True
    assert decision.model == "router-model"
    assert provider.closed


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
async def test_route_request_ignores_jev_result_even_when_it_contradicts_the_gate(monkeypatch):
    """The shadow probe is logging-only. A Jev result that confidently
    disagrees with the gate, or is an error dict, must never change the
    decision `_route_request` returns."""
    import cowork.handlers.responses as responses
    from cowork.handlers import jev_shadow

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    async def fake_decide_route(**_kwargs):
        return RouteDecision(route=DIRECT_CONTEXT, reason="test", text="hi")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    async def loud_wrong_jev(**_kwargs):
        return {"jev_choice": "needs_agent", "jev_confidence": 0.99, "jev_ms": 5}

    monkeypatch.setattr(jev_shadow, "probe", loud_wrong_jev)

    decision, _turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "hi"

    async def jev_error(**_kwargs):
        return {"jev_error": "timeout", "jev_ms": 3000}

    monkeypatch.setattr(jev_shadow, "probe", jev_error)

    decision, _turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "hi"


@pytest.mark.asyncio
async def test_gate_and_jev_probe_share_the_turns_correlation_id(monkeypatch):
    """The gate's trace context carries the turn's correlation_id, and
    the detached probe inherits it, so the real probe's Jev call is attributed
    to the conversation (origin:harness, not direct-api) and joins the gate
    decision it shadows. Only the HTTP client is faked: the context has to cross
    the gate's create_task for this to pass."""
    import asyncio
    import json

    import cowork.handlers.responses as responses
    from anton.core.llm.tracing import get_trace_context
    from cowork.handlers import jev_shadow

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    llm_block = {"provider": "minds-cloud", "api_key": "turn-key", "base_url": "https://minds.example/v1"}

    async def fake_binding():
        return None, {"correlation_id": "corr-1", "llm": llm_block}

    handler._router_binding = fake_binding
    seen = {}

    async def fake_decide_route(**_kwargs):
        seen["gate_context"] = get_trace_context()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)
    sent = asyncio.Event()

    class _Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, _url, *, headers, json):
            seen["probe_headers"] = headers
            sent.set()
            return SimpleNamespace(status_code=500, json=lambda: {})

    monkeypatch.setattr(jev_shadow.httpx, "AsyncClient", _Client)

    await handler._route_request(
        conversation_id="conv-1",
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
        trace_metadata={"cowork_server_version": "1.2.3"},
    )
    await asyncio.wait_for(sent.wait(), timeout=2)

    gate = seen["gate_context"]
    assert gate.metadata["correlation_id"] == "corr-1"
    assert gate.turn_id is None
    headers = seen["probe_headers"]
    assert "Langfuse-Session-Id" not in headers
    assert "cowork-gate" not in headers["Langfuse-Tags"]
    assert jev_shadow.JEV_SHADOW_TAG in headers["Langfuse-Tags"]
    metadata = json.loads(headers["Langfuse-Metadata"])
    assert (metadata["correlation_id"], metadata["harness"], metadata["conversation_id"]) == (
        "corr-1",
        "anton",
        "conv-1",
    )
    # The context is the gate's alone: it does not leak past the gate block.
    assert get_trace_context() is None


@pytest.mark.asyncio
async def test_gate_without_a_minted_turn_key_carries_no_correlation_id(monkeypatch):
    import cowork.handlers.responses as responses
    from anton.core.llm.tracing import get_trace_context

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    seen = {}

    async def fake_decide_route(**_kwargs):
        seen["gate_context"] = get_trace_context()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    await handler._route_request(
        conversation_id="conv-1",
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
        trace_metadata={"cowork_server_version": "1.2.3"},
    )

    assert seen["gate_context"].metadata == {"cowork_server_version": "1.2.3"}


@pytest.mark.asyncio
async def test_route_request_returns_promptly_even_when_jev_is_slow(monkeypatch):
    """The probe must be detached, not awaited alongside the gate: a ready
    gate decision returning only after Jev finishes would mean a 'shadow'
    probe delays every real turn it shadows, exactly what it must never do."""
    import asyncio
    import time

    import cowork.handlers.responses as responses
    from cowork.handlers import jev_shadow

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )

    async def fast_decide_route(**_kwargs):
        return RouteDecision(route=DIRECT_CONTEXT, reason="test", text="hi")

    monkeypatch.setattr(responses, "decide_route", fast_decide_route)

    async def slow_jev(**_kwargs):
        await asyncio.sleep(0.25)
        return {"jev_choice": "needs_agent", "jev_confidence": 0.9, "jev_ms": 250}

    monkeypatch.setattr(jev_shadow, "probe", slow_jev)

    started = time.monotonic()
    decision, _turn_llm = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )
    elapsed = time.monotonic() - started

    assert decision.route == DIRECT_CONTEXT
    assert elapsed < 0.1  # nowhere near the slow probe's 0.25s

    await asyncio.sleep(0.3)  # let the detached probe finish before teardown


@pytest.mark.asyncio
async def test_route_request_scrubs_secrets_from_history_and_current_prompt(monkeypatch):
    """the gate reads history straight from storage, bypassing the
    scrub a normal turn gets via AntonHarness._stamp_message/_scrub_user_input.
    A secret typed in an earlier turn, or in the current message, must not
    reach `decide_route` (and from there the gate's LLM) unmasked."""
    from uuid import uuid4

    from cowork.models.message_event import MessageEvent  # noqa: F401 — resolves the ORM relationship
    from cowork.models.message import Message
    from cowork.schemas.responses import Role
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    cid = uuid4()
    leaked_key = "sk-" + "a" * 30
    rows = [Message(conversation_id=cid, role=Role.user, content=f"my key is {leaked_key}")]

    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: rows),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen.update(kwargs)
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    current_key = "sk-" + "b" * 30
    await handler._route_request(
        conversation_id=cid,
        harness_input=[{"type": "text", "text": f"and my other key is {current_key}"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    blob = str(seen["history"])
    assert leaked_key not in blob
    assert current_key not in blob
    assert blob.count("[REDACTED_API_KEY]") == 2


@pytest.mark.asyncio
async def test_route_request_scrubs_a_registered_vault_secret_by_value(monkeypatch, tmp_path, request):
    """A datasource password has no API-key shape, so only its registered
    value can redact it. Covers the scrub given the registration; the call
    site is pinned by test_handle_registers_vault_secrets_before_routing."""
    from uuid import uuid4

    from anton.core.datasources.data_vault import LocalDataVault
    from anton.utils.datasources import _reset_registered_ds_vars

    from cowork.common.history_scrub import register_vault_secrets
    from cowork.models.message_event import MessageEvent  # noqa: F401 — resolves the ORM relationship
    from cowork.models.message import Message
    from cowork.schemas.responses import Role
    import cowork.handlers.responses as responses

    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    LocalDataVault(tmp_path / "vault").save("postgres", "mydb", {
        "host": "db.example.com", "port": "5432", "database": "app",
        "user": "svc", "password": "hunter2xyz",
    })
    request.addfinalizer(_reset_registered_ds_vars)

    handler = _routing_handler(monkeypatch)
    register_vault_secrets(handler.scope)

    cid = uuid4()
    rows = [Message(conversation_id=cid, role=Role.user, content="the password is hunter2xyz")]
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: rows),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen.update(kwargs)
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    await handler._route_request(
        conversation_id=cid,
        harness_input=[{"type": "text", "text": "hi"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    blob = str(seen["history"])
    assert "hunter2xyz" not in blob
    assert "[DS_" in blob


@pytest.mark.asyncio
async def test_handle_registers_vault_secrets_before_routing(monkeypatch):
    """The gate scrubs history inside _route_request, so the vault's secrets
    must be registered before it runs, not later in _build_chat_session."""
    from uuid import uuid4

    import cowork.handlers.responses as responses
    from cowork.schemas.responses import ResponsesRequest

    handler = _routing_handler(monkeypatch)
    conv_id = uuid4()
    conversation = SimpleNamespace(id=conv_id, messages=[])
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_conversation=lambda _cid: conversation),
    )
    calls = []
    monkeypatch.setattr(
        responses, "register_vault_secrets", lambda scope: calls.append(("register", scope))
    )

    class _StopHere(Exception):
        pass

    async def fake_route_request(**kwargs):
        calls.append(("route_request", None))
        raise _StopHere

    handler._route_request = fake_route_request

    with pytest.raises(_StopHere):
        await handler.handle(ResponsesRequest(input="hi", conversation=str(conv_id)))

    assert calls == [("register", handler.scope), ("route_request", None)]


@pytest.mark.asyncio
async def test_route_request_does_not_hand_the_composer_pick_to_the_gate(monkeypatch):
    """ENG-1851: the composer's per-conversation pick drives Anton's turn, not
    the gate. `_route_request` no longer accepts or forwards it."""
    import inspect

    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen.update(kwargs)
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert "model_override" not in seen
    assert "model" not in inspect.signature(handler._route_request).parameters


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
            resolved_router_model="kimi",        # the user's summarization pick
            resolved_gate_model="mindshub_air",  # what the gate actually runs on
            hub_workspace_id="ws-1",
        ),
    )
    block = {"provider": "minds-cloud", "api_key": "mdb_test", "base_url": "http://gw/v1"}
    minted = {}

    async def fake_mint(*, org_id, user_id, correlation_id, settings, workspace_id=None):
        minted["corr"] = correlation_id
        minted["workspace_id"] = workspace_id
        return block

    monkeypatch.setattr(producer, "_mint_llm_block", fake_mint)

    binding, turn_llm = await handler._router_binding()

    assert binding is not None
    assert binding.label == "minds_cloud"
    assert binding.model == "mindshub_air"
    assert type(binding.provider).__name__ == "OpenAIProvider"
    assert turn_llm == {"correlation_id": minted["corr"], "llm": block}
    # The routing gate's own pre-mint must carry the caller's picked workspace
    # too — this key is what a delegated remote turn ends up reusing as its
    # `llm` block, so skipping it here would silently exempt every hosted-org
    # turn that goes through the gate from workspace attribution.
    assert minted["workspace_id"] == "ws-1"


@pytest.mark.asyncio
async def test_router_binding_omits_workspace_id_when_none_picked(monkeypatch):
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
            resolved_gate_model="mindshub_air",
            hub_workspace_id="",
        ),
    )
    minted = {}

    async def fake_mint(*, org_id, user_id, correlation_id, settings, workspace_id=None):
        minted["workspace_id"] = workspace_id
        return {"provider": "minds-cloud", "api_key": "mdb_test", "base_url": "http://gw/v1"}

    monkeypatch.setattr(producer, "_mint_llm_block", fake_mint)

    await handler._router_binding()

    assert minted["workspace_id"] is None


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
    # The whole answer is known before the first frame, so it is in the DB
    # before the client can see a completed turn — no pending row needed.
    assert calls.index("save_assistant") < calls.index("emit")
    created = json.loads(buffer.frames[0].split("data: ", 1)[1])
    assert created["conversation_id"] == str(conv_id)
    assert created["harness"] == "cowork-direct"
    for frame in buffer.frames:
        assert frame.endswith("\n\n") and "\\n" not in frame


@pytest.mark.asyncio
async def test_streaming_direct_response_registers_its_shared_redis_buffer(monkeypatch):
    from uuid import UUID

    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    handler.scoped = SimpleNamespace(
        scope=SimpleNamespace(org_id="org-1", user_id="user-1"),
    )
    buffer = SimpleNamespace()
    handle = SimpleNamespace(buffer=buffer)
    recorded = {}

    async def fake_start(**kwargs):
        kwargs["producer_coro"].close()
        return handle

    async def fake_record(conversation_id, **kwargs):
        recorded["conversation_id"] = conversation_id
        recorded.update(kwargs)

    monkeypatch.setattr(responses, "new_buffer", lambda _cid, _turn_id: buffer)
    monkeypatch.setattr(responses.registry, "start", fake_start)
    monkeypatch.setattr(responses, "get_backend", lambda: "redis")
    monkeypatch.setattr(responses, "record_turn", fake_record)
    conversation_id = UUID("d27d3533-2e4e-4021-bb5a-6e238245974c")

    await handler._handle_direct_response(
        request=SimpleNamespace(stream=True),
        conversation_id=conversation_id,
        turn_id=3,
        original_content="Hello",
        route=RouteDecision(
            route=DIRECT_CONTEXT,
            reason="router_direct_response",
            model="m",
            text="Hi.",
        ),
    )

    assert recorded["conversation_id"] == str(conversation_id)
    assert recorded["turn_id"] == 3
    assert recorded["correlation_id"].startswith("direct-")
    assert recorded["org_id"] == "org-1"
    assert recorded["user_id"] == "user-1"


# --- history view: tool rows become markers, not holes (ENG-1851) -----------


def test_text_history_keeps_tool_rows_as_markers():
    """Persisted tool block-rows reach the gate as one-line markers.

    Before, any list-shaped content was dropped, so the gate saw a transcript
    with the work removed — an answer that came from a live query looked like
    something it could restate from context.
    """
    from cowork.handlers.response_routing import _text_history

    history = [
        {"role": "user", "content": "what was last month's revenue?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "scratchpad",
             "input": {"code": "select sum(amount) ..."}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "482913.44"},
        ]},
        {"role": "assistant", "content": "Last month's revenue was $482,913.44."},
        {"role": "user", "content": "and the month before?"},
    ]

    seen = _text_history(history)

    assert [m["role"] for m in seen] == ["user", "assistant", "user", "assistant", "user"]
    assert seen[1]["content"] == "[ran tool: scratchpad]"
    assert seen[2]["content"] == "[tool output omitted]"
    # The payload never travels: the gate learns that work happened, not what it produced.
    assert "482913.44" not in seen[1]["content"] + seen[2]["content"]
    assert "select sum" not in seen[1]["content"]


def test_text_history_flattens_mixed_text_and_tool_blocks():
    from cowork.handlers.response_routing import _text_history

    seen = _text_history([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking now."},
            {"type": "tool_use", "id": "t1", "name": "web_search", "input": {}},
        ]},
    ])

    assert seen[1]["content"] == "Checking now.\n[ran tool: web_search]"


def test_text_history_drops_rows_with_nothing_sayable():
    from cowork.handlers.response_routing import _text_history

    seen = _text_history([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "unknown_block"}]},
        {"role": "assistant", "content": 42},
        {"role": "system", "content": "not for the gate"},
    ])

    assert seen == [{"role": "user", "content": "hi"}]


def test_gate_sees_persisted_tool_rows_end_to_end():
    """Pin the coupling to the persistence shape: rows exactly as
    `save_assistant_turn` writes them, through `to_openai_message` (what
    `_route_request` calls), reach the gate with markers intact."""
    from uuid import uuid4

    from cowork.models.message_event import MessageEvent  # noqa: F401 — resolves the ORM relationship
    from cowork.models.message import Message
    from cowork.handlers.response_routing import _text_history
    from cowork.schemas.responses import Role

    cid = uuid4()
    rows = [
        Message(conversation_id=cid, role=Role.user, content="pull the sales table"),
        Message(conversation_id=cid, role=Role.assistant, content=[
            {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": "..."}},
        ]),
        Message(conversation_id=cid, role=Role.user, content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "rows: 1204"},
        ]),
        Message(conversation_id=cid, role=Role.assistant, content="Pulled 1,204 rows."),
    ]

    history = [
        m.to_openai_message().model_dump()
        for m in rows
        if m.role in {Role.user, Role.assistant}
    ]
    seen = _text_history(history)

    assert len(seen) == len(rows)
    assert seen[1]["content"] == "[ran tool: scratchpad]"
    assert seen[2]["content"] == "[tool output omitted]"


# --- attribution survives a failed gate call --------------------------------


@pytest.mark.asyncio
async def test_router_unavailable_keeps_model_attribution(monkeypatch):
    """A provider error (e.g. a 402 on a paid router pick) still fails open,
    and the decision names the model that failed so the trace is diagnosable."""
    import cowork.handlers.response_routing as routing

    class _Boom:
        router_model = "opus"

        def __init__(self):
            self.router_provider = self

        async def stream(self, **kwargs):
            raise RuntimeError("402 wallet_empty")
            yield  # unreachable; makes this an async generator, like the real provider

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="opus"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: _Boom())

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True
    assert decision.provider == "minds_cloud"
    assert decision.model == "opus"



# --- the streamed gate: budget the decision, not the answer (ENG-1851) -------


def _text(t):
    from anton.core.llm.provider import StreamTextDelta
    return StreamTextDelta(text=t)


def _tool():
    from anton.core.llm.provider import StreamToolUseStart
    return StreamToolUseStart(id="t1", name="delegate")


def _complete(stop_reason="end_turn", tool_calls=()):
    from anton.core.llm.provider import StreamComplete
    return StreamComplete(response=SimpleNamespace(stop_reason=stop_reason, tool_calls=list(tool_calls)))


class _StreamProvider:
    """A provider whose `stream()` replays scripted events, optionally slowly."""

    def __init__(self, events, *, first_delay=0.0, gap=0.0):
        self.events = events
        self.first_delay = first_delay
        self.gap = gap
        self.calls = []
        self.closed = False

    async def stream(self, **kwargs):
        import asyncio

        self.calls.append(kwargs)
        try:
            await asyncio.sleep(self.first_delay)
            for i, event in enumerate(self.events):
                if i and self.gap:
                    await asyncio.sleep(self.gap)
                yield event
        finally:
            self.closed = True


def _binding(provider):
    return RouterBinding(provider=provider, model="gate-model", label="minds_cloud")


HISTORY = [{"role": "user", "content": "Hello"}]
PREAMBLE = ("Sure — let me pull up last month's revenue for you. I'll query the sales "
            "table, sum the amounts, and format the result nicely.")


@pytest.mark.asyncio
async def test_gate_streams_with_the_delegate_tool_and_bounded_output():
    from cowork.handlers.response_routing import _DIRECT_MAX_TOKENS, _gate

    provider = _StreamProvider([_tool()])
    await _gate(_binding(provider), history=HISTORY)

    (call,) = provider.calls
    assert call["model"] == "gate-model"
    assert [t["name"] for t in call["tools"]] == ["delegate"]
    assert call["max_tokens"] == _DIRECT_MAX_TOKENS
    assert call["messages"] == HISTORY


@pytest.mark.asyncio
async def test_gate_delegates_on_a_tool_call_and_closes_the_stream():
    from cowork.handlers.response_routing import _gate

    provider = _StreamProvider([_tool(), _text("never read")])
    assert await _gate(_binding(provider), history=HISTORY) is None
    assert provider.closed


@pytest.mark.asyncio
async def test_gate_returns_the_answer_once_the_stream_ends():
    from cowork.handlers.response_routing import _gate

    provider = _StreamProvider([_text("Hi "), _text("there."), _complete()])
    assert await _gate(_binding(provider), history=HISTORY) == "Hi there."


@pytest.mark.asyncio
async def test_gate_delegates_an_empty_or_truncated_answer():
    from cowork.handlers.response_routing import _gate

    assert await _gate(_binding(_StreamProvider([_text("  "), _complete()])), history=HISTORY) is None
    assert await _gate(_binding(_StreamProvider([_complete()])), history=HISTORY) is None
    assert await _gate(_binding(_StreamProvider([])), history=HISTORY) is None
    truncated = _StreamProvider([_text("a long answer that"), _complete("max_tokens")])
    assert await _gate(_binding(truncated), history=HISTORY) is None


@pytest.mark.asyncio
async def test_gate_delegates_on_a_tool_call_reported_only_on_the_completed_response():
    """A provider that fills a tool call's id/name across deltas never emits the
    Start (anton's chat-completions branch emits it only when the first delta
    carries both), so the completed response is checked too — otherwise the gate
    answers a turn the model meant to delegate."""
    from cowork.handlers.response_routing import _gate

    provider = _StreamProvider([_text("Let me look that up."), _complete(tool_calls=[{"name": "delegate"}])])
    assert await _gate(_binding(provider), history=HISTORY) is None


@pytest.mark.asyncio
async def test_gate_budget_bounds_the_first_event_not_the_generation(monkeypatch):
    """The decision is the first event, so a delegate call exits inside the
    first-event budget however long a whole answer would have taken — the
    property the buffered gate lacked."""
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 0.05)
    monkeypatch.setattr(routing, "_GATE_IDLE_SECONDS", 1.0)

    # Five chunks 0.03s apart: ~0.15s total, three times the first-event budget.
    slow_answer = _StreamProvider([_text("a"), _text("b"), _text("c"), _text("d"), _complete()], gap=0.03)
    assert await routing._gate(_binding(slow_answer), history=HISTORY) == "abcd"

    late_start = _StreamProvider([_tool()], first_delay=0.2)
    with pytest.raises(TimeoutError):
        await routing._gate(_binding(late_start), history=HISTORY)
    assert late_start.closed


@pytest.mark.asyncio
async def test_gate_total_budget_bounds_a_trickling_stream(monkeypatch):
    """`decide_route` runs before any SSE exists, so it blocks the client's POST.
    Per-event budgets alone cannot bound that: a stream that trickles inside the
    idle window runs arbitrarily long."""
    import asyncio

    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 1.0)
    monkeypatch.setattr(routing, "_GATE_IDLE_SECONDS", 1.0)
    monkeypatch.setattr(routing, "_GATE_TOTAL_SECONDS", 0.12)

    #每 chunk well inside the idle bound, but sixteen of them overrun the total.
    trickle = _StreamProvider([_text(f"w{i} ") for i in range(16)] + [_complete()], gap=0.03)
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await routing._gate(_binding(trickle), history=HISTORY)
    assert asyncio.get_running_loop().time() - started < 0.5
    assert trickle.closed


@pytest.mark.asyncio
async def test_gate_total_budget_survives_events_that_decide_nothing(monkeypatch):
    """An event that is neither text, tool call nor completion must not re-arm
    the idle window forever."""
    import cowork.handlers.response_routing as routing
    from anton.core.llm.provider import StreamReasoningDelta

    monkeypatch.setattr(routing, "_GATE_IDLE_SECONDS", 1.0)
    monkeypatch.setattr(routing, "_GATE_TOTAL_SECONDS", 0.12)

    noise = _StreamProvider([StreamReasoningDelta(text="thinking") for _ in range(16)], gap=0.03)
    with pytest.raises(TimeoutError):
        await routing._gate(_binding(noise), history=HISTORY)


# --- Done-when #5: the two paths can never both answer -----------------------


@pytest.mark.asyncio
async def test_a_preamble_before_a_tool_call_delegates_and_says_nothing(monkeypatch):
    """The gate returns nothing until its stream ends, so a model that talks
    before delegating cannot have already spoken to the user.

    `handle()` returns a direct answer without ever building the harness, so a
    committed preamble would abandon the request: the user is told the work is
    coming and no agent ever runs."""
    import cowork.handlers.response_routing as routing

    chunks = [_text(PREAMBLE[i:i + 20]) for i in range(0, len(PREAMBLE), 20)]
    provider = _StreamProvider(chunks + [_tool()])
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: SimpleNamespace(router_provider=provider))

    decision = await decide_route(
        history=HISTORY, has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
    )

    assert len(PREAMBLE) > 120  # long enough that any commit-early rule would have fired
    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_declined_direct_response"
    assert decision.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, first_delay",
    [
        ([_tool()], 0.0),                                            # delegated outright
        ([_text("preamble"), _tool()], 0.0),                          # delegated after talking
        ([_text("x"), _complete(tool_calls=[{"name": "delegate"}])], 0.0),  # tool only on the response
        ([_text("x"), _complete("max_tokens")], 0.0),                 # overran the output budget
        ([_complete()], 0.0),                                         # said nothing
        ([_text("x"), _complete()], 0.4),                             # missed the budget
    ],
)
async def test_no_delegated_route_ever_carries_text(monkeypatch, events, first_delay):
    """The invariant behind Done-when #5, over every shape that delegates:
    `_handle_direct_response` is reached only for DIRECT_CONTEXT, and only a
    DIRECT_CONTEXT decision carries text — so the gate's answer and Anton's are
    mutually exclusive by construction."""
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 0.05)
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
    )
    provider = _StreamProvider(events, first_delay=first_delay)
    monkeypatch.setattr(routing, "build_llm_client", lambda: SimpleNamespace(router_provider=provider))

    decision = await decide_route(
        history=HISTORY, has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.text == ""


@pytest.mark.asyncio
async def test_a_direct_decision_carries_the_whole_answer(monkeypatch):
    import cowork.handlers.response_routing as routing

    provider = _StreamProvider([_text("Forty"), _text("-two."), _complete()])
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_gate_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: SimpleNamespace(router_provider=provider))

    decision = await decide_route(
        history=HISTORY, has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "Forty-two."


# ── ENG-2423: the gate must not ship another company's product ───────────────


def test_foreign_product_matcher_catches_the_observed_prod_failures():
    """The three answers this guard exists for, pasted from prod traces.

    Pasted rather than paraphrased: the Telugu one is the whole argument for
    reading the ANSWER instead of the question, and a hand-typed approximation
    would not prove the matcher survives a non-Latin script around a Latin name.
    """
    from cowork.handlers.response_routing import names_foreign_product

    # trace a2db4f78 — "how do i install this on my pc"
    assert names_foreign_product(
        "If you mean the ChatGPT desktop app:\n- **Windows:** Open Settings"
    ) == "ChatGPT"
    # trace 7249ac36 — "how do i unstall you on my computer"
    assert names_foreign_product(
        "Download it from https://chatgpt.com/download/ and drag it to Trash"
    ) == "chatgpt"
    # trace 4e513218 — Telugu "who made you?", answered in Telugu, brand in Latin
    assert names_foreign_product(
        "కాదు 😊 నన్ను **OpenAI** తయారు చేసింది. కానీ నాతో మాట్లాడేది నువ్వే కదా!"
    ) == "OpenAI"


def test_foreign_product_matcher_does_not_fire_on_ordinary_answers():
    """The guard delegates ~3.5% of answers; it must not delegate far more.

    Every string here is a real or realistic gate answer that must still ship.
    `cursor` and `llamas` guard the two ways a careless matcher over-fires:
    an ordinary English noun in the list, and a listed name as a substring.
    """
    from cowork.handlers.response_routing import names_foreign_product

    for benign in (
        "Move the cursor to the end of the line, then press Enter.",
        "Close the DB cursor in a finally block so the connection returns to the pool.",
        "A network engineer can use AI to analyse logs, alerts and packet captures.",
        "I'm Cowork, an AI assistant that can analyse data and build things.",
        "Sure — paste the sequence and I'll predict the next value.",
    ):
        assert names_foreign_product(benign) is None, benign


@pytest.mark.asyncio
async def test_gate_answer_naming_a_foreign_product_is_discarded_and_delegated():
    """The enforcement: the model answered, and we override it.

    Asserts the whole contract, because each half is separately load-bearing:
    the turn delegates (so the user never sees it), `text` is empty (so
    `_handle_direct_response` cannot persist it into history — Done-when #4),
    the reason is its own (so the guard is countable in traces), and it is NOT
    a fallback (so it stays out of the ENG-1851 outage counters).
    """
    provider = _Client(
        _response(content="If you mean the ChatGPT desktop app, open Settings → Apps.")
    )

    decision = await decide_route(
        history=[{"role": "user", "content": "how do i install this on my pc"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        binding=RouterBinding(provider=provider, model="minds-free", label="minds_cloud"),
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_answer_named_foreign_product"
    assert decision.text == ""
    assert decision.fallback is False
    assert decision.model == "minds-free"


@pytest.mark.asyncio
async def test_clean_direct_answer_still_ships():
    """The guard must not swallow the 96.5% it was not built for."""
    provider = _Client(_response(content="Yes — press Cmd+K to open the palette."))

    decision = await decide_route(
        history=[{"role": "user", "content": "is there a shortcut?"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        binding=RouterBinding(provider=provider, model="minds-free", label="minds_cloud"),
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "Yes — press Cmd+K to open the palette."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question, answer",
    [
        ("how do i install this on my pc", "Install the ChatGPT desktop app from chatgpt.com."),
        ("how do i uninstall you", "Drag ChatGPT from Applications to the Trash."),
        ("who made you?", "I was made by OpenAI."),
        ("what is MindsHub Cowork?", "I can't identify it; you may mean Claude or Gemini."),
        ("¿quién te creó?", "Fui creado por OpenAI."),
        ("你是什么产品？", "我是 ChatGPT，由 OpenAI 开发。"),
        # The denial shape — names no competitor, denies us instead. Before the
        # denial matcher these two shipped straight to the user.
        (
            "what is MindsHub Cowork?",
            "I can't reliably identify a current public product called MindsHub "
            "CoWork from the name alone.",
        ),
        (
            "is there a cowork desktop app?",
            "There is no verified Cowork desktop app that I could find.",
        ),
    ],
)
async def test_product_identity_questions_never_answer_with_a_competitor(question, answer):
    """Done-when #5: the product-identity question set.

    Covers both shapes the Done-when names — "names a competitor product **or**
    denies Cowork exists". Multilingual on purpose: the prod failure this guard
    exists for was in Telugu, and an English-only suite is exactly the suite
    that missed it.
    """
    provider = _Client(_response(content=answer))

    decision = await decide_route(
        history=[{"role": "user", "content": question}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        binding=RouterBinding(provider=provider, model="minds-free", label="minds_cloud"),
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.text == ""


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("A llama is a South American camelid.", "llama"),
        ("Claude Shannon founded information theory in 1948.", "Claude"),
        ("Gemini is a zodiac sign and also a 1960s NASA programme.", "Gemini"),
        ("The copilot sits in the right-hand seat.", "copilot"),
        ("The bard in your party can cast charm person.", "bard"),
        ("The mistral is a cold wind through southern France.", "mistral"),
    ],
)
def test_known_over_fires_are_accepted_and_pinned(sentence, name):
    """Six names in the list are also ordinary words, and DO fire.

    `grok` used to be a seventh. It came out of the list entirely: it is one of
    our own catalog aliases (`DEFAULT_ALIASES` in auth), which is the same
    reason `gpt` was already excluded, and review pointed out that keeping one
    while excluding the other made the stated rule incoherent.

    Pinned rather than fixed, deliberately. The gate runs on the router role,
    which for most users is a third-party model, so "I'm Claude, made by
    Anthropic" is a live failure for nearly every name here — dropping the
    ambiguous ones would weaken the guard against the exact thing it exists for.
    The cost is one extra hop on a path that already fails open.

    This test exists so the trade is visible in the suite instead of surfacing
    later as a surprise. If someone narrows the matcher, this test should be
    updated deliberately — not deleted to make it pass.

    (An earlier version of the benign test asserted on "Llamas", whose trailing
    boundary excludes it, and so proved nothing about the singular.)
    """
    from cowork.handlers.response_routing import names_foreign_product

    assert names_foreign_product(sentence) == name


def test_denial_matcher_catches_the_prospect_failure():
    """The ticket's worst-outcome case, and the shape the name list cannot see.

    These answers name no competitor at all — they deny our product exists.
    Done-when #5 asks for both shapes; the name list only covers one.
    """
    from cowork.handlers.response_routing import denies_our_product

    for denial in (
        # trace dd8f8dc9 — the prospect evaluating Cowork for their own SaaS
        "I can't reliably identify a current public product called MindsHub CoWork "
        "from the name alone.",
        "There is no verified Cowork desktop app that I could find.",
        "I'm not familiar with a product called Cowork.",
        "I'm not aware of any tool called MindsHub Cowork.",
        "I've never heard of Cowork by MindsDB.",
        "I could not find any product by that name — MindsHub Cowork does not appear to exist.",
        "There's no such app as MindsHub Cowork that I know of.",
        # Attributive: the name sits INSIDE the denial clause as a modifier,
        # so it is consumed by the clause's filler and the before/after rule
        # cannot see it. At least as natural as the "called Cowork" form.
        "I could not find any official Cowork application.",
        "I couldn't find any Cowork app.",
        "I could not locate a Cowork installer.",
        "I was unable to verify the Cowork product.",
        "I couldn't find any MindsHub Cowork software anywhere.",
    ):
        assert denies_our_product(denial) is not None, denial


def test_denial_matcher_leaves_true_statements_about_cowork_alone():
    """The reason this matcher is narrow rather than "product name near a negation".

    That naive form was measured discarding 5 of these 6 — including the
    browser/desktop one, which is precisely the surface-aware answer ENG-2423
    exists to produce. Saying what Cowork cannot do is not denying Cowork.
    """
    from cowork.handlers.response_routing import denies_our_product

    for true_statement in (
        "Cowork doesn't support that file type yet.",
        "Cowork can't read local folders in the browser — that's desktop only.",
        "No, Cowork does not need an API key for that.",
        "MindsHub Cowork is not the same product as MindsDB's SQL engine.",
        "I can't run that without connecting a data source in Cowork first.",
        "There is no Linux beta for Cowork at the moment.",
        "Cowork can't open that file because the path doesn't exist.",
        "I couldn't find that setting in Cowork — try Settings → Agent Harness.",
        "Cowork can't identify the file encoding automatically; specify it.",
        "The Cowork desktop app is not available from the Mac App Store.",
        # These separate the attributive rule from the obvious version of it.
        # Anchoring on the bare product name instead of `<product> <noun>` fires
        # on all of them, and the first is a real answer shape.
        "I can't locate that file in your Cowork workspace.",
        "I cannot verify the checksum of the Cowork download you pasted.",
        "I couldn't find the Cowork log file you asked about.",
    ):
        assert denies_our_product(true_statement) is None, true_statement


@pytest.mark.asyncio
async def test_answer_denying_our_product_is_discarded_with_its_own_reason():
    """Countable apart from the naming shape: they fail differently and would
    be tuned separately."""
    provider = _Client(
        _response(
            content="I can't reliably identify a current public product called "
                    "MindsHub CoWork from the name alone."
        )
    )

    decision = await decide_route(
        history=[{"role": "user", "content": "what is MindsHub Cowork?"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
        binding=RouterBinding(provider=provider, model="minds-free", label="minds_cloud"),
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_answer_denied_product"
    assert decision.text == ""
    assert decision.fallback is False


@pytest.mark.parametrize(
    "answer",
    [
        # Telugu — the trace this whole guard exists for, WITHOUT the markdown
        # bold that was masking the bug. `**OpenAI**` matched under `\w`
        # boundaries; a bare suffix did not.
        "నన్ను OpenAIతయారు చేసింది",
        "OpenAIకి",
        "मुझे OpenAIने बनाया",
        "ChatGPTを使ってください",
        "저는 ChatGPT입니다",
        "ChatGPT的桌面应用",
    ],
)
def test_brand_name_is_caught_when_a_non_latin_suffix_abuts_it(answer):
    """`\\w` is Unicode-aware, and these scripts write suffixes against the name.

    Under `(?<![\\w-])` every one of these missed: the suffix character counts as
    a word character, so the lookaround sees no boundary. The guard built for a
    Telugu prod answer did not, in fact, handle Telugu — its own regression test
    passed only because that trace carried `**` around the brand.
    """
    from cowork.handlers.response_routing import names_foreign_product

    assert names_foreign_product(answer) is not None, answer


def test_ascii_boundary_still_rejects_the_pinned_negatives():
    """The boundary narrowed; the negatives it was protecting must still hold."""
    from cowork.handlers.response_routing import names_foreign_product

    for benign in (
        "Move the cursor to the end of the line.",
        "Llamas and alpacas are both camelids.",
        'Set OPENAI_API_KEY in your .env file.',
        'openai_api_key = "sk-..."',
        "You can pick grok in the model picker.",  # our own catalog alias
    ):
        assert names_foreign_product(benign) is None, benign


def test_denial_matcher_leaves_true_negative_capability_claims_alone():
    """The adjective branch was bare proximity: our name within 120 characters
    of "no <adjective>". These are ordinary correct answers about what Cowork
    does not have, and every one of them fired before the product-noun
    requirement was added."""
    from cowork.handlers.response_routing import denies_our_product

    for true_statement in (
        "Cowork has no official Slack integration yet.",
        "There is no public API for Cowork right now.",
        "Cowork has no known issues with that file type.",
        "MindsHub has no documented rate limit for that endpoint.",
    ):
        assert denies_our_product(true_statement) is None, true_statement


def test_the_adjective_denial_still_catches_the_prod_failure():
    """...without losing "no verified Cowork desktop app", which is one of the
    ticket's three real answers. The name sits inside the denial clause there,
    so it needs its own alternative — the same shape as the attributive rule."""
    from cowork.handlers.response_routing import denies_our_product

    assert denies_our_product("There is no verified Cowork desktop app that I could find.")
    assert denies_our_product("There's no such app as MindsHub Cowork that I know of.")
