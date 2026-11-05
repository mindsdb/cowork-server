import asyncio
import json

import httpx
import pytest

from cowork.handlers import jev_route

TARGET = jev_route.JevTarget.mindshub("https://minds.example/v1/", "turn-key")
MESSAGES = [
    {"role": "user", "content": "Build a dashboard of our 2025 sales."},
    {"role": "assistant", "content": "[ran tool: query_datasource]\nRevenue was $4.81M."},
    {"role": "user", "content": "what were those headline numbers again?"},
]


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeAsyncClient:
    """Stands in for the shared httpx client; scripts one POST response or raises."""

    def __init__(self, *, response=None, exception=None, captured=None, delay=0.0):
        self._response = response
        self._exception = exception
        self._captured = captured if captured is not None else {}
        self._delay = delay

    async def post(self, url, *, headers, json, timeout):
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["json"] = json
        self._captured["timeout"] = timeout
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exception is not None:
            raise self._exception
        return self._response


def _body(choice="answer_directly", p_direct=0.97, p_search=0.1):
    answers = {"route": {
        "type": "choice", "choice": choice, "confidence": 0.9,
        "probabilities": {"answer_directly": p_direct, "needs_agent": 1 - p_direct},
    }}
    if p_search is not None:
        answers["search"] = {
            "type": "choice", "choice": "search_web" if p_search >= 0.5 else "no_search",
            "probabilities": {"search_web": p_search, "no_search": 1 - p_search},
        }
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 42, "output_tokens": 0}}


async def _decide(monkeypatch, client, *, min_p_direct=0.5, timeout_seconds=1.0, target=TARGET):
    monkeypatch.setattr(jev_route, "_http_client", lambda: client)
    return await jev_route.decide(
        target=target, messages=MESSAGES, model="jev",
        timeout_seconds=timeout_seconds, min_p_direct=min_p_direct,
    )


@pytest.mark.asyncio
async def test_decide_sends_the_route_question_with_the_latest_message_apart(monkeypatch):
    captured = {}
    result = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured))

    assert captured["url"] == "https://minds.example/v1/decisions"
    assert captured["json"] == {
        "model": "jev",
        "state": {
            "latest_user_message": "what were those headline numbers again?",
            "earlier_conversation": MESSAGES[:-1],
        },
        "questions": {"route": jev_route.ROUTE_QUESTION, "search": jev_route.SEARCH_QUESTION},
    }
    assert captured["timeout"] == 1.0
    assert result == jev_route.JevRoute(
        direct=True, p_direct=0.97, needs_search=False, ms=result.ms, model="jev-1.13.0",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("search", "needs_search"), [
    ({"choice": "search_web", "probabilities": {"search_web": 0.9, "no_search": 0.1}}, True),
    ({"choice": "no_search", "probabilities": {"search_web": 0.2, "no_search": 0.8}}, False),
    # Missing or malformed: assume a search, which only costs a longer wait.
    (None, True),
    ({"choice": "maybe", "probabilities": {"search_web": 0.1}}, True),
    ({"choice": "no_search", "probabilities": {}}, True),
])
async def test_decide_reads_whether_the_reply_needs_a_web_search(monkeypatch, search, needs_search):
    body = _body(p_search=None)
    if search is not None:
        body["answers"]["search"] = {"type": "choice", **search}
    result = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, body)))

    assert result.error == ""
    assert result.direct is True
    assert result.needs_search is needs_search


@pytest.mark.asyncio
async def test_decide_reuses_one_client_across_calls(monkeypatch):
    """One connection pool per event loop: a client per call paid a TLS
    handshake inside the gate's budget on every turn."""
    monkeypatch.setattr(jev_route, "_client", None)
    first = jev_route._http_client()
    try:
        assert jev_route._http_client() is first
    finally:
        await jev_route.close_client()
    assert first.is_closed
    second = jev_route._http_client()
    try:
        assert second is not first
    finally:
        await jev_route.close_client()


@pytest.mark.parametrize(("error", "falls_back"), [
    ("", False),
    ("http_403", True),
    ("http_402", True),
    ("http_500", True),
    ("transport_error", True),
    ("malformed_response", True),
    # The gate's time is spent: straight to the agent.
    ("timeout", False),
])
def test_which_failures_fall_back_to_the_llm_gate(error, falls_back):
    assert jev_route.JevRoute(error=error).falls_back_to_llm_gate is falls_back


@pytest.mark.asyncio
@pytest.mark.parametrize(("p_direct", "min_p_direct", "direct"), [
    (0.97, 0.5, True),
    (0.5, 0.5, True),
    (0.49, 0.5, False),
    (0.55, 0.8, False),
])
async def test_decide_routes_direct_only_at_or_above_the_threshold(monkeypatch, p_direct, min_p_direct, direct):
    """The probability decides, not the choice label: a direct answer needs
    Jev at least this sure, so the threshold can be tuned without Jev."""
    result = await _decide(
        monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body(p_direct=p_direct))),
        min_p_direct=min_p_direct,
    )

    assert result.direct is direct
    assert result.p_direct == p_direct
    assert result.error == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    {"choice": "maybe", "probabilities": {"answer_directly": 0.9}},
    {"choice": "answer_directly", "probabilities": {}},
    {"choice": "answer_directly"},
    {"choice": "answer_directly", "probabilities": {"answer_directly": True}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": "0.9"}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": 1.5}},
    {"choice": "answer_directly", "probabilities": {"answer_directly": float("nan")}},
])
async def test_decide_rejects_a_malformed_answer(monkeypatch, answer):
    body = {"answers": {"route": {"type": "choice", **answer}}}
    result = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, body)))

    assert result.error == "malformed_response"
    assert result.direct is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("client", "error"), [
    (_FakeAsyncClient(response=_FakeResponse(402, {})), "http_402"),
    (_FakeAsyncClient(exception=httpx.ConnectError("boom")), "transport_error"),
    # httpx's own timeouts fire at the same budget: no LLM gate after them.
    (_FakeAsyncClient(exception=httpx.ReadTimeout("slow")), "timeout"),
    (_FakeAsyncClient(exception=httpx.ConnectTimeout("slow")), "timeout"),
    (_FakeAsyncClient(exception=RuntimeError("boom")), "RuntimeError"),
    (_FakeAsyncClient(response=_FakeResponse(200, {"unexpected": True})), "malformed_response"),
])
async def test_decide_returns_failures_and_never_raises(monkeypatch, client, error):
    result = await _decide(monkeypatch, client)

    assert result.error == error
    assert result.direct is False


@pytest.mark.asyncio
async def test_decide_enforces_a_hard_wall_clock_timeout(monkeypatch):
    """httpx's own timeout is per phase and measures inactivity, not total
    elapsed time, so a response that trickles data forever would never trip
    it. The outer asyncio.timeout is what actually has to catch this."""
    started = asyncio.get_event_loop().time()
    result = await _decide(
        monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), delay=0.05), timeout_seconds=0.01,
    )
    elapsed = asyncio.get_event_loop().time() - started

    assert result.error == "timeout"
    assert elapsed < 0.05  # returned at the deadline, not after the slow response finished


async def _decide_headers(monkeypatch, context):
    """Run the real decision under `context` (or none) and return the headers it sent."""
    from anton.core.llm.tracing import reset_trace_context, set_trace_context

    captured = {}
    token = set_trace_context(context)
    try:
        await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured))
    finally:
        reset_trace_context(token)
    return captured["headers"]


@pytest.mark.asyncio
async def test_decide_attributes_its_trace_to_the_turn_it_routes(monkeypatch):
    """Without these headers the gateway stamps the call origin:direct-api,
    indistinguishable from the user calling Jev directly."""
    from anton.core.llm.tracing import TraceContext

    headers = await _decide_headers(
        monkeypatch,
        TraceContext(
            session_id="conv-1",
            harness="anton",
            surface="web",
            tags=("cowork-gate",),
            metadata={"cowork_server_version": "1.2.3", "correlation_id": "corr-1"},
        ),
    )

    assert headers["Authorization"] == "Bearer turn-key"
    # The user's own turn, so not hidden as a probe.
    assert "X-Minds-Request-Kind" not in headers
    # Never a session: a refused decision would mark the user's conversation
    # failed in the Sessions view, though the turn recovers by delegating.
    assert "Langfuse-Session-Id" not in headers
    assert headers["Langfuse-Tags"].split(",") == ["anton", "surface:web", jev_route.JEV_ROUTE_TAG]
    assert json.loads(headers["Langfuse-Metadata"]) == {
        "cowork_server_version": "1.2.3",
        "correlation_id": "corr-1",
        "harness": "anton",
        "surface": "web",
        "conversation_id": "conv-1",
    }


@pytest.mark.asyncio
async def test_decide_never_sends_turn_id(monkeypatch):
    """harness + turn_id makes the gateway name the trace "{harness}:turn-N",
    which would count every decision as a user turn."""
    from anton.core.llm.tracing import TraceContext

    headers = await _decide_headers(
        monkeypatch,
        TraceContext(session_id="conv-1", harness="anton", turn_id=7, metadata={"turn_id": "7"}),
    )

    assert "turn_id" not in json.loads(headers["Langfuse-Metadata"])


@pytest.mark.asyncio
async def test_decide_outside_a_turn_sends_only_its_credential(monkeypatch):
    assert await _decide_headers(monkeypatch, None) == {"Authorization": "Bearer turn-key"}


def _user(**fields):
    from cowork.common.settings.user_settings import UserSettings

    return UserSettings.model_validate(fields)


@pytest.mark.parametrize(("fields", "mode"), [
    ({"planning_provider": "minds_cloud", "minds_api_key": "k"}, "jev"),
    ({"planning_provider": "minds_cloud", "minds_api_key": "k", "jev_decisions_enabled": False}, "off"),
    ({"planning_provider": "anthropic", "anthropic_api_key": "k"}, "llm"),
    # The org's own decision model routes any provider, and the Jev opt-out
    # does not apply to it.
    ({"planning_provider": "anthropic", "anthropic_api_key": "k",
      "decision_model_url": "https://api.typesafe.ai/v1/systemone", "decision_model_api_key": "own"}, "own"),
    ({"planning_provider": "minds_cloud", "minds_api_key": "k", "jev_decisions_enabled": False,
      "decision_model_url": "https://api.typesafe.ai/v1/systemone", "decision_model_api_key": "own"}, "own"),
    # Half configured is not configured.
    ({"planning_provider": "anthropic", "anthropic_api_key": "k",
      "decision_model_url": "https://api.typesafe.ai/v1/systemone"}, "llm"),
])
def test_gate_mode(fields, mode):
    assert jev_route.gate_mode(_user(**fields)) == mode


def test_own_decision_target_unwraps_the_key_and_trims_the_url():
    settings = _user(decision_model_url=" https://api.typesafe.ai/v1/systemone ", decision_model_api_key="own-key")

    assert jev_route.own_decision_target(settings) == jev_route.JevTarget(
        url="https://api.typesafe.ai/v1/systemone", api_key="own-key",
    )


@pytest.mark.asyncio
async def test_an_own_decision_endpoint_never_receives_the_trace_headers(monkeypatch):
    """They carry user, organization and conversation ids, and mean something
    only to MindsHub's gateway."""
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    captured = {}
    token = set_trace_context(TraceContext(
        session_id="conv-1", harness="anton", metadata={"user_id": "u1", "organization_id": "o1"},
    ))
    try:
        await _decide(
            monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured),
            target=jev_route.JevTarget(url="https://api.typesafe.ai/v1/systemone", api_key="own"),
        )
    finally:
        reset_trace_context(token)

    assert captured["headers"] == {"Authorization": "Bearer own"}


_OWN = jev_route.JevTarget(url="https://decide.example.com/v1/systemone", api_key="own")


def _hosted(monkeypatch, addresses):
    monkeypatch.setattr(jev_route, "_hosted", lambda: True)
    monkeypatch.setattr(
        jev_route, "_RESOLVER",
        lambda host, port, type=None: [(None, None, None, None, (address, port)) for address in addresses],
    )


@pytest.mark.asyncio
async def test_a_hosted_server_dials_an_own_endpoint_at_its_vetted_address(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(host=request.url.host, header=request.headers["host"], sni=request.extensions.get("sni_hostname"))
        return httpx.Response(200, json=_body())

    _hosted(monkeypatch, ["93.184.216.34"])
    monkeypatch.setattr(jev_route, "_PINNED_TRANSPORT", httpx.MockTransport(handler))
    # The shared client is not used for an organization's own endpoint.
    monkeypatch.setattr(jev_route, "_http_client", lambda: pytest.fail("used the shared client"))

    route = await jev_route.decide(target=_OWN, messages=MESSAGES, model="jev", timeout_seconds=1.0, min_p_direct=0.5)

    assert route.direct and not route.error
    assert seen == {"host": "93.184.216.34", "header": "decide.example.com", "sni": "decide.example.com"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("url", "addresses"), [
    ("https://decide.example.com/v1/systemone", ["10.0.0.5"]),
    ("https://decide.example.com/v1/systemone", ["93.184.216.34", "169.254.169.254"]),
    ("http://decide.example.com/v1/systemone", ["93.184.216.34"]),
    ("https://127.0.0.1:8443/v1/systemone", []),
])
async def test_a_hosted_server_refuses_an_own_endpoint_it_must_not_dial(monkeypatch, url, addresses):
    """An organization admin sets the URL: an internal address must not
    receive a request every turn."""
    _hosted(monkeypatch, addresses)
    monkeypatch.setattr(jev_route, "_PINNED_TRANSPORT", httpx.MockTransport(lambda r: pytest.fail("dialed")))

    route = await jev_route.decide(
        target=jev_route.JevTarget(url=url, api_key="own"),
        messages=MESSAGES, model="jev", timeout_seconds=1.0, min_p_direct=0.5,
    )

    assert route.error == "endpoint_refused"
    assert route.falls_back_to_llm_gate


@pytest.mark.asyncio
async def test_mindshub_is_dialed_on_the_shared_client_when_hosted(monkeypatch):
    monkeypatch.setattr(jev_route, "_hosted", lambda: True)
    route = await _decide(monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body())))

    assert route.direct


@pytest.mark.asyncio
async def test_a_desktop_server_dials_an_own_endpoint_as_configured(monkeypatch):
    """On desktop the server is the user's own machine: a local endpoint is theirs to use."""
    monkeypatch.setattr(jev_route, "_hosted", lambda: False)
    captured = {}
    route = await _decide(
        monkeypatch, _FakeAsyncClient(response=_FakeResponse(200, _body()), captured=captured),
        target=jev_route.JevTarget(url="http://localhost:8000/v1/systemone", api_key="own"),
    )

    assert route.direct and captured["url"] == "http://localhost:8000/v1/systemone"
