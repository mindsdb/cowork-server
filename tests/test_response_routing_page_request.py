"""A request to open, go to or show a page goes to the agent, on both routers.

Only the agent has the shared browser. A router answer to "open this page"
reads the page with a fetch the person never sees, or says it can't, so the
browser panel never opens.
"""

import pytest

from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    RouteDecision,
    RouterBinding,
    decide_route,
    names_page_request,
)

PAGE_REQUESTS = [
    "Open https://checkip.amazonaws.com and tell me the IP address it shows.",
    "Open https://mindshub.ai and tell me what the page says",
    "go to example.com",
    "Can you navigate to www.wikipedia.org?",
    "show me this page",
    "Pull up the pricing page on stripe.com",
    "Visit news.ycombinator.com and list the top story",
    "Take me to the website for the Louisville Zoo",
    "Open the first link on that page",
]

NOT_PAGE_REQUESTS = [
    "Hi",
    "thanks!",
    "explain a p-value",
    "What are the open issues in our plan?",
    "Is the store open on Sundays?",
    "Show me how a p-value works",
    "What's the capital of France?",
]


def _no_gate(monkeypatch):
    import cowork.handlers.response_routing as routing

    async def gate_must_not_run(binding, *, history):
        raise AssertionError("the gate must not be called")

    monkeypatch.setattr(routing, "_gate", gate_must_not_run)
    monkeypatch.setattr(routing, "_settings_binding", lambda: (_ for _ in ()).throw(AssertionError("no binding needed")))


def _gate_spy(monkeypatch):
    import cowork.handlers.response_routing as routing

    seen = []

    async def fake_gate(binding, *, history):
        seen.append(history)
        return "Hello!"

    monkeypatch.setattr(routing, "_gate", fake_gate)
    monkeypatch.setattr(routing, "_settings_binding", lambda: RouterBinding(provider=object(), model="m", label="x"))
    return seen


def _jev_must_not_run():
    async def ask(_messages):
        pytest.fail("Jev must not be asked")

    return ask


def _jev_answers(**fields):
    from cowork.handlers import jev_route

    async def ask(_messages):
        return jev_route.JevRoute(**fields)

    return ask


async def _route_turn(prompt, **kwargs):
    return await decide_route(
        history=[{"role": "user", "content": prompt}],
        has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("on_jev", [False, True], ids=["llm_gate", "jev"])
@pytest.mark.parametrize("prompt", PAGE_REQUESTS)
async def test_a_page_request_goes_to_the_agent_without_a_router_call(monkeypatch, on_jev, prompt):
    _no_gate(monkeypatch)
    decision = await _route_turn(prompt, jev=_jev_must_not_run() if on_jev else None)
    assert decision == RouteDecision(route=DELEGATED_AGENTIC, reason="page_request")


@pytest.mark.parametrize("prompt", PAGE_REQUESTS)
def test_page_requests_are_named(prompt):
    assert names_page_request(prompt) is not None


@pytest.mark.parametrize("prompt", NOT_PAGE_REQUESTS)
def test_questions_without_a_page_are_not_page_requests(prompt):
    assert names_page_request(prompt) is None


@pytest.mark.asyncio
async def test_a_greeting_still_gets_a_direct_answer_on_jev(monkeypatch):
    _no_gate(monkeypatch)
    decision = await _route_turn("Hi", jev=_jev_answers(direct=True, p_direct=0.95, needs_search=False))
    assert (decision.route, decision.reason) == (DIRECT_CONTEXT, "jev_direct")


@pytest.mark.asyncio
async def test_a_greeting_still_gets_a_direct_answer_on_the_llm_gate(monkeypatch):
    seen = _gate_spy(monkeypatch)
    decision = await _route_turn("Hi")
    assert decision.reason == "router_direct_response" and seen


@pytest.mark.asyncio
async def test_a_change_word_keeps_its_own_reason_beside_a_page_request(monkeypatch):
    _no_gate(monkeypatch)
    decision = await _route_turn("Open example.com and submit the form")
    assert decision.reason == "write_or_approval_request"
