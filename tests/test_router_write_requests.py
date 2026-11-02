"""The router hands a change, an approval, a yes-or-no reply or any reply to its
own answer to the agent, and never ships an answer that reads as an approval
card or as finished work.

Only the agent can act, and only its write check shows the card a person has to
press. A router answer to such a turn can only imitate it, and a "yes" typed
under that imitation would reach the router again.
"""
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    DIRECT_HARNESS,
    RouteDecision,
    decide_route,
)
from tests.test_response_routing import _StreamProvider, _binding, _complete, _gate_spy, _no_gate, _text

CASES = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "response_routing" / "write_and_approval.yaml").read_text(encoding="utf-8")
)


async def _route(history, *, after_router_answer=False):
    return await decide_route(
        history=history, has_non_text_input=False, has_attachments=False,
        has_disabled_connections=False, after_router_answer=after_router_answer,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", CASES["delegates_without_router_call"] + CASES["known_over_fires"])
async def test_a_change_approval_or_bare_reply_goes_to_the_agent_without_a_router_call(monkeypatch, prompt):
    _no_gate(monkeypatch)
    decision = await _route([{"role": "user", "content": prompt}])
    assert decision == RouteDecision(route=DELEGATED_AGENTIC, reason="write_or_approval_request")


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", CASES["reaches_router"])
async def test_small_talk_and_general_questions_still_reach_the_router(monkeypatch, prompt):
    seen = _gate_spy(monkeypatch)
    decision = await _route([{"role": "user", "content": prompt}])
    assert decision.route == DIRECT_CONTEXT and seen


@pytest.mark.asyncio
async def test_only_the_latest_prompt_is_read_for_change_words(monkeypatch):
    seen = _gate_spy(monkeypatch)
    decision = await _route([
        {"role": "user", "content": "Delete the draft orders from last week"},
        {"role": "assistant", "content": "Removed 12 draft orders."},
        {"role": "user", "content": "thanks!"},
    ])
    assert decision.route == DIRECT_CONTEXT and seen


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", CASES["rejected_answers"])
async def test_a_router_answer_imitating_an_approval_or_finished_work_is_not_shipped(answer):
    """Streamed in two pieces, through the real gate, so the check reads the whole answer."""
    provider = _StreamProvider([_text(answer[:10]), _text(answer[10:]), _complete()])
    decision = await decide_route(
        history=[{"role": "user", "content": "hello there"}],
        has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
        binding=_binding(provider),
    )
    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_answer_imitated_approval"
    assert decision.text == ""
    assert decision.fallback is False


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", CASES["kept_answers"])
async def test_a_plain_router_answer_is_shipped(answer):
    provider = _StreamProvider([_text(answer), _complete()])
    decision = await decide_route(
        history=[{"role": "user", "content": "hello there"}],
        has_non_text_input=False, has_attachments=False, has_disabled_connections=False,
        binding=_binding(provider),
    )
    assert decision.route == DIRECT_CONTEXT
    assert decision.text == answer


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES["follow_ups"], ids=lambda case: case["reply"])
async def test_a_reply_to_a_router_answer_goes_to_the_agent(monkeypatch, case):
    _no_gate(monkeypatch)
    decision = await _route([
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": case["router_answer"]},
        {"role": "user", "content": case["reply"]},
    ], after_router_answer=True)
    assert decision == RouteDecision(route=DELEGATED_AGENTIC, reason="follow_up_to_router_answer")


@pytest.mark.asyncio
async def test_a_reply_to_the_agents_imitation_is_not_a_router_follow_up(monkeypatch):
    """The follow-up rule reads only the router's own answers; the agent's text is its own business."""
    seen = _gate_spy(monkeypatch)
    decision = await _route([
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "**Question card:** Proceed? Please answer yes or no"},
        {"role": "user", "content": "both of them please"},
    ])
    assert decision.route == DIRECT_CONTEXT and seen


@pytest.mark.asyncio
@pytest.mark.parametrize("harnesses, expected", [
    ((DIRECT_HARNESS,), True),
    (("anton",), False),
    ((DIRECT_HARNESS, "anton"), False),
    (("anton", DIRECT_HARNESS), True),
])
async def test_route_request_says_whether_the_last_answer_was_the_routers(monkeypatch, harnesses, expected):
    """Rows as `_read_gate_rows` returns them, oldest first: the harness is on the
    row, not in the history sent, and only the newest answer's counts."""
    import cowork.handlers.responses as responses
    from cowork.models.message import Message
    from cowork.models.message_event import MessageEvent  # noqa: F401 — resolves the ORM relationship
    from cowork.schemas.responses import Role

    seen = {}

    async def fake_decide_route(**kwargs):
        seen.update(kwargs)
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)
    conversation_id = uuid4()
    rows = []
    for harness in harnesses:
        rows += [
            Message(conversation_id=conversation_id, role=Role.user, content="hello there"),
            Message(conversation_id=conversation_id, role=Role.assistant, content="Hello!", harness=harness),
        ]
    handler = responses.ResponsesHandler()
    handler.harness_name = "anton"
    await handler._route_request(
        conversation_id=conversation_id,
        gate_rows=rows,
        harness_input=[{"type": "text", "text": "yes"}],
        has_attachments=False,
        has_disabled_connections=False,
    )
    assert seen["after_router_answer"] is expected


@pytest.mark.asyncio
async def test_a_typed_yes_under_a_saved_router_answer_reaches_the_agent_end_to_end(monkeypatch):
    """The reported sequence: a change request answered by the router before
    this fix, then a typed "yes". The decision and its reason come from the real
    `decide_route`, fed by `_route_request`."""
    import cowork.handlers.responses as responses
    from cowork.models.message import Message
    from cowork.models.message_event import MessageEvent  # noqa: F401 — resolves the ORM relationship
    from cowork.schemas.responses import Role

    _no_gate(monkeypatch)

    async def no_binding(self):
        return None, None

    monkeypatch.setattr(responses.ResponsesHandler, "_router_binding", no_binding)
    conversation_id = uuid4()
    rows = [
        Message(conversation_id=conversation_id, role=Role.user,
                content="Create two sandbox scenarios, and ask me to confirm with a question card first"),
        Message(conversation_id=conversation_id, role=Role.assistant,
                content="**Question card:** Proceed? Please answer yes or no", harness=DIRECT_HARNESS),
    ]
    handler = responses.ResponsesHandler()
    handler.harness_name = "anton"
    decision, _ = await handler._route_request(
        conversation_id=conversation_id,
        gate_rows=rows,
        harness_input=[{"type": "text", "text": "yes"}],
        has_attachments=False,
        has_disabled_connections=False,
    )
    assert decision == RouteDecision(route=DELEGATED_AGENTIC, reason="follow_up_to_router_answer")


def test_the_router_prompt_tells_it_to_hand_these_turns_over():
    from cowork.handlers.response_routing import _SYSTEM_PROMPT

    prompt = " ".join(_SYSTEM_PROMPT.split())
    assert "Delegate any request to change, send, confirm or approve something" in prompt
    assert "never say that something was done" in prompt


def test_a_whole_word_never_matches_inside_another_word():
    from cowork.handlers.response_routing import imitates_approval, names_change_or_approval

    assert names_change_or_approval("What is an address book?") is None  # "add"
    assert names_change_or_approval("Who wrote Hamlet?") is None  # "write"
    assert imitates_approval("That is a sentence about cards.") is None  # "sent", "card"
    assert names_change_or_approval("SET   UP the report") == "set up"
