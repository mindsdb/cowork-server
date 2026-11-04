"""The route gate and an open Chimera sidekick.

A sidekick left `open` handed its work back to the main agent (its last
report was needs_main or budget_exhausted). The next prompt most likely
answers it, and only the agent holds the report, so the turn goes to the
agent with no gate call, deterministically, as the other ineligible shapes do.
"""
from __future__ import annotations

import pytest

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.handlers.response_routing import DELEGATED_AGENTIC, decide_route, ineligible_reason
from cowork.handlers.responses import ResponsesHandler
from cowork.schemas.responses import ResponsesRequest
from cowork.services.conversations import ConversationService
from cowork.services.sidekick_contexts import SidekickContextService

_SHAPE = dict(has_non_text_input=False, has_attachments=False, has_disabled_connections=False)


def test_an_open_sidekick_is_an_ineligible_shape():
    assert ineligible_reason(**_SHAPE) is None
    assert ineligible_reason(**_SHAPE, has_active_sidekick=True) == "active_sidekick"


def test_the_other_shapes_keep_their_own_reasons():
    shape = {**_SHAPE, "has_attachments": True}
    assert ineligible_reason(**shape, has_active_sidekick=True) == "attachments_present"


@pytest.mark.asyncio
async def test_decide_route_delegates_without_a_gate_call():
    def _no_binding():
        raise AssertionError("the gate must not be resolved")

    decision = await decide_route(
        history=[{"role": "user", "content": "yes, go ahead"}],
        **_SHAPE,
        has_active_sidekick=True,
        binding_factory=_no_binding,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "active_sidekick"
    assert decision.fallback is False


@pytest.mark.asyncio
async def test_route_request_delegates_an_open_sidekicks_turn(monkeypatch):
    import cowork.handlers.responses as responses

    async def _gate(**_kwargs):
        raise AssertionError("the gate must not run")

    monkeypatch.setattr(responses, "decide_route", _gate)
    handler = ResponsesHandler()
    handler.harness_name = "anton"

    decision, turn_llm = await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "yes"}],
        has_attachments=False,
        has_disabled_connections=False,
        has_active_sidekick=True,
    )

    assert (decision.route, decision.reason) == (DELEGATED_AGENTIC, "active_sidekick")
    assert turn_llm is None


def _turn_start(conversation_id, *, model=None):
    handler = ResponsesHandler()
    handler.harness_name = "anton"
    with get_open_session() as db:
        return handler._read_turn_start(
            ScopedSession(db, LOCAL_SCOPE),
            request=ResponsesRequest(input="yes, go ahead", conversation=str(conversation_id), model=model),
            has_disabled_connections=False,
        )


def _conversation_with_sidekick(status: str | None):
    with get_open_session() as db:
        scoped = ScopedSession(db, LOCAL_SCOPE)
        conversation = ConversationService(scoped).create_conversation(topic="sidekick", model="chimera")
        if status is not None:
            SidekickContextService(scoped).save(conversation.id, {"sidekick_id": "sk", "messages": [], "status": status})
        return conversation.id


def test_the_first_unit_reads_an_open_sidekick_and_skips_the_gates_history():
    start = _turn_start(_conversation_with_sidekick("open"))

    assert start.active_sidekick is True
    assert start.gate_rows == []


@pytest.mark.parametrize("status", [None, "idle"])
def test_an_idle_or_absent_sidekick_leaves_the_gate_alone(status):
    start = _turn_start(_conversation_with_sidekick(status))

    assert start.active_sidekick is False


def test_a_failed_sidekick_read_routes_without_it(monkeypatch):
    def _boom(self, conversation):
        raise RuntimeError("no such table")

    monkeypatch.setattr(SidekickContextService, "is_open", _boom)

    assert _turn_start(_conversation_with_sidekick("open")).active_sidekick is False


def test_a_new_conversation_on_the_pseudo_model_stores_it_as_its_model():
    import uuid

    conversation_id = uuid.uuid4()
    _turn_start(conversation_id, model="chimera")

    with get_open_session() as db:
        stored = ConversationService(ScopedSession(db, LOCAL_SCOPE)).get_conversation(conversation_id)
        assert stored.model == "chimera"
