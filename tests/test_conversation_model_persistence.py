"""ENG-1656 follow-up: a conversation created via /responses with a
per-conversation model and/or harness pick must persist those picks onto
Conversation.model/harness, so reopening the task later remembers them —
matching how claude-code tasks already carry model/harness (App.jsx's
launchCodingModeTask). The harness pick additionally overrides which
harness actually runs the turn (request.harness in ResponsesHandler.handle),
not just which value gets stored.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class _StubHarness:
    """Minimal harness: one text delta, then a clean end of turn. Captures
    the kwargs it was called with so the test can assert the override reached
    the harness call, not just the DB write."""

    id = "stub"

    def __init__(self):
        self.calls: list[dict] = []

    def stream_response(self, **kwargs):
        self.calls.append(kwargs)
        return None

    async def formatter(self, stream, model, event_sink):
        event_sink(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "ok"},
        )
        if False:
            yield


@pytest.fixture()
def harness():
    return _StubHarness()


@pytest.fixture()
def client(harness):
    from cowork.server import create_app

    with patch("cowork.handlers.responses.get_harness", return_value=harness):
        yield TestClient(create_app())


def test_new_conversation_persists_the_picked_model(client, harness):
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "model": "picked-model"},
    )
    assert r.status_code == 200, r.text
    assert harness.calls[0]["model"] == "picked-model"
    conv_id = harness.calls[0]["conversation"].id

    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["model"] == "picked-model"
    assert body["harness"] == "anton"


def test_new_conversation_without_a_model_pick_leaves_it_null(client, harness):
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False},
    )
    assert r.status_code == 200, r.text
    assert harness.calls[0]["model"] is None

    conv_id = harness.calls[0]["conversation"].id
    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["model"] is None


def test_per_conversation_harness_pick_overrides_the_account_default(client, harness):
    # get_harness is patched module-wide to always return the same stub
    # (see the `client` fixture), so this asserts the override via the
    # persisted Conversation.harness — the same signal
    # test_new_conversation_persists_the_picked_model uses for `model`.
    # The pick must name a registered harness, so register a second one.
    from cowork.harnesses.base import _registry, register

    @register
    class _Other:
        id = "other"
        label = "Other"

    try:
        r = client.post(
            "/api/v1/responses/",
            json={"input": "hello", "stream": False, "harness": "other"},
        )
    finally:
        _registry.pop("other", None)
    assert r.status_code == 200, r.text
    conv_id = harness.calls[0]["conversation"].id

    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["harness"] == "other"


def test_unavailable_harness_pick_falls_back_to_the_account_default(client, harness):
    # A stale client cache (a harness removed since the picker last loaded)
    # must never fail the turn.
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "harness": "not-a-real-harness"},
    )
    assert r.status_code == 200, r.text
    conv_id = harness.calls[0]["conversation"].id

    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["harness"] == "anton"


def test_conversation_from_a_removed_harness_still_opens_and_continues(client, harness):
    # Rows tagged with a harness that no longer exists (Hermes) are history:
    # they list, open, keep their tag, and the next turn runs the default.
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.models.message import Message
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    try:
        conv = ConversationService(ScopedSession(session, LOCAL_SCOPE)).create_conversation(
            topic="old", harness="hermes"
        )
        session.add(Message(conversation_id=conv.id, role="user", content="hi", harness="hermes", seq=1))
        session.add(Message(conversation_id=conv.id, role="assistant", content="hello", harness="hermes", seq=2))
        session.commit()
        conv_id = str(conv.id)
    finally:
        session.close()

    r = client.get(f"/api/v1/conversations/{conv_id}")
    assert r.status_code == 200, r.text
    assert r.json()["harness"] == "hermes"

    items = client.get(f"/api/v1/conversations/{conv_id}/items")
    assert items.status_code == 200, items.text
    assert [i["harness"] for i in items.json()] == ["hermes", "hermes"]

    r = client.post(
        "/api/v1/responses/",
        json={"input": "again", "stream": False, "conversation": conv_id},
    )
    assert r.status_code == 200, r.text
    assert str(harness.calls[-1]["conversation"].id) == conv_id

    items = client.get(f"/api/v1/conversations/{conv_id}/items").json()
    assert [i["harness"] for i in items[:2]] == ["hermes", "hermes"]
    assert len(items) > 2 and all(i.get("harness") != "hermes" for i in items[2:])
