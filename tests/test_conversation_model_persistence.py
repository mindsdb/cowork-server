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
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "harness": "hermes"},
    )
    assert r.status_code == 200, r.text
    conv_id = harness.calls[0]["conversation"].id

    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["harness"] == "hermes"


def test_unavailable_harness_pick_falls_back_to_the_account_default(client, harness):
    # A stale client cache (e.g. Hermes was uninstalled since the picker
    # last loaded) must never fail the turn.
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "harness": "not-a-real-harness"},
    )
    assert r.status_code == 200, r.text
    conv_id = harness.calls[0]["conversation"].id

    r2 = client.get(f"/api/v1/conversations/{conv_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["harness"] == "anton"
