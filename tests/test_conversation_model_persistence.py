"""ENG-1656 follow-up: a conversation created via /responses with a
per-conversation model pick must persist that pick onto Conversation.model
(and its harness), so reopening the task later remembers it — matching how
claude-code tasks already carry model/harness (App.jsx's launchCodingModeTask).
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
