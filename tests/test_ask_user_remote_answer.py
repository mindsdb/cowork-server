"""POST /api/v1/responses/answer on the remote backend: the question lives in a pod."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import cowork.api.v1.endpoints.responses as endpoints
from cowork.api.v1.endpoints.responses import SharedTurn
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from cowork.turnqueue.answers import RemoteAnswerResult

CID = "conv-remote-answer"
QID = "ask:1"


@pytest.fixture(autouse=True)
def _remote_backend(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture()
def turn(monkeypatch):
    state = {"found": SharedTurn(index={"correlation_id": "corr-1", "turn_id": "1"},
                                 buffer=None, in_flight=True)}

    async def fake_shared_turn(conversation_id, scope):
        return state["found"] if conversation_id == CID else None

    monkeypatch.setattr(endpoints, "_shared_turn", fake_shared_turn)
    return state


@pytest.fixture()
def pod(monkeypatch):
    calls = []
    state = {"result": RemoteAnswerResult.ACCEPTED, "calls": calls}

    async def fake_submit(**kwargs):
        calls.append(kwargs)
        return state["result"]

    monkeypatch.setattr(endpoints, "submit_remote_answer", fake_submit)
    return state


def _post(client, conversation_id=CID, **body):
    return client.post("/api/v1/responses/answer",
                       json={"conversation_id": conversation_id, "question_id": QID, **body})


def test_accepted_answer_is_queued_for_the_turn(client, turn, pod):
    resp = _post(client, values=["pg"], text="and duckdb")
    assert resp.status_code == 200
    assert resp.json() == {"accepted": True}
    assert pod["calls"] == [{"conversation_id": CID, "correlation_id": "corr-1",
                             "question_id": QID,
                             "payload": {"values": ["pg"], "text": "and duckdb"}}]


def test_skip_is_queued(client, turn, pod):
    assert _post(client, skipped=True).status_code == 200
    assert pod["calls"][0]["payload"] == {"skipped": True}


@pytest.mark.parametrize("result,status,body", [
    (RemoteAnswerResult.INVALID_OPTION, 400, {"status": "invalid_option"}),
    (RemoteAnswerResult.NOT_FOUND, 404, {"status": "not_found"}),
    (RemoteAnswerResult.ALREADY_ANSWERED, 409, {"accepted": False, "status": "already_answered"}),
])
def test_pod_verdicts_map_to_statuses(client, turn, pod, result, status, body):
    pod["result"] = result
    resp = _post(client, values=["pg"])
    assert resp.status_code == status
    assert resp.json() == body


def test_unknown_conversation_is_not_found(client, turn, pod):
    resp = _post(client, conversation_id="someone-elses", values=["pg"])
    assert resp.status_code == 404
    assert pod["calls"] == []


def test_finished_turn_is_not_found(client, turn, pod):
    turn["found"] = turn["found"]._replace(in_flight=False)
    assert _post(client, values=["pg"]).status_code == 404
    assert pod["calls"] == []


@pytest.mark.parametrize("body,expected", [
    ({}, "empty_answer"),
    ({"skipped": True, "values": ["pg"]}, "ambiguous_answer"),
])
def test_malformed_body_is_rejected_before_queueing(client, turn, pod, body, expected):
    resp = _post(client, **body)
    assert resp.status_code == 400
    assert resp.json() == {"status": expected}
    assert pod["calls"] == []
