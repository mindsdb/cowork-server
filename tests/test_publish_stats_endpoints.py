"""Desktop stats routes under /api/v1/publish."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import publish as publish_ep
from cowork.services import artifact_stats

STATS_BODY = {
    "tracking": True,
    "period_days": 30,
    "total_views": 3,
    "last_viewed": "2026-10-19T10:22:31Z",
    "views_in_period": 3,
    "daily": [{"date": "2026-10-19", "views": 3}],
    "unique_viewers": 0,
    "viewers": [],
}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    return TestClient(app)


def _get(path: str = "/art/rep/report.html"):
    return _client().get("/api/v1/publish/stats", params={"path": path})


def _put(path: str = "/art/rep/report.html", enabled: bool = False):
    return _client().put("/api/v1/publish/stats/tracking", json={"path": path, "enabled": enabled})


@pytest.fixture
def credential(monkeypatch):
    monkeypatch.setattr(publish_ep, "_desktop_credential", lambda: ("desk-key", "https://view.example"))


@pytest.fixture
def published(monkeypatch):
    seen = []

    def resolve(raw_path, *, scope):
        seen.append(raw_path)
        return "rid-1"

    monkeypatch.setattr(artifact_stats, "report_id_for_path", resolve)
    return seen


def _upstream(monkeypatch, handler):
    requests = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(
        artifact_stats, "_new_client", lambda: httpx.Client(transport=httpx.MockTransport(record)),
    )
    return requests


def test_get_returns_the_service_json_unchanged(monkeypatch, credential, published):
    requests = _upstream(monkeypatch, lambda request: httpx.Response(200, json=STATS_BODY))

    res = _get()

    assert res.status_code == 200, res.text
    assert res.json() == STATS_BODY
    assert published == ["/art/rep/report.html"]
    assert str(requests[0].url) == "https://view.example/stats/rid-1"
    assert requests[0].headers["Authorization"] == "Bearer desk-key"


def test_put_sets_tracking_and_returns_the_flag(monkeypatch, credential, published):
    requests = _upstream(monkeypatch, lambda request: httpx.Response(200, json={"tracking": False}))

    res = _put(enabled=False)

    assert res.status_code == 200, res.text
    assert res.json() == {"tracking": False}
    assert published == ["/art/rep/report.html"]
    assert requests[0].method == "PUT"
    assert str(requests[0].url) == "https://view.example/stats/rid-1/tracking"
    assert json.loads(requests[0].content) == {"enabled": False}


def test_put_rejects_a_non_boolean_flag(credential, published):
    res = _client().put("/api/v1/publish/stats/tracking", json={"path": "/a", "enabled": "yes"})
    assert res.status_code == 422
    assert published == []


@pytest.mark.parametrize("call", [_get, _put])
def test_missing_publish_key_is_400_with_its_code(monkeypatch, published, call):
    def no_key():
        raise ValueError("Configure your provider API key in Settings before publishing")

    monkeypatch.setattr(publish_ep, "_desktop_credential", no_key)

    res = call()

    assert res.status_code == 400
    assert res.json()["detail"] == {
        "code": "publish_credential_missing",
        "message": "Configure your provider API key in Settings before publishing",
    }
    assert published == []


@pytest.mark.parametrize("call", [_get, _put])
def test_an_invalid_path_is_a_plain_400_without_the_credential_code(credential, call):
    res = call(path="bad\x00path")

    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid artifact path"


@pytest.mark.parametrize("call", [_get, _put])
def test_rejected_publish_key_is_502_with_the_reconnect_code(monkeypatch, credential, published, call):
    _upstream(monkeypatch, lambda request: httpx.Response(401, json={"error": "Invalid token"}))

    res = call()

    assert res.status_code == 502
    assert res.json()["detail"] == {
        "code": "publish_credential_rejected",
        "message": "Reconnect MindsHub to see stats.",
    }


def test_unpublished_artifact_is_404(monkeypatch, credential):
    def not_published(raw_path, *, scope):
        raise FileNotFoundError("Artifact is not published")

    monkeypatch.setattr(artifact_stats, "report_id_for_path", not_published)

    res = _get()

    assert res.status_code == 404
    assert res.json()["detail"] == "Artifact is not published"


@pytest.mark.parametrize(
    "status,body,expected_status,expected_detail",
    [
        (404, {"code": "artifact_not_found"}, 409,
         {"code": "stats_owned_by_other_account", "message": "Stats are not available for this account."}),
        (404, {"code": "stats_unavailable"}, 404,
         {"code": "stats_unavailable", "message": "Stats are not available for this artifact."}),
        (403, {"code": "permission_denied", "error": "No."}, 403,
         {"code": "permission_denied", "message": "Your current role does not allow this action."}),
        (403, {"code": "organization_inactive", "error": "This organization is archived or still being provisioned."}, 403,
         {"code": "organization_inactive", "message": "This organization is archived or still being provisioned."}),
        (402, {"code": "upgrade_required", "error": "Upgrade required"}, 402,
         {"code": "upgrade_required", "error": "Upgrade required", "message": "Upgrade required"}),
        (400, {"error": "enabled must be a boolean"}, 400, "enabled must be a boolean"),
        (503, {"error": "Auth service unreachable"}, 503,
         {"code": "stats_temporarily_unavailable", "message": "Stats are temporarily unavailable."}),
        (500, None, 502, "Stats request failed (HTTP 500)"),
    ],
)
def test_service_errors_reach_the_client_with_the_shared_mapping(
    monkeypatch, credential, published, status, body, expected_status, expected_detail,
):
    _upstream(
        monkeypatch,
        lambda request: httpx.Response(status) if body is None else httpx.Response(status, json=body),
    )

    res = _get()

    assert res.status_code == expected_status
    assert res.json()["detail"] == expected_detail
