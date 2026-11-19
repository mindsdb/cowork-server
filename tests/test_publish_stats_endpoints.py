"""Desktop stats routes under /api/v1/publish."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import publish as publish_ep
from cowork.services import artifact_stats
from tests._fakes import StatsUpstream

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


def _viewer(path: str = "/art/rep/report.html", viewer_id: str = "v_1"):
    return _client().get(f"/api/v1/publish/stats/viewers/{viewer_id}", params={"path": path})


def _delete(path: str = "/art/rep/report.html"):
    return _client().delete("/api/v1/publish/stats", params={"path": path})


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


def _upstream(monkeypatch, status, body=None):
    return StatsUpstream(status, body).install(monkeypatch)


def test_get_returns_the_service_json_unchanged(monkeypatch, credential, published):
    upstream = _upstream(monkeypatch, 200, STATS_BODY)

    res = _get()

    assert res.status_code == 200, res.text
    assert res.json() == STATS_BODY
    assert published == ["/art/rep/report.html"]
    assert upstream.calls[0].url == "https://view.example/a/rid-1/_stats"
    assert upstream.calls[0].api_key == "desk-key"


def test_put_sets_tracking_and_returns_the_flag(monkeypatch, credential, published):
    upstream = _upstream(monkeypatch, 200, {"tracking": False})

    res = _put(enabled=False)

    assert res.status_code == 200, res.text
    assert res.json() == {"tracking": False, "require_email": False}
    assert published == ["/art/rep/report.html"]
    assert upstream.calls[0].method == "PUT"
    assert upstream.calls[0].url == "https://view.example/a/rid-1/_stats/tracking"
    assert json.loads(upstream.calls[0].payload) == {"tracking": False, "enabled": False}


def test_put_rejects_a_non_boolean_flag(credential, published):
    res = _client().put("/api/v1/publish/stats/tracking", json={"path": "/a", "enabled": "yes"})
    assert res.status_code == 422
    assert published == []


@pytest.mark.parametrize("call", [_get, _put, _viewer, _delete])
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


@pytest.mark.parametrize("call", [_get, _put, _viewer, _delete])
def test_an_invalid_path_is_a_plain_400_without_the_credential_code(credential, call):
    res = call(path="bad\x00path")

    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid artifact path"


@pytest.mark.parametrize("call", [_get, _put, _viewer, _delete])
def test_rejected_publish_key_is_502_with_the_reconnect_code(monkeypatch, credential, published, call):
    _upstream(monkeypatch, 401, {"error": "Invalid token"})

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
    _upstream(monkeypatch, status, body)

    res = _get()

    assert res.status_code == expected_status
    assert res.json()["detail"] == expected_detail


# -- viewer analytics (ENG-4455) ----------------------------------------------


def test_get_passes_the_analytics_fields_through(monkeypatch, credential, published):
    body = dict(STATS_BODY, require_email=True, analytics={
        "visits": 2, "avg_seconds": 84.5, "completion": 0.5,
        "sections": [{"key": "s1", "index": 1, "label": "Intro", "avg_seconds": 21.0, "reach": 1.0}],
        "viewers": [{"id": "v_1", "kind": "anonymous", "email": None, "visits": 2, "total_seconds": 169,
                     "first_seen": "2026-10-19T10:00:00Z", "last_seen": "2026-10-19T11:00:00Z", "furthest": 1}],
    })
    _upstream(monkeypatch, 200, body)

    res = _get()

    assert res.status_code == 200
    assert res.json() == body


def test_put_sets_the_email_gate_alone(monkeypatch, credential, published):
    upstream = _upstream(monkeypatch, 200, {"tracking": True, "require_email": True})

    res = _client().put(
        "/api/v1/publish/stats/tracking", json={"path": "/art/rep/report.html", "require_email": True},
    )

    assert res.status_code == 200, res.text
    assert res.json() == {"tracking": True, "require_email": True}
    assert json.loads(upstream.calls[0].payload) == {"require_email": True}


def test_put_with_no_setting_is_422(credential, published):
    res = _client().put("/api/v1/publish/stats/tracking", json={"path": "/a"})
    assert res.status_code == 422
    assert published == []


def test_put_with_a_non_boolean_email_gate_is_422(credential, published):
    res = _client().put("/api/v1/publish/stats/tracking", json={"path": "/a", "require_email": "yes"})
    assert res.status_code == 422
    assert published == []


def test_credits_required_reaches_the_client_as_402(monkeypatch, credential, published):
    body = {"code": "credits_required", "message": "Add credits to collect viewer analytics."}
    _upstream(monkeypatch, 402, body)

    res = _put(enabled=True)

    assert res.status_code == 402
    assert res.json()["detail"] == body


def test_tracking_off_reaches_the_client_as_422(monkeypatch, credential, published):
    _upstream(monkeypatch, 422, {"code": "tracking_off"})

    res = _client().put(
        "/api/v1/publish/stats/tracking", json={"path": "/art/rep/report.html", "require_email": True},
    )

    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "tracking_off"


def test_viewer_route_returns_the_service_json(monkeypatch, credential, published):
    body = {"viewer": {"id": "v_1", "kind": "email", "email": "a@b.com"}, "visits": []}
    upstream = _upstream(monkeypatch, 200, body)

    res = _viewer(viewer_id="v_1")

    assert res.status_code == 200, res.text
    assert res.json() == body
    assert upstream.calls[0].url == "https://view.example/a/rid-1/_stats/viewers/v_1"
    assert published == ["/art/rep/report.html"]


def test_unknown_viewer_is_404(monkeypatch, credential, published):
    _upstream(monkeypatch, 404, {"code": "viewer_not_found", "error": "No such viewer"})

    res = _viewer(viewer_id="v_nope")

    assert res.status_code == 404
    assert res.json()["detail"] == {"code": "viewer_not_found", "message": "No such viewer"}


def test_delete_purges_the_analytics(monkeypatch, credential, published):
    upstream = _upstream(monkeypatch, 204, None)

    res = _delete()

    assert res.status_code == 204
    assert res.content == b""
    assert upstream.calls[0].method == "DELETE"
    assert upstream.calls[0].url == "https://view.example/a/rid-1/_stats"
