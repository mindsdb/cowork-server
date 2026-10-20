"""The publishing service's stats client, on httpx.MockTransport."""
from __future__ import annotations

import json
import ssl
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from cowork.db.scoped import LOCAL_SCOPE
from cowork.services import artifact_stats as stats
from cowork.services.product_permissions import ProductPermissionDenied
from cowork.services.publish import PublisherUnavailable

PUBLISH_URL = "https://view.example/"
STATS_BODY = {
    "tracking": True,
    "period_days": 30,
    "total_views": 2,
    "last_viewed": "2026-10-19T10:22:31Z",
    "views_in_period": 2,
    "daily": [{"date": "2026-10-19", "views": 2}],
    "unique_viewers": 0,
    "viewers": [],
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _respond(status: int, body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, json=body)

    return handler


def _call(kind: str, client: httpx.Client):
    if kind == "fetch":
        return stats.fetch_stats(publish_url=PUBLISH_URL, key="k-1", report_id="rid-1", client=client)
    return stats.set_tracking(
        publish_url=PUBLISH_URL, key="k-1", report_id="rid-1", enabled=True, client=client,
    )


# -- requests -----------------------------------------------------------------


def test_fetch_sends_the_publish_credential_and_returns_the_body_unchanged():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=STATS_BODY)

    out = stats.fetch_stats(publish_url=PUBLISH_URL, key="k-1", report_id="rid-1", client=_client(handler))

    request = seen["request"]
    assert out == STATS_BODY
    assert request.method == "GET"
    assert str(request.url) == "https://view.example/stats/rid-1"
    assert request.headers["Authorization"] == "Bearer k-1"
    assert "Anton/1.0" in request.headers["User-Agent"]


def test_set_tracking_puts_the_flag_and_returns_the_stored_value():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"tracking": False})

    out = stats.set_tracking(
        publish_url=PUBLISH_URL, key="k-1", report_id="rid-1", enabled=False, client=_client(handler),
    )

    request = seen["request"]
    assert out is False
    assert request.method == "PUT"
    assert str(request.url) == "https://view.example/stats/rid-1/tracking"
    assert request.headers["Authorization"] == "Bearer k-1"
    assert json.loads(request.content) == {"enabled": False}


def test_report_id_is_escaped_into_one_path_segment():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=STATS_BODY)

    stats.fetch_stats(publish_url=PUBLISH_URL, key="k-1", report_id="a/b", client=_client(handler))

    assert seen["url"] == "https://view.example/stats/a%2Fb"


def test_tls_verification_follows_the_anton_setting(monkeypatch):
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", "false")
    assert stats._tls_verify() is False
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", "true")
    assert isinstance(stats._tls_verify(), ssl.SSLContext)
    monkeypatch.delenv("ANTON_MINDS_SSL_VERIFY")
    assert isinstance(stats._tls_verify(), ssl.SSLContext)


@pytest.mark.parametrize("value", ["", "  ", "TRUE", "1", "yes", "maybe"])
def test_tls_verification_stays_on_unless_explicitly_off(monkeypatch, value):
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", value)
    assert isinstance(stats._tls_verify(), ssl.SSLContext)


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " OFF "])
def test_tls_verification_turns_off_only_for_off_values(monkeypatch, value):
    monkeypatch.setenv("ANTON_MINDS_SSL_VERIFY", value)
    assert stats._tls_verify() is False


# -- response mapping ---------------------------------------------------------


def test_typed_errors_stay_out_of_the_generic_route_branches():
    assert not issubclass(stats.StatsError, (RuntimeError, ValueError, OSError))


@pytest.mark.parametrize("kind", ["fetch", "set"])
@pytest.mark.parametrize(
    "status,body,error",
    [
        (404, {"code": "artifact_not_found"}, stats.StatsOwnedByOtherAccount),
        (404, {"code": "stats_unavailable"}, stats.StatsUnavailable),
        (404, {"message": "Not Found"}, RuntimeError),
        (404, None, RuntimeError),
        (403, {"code": "permission_denied", "error": "You do not have permission to manage artifacts."}, ProductPermissionDenied),
        (403, {"code": "organization_inactive", "error": "This organization is archived."}, stats.StatsForbidden),
        (403, None, stats.StatsForbidden),
        (402, {"code": "upgrade_required", "error": "Upgrade required"}, stats.StatsUpgradeRequired),
        (401, {"error": "Invalid token"}, stats.PublishCredentialRejected),
        (400, {"error": "enabled must be a boolean"}, ValueError),
        (503, {"error": "Auth service unreachable"}, stats.StatsTemporarilyUnavailable),
        (503, None, stats.StatsTemporarilyUnavailable),
        (302, None, RuntimeError),
        (409, {"error": "conflict"}, RuntimeError),
        (500, None, RuntimeError),
        (502, None, RuntimeError),
    ],
)
def test_service_statuses_map_to_typed_errors(kind, status, body, error):
    with pytest.raises(error):
        _call(kind, _client(_respond(status, body)))


def test_forbidden_keeps_the_service_code_and_message():
    body = {"code": "organization_inactive", "error": "This organization is archived."}
    with pytest.raises(stats.StatsForbidden) as excinfo:
        _call("fetch", _client(_respond(403, body)))
    assert excinfo.value.code == "organization_inactive"
    assert excinfo.value.message == "This organization is archived."


def test_forbidden_without_a_code_gets_a_generic_one():
    with pytest.raises(stats.StatsForbidden) as excinfo:
        _call("fetch", _client(_respond(403, None)))
    assert excinfo.value.code == "forbidden"
    assert excinfo.value.message == "Access to stats was refused."


def test_upgrade_required_carries_the_service_body():
    body = {"code": "upgrade_required", "error": "Upgrade required", "required_product": "mindsdb-pro-hub"}
    with pytest.raises(stats.StatsUpgradeRequired) as excinfo:
        _call("fetch", _client(_respond(402, body)))
    assert excinfo.value.body == body


def test_bad_request_carries_the_service_message():
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        _call("set", _client(_respond(400, {"error": "enabled must be a boolean"})))


@pytest.mark.parametrize("kind", ["fetch", "set"])
def test_network_failure_is_a_runtime_error(kind):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(RuntimeError):
        _call(kind, _client(handler))


@pytest.mark.parametrize("kind", ["fetch", "set"])
def test_an_unreadable_success_body_is_a_runtime_error_not_a_bad_request(kind):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(RuntimeError):
        _call(kind, _client(handler))


def test_tracking_reply_without_a_boolean_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        _call("set", _client(_respond(200, {"tracking": "yes"})))


def test_fetch_reply_that_is_not_an_object_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        _call("fetch", _client(_respond(200, [1, 2, 3])))


# -- route mapping ------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,desktop,status,detail",
    [
        (stats.StatsOwnedByOtherAccount(), True, 409,
         {"code": "stats_owned_by_other_account", "message": "Stats are not available for this account."}),
        (stats.StatsUnavailable(), False, 404,
         {"code": "stats_unavailable", "message": "Stats are not available for this artifact."}),
        (stats.StatsForbidden("organization_inactive", "Archived."), True, 403,
         {"code": "organization_inactive", "message": "Archived."}),
        (stats.StatsUpgradeRequired({"code": "upgrade_required", "error": "Upgrade required"}), True, 402,
         {"code": "upgrade_required", "error": "Upgrade required", "message": "Upgrade required"}),
        (stats.StatsUpgradeRequired({"error": {"detail": "x"}}), True, 402,
         {"error": {"detail": "x"}, "code": "upgrade_required", "message": "Upgrade required"}),
        (stats.StatsUpgradeRequired({"error": ["x"], "message": None}), True, 402,
         {"error": ["x"], "message": "Upgrade required", "code": "upgrade_required"}),
        (stats.StatsUpgradeRequired({"error": "Plan limit", "message": None}), True, 402,
         {"error": "Plan limit", "message": "Plan limit", "code": "upgrade_required"}),
        (stats.StatsUpgradeRequired({"error": "Plan limit", "message": ""}), True, 402,
         {"error": "Plan limit", "message": "Plan limit", "code": "upgrade_required"}),
        (stats.StatsUpgradeRequired({"error": "Plan limit", "message": "Upgrade now"}), True, 402,
         {"error": "Plan limit", "message": "Upgrade now", "code": "upgrade_required"}),
        (stats.StatsUpgradeRequired(None), True, 402,
         {"code": "upgrade_required", "message": "Upgrade required"}),
        (stats.StatsTemporarilyUnavailable(), False, 503,
         {"code": "stats_temporarily_unavailable", "message": "Stats are temporarily unavailable."}),
        (stats.PublishCredentialRejected(), True, 502,
         {"code": "publish_credential_rejected", "message": "Reconnect MindsHub to see stats."}),
        (stats.PublishCredentialRejected(), False, 502,
         "The publishing service rejected the stats credential."),
        (FileNotFoundError("Artifact is not published"), True, 404, "Artifact is not published"),
        (ValueError("Invalid artifact path"), True, 400, "Invalid artifact path"),
        (PublisherUnavailable("Anton publisher is unavailable"), True, 503, "Anton publisher is unavailable"),
        (RuntimeError("Stats request failed (HTTP 500)"), False, 502, "Stats request failed (HTTP 500)"),
    ],
)
def test_stats_http_exception_table(exc, desktop, status, detail):
    mapped = stats.stats_http_exception(exc, desktop=desktop)
    assert mapped.status_code == status
    assert mapped.detail == detail


# -- report id resolution -----------------------------------------------------


def _artifact(tmp_path: Path, entry: dict | None = None, *, raw: str | None = None) -> Path:
    folder = tmp_path / "artifacts" / "rep"
    folder.mkdir(parents=True)
    (folder / "report.html").write_text("<html></html>", encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}), encoding="utf-8")
    if raw is not None:
        (folder / ".published.json").write_text(raw, encoding="utf-8")
    elif entry is not None:
        (folder / ".published.json").write_text(json.dumps({"report.html": entry}), encoding="utf-8")
    return folder


def test_folder_report_id_comes_from_the_live_record(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "rid-1", "url": "u", "published": True})
    assert stats.report_id_for_folder(folder) == "rid-1"


def test_folder_record_without_published_flag_counts_as_live(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "rid-1", "url": "u"})
    assert stats.report_id_for_folder(folder) == "rid-1"


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {"report_id": "rid-1", "url": "u", "published": False},
        {"report_id": "", "last_md5": "m1", "published": False},
        {"report_id": "", "published": True},
    ],
)
def test_folder_without_a_live_record_is_not_published(tmp_path, entry):
    folder = _artifact(tmp_path, entry)
    with pytest.raises(FileNotFoundError, match="Artifact is not published"):
        stats.report_id_for_folder(folder)


@pytest.mark.parametrize("report_id", ["..", ".", "a/b", "%2e%2e", "a b", "../x", "id.1"])
def test_folder_record_with_an_unsafe_report_id_is_not_published(tmp_path, report_id):
    folder = _artifact(tmp_path, {"report_id": report_id, "url": "u", "published": True})
    with pytest.raises(FileNotFoundError, match="Artifact is not published"):
        stats.report_id_for_folder(folder)


@pytest.mark.parametrize("report_id", ["abc123", "A_b-9", "0123456789abcdef0123456789abcdef"])
def test_folder_record_accepts_service_style_report_ids(tmp_path, report_id):
    folder = _artifact(tmp_path, {"report_id": report_id, "url": "u", "published": True})
    assert stats.report_id_for_folder(folder) == report_id


def test_folder_record_for_another_file_is_not_published(tmp_path):
    folder = _artifact(tmp_path, raw=json.dumps({"other.html": {"report_id": "rid-9", "published": True}}))
    with pytest.raises(FileNotFoundError, match="Artifact is not published"):
        stats.report_id_for_folder(folder)


def test_folder_record_with_only_an_md5_has_no_stats(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "", "last_md5": "m1", "published": True})
    with pytest.raises(stats.StatsUnavailable):
        stats.report_id_for_folder(folder)


def test_unreadable_record_is_a_runtime_error(tmp_path):
    folder = _artifact(tmp_path, raw="{not json")
    with pytest.raises(RuntimeError, match="Could not read publish record"):
        stats.report_id_for_folder(folder)


def test_desktop_path_resolves_inside_the_scope_roots(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "rid-1", "url": "u", "published": True})
    with patch("cowork.services.artifacts._scan_artifact_dirs", lambda *, scope: [folder.parent]):
        assert stats.report_id_for_path(str(folder / "report.html"), scope=LOCAL_SCOPE) == "rid-1"


def test_desktop_unpublished_record_is_not_published(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "rid-1", "url": "u", "published": False})
    with patch("cowork.services.artifacts._scan_artifact_dirs", lambda *, scope: [folder.parent]):
        with pytest.raises(FileNotFoundError, match="Artifact is not published"):
            stats.report_id_for_path(str(folder / "report.html"), scope=LOCAL_SCOPE)


def test_desktop_record_with_only_an_md5_has_no_stats(tmp_path):
    folder = _artifact(tmp_path, {"report_id": "", "last_md5": "m1", "published": True})
    with patch("cowork.services.artifacts._scan_artifact_dirs", lambda *, scope: [folder.parent]):
        with pytest.raises(stats.StatsUnavailable):
            stats.report_id_for_path(str(folder / "report.html"), scope=LOCAL_SCOPE)


def test_desktop_path_with_a_nul_byte_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="Invalid artifact path"):
        stats.report_id_for_path("bad\x00path", scope=LOCAL_SCOPE)
