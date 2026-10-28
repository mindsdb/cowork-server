"""How an upload-service failure becomes the error a route answers with."""
from __future__ import annotations

import http.client
import io
import json
import logging
import urllib.error

import pytest
from fastapi import HTTPException

from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable
from cowork.services.publish import (
    PublishTooLarge,
    UploadError,
    UploadServiceUnavailable,
    _publisher_http_detail,
    _read_http_error_body,
    classify_upload_error,
)

TOO_LARGE_DETAIL = {
    "code": "artifact_too_large",
    "message": "This artifact is too large to publish. Remove or shrink large files in it, such as PDFs or images.",
}
UNAVAILABLE_DETAIL = {
    "code": "publish_unavailable",
    "message": "The publishing service is temporarily unavailable. Try again in a moment.",
}


class _BrokenBody(io.BytesIO):
    def read(self, *args):
        raise http.client.IncompleteRead(b"")


def _http_error(status: int, body, reason: str = "Rejected") -> urllib.error.HTTPError:
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    fp = body if isinstance(body, io.IOBase) else io.BytesIO(body or b"")
    return urllib.error.HTTPError("https://publish.example/upload", status, reason, {}, fp)


@pytest.mark.parametrize("body", [
    {"message": "Request Too Long"},
    b"HTTP content length exceeded 10485760 bytes.",
    b"",
    _BrokenBody(),
])
def test_a_413_is_too_large_whatever_the_body(body):
    with pytest.raises(PublishTooLarge) as error:
        classify_upload_error(_http_error(413, body, "Payload Too Large"))
    assert error.value.status_code == 413
    assert error.value.detail == TOO_LARGE_DETAIL


def test_a_503_marked_as_an_auth_outage_is_a_permission_check_failure():
    with pytest.raises(ProductPermissionUnavailable):
        classify_upload_error(_http_error(503, {"error": "down", "code": "auth_unavailable"}))


@pytest.mark.parametrize("body", [
    {"error": "Auth unavailable"},
    {"error": "Artifact ownership could not be checked", "code": "artifact_ownership_unavailable"},
    b"Service Unavailable",
    _BrokenBody(),
])
def test_any_other_503_is_the_upload_service_being_unavailable(body):
    with pytest.raises(UploadServiceUnavailable) as error:
        classify_upload_error(_http_error(503, body, "Service Unavailable"))
    assert error.value.status_code == 503
    assert error.value.detail == UNAVAILABLE_DETAIL


def test_a_confirmed_403_is_a_permission_denial():
    with pytest.raises(ProductPermissionDenied):
        classify_upload_error(_http_error(403, {"code": "permission_denied", "error": "nope"}))


@pytest.mark.parametrize("status, body, expected", [
    (403, {"code": "organization_inactive", "error": "This organization is archived or still being provisioned."},
     UploadError(403, "organization_inactive", "This organization is archived or still being provisioned.")),
    (400, {"error": "pip install failed: boom"}, UploadError(400, None, "pip install failed: boom")),
    (500, b"boom", UploadError(500, None, None)),
    (409, b"", UploadError(409, None, None)),
])
def test_other_answers_are_returned_for_the_caller_to_word(status, body, expected):
    assert classify_upload_error(_http_error(status, body)) == expected


def test_a_failure_without_an_http_answer_returns_none():
    assert classify_upload_error(OSError("disk full")) is None
    assert classify_upload_error(urllib.error.URLError("timed out")) is None


@pytest.mark.parametrize("exc", [
    ProductPermissionDenied(), ProductPermissionUnavailable(), PublishTooLarge(), UploadServiceUnavailable(),
])
def test_an_already_typed_error_is_raised_unchanged(exc):
    with pytest.raises(type(exc)) as error:
        classify_upload_error(exc)
    assert error.value is exc


@pytest.mark.parametrize("status, body", [
    (403, {"code": "permission_denied"}),
    (413, {"error": "x"}),
    (503, {"error": "x"}),
])
def test_the_refusals_that_skip_the_publish_failure_log_are_logged(status, body, caplog):
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        with pytest.raises(HTTPException):
            classify_upload_error(_http_error(status, body))
    assert f"Publish service answered HTTP {status}" in caplog.text


@pytest.mark.parametrize("status, body", [
    (500, b"boom"),
    (403, {"code": "organization_inactive", "error": "archived"}),
])
def test_an_answer_left_to_the_caller_is_not_logged_here(status, body, caplog):
    # publish_artifact logs these itself; logging here too would log them twice.
    with caplog.at_level(logging.WARNING, logger="cowork.services.publish"):
        classify_upload_error(_http_error(status, body))
    assert "Publish service answered" not in caplog.text


@pytest.mark.parametrize("body, expected", [
    ({"code": "c", "error": "e"}, ('{"code": "c", "error": "e"}', "c", "e")),
    ({"other": 1}, ('{"other": 1}', None, None)),
    ({"code": 7, "error": {"nested": True}}, ('{"code": 7, "error": {"nested": true}}', None, None)),
    ([1, 2], ("[1, 2]", None, None)),
    ({"message": "Missing Authentication Token"}, ('{"message": "Missing Authentication Token"}', None, None)),
    (b"plain text", ("plain text", None, None)),
    (_BrokenBody(), ("", None, None)),
])
def test_the_body_is_read_once_into_text_code_and_error(body, expected):
    assert _read_http_error_body(_http_error(400, body)) == expected


@pytest.mark.parametrize("body, expected", [
    ({"error": "Version not found"}, "Version not found"),
    ({"other": 1}, '{"other": 1}'),
])
def test_publisher_http_detail_keeps_its_answer(body, expected):
    assert _publisher_http_detail(_http_error(404, body)) == expected


@pytest.mark.parametrize("body, expected", [
    ({"message": "Too Many Requests"}, ('{"message": "Too Many Requests"}', None, "Too Many Requests")),
    ({"error": "e", "message": "m"}, ('{"error": "e", "message": "m"}', None, "e")),
])
def test_a_throttled_answer_takes_its_text_from_message(body, expected):
    assert _read_http_error_body(_http_error(429, body)) == expected


def test_publisher_http_detail_falls_back_to_the_exception_text():
    exc = _http_error(404, b"", "Not Found")
    assert _publisher_http_detail(exc) == str(exc)
