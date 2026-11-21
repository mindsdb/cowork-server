"""BodyLimitMiddleware (cowork/api/body_limits.py): in org tenancy the server
refuses a request body past its route's cap with 413, before the app reads it.

The request tests patch every cap down to 1 MiB, so no test sends a large
body, and drive the real ``POST /api/v1/files/`` route of an org-mode app.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from cowork.api import body_limits
from cowork.api.body_limits import BodyLimit, BodyLimitMiddleware, body_limit_for
from cowork.api.v1.endpoints import artifact_comment_images, files, project_files, skills
from cowork.api.v1.endpoints.compat import stubs
from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import TrustedHeaderMiddleware
from cowork.server import create_app
from cowork.services.skills import SkillService

_CAP = 1024 * 1024
_IDENTITY = {
    "X-User-Id": "5d1b7c3e-2a4f-4e6b-9c8d-0f1e2a3b4c5d",
    "X-Organization-Id": "9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b",
    "X-User-Email": "uploader@example.com",
}
_BOUNDARY = "body-limit-test-boundary"
_FORM = {"Content-Type": f"multipart/form-data; boundary={_BOUNDARY}"}


def _upload_body(*, size: int) -> bytes:
    """A multipart form for ``POST /api/v1/files/``, exactly ``size`` bytes long."""
    head = (
        f"--{_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "assistants\r\n"
        f"--{_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{_BOUNDARY}--\r\n".encode()
    return head + b"x" * (size - len(head) - len(tail)) + tail


def _in_two_chunks(*, body: bytes) -> Iterator[bytes]:
    """``body`` as a stream, which the client sends chunked, with no Content-Length."""
    middle = len(body) // 2
    yield body[:middle]
    yield body[middle:]


def _filled(*, path: str) -> str:
    """A route's path with each of its parameters filled in, as a request names it."""
    return re.sub(r"\{[^}]+\}", "x", path)


@pytest.fixture
def org_client(org_deployment):
    # No `with`: skipping the lifespan skips the boot migrations and workers.
    return TestClient(create_app())


@pytest.fixture
def one_mib_caps(monkeypatch):
    """Every cap at 1 MiB."""
    monkeypatch.setattr(
        body_limits,
        "UPLOAD_ROUTES",
        tuple(replace(route, max_bytes=_CAP) for route in body_limits.UPLOAD_ROUTES),
    )
    monkeypatch.setattr(body_limits, "DEFAULT_BODY_MAX_BYTES", _CAP)


@pytest.fixture
def saved_uploads(monkeypatch) -> list[str]:
    """The names of the files the files route handed to save_upload."""
    saved: list[str] = []
    save_upload = files.save_upload

    async def recording_save_upload(upload, **kwargs):
        saved.append(upload.filename)
        return await save_upload(upload, **kwargs)

    monkeypatch.setattr(files, "save_upload", recording_save_upload)
    return saved


def test_an_upload_past_its_cap_is_refused_before_the_route_runs(org_client, one_mib_caps, saved_uploads):
    origin = get_app_settings().allowed_origins[0]
    response = org_client.post(
        "/api/v1/files/",
        content=_upload_body(size=_CAP + 1),
        headers={**_IDENTITY, **_FORM, "Origin": origin},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "This upload is larger than the 1 MB limit."}
    # The refusal passes back through CORS, so the app can read its sentence.
    assert response.headers["access-control-allow-origin"] == origin
    assert saved_uploads == []


def test_an_upload_at_its_cap_still_reaches_the_route(org_client, one_mib_caps, saved_uploads):
    response = org_client.post(
        "/api/v1/files/",
        content=_upload_body(size=_CAP),
        headers={**_IDENTITY, **_FORM},
    )

    assert response.status_code == 201, response.text
    assert saved_uploads == ["a.bin"]


def test_a_chunked_upload_past_its_cap_is_refused_as_it_is_read(org_client, one_mib_caps, saved_uploads):
    response = org_client.post(
        "/api/v1/files/",
        content=_in_two_chunks(body=_upload_body(size=_CAP + 1)),
        headers={**_IDENTITY, **_FORM},
    )

    # With no Content-Length, only the count of the bytes read can refuse it.
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json() == {"detail": "This upload is larger than the 1 MB limit."}
    assert saved_uploads == []


def test_any_other_route_takes_the_default_cap(org_client, one_mib_caps):
    response = org_client.post("/api/v1/projects/", json={"name": "x" * _CAP}, headers=_IDENTITY)

    assert response.status_code == 413
    assert response.json() == {"detail": "This request is larger than the 1 MB limit."}


@pytest.mark.parametrize("content_length", ["abc", "-1", "1.5"])
def test_a_content_length_that_is_not_a_byte_count_answers_400(org_client, content_length):
    response = org_client.post(
        "/api/v1/projects/",
        content=b"{}",
        headers={**_IDENTITY, "Content-Type": "application/json", "Content-Length": content_length},
    )

    assert response.status_code == 400
    assert "Content-Length" in response.json()["detail"]


def test_org_tenancy_installs_the_cap_inside_cors_and_the_identity_layer(org_deployment):
    # user_middleware lists the outermost layer first. Inside the identity
    # layer's BaseHTTPMiddleware, a 413 raised while the body is read reaches
    # FastAPI unwrapped (BodyLimitMiddleware's docstring).
    stack = [entry.cls for entry in create_app().user_middleware]

    assert stack.index(BodyLimitMiddleware) > stack.index(CORSMiddleware)
    assert stack.index(BodyLimitMiddleware) > stack.index(TrustedHeaderMiddleware)


def test_the_desktop_takes_no_cap(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()

    stack = [entry.cls for entry in create_app().user_middleware]

    assert BodyLimitMiddleware not in stack


def test_each_upload_route_takes_its_cap_and_every_other_route_the_default(org_deployment):
    upload_caps = {
        files.upload_file: body_limits.UPLOAD_MAX_BYTES,
        project_files.upload_project_files: body_limits.UPLOAD_MAX_BYTES,
        stubs.upload_attachment: body_limits.UPLOAD_MAX_BYTES,
        skills.upload_skill: body_limits.SKILL_UPLOAD_MAX_BYTES,
        artifact_comment_images.upload_comment_image: body_limits.COMMENT_IMAGE_UPLOAD_MAX_BYTES,
    }
    mounted = [route for route in create_app().routes if isinstance(route, APIRoute)]

    assert set(upload_caps) <= {route.endpoint for route in mounted}
    for route in mounted:
        upload_cap = upload_caps.get(route.endpoint)
        for method in route.methods:
            limit = body_limit_for(method=method, path=_filled(path=route.path))
            if upload_cap is None:
                assert limit.noun == "request", (method, route.path)
            else:
                assert limit == BodyLimit(max_bytes=upload_cap, noun="upload"), (method, route.path)


def test_every_upload_row_names_a_mounted_route(org_deployment):
    mounted = [
        (method, _filled(path=route.path))
        for route in create_app().routes
        if isinstance(route, APIRoute)
        for method in route.methods
    ]

    for row in body_limits.UPLOAD_ROUTES:
        assert any(
            method == row.method and row.path.fullmatch(path) for method, path in mounted
        ), (row.method, row.path.pattern)


def test_an_upload_cap_needs_the_upload_route_whole():
    default = BodyLimit(max_bytes=body_limits.DEFAULT_BODY_MAX_BYTES, noun="request")

    # Starlette redirects the form without the trailing slash, unread.
    assert body_limit_for(method="POST", path="/api/v1/files").noun == "upload"
    assert body_limit_for(method="GET", path="/api/v1/files/") == default
    assert body_limit_for(method="POST", path="/api/v1/files/upload") == default
    assert body_limit_for(method="POST", path="/api/v1/skills/upload/extra") == default
    assert body_limit_for(method="POST", path="/prefix/api/v1/skills/upload") == default


def test_the_skill_cap_takes_every_zip_within_the_expansion_bound():
    assert body_limits.SKILL_UPLOAD_MAX_BYTES > SkillService._ZIP_MAX_UNCOMPRESSED
