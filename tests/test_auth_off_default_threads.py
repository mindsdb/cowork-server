"""Authenticated requests and file downloads take none of anyio's default threads.

Every sync dependency and every sync route runs on anyio's default threads,
40 per event loop, and so does Starlette's FileResponse. When file store work
held all 40, the next request's principal could not be read and the pod
answered nothing. These tests hold all 40 themselves and check that
authenticated requests still answer, and that a download reads its file on
the file store's own threads (``run_efs``), refused with 503 when those are
busy.

Requests go through httpx.ASGITransport on the test's own loop, so the default
threads a test holds are the ones the request would take.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from uuid import uuid4

import anyio
import anyio.to_thread
import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route

import cowork.db.session as db_session
from cowork.api.v1 import efs_file_response
from cowork.api.v1.efs_file_response import EfsFileResponse
from cowork.db import units
from cowork.db.scoped import LOCAL_SCOPE
from cowork.models.project import Project
from cowork.server import create_app
from test_artifact_resolution_off_the_loop import REVIEW, seeded_artifact  # noqa: F401

ORG = "5d6c1f0e-2b7a-4c3d-8e9f-0a1b2c3d4e5f"
USER = "6e7d2a1f-3c8b-4d4e-9f0a-1b2c3d4e5f60"
IDENTITY = {"X-User-Id": USER, "X-Organization-Id": ORG}
BUDGET_SECONDS = 5.0


@contextlib.asynccontextmanager
async def _all_tokens_held(limiter: anyio.CapacityLimiter) -> AsyncIterator[None]:
    holders = [object() for _ in range(int(limiter.total_tokens))]
    for holder in holders:
        await limiter.acquire_on_behalf_of(holder)
    try:
        yield
    finally:
        for holder in holders:
            limiter.release_on_behalf_of(holder)


def _default_threads() -> anyio.CapacityLimiter:
    return anyio.to_thread.current_default_thread_limiter()


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_an_org_member_lists_projects_while_every_default_thread_is_held(org_deployment):
    async with _client(create_app()) as client:
        async with _all_tokens_held(_default_threads()):
            response = await asyncio.wait_for(
                client.get("/api/v1/projects/", headers=IDENTITY), BUDGET_SECONDS
            )

    assert response.status_code == 200, response.text
    assert [project["name"] for project in response.json()] == ["general"]


async def test_an_artifact_review_answers_while_every_default_thread_is_held(seeded_artifact):  # noqa: F811
    async with _client(create_app()) as client:
        async with _all_tokens_held(_default_threads()):
            response = await asyncio.wait_for(client.get(REVIEW), BUDGET_SECONDS)

    assert response.status_code == 200, response.text


async def _uploaded(client: httpx.AsyncClient, content: bytes) -> str:
    response = await client.post(
        "/api/v1/files/",
        files={"file": ("notes.txt", content, "text/plain")},
        data={"purpose": "assistants"},
        headers=IDENTITY,
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def test_a_download_reads_its_file_off_the_default_threads(org_deployment):
    content = b"x" * (3 * FileResponse.chunk_size + 5)
    async with _client(create_app()) as client:
        file_id = await _uploaded(client, content)
        async with _all_tokens_held(_default_threads()):
            response = await asyncio.wait_for(
                client.get(f"/api/v1/files/{file_id}/content", headers=IDENTITY), BUDGET_SECONDS
            )

    assert response.status_code == 200, response.text
    assert response.content == content


async def test_a_download_with_every_file_store_slot_held_answers_503(org_deployment, monkeypatch):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    async with _client(create_app()) as client:
        file_id = await _uploaded(client, b"hello")
        async with _all_tokens_held(units._efs_slots()):
            response = await asyncio.wait_for(
                client.get(f"/api/v1/files/{file_id}/content", headers=IDENTITY), BUDGET_SECONDS
            )
            default_threads_taken = _default_threads().borrowed_tokens

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "server_busy"
    assert default_threads_taken == 0


async def test_a_unit_that_raises_returns_its_connection():
    engine = db_session.get_engine(db_session.settings.database.uri)

    def failing(session):
        session.get(Project, uuid4())
        raise LookupError("the unit failed")

    with pytest.raises(LookupError):
        await units.run_efs_unit(failing, scope=LOCAL_SCOPE)

    assert engine.pool.checkedout() == 0
    assert await units.run_efs_unit(lambda session: session.get(Project, uuid4()), scope=LOCAL_SCOPE) is None


# EfsFileResponse answers every request the way Starlette's FileResponse does.

_REQUESTS = [
    ("GET", {}),
    ("HEAD", {}),
    ("GET", {"range": "bytes=2-6"}),
    ("GET", {"range": "bytes=-4"}),
    ("GET", {"range": "bytes=0-2,5-7"}),
    ("HEAD", {"range": "bytes=0-2,5-7"}),
    ("GET", {"range": "bytes=50-60"}),
    ("GET", {"range": "lines=1-2"}),
    ("GET", {"range": "bytes=2-6", "if-range": '"not-this-file"'}),
]


@pytest.mark.parametrize("method,headers", _REQUESTS)
async def test_an_efs_file_response_answers_as_a_file_response_does(tmp_path, monkeypatch, method, headers):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"0123456789abcdef")
    monkeypatch.setattr(efs_file_response, "token_hex", lambda _n: "b0undary")
    monkeypatch.setattr("starlette.responses.token_hex", lambda _n: "b0undary")

    def starlette_file(_request):
        return FileResponse(path, filename="notes.txt", stat_result=None)

    def efs_file(_request):
        return EfsFileResponse(path, scope=LOCAL_SCOPE, filename="notes.txt")

    app = Starlette(routes=[Route("/starlette", starlette_file, methods=["GET", "HEAD"]),
                            Route("/efs", efs_file, methods=["GET", "HEAD"])])
    async with _client(app) as client:
        expected = await client.request(method, "/starlette", headers=headers)
        answered = await client.request(method, "/efs", headers=headers)

    assert answered.status_code == expected.status_code
    assert answered.headers == expected.headers
    assert answered.content == expected.content
