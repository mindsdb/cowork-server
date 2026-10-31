"""Authenticated requests and file store work take none of anyio's default threads.

Every sync dependency and every sync route runs on anyio's default threads,
40 per event loop, and so does Starlette's FileResponse. When file store work
held all 40, the next request's principal could not be read and the pod
answered nothing. These tests hold all 40 themselves and check that
authenticated requests and the routes that work on the file store still
answer, and that a download reads its file on the file store's own threads
(``run_efs``): refused with 503 when those are busy before its headers go
out, waiting for them after, and closing its file there however it ends.

Requests go through httpx.ASGITransport on the test's own loop, so the default
threads a test holds are the ones the request would take.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID, uuid4

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
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.units import unit_session
from cowork.models.project import Project
from cowork.server import create_app
from cowork.services.conversations import ConversationService
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


async def _uploaded_file_url(client: httpx.AsyncClient, content: bytes) -> str:
    """An uploaded file's download, served by EfsFileResponse."""
    return f"/api/v1/files/{await _uploaded(client, content)}/content"


async def _project_file_url(client: httpx.AsyncClient, content: bytes) -> str:
    """A project file's download, streamed from its pinned descriptor
    (PinnedStreamingResponse)."""
    name = f"downloads-{uuid4().hex[:8]}"
    created = await client.post("/api/v1/projects/", json={"name": name}, headers=IDENTITY)
    assert created.status_code == 201, created.text
    uploaded = await client.post(
        f"/api/v1/projects/{name}/files/upload",
        files={"files": ("notes.txt", content, "text/plain")},
        headers=IDENTITY,
    )
    assert uploaded.status_code == 200 and uploaded.json()["results"][0]["ok"], uploaded.text
    return f"/api/v1/projects/{name}/files-raw/notes.txt"


_DOWNLOADS = {"uploaded file": _uploaded_file_url, "project file": _project_file_url}


@pytest.mark.parametrize("stored", _DOWNLOADS)
async def test_a_download_reads_its_file_off_the_default_threads(org_deployment, stored):
    content = b"x" * (3 * FileResponse.chunk_size + 5)
    async with _client(create_app()) as client:
        url = await _DOWNLOADS[stored](client, content)
        async with _all_tokens_held(_default_threads()):
            response = await asyncio.wait_for(client.get(url, headers=IDENTITY), BUDGET_SECONDS)

    assert response.status_code == 200, response.text
    assert response.content == content


@pytest.mark.parametrize("stored", _DOWNLOADS)
async def test_a_download_with_every_file_store_slot_held_answers_503(org_deployment, monkeypatch, stored):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    async with _client(create_app()) as client:
        url = await _DOWNLOADS[stored](client, b"hello")
        async with _all_tokens_held(units._efs_slots()):
            response = await asyncio.wait_for(client.get(url, headers=IDENTITY), BUDGET_SECONDS)
            default_threads_taken = _default_threads().borrowed_tokens

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "server_busy"
    assert default_threads_taken == 0


async def _every_slot_held_for(seconds: float) -> None:
    async with _all_tokens_held(units._efs_slots()):
        await asyncio.sleep(seconds)


def _taking_every_slot_once_the_headers_are_sent(app, *, seconds: float):
    """``app``, with every file store slot taken for ``seconds`` once the
    response's headers have gone out."""
    holders: list[asyncio.Task] = []

    async def wrapped(scope, receive, send):
        async def send_then_take(message):
            await send(message)
            if message["type"] == "http.response.start":
                holders.append(asyncio.create_task(_every_slot_held_for(seconds)))
                await asyncio.sleep(0)

        await app(scope, receive, send_then_take)

    return wrapped, holders


@pytest.mark.parametrize("stored", _DOWNLOADS)
async def test_a_download_waits_for_a_slot_once_its_headers_are_sent(org_deployment, monkeypatch, stored):
    """After the headers a refusal could only cut the body short of what the
    response declared, so a busy file store makes the rest of the download
    wait instead. Before them, it answers 503 (the test above)."""
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    content = bytes(range(256)) * (3 * FileResponse.chunk_size // 256) + b"tail!"
    app = create_app()
    async with _client(app) as client:
        url = await _DOWNLOADS[stored](client, content)
    taking, holders = _taking_every_slot_once_the_headers_are_sent(app, seconds=2)
    async with _client(taking) as client:
        response = await asyncio.wait_for(client.get(url, headers=IDENTITY), 3 * BUDGET_SECONDS)
    await asyncio.gather(*holders)

    assert holders, "the slots were never taken"
    assert response.status_code == 200, response.text
    assert response.content == content


async def test_an_efs_file_response_refused_before_its_headers_sends_nothing(tmp_path, monkeypatch):
    """The open and its fstat take a slot before any header goes out, so a
    busy file store's refusal can still become the app's 503."""
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    path = tmp_path / "notes.txt"
    path.write_bytes(b"0123456789")
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message["type"])

    response = EfsFileResponse(path, scope=LOCAL_SCOPE)
    async with _all_tokens_held(units._efs_slots()):
        with pytest.raises(units.FilesystemBusy):
            await response({"type": "http", "method": "GET", "headers": []}, receive, send)

    assert sent == []


class _CloseRecorder:
    """A file whose close records whether it ran on a file store thread."""

    def __init__(self, file, closed_on: list[bool]) -> None:
        self._file = file
        self._closed_on = closed_on

    def seek(self, offset: int) -> int:
        return self._file.seek(offset)

    def read(self, size: int) -> bytes:
        return self._file.read(size)

    def close(self) -> None:
        self._closed_on.append(units.on_efs_thread())
        self._file.close()


async def test_an_efs_file_response_closes_its_file_on_the_file_store_threads_when_cancelled(tmp_path, monkeypatch):
    """A cancel scope's cancel in the middle of the body (a middleware's task
    group, at a shutdown) still closes the file through the file store's
    threads, never later on the event loop. run_efs_cleanup's own test covers
    repeated native cancels."""
    path = tmp_path / "notes.txt"
    path.write_bytes(b"x" * (3 * FileResponse.chunk_size))
    closed_on: list[bool] = []
    real_open = efs_file_response._open_regular_file

    def recorded_open(target):
        opened = real_open(target)
        return opened._replace(file=_CloseRecorder(opened.file, closed_on))

    monkeypatch.setattr(efs_file_response, "_open_regular_file", recorded_open)
    bodies = 0

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal bodies
        if message["type"] == "http.response.body":
            bodies += 1
            if bodies == 2:
                cancelling()

    async def serve():
        await EfsFileResponse(path, scope=LOCAL_SCOPE)({"type": "http", "method": "GET", "headers": []}, receive, send)

    with anyio.CancelScope() as scope:
        cancelling = scope.cancel
        await serve()

    assert scope.cancelled_caught
    assert closed_on == [True]


async def test_an_efs_file_response_cancelled_while_it_opens_closes_the_file_on_the_file_store_threads(
    tmp_path, monkeypatch
):
    """On a stalled mount the open is the slow step. A cancel that lands while
    it runs waits for it, and the file it opened is closed on the file store's
    threads before the cancel goes on, not left to the garbage collector on
    the event loop."""
    path = tmp_path / "notes.txt"
    path.write_bytes(b"0123456789")
    closed_on: list[bool] = []
    real_open = efs_file_response._open_regular_file

    def slow_open(target):
        time.sleep(0.3)
        opened = real_open(target)
        return opened._replace(file=_CloseRecorder(opened.file, closed_on))

    monkeypatch.setattr(efs_file_response, "_open_regular_file", slow_open)
    sent: list[str] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message["type"])

    with anyio.move_on_after(0.1) as scope:
        await EfsFileResponse(path, scope=LOCAL_SCOPE)({"type": "http", "method": "GET", "headers": []}, receive, send)

    assert scope.cancelled_caught
    assert sent == []
    assert closed_on == [True]


async def test_an_efs_file_response_ends_a_range_whose_file_shrank_after_its_stat(tmp_path, monkeypatch):
    """The range was parsed against the size the open's fstat saw. A file
    that shrinks before the reads ends its body at the bytes that remain,
    instead of reading nothing forever."""
    size = 3 * FileResponse.chunk_size
    remaining = FileResponse.chunk_size + 10
    path = tmp_path / "notes.txt"
    path.write_bytes(b"x" * size)
    real_open = efs_file_response._open_regular_file

    def open_then_shrink(target):
        opened = real_open(target)
        os.truncate(target, remaining)
        return opened

    monkeypatch.setattr(efs_file_response, "_open_regular_file", open_then_shrink)
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    request = {"type": "http", "method": "GET", "headers": [(b"range", f"bytes=0-{size - 1}".encode())]}
    await asyncio.wait_for(EfsFileResponse(path, scope=LOCAL_SCOPE)(request, receive, send), BUDGET_SECONDS)

    assert messages[0]["status"] == 206
    bodies = [message for message in messages if message["type"] == "http.response.body"]
    assert b"".join(message["body"] for message in bodies) == b"x" * remaining
    assert bodies[-1]["more_body"] is False


async def test_an_efs_file_response_serves_a_file_where_os_has_no_nonblocking_flag(tmp_path, monkeypatch):
    """Windows, where the desktop runs, has no ``os.O_NONBLOCK``. The open
    uses the portable flag from cowork.common.paths, so a download still
    answers there."""
    path = tmp_path / "notes.txt"
    path.write_bytes(b"0123456789")
    monkeypatch.delattr(os, "O_NONBLOCK")
    app = Starlette(routes=[Route("/efs", lambda _request: EfsFileResponse(path, scope=LOCAL_SCOPE))])

    async with _client(app) as client:
        response = await client.get("/efs")

    assert response.status_code == 200, response.text
    assert response.content == b"0123456789"


async def test_a_project_file_download_the_client_leaves_closes_its_descriptor_on_the_file_store_threads(
    org_deployment, monkeypatch
):
    """A client that leaves mid-body cancels the stream (ASGI 2.3's disconnect
    listener). The pinned descriptor is still closed through the file store's
    threads, not on the event loop, where a close on a stalled mount would
    stop every request."""
    from cowork.api.v1.endpoints import project_files

    closed_on: list[bool] = []
    real_pinned = project_files._pinned_regular_file

    class _RecordedExit:
        def __init__(self, cm) -> None:
            self._cm = cm

        def __exit__(self, *exc_info):
            closed_on.append(units.on_efs_thread())
            return self._cm.__exit__(*exc_info)

    def recorded(target, base, *args):
        cm, fd, st = real_pinned(target, base, *args)
        return _RecordedExit(cm), fd, st

    monkeypatch.setattr(project_files, "_pinned_regular_file", recorded)
    app = create_app()
    async with _client(app) as client:
        url = await _project_file_url(client, b"x" * (8 * FileResponse.chunk_size))
    first_body = asyncio.Event()
    requested = False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_body.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_body.set()
            await asyncio.sleep(0.05)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": url, "raw_path": url.encode(), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 50000), "server": ("test", 80),
        "headers": [(b"host", b"test")] + [(k.lower().encode(), v.encode()) for k, v in IDENTITY.items()],
    }
    await asyncio.wait_for(app(scope, receive, send), BUDGET_SECONDS)

    assert first_body.is_set()
    assert closed_on == [True]


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
    ("GET", {"range": "bytes=-100"}),
    ("GET", {"range": "bytes=5-"}),
    ("GET", {"range": "bytes=0-2,5-7"}),
    ("GET", {"range": "bytes=0-5,3-8"}),
    ("HEAD", {"range": "bytes=0-2,5-7"}),
    ("GET", {"range": "bytes=50-60"}),
    ("GET", {"range": "bytes=a-b"}),
    ("GET", {"range": "lines=1-2"}),
    ("GET", {"range": "bytes=2-6", "if-range": '"not-this-file"'}),
    ("GET", {"range": "bytes=2-6", "if-range": "{etag}"}),
]

#: The edges the reads' loops turn on: an empty file, one of exactly one
#: chunk, and one a byte past it.
_CONTENTS = {
    "short": b"0123456789abcdef",
    "empty": b"",
    "one chunk": b"c" * FileResponse.chunk_size,
    "one chunk and a byte": b"c" * FileResponse.chunk_size + b"!",
}


@pytest.mark.parametrize("content", _CONTENTS)
@pytest.mark.parametrize("method,headers", _REQUESTS)
async def test_an_efs_file_response_answers_as_a_file_response_does(tmp_path, monkeypatch, method, headers, content):
    path = tmp_path / "notes.txt"
    path.write_bytes(_CONTENTS[content])
    etag = FileResponse(path, stat_result=path.stat()).headers["etag"]
    headers = {name: value.format(etag=etag) for name, value in headers.items()}
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


# Routes that worked on the file store in a sync route, on anyio's default
# threads, run that work on the file store's own threads. Each request below
# reaches its file store work in org tenancy, where the suite's blocking guard
# (tests/_blocking_guard.py) fails a test whose file store call runs off
# run_efs, and answers while every default thread is held.


@dataclass(frozen=True)
class _OrgWorkspace:
    project: str
    project_id: str
    conversation_id: str
    message_id: str
    file_id: str
    attachment_id: str


async def _attached(client: httpx.AsyncClient, project: str, conversation_id: str, name: str) -> str:
    response = await client.post(
        f"/api/v1/attachments/{project}/{conversation_id}/upload",
        files={"files": (name, b"attached", "text/plain")},
        headers=IDENTITY,
    )
    assert response.status_code == 200, response.text
    return response.json()[0]["id"]


@pytest.fixture
async def org_workspace(org_deployment) -> _OrgWorkspace:
    """One member's project, a conversation in it with one question, an
    uploaded file and an attachment, made through the routes themselves."""
    name = f"threads-{uuid4().hex[:8]}"
    async with _client(create_app()) as client:
        project = await client.post("/api/v1/projects/", json={"name": name}, headers=IDENTITY)
        assert project.status_code == 201, project.text
        conversation = await client.post(
            "/api/v1/conversations/", json={"title": "t", "project": name}, headers=IDENTITY
        )
        assert conversation.status_code == 201, conversation.text
        conversation_id = conversation.json()["id"]
        file_id = await _uploaded(client, b"uploaded")
        attachment_id = await _attached(client, name, conversation_id, "notes.txt")
    with unit_session(scope=TenantScope(org_mode=True, org_id=ORG, user_id=USER)) as session:
        message = ConversationService(session).save_user_message(UUID(conversation_id), "hello")
    return _OrgWorkspace(
        project=name,
        project_id=project.json()["id"],
        conversation_id=conversation_id,
        message_id=str(message.id),
        file_id=file_id,
        attachment_id=attachment_id,
    )


_FILE_STORE_ROUTES = [
    ("list conversations in the default project", "GET", "/api/v1/conversations/", None, 200),
    ("create a conversation in the default project", "POST", "/api/v1/conversations/", {"title": "t"}, 201),
    ("move a conversation to the general project by name", "PATCH",
     "/api/v1/conversations/{conversation_id}", {"project": "general"}, 200),
    ("move a task and its artifacts", "POST",
     "/api/v1/conversations/{conversation_id}/move", {"project": "general"}, 200),
    ("delete a turn", "DELETE", "/api/v1/conversations/{conversation_id}/turns/{message_id}", None, 200),
    ("delete a conversation", "DELETE", "/api/v1/conversations/{conversation_id}", None, 200),
    ("delete a file", "DELETE", "/api/v1/files/{file_id}", None, 204),
    ("delete an attachment", "DELETE", "/api/v1/attachments/{attachment_id}", None, 200),
    ("delete an attachment by its conversation", "DELETE",
     "/api/v1/attachments/{project}/{conversation_id}/{attachment_id}", None, 200),
    ("move an attachment into the project", "POST",
     "/api/v1/attachments/{project}/{conversation_id}/{attachment_id}/move-to-project", None, 200),
    ("list a shared project's files", "GET", "/api/v1/shared-projects/{project_id}/files", None, 404),
    ("read a shared project's file", "GET", "/api/v1/shared-projects/{project_id}/files/notes.txt", None, 404),
    ("download a shared project's file", "GET", "/api/v1/shared-projects/{project_id}/files-raw/notes.txt", None, 404),
    ("un-pick a Drive file", "DELETE",
     "/api/v1/connectors/connections/google/drive/picked-files/f1?project={project}", None, 404),
    ("create a schedule in the default project", "POST", "/api/v1/schedules/",
     {"title": "s", "prompt": "p", "cadence": "daily", "nextRunAt": "2030-01-01T00:00:00Z"}, 201),
    ("answer a cancelled OAuth sign-in", "GET",
     "/api/v1/connectors/oauth/google-drive/callback?error=access_denied&state=s", None, 200),
]


@pytest.mark.parametrize(
    "method,path,body,expected", [route[1:] for route in _FILE_STORE_ROUTES],
    ids=[route[0] for route in _FILE_STORE_ROUTES],
)
async def test_a_route_that_works_on_the_file_store_answers_while_every_default_thread_is_held(
    org_workspace, method, path, body, expected
):
    url = path.format(**vars(org_workspace))
    async with _client(create_app()) as client:
        async with _all_tokens_held(_default_threads()):
            response = await asyncio.wait_for(
                client.request(method, url, json=body, headers=IDENTITY), BUDGET_SECONDS
            )

    assert response.status_code == expected, response.text


_DATABASE_ONLY_REQUESTS = [
    ("list every project's conversations", "GET", "/api/v1/conversations/?project=all", None, 200),
    ("list one project's conversations", "GET", "/api/v1/conversations/?project_id={project_id}", None, 200),
    ("create a conversation in a project", "POST", "/api/v1/conversations/",
     {"title": "t", "project_id": "{project_id}"}, 201),
    ("rename a conversation", "PATCH", "/api/v1/conversations/{conversation_id}", {"title": "renamed"}, 200),
    ("create a schedule in a project", "POST", "/api/v1/schedules/",
     {"title": "s", "prompt": "p", "cadence": "daily", "nextRunAt": "2030-01-01T00:00:00Z",
      "projectId": "{project_id}"}, 201),
]


@pytest.mark.parametrize(
    "method,path,body,expected", [request[1:] for request in _DATABASE_ONLY_REQUESTS],
    ids=[request[0] for request in _DATABASE_ONLY_REQUESTS],
)
async def test_a_request_that_names_its_project_by_id_answers_while_every_file_store_slot_is_held(
    org_workspace, monkeypatch, method, path, body, expected
):
    """Only a lookup that may resolve the general project reaches the file
    store, so a conversation or schedule request that names its project by
    id, asks for every project, or changes only a title is a database unit
    that file store slots cannot refuse."""
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    fields = vars(org_workspace)
    url = path.format(**fields)
    payload = None if body is None else {key: value.format(**fields) if isinstance(value, str) else value
                                         for key, value in body.items()}
    async with _client(create_app()) as client:
        async with _all_tokens_held(units._efs_slots()):
            response = await asyncio.wait_for(
                client.request(method, url, json=payload, headers=IDENTITY), BUDGET_SECONDS
            )

    assert response.status_code == expected, response.text


async def test_a_project_upload_reads_each_file_only_once_its_project_is_found(org_deployment, monkeypatch):
    """The uploaded files are read from their spools on the file store's
    threads, each just before its write, and not at all for a project that is
    not found: a request holds at most one file's bytes in memory."""
    from cowork.api.v1.endpoints import project_files

    events: list[str] = []
    real_read = tempfile.SpooledTemporaryFile.read
    real_write = project_files._write_bytes_at_project_root

    def read(self, *args):
        events.append("read")
        return real_read(self, *args)

    def write(*args, **kwargs):
        events.append("write")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "read", read)
    monkeypatch.setattr(project_files, "_write_bytes_at_project_root", write)
    files = [("files", ("a.txt", b"first", "text/plain")), ("files", ("b.txt", b"second", "text/plain"))]
    name = f"uploads-{uuid4().hex[:8]}"
    async with _client(create_app()) as client:
        created = await client.post("/api/v1/projects/", json={"name": name}, headers=IDENTITY)
        assert created.status_code == 201, created.text
        missing = await client.post("/api/v1/projects/no-such-project/files/upload", files=files, headers=IDENTITY)
        read_for_a_missing_project = list(events)
        events.clear()
        found = await client.post(f"/api/v1/projects/{name}/files/upload", files=files, headers=IDENTITY)

    assert missing.status_code == 404, missing.text
    assert read_for_a_missing_project == []
    assert found.status_code == 200, found.text
    assert [result["size"] for result in found.json()["results"]] == [5, 6]
    assert events == ["read", "write", "read", "write"]
