"""Artifact routes find their folder off the event loop, with no pooled
connection held, and a stuck file store refuses only the requests that need it.

On 28 Oct a slow shared file store froze staging pods: the artifact review
and workspace routes ran their folder lookup on the event loop, so liveness
waited behind them until the kubelet restarted the pod. These tests make the
lookup slow, or stuck, at the lowest file store helper every route reaches
(``artifact_identity.opened_artifact_root``), the way the file store was.

Requests go through httpx.ASGITransport on the test's own loop, so a lookup
that blocked the loop shows up as missing heartbeat ticks and a slow
liveness answer, not only as a slow request.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import shutil
import threading
import time
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

import cowork.db.session as db_session
from cowork.common.settings.app_settings import get_app_settings
from cowork.db import units
from cowork.server import create_app
from cowork.services import artifact_identity, artifact_roots

ARTIFACT_ID = "0f9e8d7c6b5a49388271605f4e3d2c1b"
SLOW_SECONDS = 3.0
LIVE_BUDGET_SECONDS = 0.5
LIVE = "/api/v1/health/live"
WORKSPACE = f"/api/v1/artifacts/workspace/local/{ARTIFACT_ID}"
REVIEW = f"{WORKSPACE}/review"

# Every route that resolves an artifact folder: the review and source routes,
# the owner-gated workspace routes and the private draft. Each body passes
# validation, so the request reaches the lookup.
ROUTES = [
    ("GET", REVIEW, None),
    ("GET", WORKSPACE, None),
    ("PUT", WORKSPACE, {"content": "x", "expectedRevisionId": "r1"}),
    ("POST", f"{WORKSPACE}/export", {"format": "html"}),
    ("GET", f"{WORKSPACE}/revisions", None),
    ("GET", f"{WORKSPACE}/access", None),
    ("PUT", f"{WORKSPACE}/access", {"access": {"mode": "public"}}),
    ("POST", f"{WORKSPACE}/comments-access", None),
    ("GET", f"{WORKSPACE}/revisions/r1", None),
    ("POST", f"{WORKSPACE}/revisions/r1/restore", {"expectedRevisionId": "r1"}),
    (
        "POST",
        f"{WORKSPACE}/agent-repairs",
        {
            "expectedRevisionId": "r1",
            "commentThreadId": "t1",
            "thread": [{"text": "fix it"}],
            "conversationId": str(UUID(int=7)),
        },
    ),
    ("GET", f"{WORKSPACE}/agent-repairs/rep1", None),
    ("POST", f"{WORKSPACE}/agent-repairs/release", {"commentThreadId": "t1"}),
    ("POST", f"{WORKSPACE}/agent-repairs/rep1/cancel", None),
    ("POST", f"{WORKSPACE}/agent-repairs/rep1/decision", {"status": "rejected"}),
    ("GET", f"/api/v1/artifacts/drafts/local/{ARTIFACT_ID}/index.html", None),
]

# The request a lookup belongs to: set by `_marked` on the request's task,
# copied into every worker thread the request's work runs on.
_request: contextvars.ContextVar[dict] = contextvars.ContextVar("request")


def _marked(app):
    async def marked(scope, receive, send):
        if scope["type"] == "http":
            _request.set({"route": (scope["method"], scope["path"]), "slept": False})
        await app(scope, receive, send)

    return marked


@pytest.fixture
def seeded_artifact():
    """One artifact in the desktop's general project, which `local` resolves."""
    folder = Path(get_app_settings().project.root_dir) / "general" / ".anton" / "artifacts" / "brief"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps({"id": ARTIFACT_ID, "type": "html-app", "title": "Brief", "primary": "index.html"})
    )
    (folder / "index.html").write_text("<!doctype html><p>brief</p>")
    artifact_identity._clear_identity_indexes()
    yield folder
    shutil.rmtree(folder, ignore_errors=True)
    artifact_identity._clear_identity_indexes()


@pytest.fixture
def slow_lookup(monkeypatch, seeded_artifact):
    """Each request's first artifact root lookup sleeps SLOW_SECONDS. Returns
    the routes whose lookup slept."""
    real = artifact_identity.opened_artifact_root
    reached: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def slow(source):
        request = _request.get(None)
        if request is not None and not request["slept"]:
            request["slept"] = True
            reached.append(request["route"])
            time.sleep(SLOW_SECONDS)
        with real(source) as root:
            yield root

    monkeypatch.setattr(artifact_identity, "opened_artifact_root", slow)
    return reached


async def _heartbeat(counter: list[int]) -> None:
    while True:
        await asyncio.sleep(0.01)
        counter[0] += 1


async def _live_latencies(client, loop, until) -> list[float]:
    """Ask liveness every 0.2 s and time each answer from when it was due, so
    a stalled loop counts against it even when the ask itself starts late."""
    latencies = []
    first = loop.time()
    while not until():
        due = first + 0.2 * len(latencies)
        await asyncio.sleep(max(0.0, due - loop.time()))
        response = await client.get(LIVE)
        latencies.append(loop.time() - due)
        assert response.status_code == 200
    return latencies


async def test_every_artifact_route_leaves_the_loop_and_liveness_free_while_its_lookup_is_slow(
    slow_lookup, granted_product_permissions,
):
    loop = asyncio.get_running_loop()
    ticks = [0]
    transport = httpx.ASGITransport(app=_marked(create_app()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        beat = asyncio.create_task(_heartbeat(ticks))
        started = loop.time()
        routes = [asyncio.create_task(client.request(m, p, json=b)) for m, p, b in ROUTES]
        try:
            latencies = await _live_latencies(client, loop, lambda: all(t.done() for t in routes))
            await asyncio.gather(*routes)
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
        elapsed = loop.time() - started

    assert latencies, "every route finished before liveness was asked"
    assert max(latencies) < LIVE_BUDGET_SECONDS, f"liveness waited {max(latencies):.2f}s"
    # A free loop ticks about 100 times a second here; a blocked one stops.
    assert ticks[0] >= 50 * elapsed * 0.5, f"{ticks[0]} heartbeat ticks in {elapsed:.1f}s"
    assert sorted(slow_lookup) == sorted((method, path) for method, path, _ in ROUTES)


@pytest.fixture
def stuck_lookup(monkeypatch, seeded_artifact):
    """Every artifact root lookup blocks until the test releases it, as a
    call on a hard NFS mount does while the file server doesn't answer."""
    released = threading.Event()
    state = {"blocked": 0, "most": 0}
    lock = threading.Lock()

    @contextlib.contextmanager
    def stuck(_source):
        with lock:
            state["blocked"] += 1
            state["most"] = max(state["most"], state["blocked"])
        try:
            released.wait(timeout=60)
        finally:
            with lock:
                state["blocked"] -= 1
        raise FileNotFoundError("released")
        yield  # pragma: no cover

    monkeypatch.setattr(artifact_identity, "opened_artifact_root", stuck)
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    yield state, released
    released.set()


async def test_a_stuck_file_store_refuses_artifact_reviews_and_nothing_else(stuck_lookup):
    state, released = stuck_lookup
    loop = asyncio.get_running_loop()
    extra = 4
    transport = httpx.ASGITransport(app=_marked(create_app()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started = loop.time()
        reviews = [asyncio.create_task(client.get(REVIEW)) for _ in range(units.EFS_SLOTS + extra)]
        refused: list[tuple[httpx.Response, float]] = []
        try:
            while len(refused) < extra and loop.time() - started < 5:
                for task in [t for t in reviews if t.done()]:
                    reviews.remove(task)
                    refused.append((task.result(), loop.time() - started))
                await asyncio.sleep(0.05)

            live_started = loop.time()
            live = await client.get(LIVE)
            live_seconds = loop.time() - live_started
            projects_started = loop.time()
            projects = await client.get("/api/v1/projects/")
            projects_seconds = loop.time() - projects_started
            blocked_now = state["blocked"]
        finally:
            released.set()
            await asyncio.gather(*reviews)

    assert len(refused) == extra
    for response, seconds in refused:
        assert response.status_code == 503, response.text
        assert response.json()["code"] == "server_busy"
        assert response.headers["retry-after"] == "1"
        assert seconds <= 1 + 1
    assert live.status_code == 200 and live_seconds < LIVE_BUDGET_SECONDS
    assert projects.status_code == 200 and projects_seconds < LIVE_BUDGET_SECONDS
    assert blocked_now == units.EFS_SLOTS
    assert state["most"] == units.EFS_SLOTS


async def test_a_lookup_waiting_on_the_file_store_holds_no_pooled_connection(stuck_lookup):
    state, released = stuck_lookup
    engine = db_session.get_engine(db_session.settings.database.uri)
    transport = httpx.ASGITransport(app=_marked(create_app()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        review = asyncio.create_task(client.get(REVIEW))
        try:
            for _ in range(100):
                if state["blocked"]:
                    break
                await asyncio.sleep(0.02)
            assert state["blocked"] == 1, "the review never reached its lookup"
            checked_out = engine.pool.checkedout()
        finally:
            released.set()
            await review

    assert checked_out == 0


class _FileCalls:
    """Counts filesystem calls under one directory, the ones a lookup makes."""

    NAMES = ("stat", "lstat", "fstat", "open", "scandir", "listdir")

    def __init__(self, monkeypatch, root: Path):
        self.count = 0
        self.counting = False
        root_text = str(root)
        for name in self.NAMES:
            real = getattr(os, name)

            def counted(path, *args, _real=real, **kwargs):
                # A call on a descriptor, or on a name relative to one
                # (dir_fd), is the identity index's own pinned walk.
                if self.counting and (
                    isinstance(path, int)
                    or kwargs.get("dir_fd") is not None
                    or (isinstance(path, (str, os.PathLike)) and os.fspath(path).startswith(root_text))
                ):
                    self.count += 1
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(os, name, counted)


def test_a_warm_lookup_skips_conversation_folders_that_hold_no_artifacts(tmp_path, monkeypatch):
    """Org mode, a project with 100 conversation folders and no legacy
    artifacts: the folders are not roots, so a lookup doesn't index each one."""
    monkeypatch.setattr(artifact_roots, "_org_mode", lambda: True)
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "brief"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"id": ARTIFACT_ID, "type": "document"}))
    for _ in range(100):
        (project / "conversations" / str(uuid4())).mkdir(parents=True)
    row = artifact_roots.ProjectRow(id=UUID(int=1), name="project", path=str(project), external=False)
    plan = artifact_roots.RootsPlan(projects=(row,), owner_filter=True)
    artifact_identity._clear_identity_indexes()

    def lookup():
        found = artifact_roots.discover_roots(plan)
        assert found.conversation_ids() == []
        return artifact_identity.resolve_artifact_folder(found.sources(()), ARTIFACT_ID)

    lookup()  # cold: builds the index
    calls = _FileCalls(monkeypatch, tmp_path)
    calls.counting = True
    _source, resolved, _metadata = lookup()
    calls.counting = False

    assert resolved.name == "brief"
    assert calls.count <= 300, f"{calls.count} filesystem calls for one warm lookup"


@pytest.mark.parametrize(
    ("method", "path", "body", "step"),
    [
        ("POST", f"{WORKSPACE}/export", {"format": "html"}, "export_primary_file"),
        ("DELETE", f"/api/v1/artifacts/{ARTIFACT_ID}", None, "_delete_artifact_from_source"),
    ],
    ids=["export", "delete"],
)
async def test_a_busy_file_store_answers_503_not_a_failed_operation(
    seeded_artifact, granted_product_permissions, monkeypatch, method, path, body, step,
):
    """Both routes turn an unexpected error into a 500. A busy file store is
    not one: the client should wait Retry-After and try again."""
    from cowork.api.v1.endpoints import artifact_workspace, artifacts
    from cowork.services.projects import GENERAL_PROJECT_ID

    def busy(*_args, **_kwargs):
        raise units.FilesystemBusy("no shared file store slot freed within POOL_TIMEOUT")

    module = artifact_workspace if step == "export_primary_file" else artifacts
    monkeypatch.setattr(module, step, busy)
    params = {"project_id": str(GENERAL_PROJECT_ID)} if method == "DELETE" else None
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(method, path, json=body, params=params)

    assert response.status_code == 503, response.text
    assert response.json()["code"] == "server_busy"
    assert "retry-after" in response.headers


async def test_taking_the_live_publish_lock_waits_out_a_busy_file_store(monkeypatch, tmp_path):
    from cowork.api.v1.endpoints import artifact_workspace
    from cowork.db.scoped import LOCAL_SCOPE

    answers = iter([units.FilesystemBusy("busy"), True])

    async def run_efs(fn, *, scope, deadline=True):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(artifact_workspace, "run_efs", run_efs)

    assert await artifact_workspace._acquire_live_publish_lock(tmp_path / "report", LOCAL_SCOPE) is True
