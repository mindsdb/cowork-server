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
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from sqlmodel import Session
from starlette.types import ASGIApp

import cowork.db.session as db_session
from cowork.api.v1 import artifact_scope
from cowork.api.v1.endpoints import artifact_workspace
from cowork.common.settings.app_settings import get_app_settings
from cowork.db import units
from cowork.db.scoped import TenantScope
from cowork.draft_host import app as draft_app
from cowork.models.project import Project
from cowork.server import create_app
from cowork.services import artifact_files, artifact_identity, artifact_roots
from cowork.services.artifact_identity import ensure_full_id

import test_artifacts_api_tenancy as tenancy
from draft_host_support import DRAFT_HOST, FETCH, configure_draft_env, draft_token
from test_artifact_roots import _count_hops
from test_artifacts_api_tenancy import session  # noqa: F401  (fixture)

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
            headers = dict(scope["headers"])
            _request.set({
                "route": (scope["method"], scope["path"]),
                "user": headers.get(b"x-user-id", b"").decode(),
                "slept": False,
            })
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


class _DraftRequest(NamedTuple):
    """A request for a draft's index.html, and the app that serves it."""

    app: ASGIApp
    base_url: str
    path: str
    headers: dict[str, str]


@pytest.fixture(params=["drafts", "draft_host"])
def owners_draft(request, session, tmp_path, cleanup_tmp_projects, granted_product_permissions, monkeypatch):
    """USER_A's draft, whose index.html reads "brief", as USER_A asks for it:
    through the authenticated /drafts route, or through the draft host with a
    link issued to them."""
    configure_draft_env(monkeypatch)
    row, folder = tenancy._project_with_artifact(
        session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash"
    )
    (folder / "index.html").write_text("<!doctype html><p>brief</p>")
    artifact_id, _metadata = ensure_full_id(folder)
    if request.param == "drafts":
        yield _DraftRequest(
            create_app(),
            "http://test",
            f"/api/v1/artifacts/drafts/{row.id}/{artifact_id}/index.html",
            tenancy._IDENTITY,
        )
    else:
        token = draft_token(sub=tenancy.USER_A, org=tenancy.ORG_A, proj=str(row.id), art=artifact_id)
        yield _DraftRequest(draft_app.create_draft_app(), f"https://{DRAFT_HOST}", f"/_t/{token}/index.html", FETCH)
    get_app_settings.cache_clear()


@pytest.mark.parametrize("step", ["open", "preview"])
async def test_a_draft_leaves_the_loop_and_liveness_free_while_its_file_is_slow(owners_draft, monkeypatch, step):
    """Once its lookup lets the caller in, the private draft and the draft
    host open the file and build the preview from it, in the lookup's file
    store call. Each step is made slow here, and the loop keeps ticking."""
    module, name = (
        (artifact_files, "open_pinned_draft_file") if step == "open"
        else (artifact_workspace, "_preview_html_from_fd")
    )
    real = getattr(module, name)

    def slow(*args, **kwargs):
        time.sleep(SLOW_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, slow)
    loop = asyncio.get_running_loop()
    ticks = [0]
    api = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    drafts = httpx.ASGITransport(app=owners_draft.app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(transport=api, base_url="http://test") as client,
        httpx.AsyncClient(transport=drafts, base_url=owners_draft.base_url) as draft_client,
    ):
        beat = asyncio.create_task(_heartbeat(ticks))
        started = loop.time()
        draft = asyncio.create_task(draft_client.get(owners_draft.path, headers=owners_draft.headers))
        try:
            latencies = await _live_latencies(client, loop, draft.done)
            response = await draft
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
        elapsed = loop.time() - started

    assert response.status_code == 200, response.text
    assert "brief" in response.text
    assert elapsed >= SLOW_SECONDS
    assert latencies, "the draft answered before liveness was asked"
    assert max(latencies) < LIVE_BUDGET_SECONDS, f"liveness waited {max(latencies):.2f}s"
    # A free loop ticks about 100 times a second here; a blocked one stops.
    assert ticks[0] >= 50 * elapsed * 0.5, f"{ticks[0]} heartbeat ticks in {elapsed:.1f}s"


async def test_a_draft_builds_its_preview_in_the_slot_it_opened_the_file_in(owners_draft, monkeypatch):
    """Every other file store slot fills while the draft opens its file, and
    the next one to free is promised to another caller. The draft still
    answers with its preview: it never waits for a second slot after the
    open, so it is not refused with 503 once its file is open."""
    real = artifact_files.open_pinned_draft_file
    opening = threading.Event()
    filled = threading.Event()

    def open_while_the_store_fills(*args, **kwargs):
        opening.set()
        filled.wait(timeout=10)
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_files, "open_pinned_draft_file", open_while_the_store_fills)
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    slots = units._efs_slots()
    holders = [object() for _ in range(units.EFS_SLOTS)]
    next_free: asyncio.Task | None = None
    transport = httpx.ASGITransport(app=owners_draft.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url=owners_draft.base_url) as client:
        draft = asyncio.create_task(client.get(owners_draft.path, headers=owners_draft.headers))
        try:
            assert await asyncio.to_thread(opening.wait, 5), "the draft never opened its file"
            # Every slot but the draft's, then a wait for the draft's.
            for holder in holders[:-1]:
                await slots.acquire_on_behalf_of(holder)
            next_free = asyncio.create_task(slots.acquire_on_behalf_of(holders[-1]))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while slots.statistics().tasks_waiting == 0:
                assert loop.time() < deadline, "the draft held no file store slot while it opened"
                await asyncio.sleep(0.01)
            filled.set()
            response = await asyncio.wait_for(draft, timeout=5)
        finally:
            filled.set()
            if next_free is not None:
                await asyncio.gather(next_free, return_exceptions=True)
            for holder in holders:
                if holder in slots.statistics().borrowers:
                    slots.release_on_behalf_of(holder)
            await asyncio.gather(draft, return_exceptions=True)

    assert response.status_code == 200, response.text
    assert "brief" in response.text


def _stick_lookups(monkeypatch, *, sticks=lambda _request: True):
    """Make each artifact root lookup whose request ``sticks`` picks block
    until the returned event is set, as a call on a hard NFS mount does while
    the file server doesn't answer. Other lookups run. Returns the counts of
    lookups blocked now and at most, and the event. Also cuts POOL_TIMEOUT to
    1 s."""
    real = artifact_identity.opened_artifact_root
    released = threading.Event()
    state = {"blocked": 0, "most": 0}
    lock = threading.Lock()

    @contextlib.contextmanager
    def stuck(source):
        if not sticks(_request.get(None)):
            with real(source) as root:
                yield root
            return
        with lock:
            state["blocked"] += 1
            state["most"] = max(state["most"], state["blocked"])
        try:
            released.wait(timeout=60)
        finally:
            with lock:
                state["blocked"] -= 1
        raise FileNotFoundError("released")

    monkeypatch.setattr(artifact_identity, "opened_artifact_root", stuck)
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)
    return state, released


@pytest.fixture
def stuck_lookup(monkeypatch, seeded_artifact):
    """Every artifact root lookup blocks until the test releases it."""
    state, released = _stick_lookups(monkeypatch)
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


ORG = "5c0a1e5e-0000-4000-8000-000000000001"
FIRST_MEMBER = "5c0a1e5e-0000-4000-8000-0000000000a1"
SECOND_MEMBER = "5c0a1e5e-0000-4000-8000-0000000000b2"


def _as(user_id: str) -> dict[str, str]:
    """What the gateway injects on an org-mode request from ``user_id``."""
    return {"X-User-Id": user_id, "X-Organization-Id": ORG}


@pytest.fixture
def org_project_review(org_deployment, cleanup_tmp_projects, tmp_path):
    """An artifact in the shared base of one of ORG's projects, which every
    member may review. Returns the review route's path."""
    project_id = uuid4()
    project = tmp_path / "shared"
    folder = project / ".anton" / "artifacts" / "brief"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"id": ARTIFACT_ID, "type": "html-app", "title": "Brief", "primary": "index.html"})
    )
    (folder / "index.html").write_text("<!doctype html><p>brief</p>")
    with Session(db_session.get_engine(get_app_settings().database.uri)) as session:
        session.add(Project(id=project_id, name="shared", path=str(project), org_id=ORG))
        session.commit()
    artifact_identity._clear_identity_indexes()
    yield f"/api/v1/artifacts/workspace/{project_id}/{ARTIFACT_ID}/review"
    artifact_identity._clear_identity_indexes()


@pytest.fixture
def first_members_lookups_stuck(monkeypatch, org_project_review):
    """FIRST_MEMBER's artifact root lookups block until the test releases
    them; SECOND_MEMBER's run."""
    state, released = _stick_lookups(
        monkeypatch, sticks=lambda request: request is not None and request["user"] == FIRST_MEMBER
    )
    yield state, released
    released.set()


async def test_a_second_member_gets_a_slot_while_the_first_holds_their_share(
    org_project_review, first_members_lookups_stuck, granted_product_permissions,
):
    """Two members of one organization. The first fires as many reviews as
    there are file store slots, and every one sticks on the file store: the
    first member holds their share, the rest of their reviews are refused,
    and the second member's review still gets a slot and answers."""
    state, released = first_members_lookups_stuck
    loop = asyncio.get_running_loop()
    held = units.EFS_SHARE
    transport = httpx.ASGITransport(app=_marked(create_app()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        firsts = [
            asyncio.create_task(client.get(org_project_review, headers=_as(FIRST_MEMBER)))
            for _ in range(units.EFS_SLOTS)
        ]
        refused: list[httpx.Response] = []
        try:
            started = loop.time()
            while len(refused) < units.EFS_SLOTS - held and loop.time() - started < 5:
                for task in [t for t in firsts if t.done()]:
                    firsts.remove(task)
                    refused.append(task.result())
                await asyncio.sleep(0.05)
            blocked_before = state["blocked"]
            second = await client.get(org_project_review, headers=_as(SECOND_MEMBER))
            blocked_after = state["blocked"]
        finally:
            released.set()
            await asyncio.gather(*firsts)

    assert second.status_code == 200, second.text
    assert blocked_before == blocked_after == held
    assert len(refused) == units.EFS_SLOTS - held
    for response in refused:
        assert response.status_code == 503, response.text
        assert response.json()["code"] == "server_busy"


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


@pytest.mark.parametrize("roots", ["owner_filtered", "every_one_owned", "unfiltered"])
def test_a_warm_lookup_skips_conversation_folders_that_hold_no_artifacts(tmp_path, monkeypatch, roots):
    """Org mode, a project with 100 conversation folders and no legacy
    artifacts: the folders are not roots, so a lookup doesn't index each one.

    The saving shows where every folder would be searched: a reviewer's
    unfiltered lookup, or a caller who owns every conversation. There an
    empty folder counted as a root costs about 1,039 calls. The owner filter
    drops another member's folders either way, so its lookup reads the same
    with or without the saving."""
    monkeypatch.setattr(artifact_roots, "_org_mode", lambda: True)
    project = tmp_path / "project"
    folder = project / ".anton" / "artifacts" / "brief"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"id": ARTIFACT_ID, "type": "document"}))
    conversation_ids = [uuid4() for _ in range(100)]
    for conversation_id in conversation_ids:
        (project / "conversations" / str(conversation_id)).mkdir(parents=True)
    row = artifact_roots.ProjectRow(id=UUID(int=1), name="project", path=str(project), external=False)
    owned = frozenset(conversation_ids) if roots == "every_one_owned" else frozenset()
    plan = artifact_roots.RootsPlan(projects=(row,), owner_filter=True, owned=owned)
    artifact_identity._clear_identity_indexes()

    def lookup():
        found = artifact_roots.discover_roots(plan)
        assert [root for root in found.roots if root.legacy] == []
        sources = found.unfiltered() if roots == "unfiltered" else found.sources()
        return artifact_identity.resolve_artifact_folder(sources, ARTIFACT_ID)

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


# ─── one file store wait per read ──────────────────────────────────────────
#
# Each run_efs call waits for a file store slot and can refuse the request
# with 503 when none frees in time, so a route that waited again after its
# lookup could be refused with the lookup's work done. A route whose
# authorization ends with the review reads in the lookup's own call. An
# owner-only route checks ownership in a database unit after the lookup, so
# it reads in one more call, and only once that check has passed.

# The project row, the owned conversations and the location row, then the
# lookup.
LOOKUP = ["run_db", "run_efs"]
# A lookup whose location row does not answer it: the scan, then the unit that
# records what the scan found, whose refusal is only logged.
SCANNED_LOOKUP = [*LOOKUP, "record"]
# The lookup, the owner check, then the route's one read.
OWNER_READ = [*LOOKUP, "run_db", "run_efs"]


@pytest.fixture
async def owned(session, tmp_path, org_deployment, cleanup_tmp_projects, granted_product_permissions, monkeypatch):
    """USER_A's artifact in an ORG_A project, opened for same-org review.
    Returns (project_id, artifact_id)."""
    row, _folder, card = await tenancy._grant_draft_review(session, tmp_path, monkeypatch)
    return str(row.id), card["id"]


def _caller(user_id: str = tenancy.USER_A) -> TenantScope:
    """The tenant scope a request of ``user_id`` of ORG_A carries."""
    return TenantScope(org_mode=True, org_id=tenancy.ORG_A, user_id=user_id)


def _access_reads(monkeypatch) -> list[Path]:
    """Record each read of an artifact's access state, by folder."""
    from cowork.services import artifacts

    reads: list[Path] = []
    real = artifacts._published_access_for

    def published_access_for(folder, primary, **kwargs):
        reads.append(folder)
        return real(folder, primary, **kwargs)

    monkeypatch.setattr(artifacts, "_published_access_for", published_access_for)
    return reads


def _on_efs_threads(monkeypatch, module, name: str) -> list[bool]:
    """Record, for each call of ``module.name``, whether it ran in a file
    store call (run_efs) rather than on the event loop."""
    calls: list[bool] = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append(units.on_efs_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


@pytest.mark.parametrize(("user_id", "role"), [(tenancy.USER_A, "owner"), (tenancy.USER_A2, "reviewer")])
async def test_a_review_entry_reads_its_revision_in_the_lookups_file_store_call(
    owned, monkeypatch, user_id, role,
):
    project_id, artifact_id = owned
    reads = _on_efs_threads(monkeypatch, artifact_workspace, "current_source")
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    entry = await artifact_workspace.artifact_review_entry(project_id, artifact_id, _caller(user_id))

    assert entry["capabilities"]["role"] == role
    assert entry["currentRevision"]["artifactId"] == artifact_id
    assert reads == [True]
    # Then the capabilities, in a database unit.
    assert hops == [*LOOKUP, "run_db"]


@pytest.mark.parametrize("error", [FileNotFoundError, OSError, ValueError, TimeoutError])
async def test_a_review_entry_with_no_readable_source_still_answers_without_a_revision(
    owned, monkeypatch, error,
):
    project_id, artifact_id = owned

    def unreadable(*_args):
        raise error("binary or oversized")

    monkeypatch.setattr(artifact_workspace, "current_source", unreadable)
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    entry = await artifact_workspace.artifact_review_entry(project_id, artifact_id, _caller(tenancy.USER_A2))

    assert entry["currentRevision"] is None
    assert entry["capabilities"]["role"] == "reviewer"
    assert hops == [*LOOKUP, "run_db"]


async def test_a_draft_link_names_its_file_in_the_lookups_file_store_call(
    session, tmp_path, cleanup_tmp_projects, granted_product_permissions, monkeypatch,
):
    configure_draft_env(monkeypatch)
    row, folder = tenancy._project_with_artifact(
        session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)
    names = _on_efs_threads(monkeypatch, artifact_workspace, "_draft_link_file_parts")
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    try:
        response = await artifact_workspace.create_draft_link(str(row.id), artifact_id, _caller())
    finally:
        get_app_settings.cache_clear()

    assert response.status_code == 200, response.body
    assert json.loads(response.body)["url"].endswith("/index.html")
    assert names == [True]
    assert hops == SCANNED_LOOKUP


class _PinnedOpen(NamedTuple):
    """One pinned open of a draft file, as its spy saw it."""

    parts: tuple[str, ...]
    on_efs_thread: bool


def _pinned_opens(monkeypatch) -> list[_PinnedOpen]:
    """Record each pinned open of a draft file, by its path parts, and
    whether it ran in a file store call."""
    opens: list[_PinnedOpen] = []
    real = artifact_files.open_pinned_draft_file

    def open_pinned_draft_file(source, folder, parts):
        opens.append(_PinnedOpen(tuple(parts), units.on_efs_thread()))
        return real(source, folder, parts)

    monkeypatch.setattr(artifact_files, "open_pinned_draft_file", open_pinned_draft_file)
    return opens


async def _preview(route: str, project_id: str, artifact_id: str, caller):
    """The draft's index.html as ``caller`` asks for it: through the
    authenticated /drafts route, or through the draft host with a link
    issued to the same person."""
    if route == "drafts":
        return await artifact_workspace.serve_private_draft(
            project_id, artifact_id, "index.html", SimpleNamespace(query_params={}), caller
        )
    grant = draft_app._DraftGrant(
        user_id=str(caller.user_id),
        org_id=str(caller.org_id),
        project_ref=project_id,
        artifact_id=artifact_id,
    )
    request = SimpleNamespace(path_params={"rel_path": "index.html"}, query_params={})
    return await draft_app._draft_response(request, grant)


@pytest.mark.parametrize("route", ["drafts", "draft_host"])
@pytest.mark.parametrize("user_id", [tenancy.USER_A, tenancy.USER_A2], ids=["owner", "reviewer"])
async def test_a_draft_file_is_opened_in_the_lookups_file_store_call(owned, monkeypatch, route, user_id):
    """Every asset of a preview is a request of its own, so the path checks
    and the open run in the lookup's file store call: one wait for a slot."""
    project_id, artifact_id = owned
    opens = _pinned_opens(monkeypatch)
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    response = await _preview(route, project_id, artifact_id, _caller(user_id))

    assert response.status_code == 200
    assert opens == [_PinnedOpen(("index.html",), True)]
    assert hops == LOOKUP


@pytest.mark.parametrize("route", ["drafts", "draft_host"])
async def test_a_draft_file_is_never_opened_for_a_caller_the_lookup_refuses(
    session, tmp_path, org_deployment, cleanup_tmp_projects, granted_product_permissions, monkeypatch, route,
):
    row, folder = tenancy._project_with_artifact(
        session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash"
    )
    artifact_id, _metadata = ensure_full_id(folder)
    opens = _pinned_opens(monkeypatch)
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    # A co-member, and the owner never opened the draft for review.
    with pytest.raises(HTTPException) as refused:
        await _preview(route, str(row.id), artifact_id, _caller(tenancy.USER_A2))

    assert refused.value.status_code == 404
    assert opens == []
    assert hops == SCANNED_LOOKUP


async def _edit(project_id: str, artifact_id: str, caller):
    """The owner's save of a new source, ready to call."""
    current = await artifact_workspace.artifact_source(project_id, artifact_id, caller, path=None)
    return partial(
        artifact_workspace.update_artifact_source,
        project_id,
        artifact_id,
        artifact_workspace._SourceUpdateBody(
            content="<html><h1>Edited</h1></html>",
            expectedRevisionId=current["revision"]["id"],
            path="index.html",
        ),
        caller,
    )


async def _restore_the_first_revision(project_id: str, artifact_id: str, caller):
    """After one edit, the owner's restore of the source before it, ready to call."""
    await (await _edit(project_id, artifact_id, caller))()
    revisions = (await artifact_workspace.artifact_revisions(project_id, artifact_id, caller, path=None))[
        "revisions"
    ]
    first, head = revisions[-1], revisions[0]
    return partial(
        artifact_workspace.restore_artifact_revision,
        project_id,
        artifact_id,
        first["id"],
        artifact_workspace._RestoreBody(expectedRevisionId=head["id"]),
        caller,
    )


@pytest.mark.parametrize("route", ["workspace", "save", "access", "comments_access", "restore"])
async def test_an_owner_route_reads_once_after_its_owner_check(owned, monkeypatch, route):
    project_id, artifact_id = owned
    synced: list[Path] = []

    async def sync_live_artifact(_scope, folder, *, project_id=None):
        synced.append(folder)

    # A save that changed the source republishes a live artifact afterwards:
    # its own operation, with the publish lock's waits, not the route's read.
    monkeypatch.setattr(artifact_workspace, "_sync_live_artifact", sync_live_artifact)
    caller = _caller()
    if route == "save":
        call = await _edit(project_id, artifact_id, caller)
    elif route == "restore":
        call = await _restore_the_first_revision(project_id, artifact_id, caller)
    else:
        call = {
            "workspace": partial(artifact_workspace.artifact_source, project_id, artifact_id, caller, path=None),
            "access": partial(artifact_workspace.artifact_access, project_id, artifact_id, caller),
            "comments_access": partial(
                artifact_workspace.enable_artifact_comments, project_id, artifact_id, caller
            ),
        }[route]
    synced.clear()
    access_reads = _access_reads(monkeypatch)
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    await call()

    assert hops == OWNER_READ
    assert len(synced) == (1 if route in ("save", "restore") else 0)
    assert len(access_reads) == (1 if route == "access" else 0)


@pytest.mark.parametrize(("granted", "status_code"), [(True, 403), (False, 404)], ids=["reviewer", "no_grant"])
async def test_a_non_owner_is_refused_access_before_any_read(
    session, tmp_path, org_deployment, cleanup_tmp_projects, granted_product_permissions, monkeypatch,
    granted, status_code,
):
    if granted:
        row, _folder, card = await tenancy._grant_draft_review(session, tmp_path, monkeypatch)
        artifact_id = card["id"]
    else:
        row, folder = tenancy._project_with_artifact(
            session, tmp_path, name="mine", org_id=tenancy.ORG_A, slug="dash"
        )
        artifact_id, _metadata = ensure_full_id(folder)
    access_reads = _access_reads(monkeypatch)
    hops = _count_hops(monkeypatch, artifact_scope, artifact_workspace)

    with pytest.raises(HTTPException) as refused:
        await artifact_workspace.artifact_access(str(row.id), artifact_id, _caller(tenancy.USER_A2))

    assert refused.value.status_code == status_code
    # A reviewer's lookup opens the row's folder and the owner check refuses;
    # with no grant, the lookup scans, records its folder and refuses. Either
    # way the access state is never read.
    assert access_reads == []
    assert hops == ([*LOOKUP, "run_db"] if granted else SCANNED_LOOKUP)
