"""POST .../quick-edits (ENG-4227): the viewer's chat bar edits a draft directly.

anton's router and quick editor are faked with the contract's signatures
(``route_request(message, *, files, kind, base_url, api_key, ...)`` and
``run_quick_edit(folder, message, *, kind, llm, checker, history,
on_progress)``); the revision journal, the preview check broker and the route
are the real ones, on a real artifact folder.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from _async_stub import awaitable
from cowork.api.v1.endpoints import artifact_quick_edit as quick_ep
from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.services import quick_edit as quick_edit_service
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_revisions import (
    create_agent_repair,
    current_source,
    list_revisions,
    record_quick_edit,
)
from cowork.services.quick_edit import AntonQuickEdit, Decisions, QuickEditBackend
from cowork.streaming.preview_checks import CHECK_ID_PREFIX, preview_checks

CONVERSATION = "33333333-3333-4333-8333-333333333333"
ORIGINAL = "<!doctype html><h1>Sales</h1>\n"
EDITED = "<!doctype html><h1 style=\"color:blue\">Sales</h1>\n"


_real_decisions_for_scope = quick_edit_service.decisions_for_scope
_real_resolve_backend = quick_edit_service.resolve_backend


# ── fakes for anton's API ────────────────────────────────────────────────


@dataclass
class RouteDecision:
    route: str
    confidence: float
    reason: str = "jev"
    ms: int = 12


@dataclass
class CheckResult:
    status: str
    channel: str = "preview"
    findings: list = field(default_factory=list)
    screenshot_b64: str | None = "aGVsbG8="
    ms: int = 0
    note: str = ""


@dataclass
class QuickEditResult:
    status: str
    changed: list[str]
    summary: str
    escalate_reason: str | None = None
    check: CheckResult | None = None
    rounds: int = 0
    llm_calls: int = 1
    ms: int = 5


class FakePreviewChecker:
    """anton's checker chain ending in the preview: asks the host's
    ``preview_checker(slug, timeout_s)`` and reads the answer."""

    def __init__(self, preview_checker, slug):
        self.ask = preview_checker
        self.slug = slug

    async def check(self, ws, changed):
        answer = await self.ask(self.slug, 10)
        if answer is None:
            return CheckResult("unverified", note="no open preview answered the check")
        return CheckResult(answer["status"], findings=[
            SimpleNamespace(kind=e["type"], message=e["message"], source=e.get("file") or None)
            for e in answer.get("errors") or []
        ])


@dataclass
class FakeAnton:
    """Records calls; ``edit`` decides what run_quick_edit does."""

    route: RouteDecision = field(default_factory=lambda: RouteDecision("edit", 0.93))
    edit: object = None
    route_calls: list = field(default_factory=list)
    edit_calls: list = field(default_factory=list)

    async def route_request(self, message, *, files, kind, base_url, api_key, model="jev-1.13.0", timeout_s=3.0,
                            http=None, name="", description="", data="", history=None):
        self.route_calls.append(dict(message=message, files=files, kind=kind, base_url=base_url, api_key=api_key,
                                     model=model, name=name, description=description, data=data, history=history))
        return self.route

    async def run_quick_edit(self, folder, message, *, kind, llm, checker, history=None, on_progress=None, max_fix_rounds=2):
        self.edit_calls.append(dict(folder=folder, message=message, kind=kind, llm=llm, history=history))
        return await self.edit(folder, checker, on_progress)

    def api(self) -> AntonQuickEdit:
        checkers = SimpleNamespace(select_checker=lambda kind, *, slug, preview_checker: FakePreviewChecker(preview_checker, slug))
        workspace_cls = lambda folder, *, kind: SimpleNamespace(  # noqa: E731
            map=lambda: [SimpleNamespace(path=p.name, bytes=p.stat().st_size) for p in sorted(Path(folder).iterdir()) if p.is_file()]
        )
        return AntonQuickEdit(
            route_request=self.route_request,
            run_quick_edit=self.run_quick_edit,
            checkers=checkers,
            workspace_cls=workspace_cls,
            # anton's rule: a decisions model only on MindsHub here.
            describe_artifact=lambda folder: {"name": "Sales", "description": "Revenue by region", "data": "{rows: [...]}"},
            route_extras=frozenset({"name", "description", "data", "history"}),
            decisions_available=lambda settings: settings.planning_provider == "minds-cloud",
            decisions_config=lambda settings: (
                ("https://hub.example/v1", settings.minds_api_key, "jev-1.13.0")
                if settings.planning_provider == "minds-cloud" else None
            ),
        )


async def answer_the_check(status="passed", errors=None):
    """The viewer: answer the first preview check that opens, as the web's
    POST /api/v1/artifacts/checks/{check_id} does."""
    for _ in range(500):
        for check_id in list(preview_checks._pending):
            preview_checks.submit(check_id, {"status": status, "errors": errors or [], "ms": 7})
            return
        await asyncio.sleep(0.01)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def artifact(tmp_path):
    folder = tmp_path / "sales-dashboard"
    folder.mkdir()
    metadata = {
        "id": "a1b2c3d4",
        "slug": "sales-dashboard",
        "createdAt": "2026-11-12T12:00:00+00:00",
        "name": "Sales",
        "type": "html-app",
        "primary": "index.html",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "index.html").write_text(ORIGINAL, encoding="utf-8")
    artifact_id, metadata = ensure_full_id(folder, metadata)
    return folder, metadata, artifact_id


@pytest.fixture(autouse=True)
def _clean():
    yield
    preview_checks.reset()
    quick_edit_service.quick_edit_locks.reset()
    get_app_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    # No `with`: the app's startup (migrations) is not what these test.
    yield TestClient(create_app())


@pytest.fixture
def fake_anton(monkeypatch, artifact):
    folder, metadata, _ = artifact
    anton = FakeAnton()
    monkeypatch.setattr(quick_edit_service, "anton_quick_edit", lambda: anton.api())
    monkeypatch.setattr(
        quick_edit_service,
        "resolve_backend",
        awaitable(lambda *_a, **_k: QuickEditBackend(llm="the-llm", route_base_url="https://hub.example/v1", route_api_key="k")),
    )
    monkeypatch.setattr(
        quick_edit_service, "decisions_for_scope",
        awaitable(lambda anton, scope: ("the-user", Decisions(available=True, model="jev-1.13.0"))),
    )
    monkeypatch.setattr(quick_ep, "_owner_workspace", awaitable(lambda *_a: (None, folder, metadata, {"canEdit": True})))
    monkeypatch.setattr(
        workspace_ep, "_owner_workspace",
        awaitable(lambda *_a: (SimpleNamespace(project_id=None), folder, metadata, {"canEdit": True})),
    )
    monkeypatch.setattr(workspace_ep, "_sync_live_artifact", awaitable(lambda *_a, **_k: None))
    return anton


def events_of(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def post(client, artifact_id, **body):
    body.setdefault("message", "Make the title blue")
    return client.post(f"/api/v1/artifacts/workspace/project/{artifact_id}/quick-edits", json=body)


# ── happy path ────────────────────────────────────────────────────────────


def test_an_applied_edit_streams_route_progress_check_and_saves_one_revision(client, fake_anton, artifact):
    folder, metadata, artifact_id = artifact
    initial = current_source(folder, metadata, artifact_id)["revision"]

    async def edit(folder, checker, on_progress):
        on_progress({"stage": "reading"})
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        on_progress({"stage": "checking"})
        web = asyncio.ensure_future(answer_the_check("passed"))
        check = await checker.check(None, ["index.html"])
        await web
        return QuickEditResult("applied", ["index.html"], "Made the title blue.", check=check)

    fake_anton.edit = edit
    response = post(
        client, artifact_id,
        expectedRevisionId=initial["id"],
        history=[{"role": "user", "text": "Add a title"}, {"role": "assistant", "text": "Added."}],
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = events_of(response.text)
    names = [name for name, _ in events]
    assert names == ["route", "progress", "progress", "check_request", "done"]
    assert events[0][1] == {"route": "edit", "confidence": 0.93, "ms": events[0][1]["ms"], "reason": "jev"}
    assert [data["stage"] for name, data in events if name == "progress"] == ["reading", "checking"]
    check_request = events[3][1]
    assert check_request["check_id"].startswith(CHECK_ID_PREFIX)
    assert check_request["artifact_slug"] == folder.name
    assert check_request["timeout_s"] == 10

    done = events[-1][1]
    revisions = list_revisions(folder)
    assert len(revisions) == 2
    assert done["status"] == "applied"
    assert done["summary"] == "Made the title blue."
    assert done["revisionId"] == revisions[0]["id"]
    assert done["previousRevisionId"] == initial["id"]
    assert done["changed"] == ["index.html"]
    assert done["check"]["status"] == "passed"
    assert "screenshot_b64" not in done["check"]
    assert revisions[0]["summary"] == "Quick edit: Made the title blue."
    assert revisions[0]["baseRevisionId"] == initial["id"]

    # The router saw the message and the artifact's files; the editor got the
    # bar's thread and the caller's LLM.
    assert fake_anton.route_calls[0]["files"] == ["index.html", "metadata.json"]
    assert fake_anton.route_calls[0]["kind"] == "html-app"
    assert fake_anton.edit_calls[0]["llm"] == "the-llm"
    assert fake_anton.edit_calls[0]["history"][0] == {"role": "user", "text": "Add a title"}

    # Undo is the existing restore route.
    undo = client.post(
        f"/api/v1/artifacts/workspace/project/{artifact_id}/revisions/{done['previousRevisionId']}/restore",
        json={"expectedRevisionId": done["revisionId"]},
    )
    assert undo.status_code == 200, undo.text
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL
    # The lock and the check are gone.
    assert not quick_edit_service.quick_edit_locks._held
    assert not preview_checks._pending


def test_a_failed_preview_answer_reaches_done_with_its_findings(client, fake_anton, artifact):
    _folder, _metadata, artifact_id = artifact

    async def edit(folder, checker, on_progress):
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        web = asyncio.ensure_future(answer_the_check("failed", [{"type": "error", "message": "x is undefined", "file": "index.html", "line": 3, "url": ""}]))
        check = await checker.check(None, ["index.html"])
        await web
        return QuickEditResult("applied", ["index.html"], "Made the title blue.", check=check)

    fake_anton.edit = edit
    done = events_of(post(client, artifact_id).text)[-1][1]
    assert done["check"]["status"] == "failed"
    assert done["check"]["findings"] == [{"kind": "error", "message": "x is undefined", "source": "index.html"}]


def test_an_applied_edit_is_noted_in_the_artifacts_conversation(client, fake_anton, artifact, monkeypatch):
    _folder, _metadata, artifact_id = artifact
    notes = []

    async def note(scope, *, conversation_id, message, text):
        notes.append((str(conversation_id), message, text))
        return True

    monkeypatch.setattr(quick_edit_service, "record_conversation_note", note)

    async def edit(folder, checker, on_progress):
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        return QuickEditResult("applied", ["index.html"], "Made the title blue.")

    fake_anton.edit = edit
    done = events_of(post(client, artifact_id, conversationId=CONVERSATION).text)[-1][1]
    assert done["status"] == "applied"
    assert len(notes) == 1
    conversation_id, message, text = notes[0]
    assert (conversation_id, message) == (CONVERSATION, "Make the title blue")
    assert "Made the title blue." in text and "index.html" in text and "revision 2" in text


# ── the other outcomes ───────────────────────────────────────────────────


def test_the_agent_route_ends_the_stream_without_editing(client, fake_anton, artifact):
    folder, _metadata, artifact_id = artifact
    fake_anton.route = RouteDecision("agent", 0.97)
    fake_anton.edit = None  # would raise if called

    events = events_of(post(client, artifact_id, conversationId=CONVERSATION).text)

    assert [name for name, _ in events] == ["route", "done"]
    assert events[0][1]["route"] == "agent"
    assert events[1][1] == {
        "status": "agent", "summary": "", "revisionId": None, "previousRevisionId": None,
        "changed": [], "check": None, "editId": None, "undoable": False, "ms": events[1][1]["ms"],
    }
    assert fake_anton.edit_calls == []
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL


def test_force_route_skips_the_router(client, fake_anton, artifact):
    _folder, _metadata, artifact_id = artifact
    events = events_of(post(client, artifact_id, forceRoute="agent").text)
    assert events[0][1]["route"] == "agent" and events[0][1]["confidence"] == 1.0
    assert fake_anton.route_calls == []


def test_without_a_decision_endpoint_the_message_goes_to_the_agent(client, fake_anton, artifact, monkeypatch):
    _folder, _metadata, artifact_id = artifact
    monkeypatch.setattr(quick_edit_service, "resolve_backend", awaitable(lambda *_a, **_k: QuickEditBackend(llm="the-llm")))
    events = events_of(post(client, artifact_id).text)
    assert events[0][1]["route"] == "agent"
    assert events[0][1]["reason"] == "no_decisions_model"
    assert events[-1][1]["status"] == "agent"
    assert fake_anton.route_calls == []


@pytest.mark.parametrize("force", [None, "edit"])
def test_no_decisions_model_goes_straight_to_the_chat(client, fake_anton, artifact, monkeypatch, force):
    """A person on their own OpenAI or Anthropic keys has no decisions model:
    neither Jev, nor the LLM, nor the editor runs, even with the edit forced."""
    folder, _metadata, artifact_id = artifact
    monkeypatch.setattr(
        quick_edit_service, "decisions_for_scope", awaitable(lambda anton, scope: (None, Decisions(False))),
    )
    backends = []
    monkeypatch.setattr(quick_edit_service, "resolve_backend", awaitable(lambda *a: backends.append(a)))
    fake_anton.edit = None

    events = events_of(post(client, artifact_id, **({"forceRoute": force} if force else {})).text)

    assert events[0] == ("route", {"route": "agent", "confidence": 1.0, "ms": 0, "reason": "no_decisions_model"})
    assert [name for name, _ in events] == ["route", "done"]
    assert events[1][1]["status"] == "agent"
    assert fake_anton.route_calls == [] and fake_anton.edit_calls == [] and backends == []
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL


def test_an_organization_that_turned_jev_off_has_no_decisions():
    from cowork.common.settings.user_settings import Provider, UserSettings

    user = UserSettings(planning_provider=Provider.MINDS_CLOUD, minds_api_key="k", jev_decisions_enabled=False)
    assert quick_edit_service.decisions_for(FakeAnton().api(), user).available is False


def test_on_mindshub_the_router_gets_anton_s_decisions_model(client, fake_anton, artifact):
    _folder, _metadata, artifact_id = artifact
    fake_anton.route = RouteDecision("agent", 0.6, reason="uncertain")
    events = events_of(post(client, artifact_id, history=[{"role": "user", "text": "Add a title"}]).text)
    call = fake_anton.route_calls[0]
    assert call["model"] == "jev-1.13.0"
    # The artifact's context (anton's describe_artifact) and the bar's thread.
    assert (call["name"], call["description"], call["data"]) == ("Sales", "Revenue by region", "{rows: [...]}")
    assert call["history"] == [{"role": "user", "text": "Add a title"}]
    assert events[0][1]["reason"] == "uncertain"


@pytest.mark.parametrize(
    "provider,key,expected",
    [("minds_cloud", "minds_api_key", True), ("anthropic", "anthropic_api_key", False), ("openai", "openai_api_key", False)],
)
def test_decisions_are_read_from_anton_settings_built_like_a_turn(provider, key, expected):
    from cowork.common.settings.user_settings import Provider, UserSettings

    user = UserSettings(planning_provider=Provider(provider), **{key: "k"})
    decisions = quick_edit_service.decisions_for(FakeAnton().api(), user)
    assert decisions.available is expected
    assert decisions.model == ("jev-1.13.0" if expected else None)


def test_an_anton_without_the_decisions_check_has_no_decisions():
    from cowork.common.settings.user_settings import Provider, UserSettings

    api = FakeAnton().api()
    bare = AntonQuickEdit(route_request=api.route_request, run_quick_edit=api.run_quick_edit, checkers=None, workspace_cls=None)
    user = UserSettings(planning_provider=Provider.MINDS_CLOUD, minds_api_key="k")
    assert quick_edit_service.decisions_for(bare, user).available is False
    assert quick_edit_service.decisions_for(None, user).available is False


@pytest.mark.parametrize("status", ["escalate", "failed", "noop"])
def test_an_edit_that_did_not_apply_saves_no_revision(client, fake_anton, artifact, status):
    folder, metadata, artifact_id = artifact
    current_source(folder, metadata, artifact_id)

    async def edit(folder, checker, on_progress):
        # anton restores the folder on escalate and failed; noop wrote nothing.
        return QuickEditResult(status, [], "This needs the sales data.", escalate_reason="needs data" if status == "escalate" else None)

    fake_anton.edit = edit
    done = events_of(post(client, artifact_id).text)[-1][1]

    assert done["status"] == status
    assert done["summary"] == "This needs the sales data."
    assert done["revisionId"] is None and done["previousRevisionId"] is None
    assert len(list_revisions(folder)) == 1
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL


def test_an_edit_past_its_time_is_cancelled_and_reported_failed(client, fake_anton, artifact, monkeypatch):
    folder, _metadata, artifact_id = artifact
    monkeypatch.setattr(quick_ep, "EDIT_TIMEOUT_S", 0.2)
    cancelled = []

    async def edit(folder, checker, on_progress):
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            (folder / "index.html").write_text(ORIGINAL, encoding="utf-8")  # anton restores
            cancelled.append(True)
            raise

    fake_anton.edit = edit
    done = events_of(post(client, artifact_id).text)[-1][1]
    assert done["status"] == "failed"
    assert cancelled == [True]
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL


def test_an_editor_crash_is_an_error_event(client, fake_anton, artifact):
    _folder, _metadata, artifact_id = artifact

    async def edit(folder, checker, on_progress):
        raise RuntimeError("provider exploded with a secret in it")

    fake_anton.edit = edit
    events = events_of(post(client, artifact_id).text)
    assert events[-1][0] == "error"
    assert "secret" not in events[-1][1]["message"]
    assert not quick_edit_service.quick_edit_locks._held


# ── refusals before the stream ───────────────────────────────────────────


def test_a_second_quick_edit_on_the_same_artifact_is_409(client, fake_anton, artifact):
    _folder, _metadata, artifact_id = artifact
    assert quick_edit_service.quick_edit_locks.acquire(quick_ep._lock_key(LOCAL_SCOPE, artifact_id))
    response = post(client, artifact_id)
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "quick_edit_running"


def test_a_pending_agent_repair_is_409(client, fake_anton, artifact):
    folder, metadata, artifact_id = artifact
    head = current_source(folder, metadata, artifact_id)["revision"]["id"]
    repair = create_agent_repair(
        folder, metadata, artifact_id, expected_revision_id=head, comment_thread_id="thread-1",
        selector=None, thread=[{"text": "Fix"}], conversation_id=CONVERSATION,
    )["repair"]

    response = post(client, artifact_id, expectedRevisionId=head)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "repair_pending"
    assert detail["repairId"] == repair["id"]
    assert detail["commentThreadId"] == "thread-1"
    # Refused before anything ran, and the lock is free again.
    assert fake_anton.route_calls == []
    assert not quick_edit_service.quick_edit_locks._held


def test_a_stale_revision_is_409_in_the_draft_puts_shape(client, fake_anton, artifact):
    folder, metadata, artifact_id = artifact
    head = current_source(folder, metadata, artifact_id)["revision"]
    response = post(client, artifact_id, expectedRevisionId="not-the-head")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["currentRevision"]["id"] == head["id"]
    assert set(detail) == {"message", "currentRevision"}
    assert fake_anton.route_calls == []
    assert not quick_edit_service.quick_edit_locks._held


def test_without_anton_support_the_route_is_501_and_the_capability_false(client, artifact, monkeypatch):
    # An anton without the quick editor: hide its modules from the import, so
    # this holds whichever anton is installed.
    import sys

    folder, metadata, artifact_id = artifact
    monkeypatch.setattr(quick_ep, "_owner_workspace", awaitable(lambda *_a: (None, folder, metadata, {"canEdit": True})))
    monkeypatch.setitem(sys.modules, "anton.core.artifact_editor.quick", None)
    monkeypatch.setitem(sys.modules, "anton.core.artifact_editor.router", None)
    assert quick_edit_service.anton_quick_edit() is None

    response = post(client, artifact_id)

    assert response.status_code == 501
    assert quick_edit_service.quick_edit_available() is False


def test_the_setting_turns_quick_edits_off(client, fake_anton, artifact, monkeypatch):
    _folder, _metadata, artifact_id = artifact
    monkeypatch.setenv("COWORK_QUICK_EDIT_ENABLED", "false")
    get_app_settings.cache_clear()
    assert post(client, artifact_id).status_code == 501
    assert quick_edit_service.quick_edit_available() is False


async def test_the_capability_follows_the_agent_permissions_and_support(monkeypatch):
    async def allowed(scope, permission):
        return True

    monkeypatch.setattr(workspace_ep, "has_product_permission", allowed)
    caps = {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True}
    monkeypatch.setattr(workspace_ep, "quick_edit_available", lambda: True)
    monkeypatch.setattr(workspace_ep, "direct_edit_available", awaitable(lambda scope: True))
    current = await workspace_ep._current_capabilities(LOCAL_SCOPE, caps)
    assert (current["canQuickEdit"], current["quickEditDirect"]) == (True, True)
    # No decisions model: the bar still shows, as a shortcut into the chat.
    monkeypatch.setattr(workspace_ep, "direct_edit_available", awaitable(lambda scope: False))
    current = await workspace_ep._current_capabilities(LOCAL_SCOPE, caps)
    assert (current["canQuickEdit"], current["quickEditDirect"]) == (True, False)
    monkeypatch.setattr(workspace_ep, "quick_edit_available", lambda: False)
    assert (await workspace_ep._current_capabilities(LOCAL_SCOPE, caps))["canQuickEdit"] is False
    monkeypatch.setattr(workspace_ep, "quick_edit_available", lambda: True)
    reviewer = {"canEdit": False, "canAddressWithAgent": False, "canResolveComments": False}
    assert (await workspace_ep._current_capabilities(LOCAL_SCOPE, reviewer))["canQuickEdit"] is False


async def test_permissions_are_checked_in_the_agent_repair_order(monkeypatch):
    from cowork.services import product_permissions as permissions

    calls = []

    async def allowed(scope, permission):
        calls.append(permission)
        return permission == "artifact.manage"

    monkeypatch.setattr(permissions, "has_product_permission", allowed)
    org = TenantScope(org_mode=True, org_id="org", user_id="user")
    with pytest.raises(permissions.ProductPermissionDenied):
        await quick_ep.quick_edit_artifact(
            "project", "artifact", quick_ep._QuickEditBody(message="x"), SimpleNamespace(), org,
        )
    assert calls == ["artifact.manage", "product.execute"]


# ── the client going away ────────────────────────────────────────────────


async def test_a_client_that_goes_away_cancels_the_edit_and_frees_the_artifact(artifact, monkeypatch):
    folder, metadata, artifact_id = artifact
    started = asyncio.Event()
    restored = []

    async def edit(folder, checker, on_progress):
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            (folder / "index.html").write_text(ORIGINAL, encoding="utf-8")  # anton restores
            restored.append(True)
            raise

    anton = FakeAnton(edit=edit)
    monkeypatch.setattr(quick_ep, "_DISCONNECT_POLL_S", 0.05)
    monkeypatch.setattr(quick_edit_service, "decisions_for_scope", awaitable(lambda a, s: (None, Decisions(True))))
    key = quick_ep._lock_key(LOCAL_SCOPE, artifact_id)
    assert quick_edit_service.quick_edit_locks.acquire(key)
    run = quick_ep.QuickEditRun(
        anton=anton.api(), scope=LOCAL_SCOPE, folder=folder, metadata=metadata, artifact_id=artifact_id,
        body=quick_ep._QuickEditBody(message="Make the title blue"), source_path="index.html",
        base_revision_id="rev", started=0.0, lock_key=key,
        resolve_backend=awaitable(lambda *_a, **_k: QuickEditBackend(llm="llm", route_base_url="https://h/v1", route_api_key="k")),
    )

    class Request:
        async def is_disconnected(self):
            return started.is_set()

    frames = [frame async for frame in quick_ep.quick_edit_events(run, Request())]

    assert [name for name, _ in events_of("".join(frames))] == ["route"]
    assert restored == [True]
    assert (folder / "index.html").read_text(encoding="utf-8") == ORIGINAL
    assert not quick_edit_service.quick_edit_locks.held(key)
    assert len(list_revisions(folder)) == 0


async def test_a_closed_response_body_cancels_the_edit_too(artifact, monkeypatch):
    """Starlette closes the body generator when the send fails."""
    folder, metadata, artifact_id = artifact
    monkeypatch.setattr(quick_edit_service, "decisions_for_scope", awaitable(lambda a, s: (None, Decisions(True))))
    restored = []

    async def edit(folder, checker, on_progress):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            restored.append(True)
            raise

    key = quick_ep._lock_key(LOCAL_SCOPE, artifact_id)
    quick_edit_service.quick_edit_locks.acquire(key)
    run = quick_ep.QuickEditRun(
        anton=FakeAnton(edit=edit).api(), scope=LOCAL_SCOPE, folder=folder, metadata=metadata,
        artifact_id=artifact_id, body=quick_ep._QuickEditBody(message="x"), source_path="index.html",
        base_revision_id="rev", started=0.0, lock_key=key,
        resolve_backend=awaitable(lambda *_a, **_k: QuickEditBackend(llm="llm", route_base_url="https://h/v1", route_api_key="k")),
    )

    class Request:
        async def is_disconnected(self):
            return False

    body = quick_ep.quick_edit_events(run, Request())
    assert (await body.__anext__()).startswith("event: route")
    await asyncio.sleep(0.05)
    await body.aclose()
    assert restored == [True]
    assert not quick_edit_service.quick_edit_locks.held(key)


# ── the preview check answer, authorized against the quick edit's owner ──


@pytest.mark.parametrize(
    "org,user,expected",
    [
        ("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "11111111-1111-1111-1111-111111111111", 200),
        ("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "22222222-2222-2222-2222-222222222222", 404),
        ("0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1", "11111111-1111-1111-1111-111111111111", 404),
    ],
)
async def test_a_quick_edit_check_is_answered_only_by_its_owner(monkeypatch, org, user, expected):
    """No turn holds a quick edit's check: the existing answer endpoint
    authorizes it against the request that opened it."""
    from cowork.server import create_app
    from cowork.streaming.turn_index import TurnOwner

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    owner = TurnOwner(org_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8", user_id="11111111-1111-1111-1111-111111111111")
    check_id, future = preview_checks.open("quick-edit:abc", owner=owner)

    response = TestClient(create_app()).post(
        f"/api/v1/artifacts/checks/{check_id}",
        json={"status": "passed"},
        headers={"X-Organization-Id": org, "X-User-Id": user},
    )

    assert response.status_code == expected, response.text
    assert future.done() is (expected == 200)


def test_the_owner_rule_matches_turns(monkeypatch):
    from cowork.api.v1.endpoints.responses import _owner_matches
    from cowork.streaming.turn_index import TurnOwner

    owner = TurnOwner(org_id="org-1", user_id="user-1")
    assert _owner_matches(owner, TenantScope(org_mode=True, org_id="org-1", user_id="user-1"))
    assert not _owner_matches(owner, TenantScope(org_mode=True, org_id="org-1", user_id="user-2"))
    assert not _owner_matches(owner, TenantScope(org_mode=True, org_id="org-2", user_id="user-1"))


async def test_the_broker_files_a_quick_edit_check_under_its_owner():
    from cowork.streaming.turn_index import TurnOwner

    owner = TurnOwner(org_id="o", user_id="u")
    check_id, _future = preview_checks.open("quick-edit:abc", owner=owner)
    assert preview_checks.conversation_of(check_id) == "quick-edit:abc"
    assert preview_checks.owner_of(check_id) == owner
    turn_check, _ = preview_checks.open("conv-1")
    assert preview_checks.owner_of(turn_check) is None


# ── the revision journal ─────────────────────────────────────────────────


def test_a_viewer_capture_of_the_edit_is_relabelled_not_duplicated(artifact):
    folder, metadata, artifact_id = artifact
    base = current_source(folder, metadata, artifact_id)["revision"]["id"]
    (folder / "index.html").write_text(EDITED, encoding="utf-8")
    # The viewer reloaded mid-edit: its read captured the write.
    captured = current_source(folder, metadata, artifact_id)["revision"]
    assert captured["summary"] == "Captured current draft"

    recorded = record_quick_edit(
        folder, metadata, artifact_id, rel_path="index.html", base_revision_id=base, summary="Blue title",
    )

    assert recorded["id"] == captured["id"]
    assert recorded["summary"] == "Quick edit: Blue title"
    assert recorded["actor"]["kind"] == "agent"
    assert len(list_revisions(folder)) == 2


def test_an_edit_that_left_the_source_alone_records_nothing(artifact):
    folder, metadata, artifact_id = artifact
    base = current_source(folder, metadata, artifact_id)["revision"]["id"]
    assert record_quick_edit(
        folder, metadata, artifact_id, rel_path="index.html", base_revision_id=base, summary="Styles only",
    ) is None
    assert len(list_revisions(folder)) == 1


# ── the conversation note ────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    from cowork.models.conversation import Conversation
    from cowork.models.project import Project

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as raw:
        session = ScopedSession(raw, LOCAL_SCOPE)
        project = Project(name="P", path=str(tmp_path / "p"))
        session.add(project)
        session.commit()
        session.refresh(project)
        conversation = Conversation(project_id=project.id, topic="t")
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        yield session, conversation


def test_the_note_is_a_turn_the_agent_replays(db):
    from cowork.services.conversations import ConversationService

    session, conversation = db
    text = quick_edit_service.conversation_note_text(
        summary="Made the title blue.", changed=["index.html"], revision={"number": 3},
    )
    assert quick_edit_service._save_note(session, conversation_id=conversation.id, message="Make the title blue", text=text)

    messages = ConversationService(session).get_ordered_messages(conversation.id)
    assert [(m.role, m.harness) for m in messages] == [("user", None), ("assistant", quick_edit_service.QUICK_EDIT_HARNESS)]
    assert messages[0].content == "Make the title blue"
    assert "Made the title blue." in messages[1].content and "revision 3" in messages[1].content


def test_no_note_lands_in_the_middle_of_a_running_turn(db):
    from cowork.services.conversations import ConversationService

    session, conversation = db
    service = ConversationService(session)
    service.save_user_message(conversation.id, "Analyse sales", pending=True)
    assert not quick_edit_service._save_note(session, conversation_id=conversation.id, message="m", text="t")
    assert len(service.get_ordered_messages(conversation.id, include_pending=True)) == 1


def test_no_note_in_a_conversation_the_caller_does_not_own(db):
    session, _conversation = db
    with pytest.raises(ValueError):
        quick_edit_service._save_note(session, conversation_id=uuid.uuid4(), message="m", text="t")


# ── undo: every file the edit changed ────────────────────────────────────


def _multi_file_edit(folder, checker, on_progress):
    """A fullstack-style edit: the tracked page, a script, and a new file."""

    async def edit(*_a):
        (folder / "index.html").write_text(EDITED, encoding="utf-8")
        (folder / "app.js").write_text("render('blue');\n", encoding="utf-8")
        (folder / "lib").mkdir(exist_ok=True)
        (folder / "lib" / "theme.js").write_text("export const blue = 1;\n", encoding="utf-8")
        return QuickEditResult("applied", ["index.html", "app.js", "lib/theme.js"], "Made it blue.")

    return edit


@pytest.fixture
def fullstack(artifact):
    folder, metadata, artifact_id = artifact
    (folder / "app.js").write_text("render('black');\n", encoding="utf-8")
    return folder, metadata, artifact_id


def _apply(client, fake_anton, folder, artifact_id):
    async def edit(f, checker, on_progress):
        return await _multi_file_edit(f, checker, on_progress)()

    fake_anton.edit = edit
    return events_of(post(client, artifact_id).text)[-1][1]


def undo(client, artifact_id, edit_id, **body):
    return client.post(f"/api/v1/artifacts/workspace/project/{artifact_id}/quick-edits/{edit_id}/undo", json=body)


def test_undo_restores_every_file_and_removes_created_ones(client, fake_anton, fullstack):
    folder, metadata, artifact_id = fullstack
    done = _apply(client, fake_anton, folder, artifact_id)
    assert done["undoable"] is True and len(done["editId"]) == 32
    assert (folder / "app.js").read_text() == "render('blue');\n"

    response = undo(client, artifact_id, done["editId"], expectedRevisionId=done["revisionId"])

    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted(body["restored"]) == ["app.js", "index.html"]
    assert body["removed"] == ["lib/theme.js"]
    assert body["revision"]["summary"] == "Undid quick edit"
    assert (folder / "index.html").read_text() == ORIGINAL
    assert (folder / "app.js").read_text() == "render('black');\n"
    assert not (folder / "lib" / "theme.js").exists()
    assert list_revisions(folder)[0]["id"] == body["revision"]["id"]
    # Once only.
    again = undo(client, artifact_id, done["editId"], expectedRevisionId=body["revision"]["id"])
    assert again.status_code == 409 and again.json()["detail"]["reason"] == "already_undone"


def test_undo_of_an_edit_that_left_the_source_alone_records_no_revision(client, fake_anton, fullstack):
    folder, metadata, artifact_id = fullstack

    async def edit(f, checker, on_progress):
        (f / "app.js").write_text("render('blue');\n", encoding="utf-8")
        return QuickEditResult("applied", ["app.js"], "Script only.")

    fake_anton.edit = edit
    done = events_of(post(client, artifact_id).text)[-1][1]
    assert done["revisionId"] is None and done["undoable"] is True
    revisions = len(list_revisions(folder))

    response = undo(client, artifact_id, done["editId"])

    assert response.status_code == 200, response.text
    assert response.json()["revision"] is None
    assert (folder / "app.js").read_text() == "render('black');\n"
    assert len(list_revisions(folder)) == revisions


def test_undo_refuses_when_a_file_changed_since_and_changes_nothing(client, fake_anton, fullstack):
    folder, metadata, artifact_id = fullstack
    done = _apply(client, fake_anton, folder, artifact_id)
    (folder / "app.js").write_text("render('green');\n", encoding="utf-8")  # a later edit

    response = undo(client, artifact_id, done["editId"], expectedRevisionId=done["revisionId"])

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "files_changed"
    assert response.json()["detail"]["paths"] == ["app.js"]
    assert (folder / "index.html").read_text() == EDITED
    assert (folder / "lib" / "theme.js").exists()


def test_undo_with_a_stale_head_is_409(client, fake_anton, fullstack):
    folder, metadata, artifact_id = fullstack
    done = _apply(client, fake_anton, folder, artifact_id)
    response = undo(client, artifact_id, done["editId"], expectedRevisionId="stale")
    assert response.status_code == 409
    assert set(response.json()["detail"]) == {"message", "currentRevision"}
    assert (folder / "app.js").read_text() == "render('blue');\n"


def test_undo_of_an_unknown_edit_is_404(client, fake_anton, fullstack):
    _folder, _metadata, artifact_id = fullstack
    assert undo(client, artifact_id, "0" * 32).status_code == 404
    assert undo(client, artifact_id, "../../etc").status_code == 404


def test_undo_is_refused_while_a_quick_edit_runs(client, fake_anton, fullstack):
    _folder, _metadata, artifact_id = fullstack
    quick_edit_service.quick_edit_locks.acquire(quick_ep._lock_key(LOCAL_SCOPE, artifact_id))
    response = undo(client, artifact_id, "0" * 32)
    assert response.status_code == 409 and response.json()["detail"]["reason"] == "quick_edit_running"


def test_undo_failing_midway_puts_back_what_it_changed(fullstack, monkeypatch):
    """All or nothing: a write that fails rolls back the files already restored."""
    from cowork.services import artifact_quick_edits as undo_store

    folder, metadata, artifact_id = fullstack
    paths = ["index.html", "app.js"]
    before = undo_store.capture(folder, paths)
    (folder / "index.html").write_text(EDITED, encoding="utf-8")
    (folder / "app.js").write_text("render('blue');\n", encoding="utf-8")
    edit_id = undo_store.new_edit_id()
    assert undo_store.save_undo(
        folder, edit_id=edit_id, before=before, changed=paths, source_path="index.html",
        revision_id=None, previous_revision_id=None,
    ) == paths

    real_write = undo_store._write
    calls = []

    def flaky(root, rel_path, file):
        calls.append(rel_path)
        if len(calls) == 2:
            raise OSError("disk full")
        real_write(root, rel_path, file)

    monkeypatch.setattr(undo_store, "_write", flaky)
    with pytest.raises(OSError):
        undo_store.undo(folder, metadata, artifact_id, edit_id, expected_revision_id=None, actor_id=None)
    assert (folder / "index.html").read_text() == EDITED
    assert (folder / "app.js").read_text() == "render('blue');\n"


def test_only_the_last_ten_undos_are_kept(artifact):
    from cowork.services import artifact_quick_edits as undo_store

    folder, _metadata, _artifact_id = artifact
    ids = []
    for i in range(12):
        before = undo_store.capture(folder, ["index.html"])
        (folder / "index.html").write_text(f"v{i}\n", encoding="utf-8")
        edit_id = undo_store.new_edit_id()
        undo_store.save_undo(folder, edit_id=edit_id, before=before, changed=["index.html"],
                             source_path="index.html", revision_id=None, previous_revision_id=None)
        ids.append(edit_id)
    kept = {r["id"] for r in undo_store._records(folder)}
    assert kept == set(ids[2:])
    blobs = list((folder / ".revisions" / "quick-edits" / "blobs").iterdir())
    assert len(blobs) == 10


def test_a_link_planted_in_the_artifact_is_never_followed(fullstack, tmp_path):
    from cowork.services import artifact_quick_edits as undo_store

    folder, _metadata, _artifact_id = fullstack
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (folder / "link.js").symlink_to(outside)
    assert undo_store.capture(folder, ["index.html", "link.js"]) is None


def test_ambient_mindshub_config_does_not_make_a_byok_user_look_like_mindshub(monkeypatch):
    """AntonSettings reads ANTON_* and ~/.anton/.env; only what the person
    configured may decide whether the bar can edit directly."""
    from cowork.common.settings.user_settings import Provider, UserSettings

    monkeypatch.setenv("ANTON_MINDS_API_KEY", "left-over")
    monkeypatch.setenv("ANTON_OPENAI_BASE_URL", "https://api.mindshub.ai/v1")

    def mindshub_rule(settings):  # anton's: a MindsHub base URL and a key
        return bool(settings.minds_api_key) and "mindshub" in (settings.openai_base_url or "")

    api = FakeAnton().api()
    api = AntonQuickEdit(
        route_request=api.route_request, run_quick_edit=api.run_quick_edit, checkers=None, workspace_cls=None,
        decisions_available=mindshub_rule,
        decisions_config=lambda s: ("https://api.mindshub.ai/v1", s.minds_api_key, "jev-1.13.0") if mindshub_rule(s) else None,
    )
    byok = UserSettings(planning_provider=Provider.ANTHROPIC, anthropic_api_key="k")
    assert quick_edit_service.decisions_for(api, byok).available is False
    hub = UserSettings(planning_provider=Provider.OPENAI_COMPATIBLE, openai_base_url="https://api.mindshub.ai/v1", minds_api_key="mk")
    assert quick_edit_service.decisions_for(api, hub).available is True


def _anton_rule_api(anton: FakeAnton) -> AntonQuickEdit:
    """anton's decisions rule (anton.core.llm.jev): ANTON_DECISIONS_MODEL, a
    MindsHub host, and a key; the installed anton predates it."""
    import os
    from urllib.parse import urlparse

    def config(settings):
        model = os.environ.get("ANTON_DECISIONS_MODEL")
        base = (settings.openai_base_url or "").rstrip("/")
        if not model or urlparse(base).hostname != "api.mindshub.ai":
            return None
        key = settings.minds_api_key or settings.openai_api_key
        return (base, key, model) if key else None

    api = anton.api()
    return AntonQuickEdit(
        route_request=api.route_request, run_quick_edit=api.run_quick_edit, checkers=api.checkers,
        workspace_cls=api.workspace_cls, decisions_available=lambda s: config(s) is not None, decisions_config=config,
    )


@pytest.mark.parametrize("host,direct", [("https://api.mindshub.ai/v1", True), ("https://llm.example.com/v1", False)])
def test_openai_compatible_on_mindshub_is_direct_and_elsewhere_goes_to_the_chat(
    client, fake_anton, artifact, monkeypatch, host, direct,
):
    """A self-hosted deployment pointed straight at MindsHub as an
    openai-compatible provider: anton's decisions_config alone decides."""
    from cowork.common.settings.user_settings import Provider, UserSettings

    _folder, _metadata, artifact_id = artifact
    monkeypatch.setenv("ANTON_DECISIONS_MODEL", "jev-1.13.0")
    user = UserSettings(planning_provider=Provider.OPENAI_COMPATIBLE, openai_base_url=host, openai_api_key="ok")
    api = _anton_rule_api(fake_anton)
    monkeypatch.setattr(quick_edit_service, "anton_quick_edit", lambda: api)
    monkeypatch.setattr(quick_edit_service, "load_user_settings", awaitable(lambda scope: user))
    # The real decisions_for_scope and resolve_backend: only the LLM build is stubbed.
    monkeypatch.setattr(quick_edit_service, "decisions_for_scope", _real_decisions_for_scope)
    monkeypatch.setattr(quick_edit_service, "resolve_backend", _real_resolve_backend)
    monkeypatch.setattr("cowork.services.providers.build_llm_client", lambda **_k: "the-llm")
    fake_anton.route = RouteDecision("agent", 0.9)

    async def allowed(scope, permission):
        return True

    monkeypatch.setattr(workspace_ep, "has_product_permission", allowed)
    caps = asyncio.run(workspace_ep._current_capabilities(
        LOCAL_SCOPE, {"canEdit": True, "canAddressWithAgent": True, "canResolveComments": True},
    ))
    assert caps["canQuickEdit"] is True
    assert caps["quickEditDirect"] is direct

    events = events_of(post(client, artifact_id).text)
    if direct:
        assert fake_anton.route_calls[0]["base_url"] == "https://api.mindshub.ai/v1"
        assert fake_anton.route_calls[0]["api_key"] == "ok"
        assert fake_anton.route_calls[0]["model"] == "jev-1.13.0"
        assert events[0][1]["reason"] == "jev"
    else:
        assert fake_anton.route_calls == []
        assert events[0][1]["reason"] == "no_decisions_model"
