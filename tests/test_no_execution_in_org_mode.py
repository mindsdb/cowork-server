from pathlib import Path

import pytest

from cowork.services import artifacts


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_backend_launch_refused_in_org_mode(org_mode, tmp_path, monkeypatch):
    """No subprocess, no port probe, no anton import. Just a refusal."""
    def _boom(*_a, **_k):
        raise AssertionError("_probe_port must not run in org mode")

    monkeypatch.setattr(artifacts, "_probe_port", _boom)

    running, detail, port = await artifacts._ensure_backend_running(tmp_path / "slug", 8000)

    assert running is False
    assert port == 0
    assert "not available" in detail.lower()
    assert artifacts._LAUNCHED_BACKENDS == {}


@pytest.mark.asyncio
async def test_backend_launch_still_works_on_desktop(monkeypatch, tmp_path):
    """Desktop keeps the launcher. Guard against the kill switch being unconditional."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    monkeypatch.setattr(artifacts, "_probe_port", lambda _p: True)

    running, detail, port = await artifacts._ensure_backend_running(tmp_path / "slug", 8000)

    assert running is True
    assert detail == "already_running"
    get_app_settings.cache_clear()


def test_reveal_in_file_manager_refused_in_org_mode(org_mode, tmp_path):
    with pytest.raises(RuntimeError, match="not available"):
        artifacts.reveal_in_file_manager(tmp_path)


# ─── I1: open_artifact endpoint ────────────────────────────────────

@pytest.mark.asyncio
async def test_open_artifact_endpoint_refused_in_org_mode(org_mode):
    """Regression guard: this endpoint's guard had no test, so deleting it
    changed nothing in the full suite."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _PathBody, open_artifact

    with pytest.raises(HTTPException) as exc:
        await open_artifact(_PathBody(path="/tmp/whatever"))
    assert exc.value.status_code == 403


# ─── I4: mount_preview's proxy branch must not register a token ───

@pytest.mark.asyncio
async def test_mount_preview_proxy_refused_in_org_mode(org_mode, tmp_path):
    """A fullstack artifact's `port` is read fresh from its own (agent-
    writable) metadata.json by the proxy route on every request. Registering
    the token before refusing would leave that SSRF pivot live even though
    `_ensure_backend_running` itself refuses to launch anything."""
    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 54321}', encoding="utf-8")
    primary = root / "index.html"
    primary.write_text("<html></html>", encoding="utf-8")

    before = dict(artifacts._PREVIEW_MOUNTS)
    with pytest.raises(ValueError, match="not available"):
        await artifacts.mount_preview(primary)
    assert artifacts._PREVIEW_MOUNTS == before


@pytest.mark.asyncio
async def test_mount_preview_proxy_still_registers_on_desktop(monkeypatch, tmp_path):
    """Guard against the I4 fix being unconditional: desktop still needs the
    token registered so the proxy route (and a subsequent backend launch)
    can find the artifact root."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 54321}', encoding="utf-8")
    primary = root / "index.html"
    primary.write_text("<html></html>", encoding="utf-8")

    payload = await artifacts.mount_preview(primary)

    assert payload["kind"] == "proxy"
    assert payload["token"] in artifacts._PREVIEW_MOUNTS
    get_app_settings.cache_clear()


# ─── I5: PDF/Word export can SSRF via the source HTML's own URIs ──

@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["pdf", "docx", "PDF", ".docx"])
async def test_export_pdf_docx_refused_in_org_mode(org_mode, fmt):
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _ExportBody, export_artifact_endpoint

    with pytest.raises(HTTPException) as exc:
        await export_artifact_endpoint(_ExportBody(path="/tmp/whatever.md", format=fmt))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_export_html_not_refused_in_org_mode(org_mode, tmp_path):
    """Guard against the I5 fix being a blanket export ban: HTML export does
    no URI resolution (no fetch, no local read beyond the source itself), so
    it stays available. The 404 below (not 403) proves the request reached
    path resolution instead of being stopped by the export guard."""
    from fastapi import HTTPException

    from cowork.api.v1.endpoints.artifacts import _ExportBody, export_artifact_endpoint

    with pytest.raises(HTTPException) as exc:
        await export_artifact_endpoint(
            _ExportBody(path=str(tmp_path / "missing.md"), format="html")
        )
    assert exc.value.status_code == 404


# ─── C2: apply_env_to_process would pollute this whole process's env ──

def test_apply_workspace_env_refused_in_org_mode(org_mode):
    from unittest.mock import Mock

    from cowork.harnesses.anton_harness.harness import _apply_workspace_env_if_safe

    fake_workspace = Mock()
    applied = _apply_workspace_env_if_safe(fake_workspace)

    assert applied is False
    fake_workspace.apply_env_to_process.assert_not_called()


def test_apply_workspace_env_still_works_on_desktop(monkeypatch):
    """Desktop still needs its own .env loaded into the process (e.g. a
    locally-set API key). Guard against the kill switch being unconditional."""
    from unittest.mock import Mock

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    from cowork.harnesses.anton_harness.harness import _apply_workspace_env_if_safe

    fake_workspace = Mock()
    applied = _apply_workspace_env_if_safe(fake_workspace)

    assert applied is True
    fake_workspace.apply_env_to_process.assert_called_once()
    get_app_settings.cache_clear()


def test_code_project_commands_refused_in_org_mode(org_mode, monkeypatch):
    """Project-configured commands are local desktop behavior, never org execution."""
    from cowork.coding import project_workspaces
    from cowork.coding.project_models import CodeProject, ProjectFolder

    monkeypatch.setattr(
        project_workspaces.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess must not run")),
    )

    with pytest.raises(RuntimeError, match="not available"):
        project_workspaces.ProjectCommandRunner().run(
            CodeProject(
                id="blocked",
                name="Blocked",
                folders=[ProjectFolder(id="folder", name="Folder", path="/tmp/folder")],
            ),
            (),
            "validate",
            {},
        )

# ─── C3: stream_response is the single choke point for in-process turns ──

@pytest.mark.asyncio
async def test_stream_response_refused_in_org_mode(org_mode):
    """AntonHarness.stream_response is reachable in org mode via three paths
    that all bypass the remote-worker routing gate: the legacy non-streaming
    branch in handlers/responses.py (ResponsesRequest.stream defaults to
    False), _produce/_run_turn's in-process fallback whenever
    COWORK_TURN_BACKEND isn't "remote", and the channel-ingress runtime.
    Refusing here, before `conversation` is ever touched, closes all three at
    once. `conversation=None` proves the refusal precedes any use of it."""
    from cowork.harnesses.anton_harness.harness import AntonHarness

    harness = AntonHarness()
    stream = harness.stream_response(conversation=None, input=[])
    with pytest.raises(RuntimeError, match="remote worker"):
        await stream.__anext__()


# --- Round 2, Finding 1: build_chat_session is the only sanctioned --------
# --- ChatSession(...) constructor ------------------------------------------

def test_build_chat_session_refused_in_org_mode(org_mode):
    """Both known constructors (the probe and the anton harness turn path)
    were rewritten to call build_chat_session instead of ChatSession(...)
    directly. This is the one place that refuses, so proving the refusal
    here covers both callers without needing anton's real session plumbing."""
    from cowork.common import chat_session

    with pytest.raises(RuntimeError, match="dispatched to a worker"):
        chat_session.build_chat_session(object())


def test_build_chat_session_still_works_on_desktop(monkeypatch):
    """Guard against the kill switch being unconditional: desktop still needs
    a real ChatSession. anton.core.session.ChatSession is stubbed, not the
    real thing, so this doesn't need a real ChatSessionConfig."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    import anton.core.session as anton_session
    from cowork.common import chat_session

    sentinel = object()
    monkeypatch.setattr(anton_session, "ChatSession", lambda config: sentinel)

    result = chat_session.build_chat_session(object())

    assert result is sentinel
    get_app_settings.cache_clear()


# --- Round 2, Finding 2: the proxy forwarder itself must refuse, not just -
# --- mount_preview's proxy branch ------------------------------------------

@pytest.mark.asyncio
async def test_proxy_artifact_request_refused_in_org_mode(org_mode, tmp_path):
    """mount_preview's STATIC branch (HTML preview, not a backend) registers
    a token without checking _org_mode, since static preview isn't execution.
    But metadata.json lives inside the artifact directory, which the org's
    own agent controls, so nothing stops it writing {"port": 6379} into that
    same file after the token exists. The proxy route re-reads metadata.json
    on every request, so the guard has to live in proxy_artifact_request
    itself: a token that was minted by the static branch, then repointed at
    a port after the fact, must still be refused."""
    from starlette.requests import Request as StarletteRequest

    from cowork.services import artifacts, preview_proxy

    root = tmp_path / "artifact_root"
    root.mkdir()
    (root / "metadata.json").write_text('{"port": 6379}', encoding="utf-8")
    token = "deadbeefcafef00d"
    artifacts._PREVIEW_MOUNTS[token] = root
    try:
        scope = {
            "type": "http", "method": "GET", "path": "/",
            "headers": [], "query_string": b"",
        }
        request = StarletteRequest(scope)

        response = await preview_proxy.proxy_artifact_request(token, "", request)

        assert response.status_code == 403
    finally:
        del artifacts._PREVIEW_MOUNTS[token]


# --- Round 2, Finding 4: .env -> DB migration must not read the shared ----
# --- tree root into unscoped global rows ------------------------------------

def test_migrate_env_to_db_gated_off_in_org_mode(org_mode):
    import cowork.migrations as migrations_mod
    from cowork.dev_setup import _migrate_env_to_db_if_local

    calls = []
    original = migrations_mod.migrate_env_to_db
    migrations_mod.migrate_env_to_db = lambda session: calls.append(session)
    try:
        _migrate_env_to_db_if_local(object())
    finally:
        migrations_mod.migrate_env_to_db = original
    assert calls == [], "org mode must not migrate .env into unscoped DB rows"


def test_migrate_env_to_db_still_runs_on_desktop(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    import cowork.migrations as migrations_mod
    from cowork.dev_setup import _migrate_env_to_db_if_local

    calls = []
    monkeypatch.setattr(migrations_mod, "migrate_env_to_db", lambda session: calls.append(session))
    sentinel_session = object()
    _migrate_env_to_db_if_local(sentinel_session)

    assert calls == [sentinel_session]
    get_app_settings.cache_clear()


# --- Round 2, Finding 5: reveal_artifact's except clause must not relabel -
# --- every RuntimeError as a policy refusal ---------------------------------

@pytest.mark.asyncio
async def test_reveal_artifact_endpoint_maps_execution_refused_to_403(monkeypatch, tmp_path):
    """The endpoint used to catch bare RuntimeError, so reveal_in_file_manager's
    org-mode refusal and a genuine RuntimeError from the platform call
    underneath both landed on the same 403. Only the typed ExecutionRefused
    should reach 403; anything else must fall through to the generic 500."""
    import cowork.api.v1.endpoints.artifacts as artifacts_ep
    from fastapi import HTTPException

    from cowork.services.artifacts import ExecutionRefused

    monkeypatch.setattr(artifacts_ep, "_resolve_reveal_path", lambda path, session: tmp_path)

    def _refuse(_path):
        raise ExecutionRefused("not available on this deployment")
    monkeypatch.setattr(artifacts_ep, "reveal_in_file_manager", _refuse)

    with pytest.raises(HTTPException) as exc:
        await artifacts_ep.reveal_artifact(artifacts_ep._PathBody(path=str(tmp_path)), session=None)
    assert exc.value.status_code == 403

    def _boom(_path):
        raise RuntimeError("os.startfile blew up")
    monkeypatch.setattr(artifacts_ep, "reveal_in_file_manager", _boom)

    with pytest.raises(HTTPException) as exc:
        await artifacts_ep.reveal_artifact(artifacts_ep._PathBody(path=str(tmp_path)), session=None)
    assert exc.value.status_code == 500


# --- Round 2, Finding 6: non-streaming /responses must refuse cleanly in --
# --- org mode, not surface an unhandled RuntimeError as a 500 --------------

@pytest.mark.asyncio
async def test_handle_refuses_non_streaming_turn_in_org_mode(org_mode):
    """ResponsesRequest.stream defaults to False, so a client reaches the
    non-streaming branch of handle() just by omitting the field. That branch
    drives AntonHarness.stream_response synchronously in this process, which
    raises RuntimeError in org mode (C3 above), previously unhandled here,
    surfacing as an opaque 500. The guard must raise before touching the
    harness at all."""
    from uuid import uuid4

    from fastapi import HTTPException

    from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    class _FakeConversation:
        def __init__(self, conv_id):
            self.id = conv_id
            self.messages = []

    class _FakeConversationService:
        def __init__(self, scoped):
            pass

        def get_conversation(self, conv_id):
            return _FakeConversation(conv_id)

    import cowork.handlers.responses as responses_mod
    original = responses_mod.ConversationService
    responses_mod.ConversationService = _FakeConversationService
    try:
        handler = object.__new__(ResponsesHandler)
        handler.principal = None
        handler.scoped = object()

        async def _fake_route_request(**kwargs):
            return RouteDecision(route=DELEGATED_AGENTIC, reason="test"), None
        handler._route_request = _fake_route_request

        class _BoomHarness:
            id = "anton"

            def stream_response(self, **kwargs):
                raise AssertionError(
                    "must refuse before driving the in-process harness"
                )
        handler._get_harness = lambda: _BoomHarness()

        request = ResponsesRequest(input="hi", conversation=str(uuid4()), stream=False)

        with pytest.raises(HTTPException) as exc:
            await handler.handle(request)
        assert exc.value.status_code == 501
    finally:
        responses_mod.ConversationService = original
