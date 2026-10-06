from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.db.scoped import TenantScope
from cowork.services import product_permissions as permissions

ORG = TenantScope(
    org_mode=True,
    org_id="11111111-1111-4111-8111-111111111111",
    user_id="22222222-2222-4222-8222-222222222222",
)


def authority(monkeypatch, *, body=None, status=200, error=None):
    requests = []

    def handle(request):
        requests.append(request)
        if error:
            raise error
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        permissions.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    monkeypatch.setattr(
        permissions,
        "TurnQueueSettings",
        lambda: SimpleNamespace(
            auth_internal_base_url="https://auth.internal",
            auth_internal_secret="fixture-only",
        ),
    )
    return requests


@pytest.mark.parametrize("permission", ["product.execute", "artifact.manage"])
@pytest.mark.parametrize("allowed", [True, False])
async def test_decision_uses_verified_identity_and_internal_host(
    monkeypatch, permission, allowed
):
    import json

    requests = authority(monkeypatch, body={"allowed": allowed})
    assert await permissions.has_product_permission(ORG, permission) is allowed
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://auth.internal/internal/permissions/authorize/"
    assert json.loads(request.content) == {
        "user_id": ORG.user_id,
        "organization_id": ORG.org_id,
        "permission": permission,
    }
    assert request.headers["x-internal-auth"] == "fixture-only"
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
    "body,status",
    [
        ({}, 200),
        ({"allowed": "false"}, 200),
        ({"allowed": 1}, 200),
        ({"allowed": True, "other": 1}, 200),
        (None, 200),
        ({"allowed": True}, 403),
        ({"allowed": True}, 503),
        ({"allowed": True}, 302),
    ],
)
async def test_unknown_authority_is_unavailable_and_never_denied_or_allowed(
    monkeypatch, body, status
):
    authority(monkeypatch, body=body, status=status)
    with pytest.raises(permissions.ProductPermissionUnavailable):
        await permissions.require_product_permission(ORG, "product.execute")


async def test_transport_outage_is_unavailable(monkeypatch):
    authority(monkeypatch, error=httpx.ConnectError("Offline"))
    with pytest.raises(permissions.ProductPermissionUnavailable):
        await permissions.require_product_permission(ORG, "product.execute")


async def test_desktop_keeps_its_single_user_boundary(monkeypatch):
    check = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr(permissions, "TurnQueueSettings", check)
    await permissions.require_product_permission(
        TenantScope(org_mode=False), "product.execute"
    )
    check.assert_not_called()


async def test_missing_internal_configuration_fails_closed(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "TurnQueueSettings",
        lambda: SimpleNamespace(auth_internal_base_url="", auth_internal_secret=""),
    )
    with pytest.raises(permissions.ProductPermissionUnavailable):
        await permissions.require_product_permission(ORG, "product.execute")


@pytest.mark.parametrize("stream", [True, False])
async def test_response_denial_precedes_router_harness_and_persistence(
    monkeypatch, stream
):
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    authority(monkeypatch, body={"allowed": False})
    handler = object.__new__(ResponsesHandler)
    handler.scope = ORG
    handler._router_binding = AsyncMock(
        side_effect=AssertionError("Router must not run")
    )
    handler.session = Mock()
    with pytest.raises(permissions.ProductPermissionDenied) as error:
        await handler.handle(ResponsesRequest(input="Run the model", stream=stream))
    assert error.value.status_code == 403
    handler._router_binding.assert_not_called()
    handler.session.assert_not_called()


async def test_queued_reuse_denial_precedes_redis_even_with_an_existing_key(
    monkeypatch,
):
    from cowork.turnqueue import producer

    authority(monkeypatch, body={"allowed": False})
    redis = Mock(side_effect=AssertionError("Do not enqueue"))
    monkeypatch.setattr(producer, "get_redis", redis)
    with pytest.raises(permissions.ProductPermissionDenied):
        async for _ in producer.stream_remote_replies(
            org_id=ORG.org_id,
            user_id=ORG.user_id,
            conversation_id="conversation",
            input_text="Run",
            model="free-model",
            llm={"api_key": "old-key"},
        ):
            pass
    redis.assert_not_called()


def test_artifact_http_write_denied_before_resource_or_disk_access(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace
    from cowork.db.scoped import get_scoped_session

    authority(monkeypatch, body={"allowed": False})
    app = FastAPI()
    app.include_router(artifact_workspace.router)
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=ORG)
    disk = Mock(side_effect=AssertionError("No resource or disk access"))
    monkeypatch.setattr(artifact_workspace, "_owner_workspace", disk)
    with TestClient(app) as client:
        response = client.put(
            "/workspace/project/33333333-3333-4333-8333-333333333333",
            json={
                "content": "changed",
                "expectedRevisionId": "rev",
                "path": "index.html",
            },
        )
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "permission_denied"
    disk.assert_not_called()


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (
            403,
            {"code": "permission_denied", "detail": "No execution"},
            permissions.ProductPermissionDenied,
        ),
        (401, {"code": "permission_denied"}, permissions.ProductPermissionUnavailable),
        (
            403,
            {"code": "internal_secret_invalid"},
            permissions.ProductPermissionUnavailable,
        ),
        (403, {}, permissions.ProductPermissionUnavailable),
        (403, None, permissions.ProductPermissionUnavailable),
        (403, ["permission_denied"], permissions.ProductPermissionUnavailable),
        (503, {}, permissions.ProductPermissionUnavailable),
        (500, {}, permissions.ProductPermissionUnavailable),
        (302, {}, permissions.ProductPermissionUnavailable),
        (200, {}, permissions.ProductPermissionUnavailable),
        (200, {"key": ""}, permissions.ProductPermissionUnavailable),
        (200, {"key": "  "}, permissions.ProductPermissionUnavailable),
        (200, {"key": 1}, permissions.ProductPermissionUnavailable),
        (200, ["key"], permissions.ProductPermissionUnavailable),
    ],
)
async def test_turn_key_only_maps_a_confirmed_authorization_denial(
    monkeypatch, status, body, expected
):
    from cowork.turnqueue.auth_keys import mint_turn_key

    authority(monkeypatch, body=body, status=status)
    settings = permissions.TurnQueueSettings()
    with pytest.raises(expected):
        await mint_turn_key(
            user_id=ORG.user_id,
            org_id=ORG.org_id,
            correlation_id="turn",
            ttl_seconds=60,
            settings=settings,
        )


async def test_turn_key_transport_and_missing_configuration_are_unavailable(
    monkeypatch,
):
    from cowork.turnqueue.auth_keys import mint_turn_key

    authority(monkeypatch, error=httpx.ConnectError("Offline"))
    for settings in [
        permissions.TurnQueueSettings(),
        SimpleNamespace(auth_internal_base_url="", auth_internal_secret=""),
        SimpleNamespace(
            auth_internal_base_url="https://auth.internal", auth_internal_secret=""
        ),
    ]:
        with pytest.raises(permissions.ProductPermissionUnavailable):
            await mint_turn_key(
                user_id=ORG.user_id,
                org_id=ORG.org_id,
                correlation_id="turn",
                ttl_seconds=60,
                settings=settings,
            )


@pytest.mark.parametrize(
    "error_type",
    [permissions.ProductPermissionDenied, permissions.ProductPermissionUnavailable],
)
async def test_routing_gate_does_not_delegate_authorization_failures(
    monkeypatch, error_type
):
    import cowork.handlers.responses as responses

    handler = object.__new__(responses.ResponsesHandler)
    handler.scope, handler.scoped, handler.harness_name = ORG, Mock(), "anton"
    handler._router_binding = AsyncMock(side_effect=error_type())
    route = AsyncMock(side_effect=AssertionError("No fallback model"))
    monkeypatch.setattr(responses, "decide_route", route)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda session: SimpleNamespace(get_ordered_messages=lambda cid: []),
    )
    with pytest.raises(error_type):
        await handler._route_request(
            conversation_id=None,
            harness_input=[{"type": "text", "text": "Run"}],
            has_attachments=False,
            has_disabled_connections=False,
        )
    route.assert_not_called()


@pytest.mark.parametrize(
    "model", ["default", "free-model", "paid-model", "stored-provider-model"]
)
async def test_model_selection_cannot_bypass_admission(monkeypatch, model):
    from cowork.handlers.responses import ResponsesHandler
    from cowork.schemas.responses import ResponsesRequest

    authority(monkeypatch, body={"allowed": False})
    handler = object.__new__(ResponsesHandler)
    handler.scope = ORG
    with pytest.raises(permissions.ProductPermissionDenied):
        await handler.handle(ResponsesRequest(input="Run", model=model))


@pytest.mark.parametrize(
    "suffix,method,body",
    [
        ("", "put", {"content": "changed", "expectedRevisionId": "rev"}),
        ("/access", "put", {"access": {"mode": "public"}}),
        ("/comments-access", "post", None),
        ("/revisions/rev/restore", "post", {"expectedRevisionId": "rev"}),
        (
            "/agent-repairs",
            "post",
            {
                "expectedRevisionId": "rev",
                "commentThreadId": "thread",
                "thread": [{"text": "Fix"}],
                "conversationId": "33333333-3333-4333-8333-333333333333",
            },
        ),
        ("/agent-repairs/release", "post", {"commentThreadId": "thread"}),
        ("/agent-repairs/repair/cancel", "post", {}),
        ("/agent-repairs/repair/decision", "post", {"status": "accepted"}),
    ],
)
@pytest.mark.parametrize("allowed,status", [(False, 403), (None, 503)])
def test_every_artifact_mutation_checks_authority_before_resolving_files(
    monkeypatch, suffix, method, body, allowed, status
):
    from cowork.api.v1.endpoints import artifact_workspace
    from cowork.db.scoped import get_scoped_session

    authority(monkeypatch, body={"allowed": allowed})
    app = FastAPI()
    app.include_router(artifact_workspace.router)
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=ORG)
    resolve = Mock(side_effect=AssertionError("No artifact mutation"))
    monkeypatch.setattr(artifact_workspace, "_owner_workspace", resolve)
    with TestClient(app) as client:
        response = client.request(
            method,
            f"/workspace/project/33333333-3333-4333-8333-333333333333{suffix}",
            json=body,
        )
    assert response.status_code == status, response.text
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        ".anton/artifacts/report/index.html",
        ".ANTON/ARTIFACTS/report/metadata.json",
        ".anton/artifacts",
        ".anton",
        "conversations",
        "conversations/33333333-3333-4333-8333-333333333333",
        "conversations/33333333-3333-4333-8333-333333333333/.anton",
        "conversations/33333333-3333-4333-8333-333333333333/.anton/artifacts/report/index.html",
    ],
)
@pytest.mark.parametrize("method", ["put", "delete"])
def test_generic_file_routes_cannot_mutate_artifact_bytes_or_ancestors(
    monkeypatch, path, method
):
    from cowork.api.v1.endpoints import project_files
    from cowork.db.scoped import get_scoped_session

    authority(monkeypatch, body={"allowed": False})
    app = FastAPI()
    app.include_router(project_files.router)
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=ORG)
    write = Mock(side_effect=AssertionError("No disk access"))
    monkeypatch.setattr(project_files, "_require_workspace_path", write)
    with TestClient(app) as client:
        response = client.request(
            method, f"/project/files/{path}", json={"content": "tampered"}
        )
    assert response.status_code == 403, response.text
    write.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "notes.txt",
        ".anton/memory/rules.md",
        "conversations/id/notes.txt",
        "conversations/door.html",
    ],
)
async def test_generic_non_artifact_paths_keep_their_existing_resource_authority(
    monkeypatch, path
):
    from cowork.api.v1.endpoints import project_files

    call = authority(monkeypatch, body={"allowed": False})
    await project_files._authorize_artifact_file_mutation(
        project_files._validated_project_path(path), SimpleNamespace(scope=ORG)
    )
    assert call == []


@pytest.mark.parametrize(
    "path",
    ["conversations", "conversations/33333333-3333-4333-8333-333333333333"],
)
async def test_artifact_parent_paths_continue_after_a_current_grant(monkeypatch, path):
    from cowork.api.v1.endpoints import project_files

    calls = authority(monkeypatch, body={"allowed": True})
    await project_files._authorize_artifact_file_mutation(
        project_files._validated_project_path(path), SimpleNamespace(scope=ORG)
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "can_edit,can_execute", [(True, True), (True, False), (False, True), (False, False)]
)
async def test_artifact_capabilities_reflect_current_role_and_execution(
    monkeypatch, can_edit, can_execute
):
    from cowork.api.v1.endpoints import artifact_workspace

    async def allowed(scope, permission):
        assert scope is ORG
        return can_edit if permission == "artifact.manage" else can_execute

    monkeypatch.setattr(artifact_workspace, "has_product_permission", allowed)
    original = {
        "canEdit": True,
        "canAddressWithAgent": True,
        "canResolveComments": True,
        "canPreview": True,
    }
    actual = await artifact_workspace._current_capabilities(
        SimpleNamespace(scope=ORG), original
    )
    assert actual == {
        "canEdit": can_edit,
        "canAddressWithAgent": can_edit and can_execute,
        "canResolveComments": can_edit,
        "canPreview": True,
    }
    assert original["canEdit"] is True


async def test_artifact_role_permissions_never_elevate_a_reviewer(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace

    call = authority(monkeypatch, body={"allowed": True})
    original = {
        "canEdit": False,
        "canAddressWithAgent": False,
        "canResolveComments": False,
        "canPreview": True,
    }
    assert (
        await artifact_workspace._current_capabilities(
            SimpleNamespace(scope=ORG), original
        )
        == original
    )
    assert call == []


def test_artifact_delete_checks_role_before_resolving_a_slug(monkeypatch):
    from cowork.api.v1.endpoints import artifacts
    from cowork.db.scoped import get_scoped_session

    authority(monkeypatch, body={"allowed": False})
    app = FastAPI()
    app.include_router(artifacts.router)
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=ORG)
    resolve = Mock(side_effect=AssertionError("No artifact lookup"))
    monkeypatch.setattr(artifacts, "scoped_project_id_for_request", resolve)
    with TestClient(app) as client:
        response = client.delete(
            "/report", params={"project_id": "33333333-3333-4333-8333-333333333333"}
        )
    assert response.status_code == 403, response.text
    resolve.assert_not_called()


@pytest.mark.parametrize("is_manual", [True, False])
async def test_schedules_recheck_stored_owner_before_creating_a_conversation(
    monkeypatch, is_manual
):
    from datetime import datetime, timezone
    from uuid import uuid4
    from sqlmodel import Session, select
    from cowork import scheduler
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import ScopedSession
    from cowork.db.session import get_engine
    from cowork.models.project import Project
    from cowork.models.schedule import ScheduleRun
    from cowork.schemas.schedules import RunStatus
    from cowork.services.schedules import ScheduleService
    from cowork.services.conversations import ConversationService
    from cowork.handlers import responses

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    requests = authority(monkeypatch, body={"allowed": False})
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as raw:
        scoped = ScopedSession(raw, ORG)
        project = Project(
            id=uuid4(), name="denied-schedule", path="/tmp/denied-schedule"
        )
        scoped.add(project)
        scoped.commit()
        scoped.refresh(project)
        schedule = ScheduleService(scoped).create_schedule(
            title="Denied",
            prompt="Run",
            cadence="daily",
            next_run_at=datetime.now(timezone.utc),
            model="default",
            project_id=project.id,
        )
        schedule_id = schedule.id
    create = Mock(side_effect=AssertionError("No conversation creation"))
    handler = Mock(side_effect=AssertionError("No handler or model"))
    monkeypatch.setattr(ConversationService, "create_conversation", create)
    monkeypatch.setattr(responses, "ResponsesHandler", handler)
    try:
        await scheduler.execute_schedule(schedule_id, is_manual=is_manual)
        with Session(engine) as raw:
            runs = raw.exec(
                select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)
            ).all()
            assert len(runs) == 1 and runs[0].status == RunStatus.failed
            assert "permission_denied" in runs[0].error
            assert runs[0].conversation_id is None
        assert len(requests) == 1
        create.assert_not_called()
        handler.assert_not_called()
    finally:
        get_app_settings.cache_clear()


@pytest.mark.parametrize(
    "scope",
    [
        TenantScope(org_mode=True),
        TenantScope(org_mode=True, org_id=ORG.org_id),
        TenantScope(org_mode=True, user_id=ORG.user_id),
    ],
)
async def test_missing_verified_tenant_identity_cannot_obtain_a_decision(
    monkeypatch, scope
):
    calls = authority(monkeypatch, body={"allowed": True})
    with pytest.raises(permissions.ProductPermissionDenied):
        await permissions.require_product_permission(scope, "product.execute")
    assert calls == []


@pytest.mark.parametrize("error", [TimeoutError(), ValueError("Invalid JSON")])
async def test_invalid_or_timed_out_permission_response_is_unavailable(
    monkeypatch, error
):
    authority(monkeypatch, error=error)
    with pytest.raises(permissions.ProductPermissionUnavailable):
        await permissions.require_product_permission(ORG, "artifact.manage")


async def test_a_role_grant_does_not_override_narrow_artifact_operation_flags(
    monkeypatch,
):
    from cowork.api.v1.endpoints import artifact_workspace

    authority(monkeypatch, body={"allowed": True})
    narrow = {
        "canEdit": True,
        "canAddressWithAgent": False,
        "canResolveComments": False,
    }
    assert (
        await artifact_workspace._current_capabilities(
            SimpleNamespace(scope=ORG), narrow
        )
        == narrow
    )


async def test_agent_repair_requires_execution_after_artifact_management(monkeypatch):
    from cowork.api.v1.endpoints import artifact_workspace

    calls = []

    async def allowed(scope, permission):
        calls.append(permission)
        return permission == "artifact.manage"

    monkeypatch.setattr(permissions, "has_product_permission", allowed)
    with pytest.raises(permissions.ProductPermissionDenied):
        await artifact_workspace.request_agent_repair(
            "project", "artifact", Mock(), SimpleNamespace(scope=ORG)
        )
    assert calls == ["artifact.manage", "product.execute"]


async def test_hosted_queue_cannot_hide_missing_identity_as_desktop(monkeypatch):
    from cowork.turnqueue import producer
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    calls = authority(monkeypatch, body={"allowed": True})
    redis = Mock(side_effect=AssertionError("Do not enqueue"))
    monkeypatch.setattr(producer, "get_redis", redis)
    try:
        with pytest.raises(permissions.ProductPermissionDenied):
            async for _ in producer.stream_remote_replies(
                org_id=None,
                user_id=None,
                conversation_id="conversation",
                input_text="Run",
                model="free-model",
                llm={"api_key": "old-key"},
            ):
                pass
        assert calls == []
        redis.assert_not_called()
    finally:
        get_app_settings.cache_clear()


def test_generic_write_detaches_a_hardlink_to_a_protected_artifact(
    monkeypatch, tmp_path
):
    import os
    import stat
    from contextlib import contextmanager
    from cowork.api.v1.endpoints import project_files
    from cowork.db.scoped import get_scoped_session

    base = tmp_path / "project"
    artifact = base / ".anton" / "artifacts" / "report" / "index.html"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("protected")
    artifact.chmod(0o750)
    alias = base / "ordinary.html"
    os.link(artifact, alias)
    requests = authority(monkeypatch, body={"allowed": False})

    @contextmanager
    def inventory(scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    monkeypatch.setattr(project_files, "_opened_project_directory_inventory", inventory)
    monkeypatch.setattr(project_files, "_project_dir", lambda *args: base)
    app = FastAPI()
    app.include_router(project_files.router)
    app.dependency_overrides[get_scoped_session] = lambda: SimpleNamespace(scope=ORG)
    with TestClient(app) as client:
        response = client.put(
            "/project/files/ordinary.html", json={"content": "changed alias"}
        )
    assert response.status_code == 200, response.text
    assert requests == []
    assert alias.read_text() == "changed alias"
    assert artifact.read_text() == "protected"
    assert alias.stat().st_ino != artifact.stat().st_ino
    assert stat.S_IMODE(alias.stat().st_mode) == 0o750
    assert not list(base.glob(".cowork-file-*"))


@pytest.mark.parametrize("failure", ["write", "zero-write", "replace"])
def test_atomic_generic_write_keeps_original_bytes_and_cleans_failed_temporary(
    monkeypatch, tmp_path, failure
):
    from cowork.api.v1.endpoints import project_files

    original = tmp_path / "notes.txt"
    original.write_text("original")

    def fail(*args):
        raise OSError("fixture failure")

    if failure == "write":
        monkeypatch.setattr(project_files.os, "write", fail)
    elif failure == "zero-write":
        monkeypatch.setattr(project_files.os, "write", lambda *args: 0)
    else:
        monkeypatch.setattr(project_files, "dir_replace", fail)
    with project_files.pinned_dir(tmp_path) as root:
        with pytest.raises(OSError):
            project_files._write_bytes_at_project_root(
                root, project_files._validated_project_path("notes.txt"), b"replacement"
            )
    assert original.read_text() == "original"
    assert list(tmp_path.iterdir()) == [original]


def test_atomic_write_supports_a_platform_without_descriptor_chmod(
    monkeypatch, tmp_path
):
    from cowork.api.v1.endpoints import project_files

    original = tmp_path / "notes.txt"
    original.write_text("original")
    monkeypatch.delattr(project_files.os, "fchmod")
    with project_files.pinned_dir(tmp_path) as root:
        project_files._write_bytes_at_project_root(
            root, project_files._validated_project_path("notes.txt"), b"replacement"
        )
    assert original.read_bytes() == b"replacement"


@pytest.mark.parametrize("purpose", ["execution", "artifact_publish"])
async def test_mint_declares_a_single_server_selected_purpose(monkeypatch, purpose):
    import json
    from cowork.turnqueue.auth_keys import mint_turn_key

    requests = authority(monkeypatch, body={"key": "synthetic-key"}, status=201)
    key = await mint_turn_key(
        user_id=ORG.user_id,
        org_id=ORG.org_id,
        correlation_id="instance",
        ttl_seconds=60,
        settings=permissions.TurnQueueSettings(),
        purpose=purpose,
    )
    assert key == "synthetic-key"
    body = json.loads(requests[0].content)
    assert body["purpose"] == purpose
    assert body["user_id"] == ORG.user_id
    assert body["organization_id"] == ORG.org_id
    assert body["instance_id"] == "instance"
    assert "scopes" not in body


async def test_decisions_are_live_between_consecutive_actions(monkeypatch):
    decision = {"allowed": True}
    requests = authority(monkeypatch, body=decision)
    await permissions.require_product_permission(ORG, "product.execute")
    decision["allowed"] = False
    with pytest.raises(permissions.ProductPermissionDenied):
        await permissions.require_product_permission(ORG, "product.execute")
    assert len(requests) == 2
