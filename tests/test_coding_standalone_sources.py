from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coding_service_fakes import CREDS, FakeEngine, service_with
from cowork.api.v1.endpoints import coding
from cowork.coding.contracts import SessionCreateRequest, SourceContext
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import SourceContextRequest, WorkItemSearchRequest
from cowork.coding.workspace import WorkspaceError
from cowork.schemas.connectors import ConnectionSummaryResponse


SOURCE = SourceContext(
    provider="linear", kind="issue", url="https://linear.app/mindsdb/issue/ENG-2382/fix-paste",
    title="Fix paste", external_id="ENG-2382", body="Keep pasted URLs in the task description.",
    connection_name="work",
)


@pytest.mark.parametrize("operation", ["read", "search"])
@pytest.mark.parametrize("state", ["missing", "expired", "ambiguous", "wrong-account"])
def test_standalone_sources_only_use_an_available_selected_account(operation: str, state: str) -> None:
    with httpx.MockTransport(lambda _: pytest.fail("unauthorized provider request")) as transport:
        integration = DeveloperIntegrationService(None, transport=transport)
        names = [] if state == "missing" else ["work", "personal"] if state == "ambiguous" else ["work"]
        integration.connections = SimpleNamespace(
            list=lambda: [ConnectionSummaryResponse(engine="linear", name=name) for name in names],
            runtime_fields=lambda *_: {"status": "needs_reconnect"} if state == "expired" else {"api_key": "secret"},
        )
        name = "someone-else" if state == "wrong-account" else None
        request = SourceContextRequest(provider="linear", kind="issue", url=SOURCE.url, connection_name=name) if operation == "read" else WorkItemSearchRequest(provider="linear", connection_name=name)
        with pytest.raises(WorkspaceError, match="Connect|Reconnect|Choose"):
            getattr(integration, operation)(None, request)
        integration.close()


@pytest.fixture
def source_app():
    app = FastAPI()
    app.include_router(coding.router, prefix="/coding")
    integrations = SimpleNamespace(read=Mock(return_value=SOURCE), search=Mock(return_value={"items": [], "incomplete": False}))
    app.dependency_overrides[coding._integration_service] = lambda: integrations
    return app, integrations


@pytest.mark.parametrize(("path", "method", "body"), [
    ("/source-context", "read", {"provider": "linear", "kind": "issue", "url": SOURCE.url, "connection_name": "work"}),
    ("/work-items/search", "search", {"provider": "linear", "query": "paste", "connection_name": "work"}),
])
def test_standalone_source_routes_are_account_scoped_and_keep_local_guards(source_app, monkeypatch, path, method, body) -> None:
    app, integrations = source_app
    call = getattr(integrations, method)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.post(f"/coding{path}", json=body).status_code == 200
        assert call.call_args.args[0] is None
        assert call.call_args.args[1].connection_name == "work"
        call.reset_mock()
        assert client.post(f"/coding{path}", json=body, headers={"Origin": "https://untrusted.invalid"}).status_code == 403
        call.assert_not_called()
        call.side_effect = WorkspaceError("Reconnect Linear before using this source")
        failed = client.post(f"/coding{path}", json=body)
        assert failed.status_code == 409
        assert failed.json()["detail"] == "Reconnect Linear before using this source"
        call.reset_mock()
    with TestClient(app, client=("203.0.113.1", 50000)) as client:
        assert client.post(f"/coding{path}", json=body).status_code == 403
        call.assert_not_called()
    monkeypatch.setattr("cowork.common.settings.app_settings.get_app_settings", lambda: SimpleNamespace(tenancy_mode="org"))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.post(f"/coding{path}", json=body).status_code == 403
        call.assert_not_called()


def test_folder_task_persists_linked_context_and_delivers_it_to_the_agent(tmp_path: Path) -> None:
    folder = tmp_path / "app"
    folder.mkdir()
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    session = service.create_session(SessionCreateRequest(
        path=str(folder), allow_direct_folder=True, prompt="Fix this issue", engine_id="fake",
        source_contexts=[SOURCE],
    ), CREDS, default_engine="fake", default_model="fake-model")
    assert session.project_id is None
    assert session.source_contexts == [SOURCE]
    assert service.store.load_session(session.id).source_contexts == [SOURCE]
    assert service.control.store.get_task(session.task_id).source_contexts == [SOURCE]
    assert "untrusted reference data" in session.developer_instructions
    assert SOURCE.body in session.developer_instructions
    assert engine.opened.wait(timeout=3)
    assert SOURCE.body in engine.configs[0].developer_instructions
