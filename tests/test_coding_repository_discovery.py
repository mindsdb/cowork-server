from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import coding
from cowork.coding.integrations import GitPushCredentials, local_repository_credentials
from cowork.coding.project_models import CodeProject, ProjectConnection, RepositoryResource
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.workspace import WorkspaceError, WorkspaceManager
from test_coding_integrations import public_resolver, service

FIELDS = {("github", "work"): {"access_token": "test-secret"}}
REPOSITORY = {"full_name": "acme/private", "private": True, "default_branch": "staging", "archived": False}


def test_lists_personal_collaborator_and_organisation_repositories_without_exposing_secrets():
    def handler(request):
        assert request.url.path == "/user/repos"
        assert dict(request.url.params) == {"affiliation": "owner,collaborator,organization_member", "sort": "updated", "per_page": "100", "page": "2"}
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(200, json=[REPOSITORY], headers={"Link": '<https://api.github.com/user/repos?page=3>; rel="next"'})
    integration = service(handler, FIELDS)
    try:
        page = integration.repositories("work", 2)
        assert page.next_page == 3
        assert page.items[0].model_dump() == {
            "full_name": "acme/private", "private": True, "default_branch": "staging", "archived": False,
            "connection_name": "work", "clone_url": "https://github.com/acme/private.git",
        }
        assert "test-secret" not in page.model_dump_json()
    finally:
        integration.close()


@pytest.mark.parametrize("host", ["ghe.example.com", "[2606:4700::6810:1]"])
@pytest.mark.parametrize("port", ["", ":8443"])
def test_enterprise_discovery_uses_the_validated_host_and_not_supplied_clone_urls(host, port):
    def handler(request):
        assert request.url.path == "/api/v3/user/repos"
        assert request.headers["Host"] == f"{host}{port}"
        return httpx.Response(200, json=[{**REPOSITORY, "clone_url": "https://attacker.example/steal", "archived": True}])
    integration = service(handler, {("github", "work"): {"access_token": "secret", "base_url": f"https://{host}{port}"}}, resolver=public_resolver)
    try:
        page = integration.repositories("work")
        assert page.items[0].clone_url == f"https://{host}{port}/acme/private.git"
        resource = RepositoryResource(id="repo", name="Private", source_url=page.items[0].clone_url)
        assert resource.source_url == page.items[0].clone_url
        assert page.items[0].archived
        assert page.next_page is None
    finally:
        integration.close()


@pytest.mark.parametrize("payload", [[None, {}, {"full_name": "../private"}, {"full_name": "https://evil.test/x"}], []])
def test_invalid_repository_objects_are_not_offered(payload):
    integration = service(lambda _: httpx.Response(200, json=payload), FIELDS)
    try:
        assert integration.repositories("work").items == []
    finally:
        integration.close()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_provider_failures_are_actionable_and_do_not_leak_the_response_body(status):
    integration = service(lambda _: httpx.Response(status, json={"error": "test-secret"}), FIELDS)
    try:
        with pytest.raises(WorkspaceError) as raised:
            integration.repositories("work")
        assert "test-secret" not in str(raised.value)
        assert "permission" in str(raised.value) if status in {401, 403} else f"HTTP {status}" in str(raised.value)
    finally:
        integration.close()


@pytest.mark.parametrize("fields,name", [({}, "work"), (FIELDS, "another-account"), ({("github", "work"): {"status": "needs_reconnect"}}, "work")])
def test_missing_or_unavailable_connections_never_make_an_upstream_request(fields, name):
    integration = service(lambda _: pytest.fail("unexpected upstream request"), fields)
    try:
        with pytest.raises(WorkspaceError):
            integration.repositories(name)
    finally:
        integration.close()


def test_repository_endpoint_uses_scoped_integration_and_normalizes_errors():
    integration = service(lambda _: httpx.Response(200, json=[REPOSITORY]), FIELDS)
    try:
        assert coding.list_github_repositories(integration, "work").items[0].connection_name == "work"
    finally:
        integration.close()


def test_repository_route_preserves_loopback_origin_and_tenant_guards(monkeypatch):
    app = FastAPI()
    app.include_router(coding.router, prefix="/coding")
    integration = SimpleNamespace(repositories=Mock(return_value={"items": [], "next_page": None}))
    app.dependency_overrides[coding._integration_service] = lambda: integration
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/coding/github/repositories", params={"connection_name": "work", "page": 2}).status_code == 200
        integration.repositories.assert_called_once_with("work", 2)
        integration.repositories.reset_mock()
        assert client.get("/coding/github/repositories?connection_name=work", headers={"Origin": "https://evil.invalid"}).status_code == 403
        for query in ["", "?connection_name=work&page=0", "?connection_name=work&page=10001"]:
            assert client.get(f"/coding/github/repositories{query}").status_code == 422
        integration.repositories.assert_not_called()
    with TestClient(app, client=("203.0.113.1", 50000)) as client:
        assert client.get("/coding/github/repositories?connection_name=work").status_code == 403
    monkeypatch.setattr("cowork.common.settings.app_settings.get_app_settings", lambda: SimpleNamespace(tenancy_mode="org"))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/coding/github/repositories?connection_name=work").status_code == 403
    integration.repositories.assert_not_called()


def test_invalid_provider_response_is_not_an_empty_success():
    integration = service(lambda _: httpx.Response(200, json={"unexpected": "object"}), FIELDS)
    try:
        with pytest.raises(WorkspaceError, match="could not list repositories"):
            integration.repositories("work")
    finally:
        integration.close()


def test_pagination_never_follows_an_untrusted_link():
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=[], headers={"Link": '<https://evil.invalid/steal>; rel="next"'})
    integration = service(handler, FIELDS)
    try:
        next_page = integration.repositories("work").next_page
        assert next_page == 2
        integration.repositories("work", next_page)
        assert all(url.startswith("https://api.github.com/user/repos?") for url in calls)
    finally:
        integration.close()


def test_clone_and_refresh_receive_ephemeral_credentials_but_local_git_operations_do_not(tmp_path):
    resource = RepositoryResource(id="repo", name="Private", source_url="https://github.com/acme/private.git", connector_name="work")
    calls = []
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path))
    encoded = base64.b64encode(b"x-access-token:secret").decode()
    credentials = GitPushCredentials(resource.source_url, {
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"http.{resource.source_url}.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded}",
    })

    def run(cwd, *args, **kwargs):
        calls.append((args, kwargs.get("environment")))
        if args[0] == "clone":
            Path(args[-1]).mkdir()
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    manager.workspaces.git.run = run
    root = manager._repository_cache(resource, credentials)
    assert root.is_dir()
    assert manager._repository_cache(resource, credentials) == root
    for args, environment in calls:
        assert "secret" not in str(args) and encoded not in str(args)
        if args[0] in {"clone", "fetch"}:
            assert environment["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {encoded}"
            assert environment["GIT_CONFIG_VALUE_1"] == "false"
        else:
            assert environment is None
    assert [args[0] for args, _ in calls][:2] == ["clone", "fetch"]


def test_existing_local_checkout_does_not_need_connector_credentials(tmp_path):
    resource = RepositoryResource(id="repo", name="Local", local_path=str(tmp_path), source_url="https://github.com/acme/private.git", connector_name="work")
    project = CodeProject(id="project", name="Project", resources=[resource])
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"), repository_credentials=lambda *_: pytest.fail("local checkout should not contact GitHub"))
    assert manager._runtime_folder(resource, project).path == str(tmp_path)


def test_manual_public_repository_does_not_inherit_a_projects_github_connection(tmp_path):
    project = CodeProject(
        id="project", name="Project",
        resources=[RepositoryResource(id="public", name="Public", source_url="https://github.com/acme/public.git")],
        connections=[ProjectConnection(provider="github", name="unavailable-or-enterprise")],
    )
    # Saving and loading must not bind a pasted URL to the sole connection.
    project = CodeProject.model_validate_json(project.model_dump_json())
    resource = project.resources[0]
    resolve_credentials = Mock(side_effect=WorkspaceError("Connection unavailable"))
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path), repository_credentials=resolve_credentials)

    def run(cwd, *args, **kwargs):
        if args[0] == "clone":
            Path(args[-1]).mkdir()
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    manager.workspaces.git.run = Mock(side_effect=run)
    first = manager._runtime_folder(resource, project)
    assert manager._runtime_folder(resource, project).path == first.path
    assert resource.connector_name is None
    resolve_credentials.assert_not_called()
    calls = manager.workspaces.git.run.call_args_list
    assert [call.args[1] for call in calls][:2] == ["clone", "fetch"]
    assert all(call.kwargs.get("environment") is None for call in calls)


def test_explicit_repository_connection_survives_reload_and_does_not_fall_back_on_auth_failure(tmp_path):
    project = CodeProject(
        id="project", name="Project",
        resources=[RepositoryResource(id="private", name="Private", source_url="https://github.com/acme/private.git", connector_name="work")],
        connections=[ProjectConnection(provider="github", name="work")],
    )
    project = CodeProject.model_validate_json(project.model_dump_json())
    resource = project.resources[0]
    resolve_credentials = Mock(side_effect=WorkspaceError("Connection unavailable"))
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path), repository_credentials=resolve_credentials)
    manager.workspaces.git.run = Mock()

    assert resource.connector_name == "work"
    with pytest.raises(WorkspaceError, match="Connection unavailable"):
        manager._runtime_folder(resource, project)
    resolve_credentials.assert_called_once_with(project, resource)
    manager.workspaces.git.run.assert_not_called()


def test_removed_connector_does_not_fall_back_to_another_projects_connection():
    resource = RepositoryResource(id="repo", name="Private", source_url="https://github.com/acme/private.git", connector_name="removed")
    project = CodeProject(id="project", name="Project", resources=[resource], connections=[ProjectConnection(provider="github", name="other")])
    with pytest.raises(WorkspaceError, match="this repository's GitHub connection"):
        local_repository_credentials(project, resource)


def test_local_credential_resolution_is_disabled_in_org_mode(monkeypatch):
    import cowork.coding.integrations as integrations
    from cowork.db.scoped import MissingTenantScopeError
    resource = RepositoryResource(id="repo", name="Private", source_url="https://github.com/acme/private.git", connector_name="work")
    project = CodeProject(id="project", name="Project", resources=[resource], connections=[ProjectConnection(provider="github", name="work")])
    def refuse():
        raise MissingTenantScopeError("no background scope")
    monkeypatch.setattr(integrations, "scope_for_background_context", refuse)
    with pytest.raises(MissingTenantScopeError):
        local_repository_credentials(project, resource)
