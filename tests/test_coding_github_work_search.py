from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import coding
from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import CodeProject, RepositoryResource, WorkItemSearchRequest
from cowork.coding.workspace import WorkspaceError
from cowork.schemas.connectors import ConnectionSummaryResponse


def item(number: int, kind: str, day: int) -> dict:
    result = {
        "number": number,
        "title": f"Work {number}",
        "html_url": f"https://github.com/mindsdb/cowork/{'pull' if kind == 'pr' else 'issues'}/{number}",
        "repository_url": "https://api.github.com/repos/mindsdb/cowork",
        "state": "open",
        "updated_at": f"2026-09-{day:02}T12:00:00Z",
        "assignees": [{"login": "ianu82"}],
    }
    if kind == "pr":
        result["pull_request"] = {"url": f"https://api.github.com/repos/mindsdb/cowork/pulls/{number}"}
    return result


@pytest.fixture
def github_search():
    requests = []
    responses = {
        "is:issue": httpx.Response(200, json={"items": [item(1, "issue", 5), item(2, "issue", 3)]}),
        "is:pull-request": httpx.Response(200, json={"items": [item(3, "pr", 6), item(4, "pr", 4)]}),
    }

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/search/issues"
        assert request.headers["Authorization"] == "Bearer test-github-token"
        assert request.url.params["sort"] == "updated"
        assert request.url.params["order"] == "desc"
        qualifier = request.url.params["q"].split()[0]
        # Reproduce GitHub's app-token restriction, not an always-successful mock.
        return responses.get(qualifier, httpx.Response(422, json={
            "message": "Query must include 'is:issue' or 'is:pull-request'",
        }))

    integration = DeveloperIntegrationService(None, transport=httpx.MockTransport(handler))
    integration.connections = SimpleNamespace(
        list=lambda: [ConnectionSummaryResponse(engine="github", name="work")],
        runtime_fields=lambda *_: {"access_token": "test-github-token"},
    )
    yield integration, requests, responses
    integration.close()


@pytest.mark.parametrize("project_id", [None, "project"])
@pytest.mark.parametrize("query", ["", "  ", "  repo:mindsdb/cowork delivery  "])
def test_search_routes_combine_issue_and_pr_results(github_search, monkeypatch, project_id, query):
    integration, requests, _ = github_search
    project = CodeProject(id="project", name="Product", resources=[
        RepositoryResource(id="repo", name="cowork", source_url="https://github.com/mindsdb/cowork.git"),
    ])
    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(projects=SimpleNamespace(get=lambda _: project)))
    app = FastAPI()
    app.include_router(coding.router, prefix="/coding")
    app.dependency_overrides[coding._integration_service] = lambda: integration
    route = f"/coding/projects/{project_id}/work-items/search" if project_id else "/coding/work-items/search"

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(route, json={"provider": "github", "connection_name": "work", "query": query, "limit": 3})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [entry["external_id"] for entry in payload["items"]] == [
        "mindsdb/cowork#3", "mindsdb/cowork#1", "mindsdb/cowork#4",
    ]
    assert [entry["kind"] for entry in payload["items"]] == ["pull_request", "issue", "pull_request"]
    assert all(entry["connection_name"] == "work" and entry["assignee"] == "ianu82" for entry in payload["items"])
    assert payload["incomplete"] is False
    search = f"{query.strip()} is:open" if query.strip() else "is:open assignee:@me"
    assert [request.url.params["q"] for request in requests] == [f"is:issue {search}", f"is:pull-request {search}"]
    assert all(request.url.params["per_page"] == "3" for request in requests)
    assert project.connections == []


@pytest.mark.parametrize("incomplete_kind", [None, "is:issue", "is:pull-request"])
def test_empty_results_preserve_either_searches_incomplete_flag(github_search, incomplete_kind):
    integration, _, responses = github_search
    for kind in responses:
        responses[kind] = httpx.Response(200, json={"items": [], "incomplete_results": kind == incomplete_kind})
    page = integration.search(None, WorkItemSearchRequest(provider="github"))
    assert page.items == []
    assert page.incomplete is (incomplete_kind is not None)


@pytest.mark.parametrize("qualifier", ["is:issue", "is:pr", "type:pr"])
def test_explicit_user_type_filters_are_retained_and_overlapping_results_are_deduplicated(github_search, qualifier):
    integration, requests, responses = github_search
    kind = "issue" if qualifier == "is:issue" else "pr"
    # GitHub honors the last type qualifier: both requests can return the same work.
    for key in responses:
        responses[key] = httpx.Response(200, json={"items": [item(8, kind, 7)]})
    page = integration.search(None, WorkItemSearchRequest(provider="github", query=f"{qualifier} fix"))
    assert len(page.items) == 1
    assert page.items[0].kind == ("issue" if kind == "issue" else "pull_request")
    assert all(request.url.params["q"].endswith(f"{qualifier} fix is:open") for request in requests)


@pytest.mark.parametrize("failed_kind", ["is:issue", "is:pull-request"])
@pytest.mark.parametrize("status", [401, 403, 422, 429, 503])
def test_failure_in_either_search_is_not_hidden_as_a_successful_partial_list(github_search, failed_kind, status):
    integration, requests, responses = github_search
    responses[failed_kind] = httpx.Response(status, json={"message": "Request rejected"})
    message = "expired or lacks permission" if status in {401, 403} else f"HTTP {status}"
    with pytest.raises(WorkspaceError, match=message):
        integration.search(None, WorkItemSearchRequest(provider="github"))
    assert len(requests) == (1 if failed_kind == "is:issue" else 2)


def test_missing_urls_and_non_object_entries_are_ignored(github_search):
    integration, _, responses = github_search
    responses["is:issue"] = httpx.Response(200, json={"items": [None, "invalid", {}, item(1, "issue", 5)]})
    responses["is:pull-request"] = httpx.Response(200, json={"items": None})
    page = integration.search(None, WorkItemSearchRequest(provider="github"))
    assert [entry.external_id for entry in page.items] == ["mindsdb/cowork#1"]


def test_limit_applies_after_combining_both_searches(github_search):
    integration, _, responses = github_search
    responses["is:issue"] = httpx.Response(200, json={"items": [item(number, "issue", 5) for number in range(1, 51)]})
    responses["is:pull-request"] = httpx.Response(200, json={"items": [item(number, "pr", 6) for number in range(51, 101)]})
    page = integration.search(None, WorkItemSearchRequest(provider="github", limit=50))
    assert len(page.items) == 50
    assert all(entry.kind == "pull_request" for entry in page.items)
