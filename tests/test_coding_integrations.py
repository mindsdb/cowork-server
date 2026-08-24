from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cowork.coding.integrations import DeveloperIntegrationService
from cowork.coding.project_models import (
    CodeProject,
    ProjectConnection,
    ProjectFolder,
    PublishRequest,
    PullRequestActionRequest,
    SourceContextRequest,
)
from cowork.coding.workspace import WorkspaceError
from cowork.schemas.connectors import ConnectionSummaryResponse


class FakeConnections:
    def __init__(self, fields: dict[tuple[str, str], dict]) -> None:
        self.fields = fields

    def list(self):
        return [
            ConnectionSummaryResponse(engine=provider, name=name, display_name=name, status=values.get("status"))
            for (provider, name), values in self.fields.items()
        ]

    def runtime_fields(self, provider: str, name: str):
        return self.fields.get((provider, name))


def project(tmp_path: Path) -> CodeProject:
    return CodeProject(
        id="project",
        name="Product",
        folders=[ProjectFolder(id="folder", name="Folder", path=str(tmp_path))],
        connections=[
            ProjectConnection(provider="github", name="github-work", label="GitHub work"),
            ProjectConnection(provider="linear", name="linear-work", label="Linear work"),
            ProjectConnection(provider="slack", name="slack-work", label="Slack work"),
        ],
    )


def service(handler, fields: dict[tuple[str, str], dict]) -> DeveloperIntegrationService:
    integration = DeveloperIntegrationService(None, httpx.Client(transport=httpx.MockTransport(handler)))
    integration.connections = FakeConnections(fields)  # type: ignore[assignment]
    return integration


def test_connected_github_issue_becomes_normalized_source_context(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[{
                "id": 7,
                "body": "Design: [mockup](https://github.com/user-attachments/assets/mockup.png)",
                "html_url": "https://github.com/mindsdb/cowork/issues/42#issuecomment-7",
                "created_at": "2026-08-24T09:00:00Z",
                "user": {"login": "reviewer"},
            }])
        assert request.url.path == "/repos/mindsdb/cowork/issues/42"
        return httpx.Response(200, json={
            "title": "Make Code projects first class",
            "body": "Project context should persist.",
            "html_url": "https://github.com/mindsdb/cowork/issues/42",
            "state": "open",
            "user": {"login": "ian"},
        })

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    context = integration.read(
        project(tmp_path),
        SourceContextRequest(
            provider="github",
            kind="issue",
            url="https://github.com/mindsdb/cowork/issues/42",
        ),
    )

    assert context.external_id == "mindsdb/cowork#42"
    assert context.title == "Make Code projects first class"
    assert context.connection_name == "github-work"
    assert context.state == "open"
    assert context.author == "ian"
    assert context.comments[0].author == "reviewer"
    assert context.attachments[0].url.endswith("mockup.png")


def test_linear_and_slack_reads_use_their_connected_credentials(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.linear.app":
            assert request.headers["Authorization"] == "linear-secret"
            return httpx.Response(200, json={"data": {"issue": {
                "id": "linear-id", "identifier": "ENG-19", "title": "Multi-repo delivery",
                "description": "Ship both repositories.", "url": "https://linear.app/work/issue/ENG-19",
            }}})
        assert request.url.path == "/api/conversations.replies"
        assert request.url.params["channel"] == "C123"
        assert request.url.params["ts"] == "1234567890.123456"
        return httpx.Response(200, json={"ok": True, "messages": [{"text": "First"}, {"text": "Second"}]})

    integration = service(handler, {
        ("linear", "linear-work"): {"api_key": "linear-secret"},
        ("slack", "slack-work"): {"bot_token": "slack-secret"},
    })
    current = project(tmp_path)
    linear = integration.read(current, SourceContextRequest(
        provider="linear", kind="issue", url="https://linear.app/work/issue/ENG-19",
    ))
    slack = integration.read(current, SourceContextRequest(
        provider="slack", kind="conversation", url="https://workspace.slack.com/archives/C123/p1234567890123456",
    ))

    assert linear.external_id == "ENG-19"
    assert slack.body == "First\n\nSecond"
    assert len(requests) == 2


def test_external_publish_requires_confirmation_and_posts_only_after_it(tmp_path: Path) -> None:
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(201, json={"html_url": "https://github.com/mindsdb/cowork/issues/42#issuecomment-1"})

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    body = PublishRequest(
        provider="github",
        action="result",
        target_url="https://github.com/mindsdb/cowork/issues/42",
        text="Implemented and verified.",
    )
    with pytest.raises(WorkspaceError, match="Confirm"):
        integration.publish(project(tmp_path), body)
    assert posted == []

    delivery = integration.publish(project(tmp_path), body.model_copy(update={"confirmed": True}))
    assert posted == [{"body": "Implemented and verified."}]
    assert delivery.status == "published"


def test_linear_publish_resolves_the_issue_identifier_before_creating_a_comment(tmp_path: Path) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if "query CodeIssue" in body["query"]:
            assert body["variables"] == {"id": "ENG-19"}
            return httpx.Response(200, json={"data": {"issue": {
                "id": "2d70a5b3-issue-uuid",
                "identifier": "ENG-19",
                "title": "Multi-repo delivery",
                "description": "Ship both repositories.",
                "url": "https://linear.app/work/issue/ENG-19",
            }}})
        assert body["variables"]["input"]["issueId"] == "2d70a5b3-issue-uuid"
        return httpx.Response(200, json={"data": {"commentCreate": {
            "success": True,
            "comment": {"url": "https://linear.app/work/issue/ENG-19#comment-1"},
        }}})

    integration = service(handler, {("linear", "linear-work"): {"api_key": "linear-secret"}})
    delivery = integration.publish(project(tmp_path), PublishRequest(
        provider="linear",
        action="result",
        target_url="https://linear.app/work/issue/ENG-19",
        text="Implemented and verified.",
        confirmed=True,
    ))

    assert len(requests) == 2
    assert delivery.external_url == "https://linear.app/work/issue/ENG-19#comment-1"


def test_github_draft_pull_request_uses_project_connection_and_remote_identity(tmp_path: Path) -> None:
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/mindsdb/cowork/pulls"
        assert request.headers["Authorization"] == "Bearer secret"
        if request.method == "GET":
            assert request.url.params["head"] == "mindsdb:cowork/product/task"
            return httpx.Response(200, json=[])
        posted.append(json.loads(request.content))
        return httpx.Response(201, json={"html_url": "https://github.com/mindsdb/cowork/pull/99"})

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    link = integration.create_draft_pull_request(
        project(tmp_path),
        repository_url="git@github.com:mindsdb/cowork.git",
        title="Project delivery",
        body="Verified locally.",
        head="cowork/product/task",
        base_branch="staging",
        connection_name="github-work",
    )

    assert link == "https://github.com/mindsdb/cowork/pull/99"
    assert posted == [{
        "title": "Project delivery",
        "body": "Verified locally.",
        "head": "cowork/product/task",
        "base": "staging",
        "draft": True,
    }]


def test_github_draft_pull_request_retry_reuses_the_task_branch(tmp_path: Path) -> None:
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[{
                "html_url": "https://github.com/mindsdb/cowork/pull/99",
                "head": {"ref": "cowork/product/task"},
            }])
        posted.append(json.loads(request.content))
        return httpx.Response(500)

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    link = integration.create_draft_pull_request(
        project(tmp_path),
        repository_url="https://github.com/mindsdb/cowork.git",
        title="Project delivery",
        body="Verified locally.",
        head="cowork/product/task",
        base_branch="staging",
        connection_name="github-work",
    )

    assert link == "https://github.com/mindsdb/cowork/pull/99"
    assert posted == []


def test_github_pull_request_status_condenses_review_and_ci_state(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/pulls/99"):
            return httpx.Response(200, json={
                "state": "open", "draft": False, "merged": False, "head": {"sha": "head-sha"},
                "title": "Project delivery", "html_url": "https://github.com/mindsdb/cowork/pull/99",
                "updated_at": "2026-08-24T10:00:00Z",
            })
        if request.url.path.endswith("/pulls/99/reviews"):
            return httpx.Response(200, json=[{
                "id": 1, "state": "APPROVED", "body": "Looks good.",
                "html_url": "https://github.com/mindsdb/cowork/pull/99#pullrequestreview-1",
                "user": {"login": "reviewer"},
            }])
        if request.url.path.endswith("/pulls/99/comments"):
            return httpx.Response(200, json=[{
                "id": 2, "body": "Please keep this typed.", "path": "src/app.ts",
                "html_url": "https://github.com/mindsdb/cowork/pull/99#discussion_r2",
                "user": {"login": "maintainer"},
            }])
        if request.url.path.endswith("/commits/head-sha/check-runs"):
            return httpx.Response(200, json={
                "check_runs": [{
                    "name": "test", "status": "completed", "conclusion": "success",
                    "details_url": "https://github.com/mindsdb/cowork/actions/runs/1",
                }],
            })
        if request.url.path.endswith("/commits/head-sha/status"):
            return httpx.Response(200, json={"state": "success"})
        return httpx.Response(404)

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    status = integration.pull_request_status(
        project(tmp_path),
        "https://github.com/mindsdb/cowork/pull/99",
        "github-work",
    )

    assert status.state == "open"
    assert status.review_state == "approved"
    assert status.ci_state == "passing"
    assert status.title == "Project delivery"
    assert status.updated_at == "2026-08-24T10:00:00Z"
    assert [(item.name, item.state) for item in status.checks] == [("test", "passing")]
    assert [(item.author, item.path) for item in status.feedback] == [
        ("reviewer", ""),
        ("maintainer", "src/app.ts"),
    ]


def test_github_pull_request_actions_require_confirmation_and_refresh_status(tmp_path: Path) -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/pulls/99/merge"):
            return httpx.Response(200, json={"merged": True})
        if request.url.path.endswith("/pulls/99"):
            return httpx.Response(200, json={
                "state": "closed", "merged": True, "draft": False,
                "title": "Project delivery", "html_url": "https://github.com/mindsdb/cowork/pull/99",
                "head": {"sha": ""},
            })
        if request.url.path.endswith("/pulls/99/reviews") or request.url.path.endswith("/pulls/99/comments"):
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    integration = service(handler, {("github", "github-work"): {"access_token": "secret"}})
    request = PullRequestActionRequest(
        action="merge",
        target_url="https://github.com/mindsdb/cowork/pull/99",
        connection_name="github-work",
    )
    with pytest.raises(WorkspaceError, match="Confirm"):
        integration.pull_request_action(project(tmp_path), request)

    status = integration.pull_request_action(
        project(tmp_path),
        request.model_copy(update={"confirmed": True}),
    )

    assert status.state == "merged"
    assert requests[0] == ("PUT", "/repos/mindsdb/cowork/pulls/99/merge")


def test_github_delivery_rejects_a_remote_from_another_host(tmp_path: Path) -> None:
    integration = service(lambda _: httpx.Response(500), {
        ("github", "github-work"): {"access_token": "secret"},
    })

    with pytest.raises(WorkspaceError, match="does not match"):
        integration.create_draft_pull_request(
            project(tmp_path),
            repository_url="git@gitlab.com:mindsdb/cowork.git",
            title="Project delivery",
            body="Verified locally.",
            head="cowork/product/task",
            base_branch="staging",
            connection_name="github-work",
        )


def test_github_push_credentials_keep_the_token_out_of_the_remote_url(tmp_path: Path) -> None:
    integration = service(lambda _: httpx.Response(500), {
        ("github", "github-work"): {"access_token": "secret"},
    })

    credentials = integration.git_push_credentials(
        project(tmp_path),
        "git@github.com:mindsdb/cowork.git",
        "github-work",
    )

    assert credentials.remote_url == "https://github.com/mindsdb/cowork.git"
    assert "secret" not in credentials.remote_url
    assert credentials.environment["GIT_CONFIG_KEY_0"].endswith(".extraHeader")
    assert credentials.environment["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


def test_missing_expired_and_ambiguous_connections_have_actionable_errors(tmp_path: Path) -> None:
    current = project(tmp_path)
    current.connections.append(ProjectConnection(provider="github", name="github-second", label="Second"))
    integration = service(lambda _: httpx.Response(500), {
        ("github", "github-work"): {"status": "needs_reconnect", "access_token": "old"},
    })

    statuses = integration.statuses(current)
    assert [(item.connection_name, item.status) for item in statuses] == [
        ("github-work", "reconnect"),
        ("linear-work", "missing"),
        ("slack-work", "missing"),
        ("github-second", "missing"),
    ]
    with pytest.raises(WorkspaceError, match="Choose which GitHub connection"):
        integration.read(current, SourceContextRequest(
            provider="github", kind="issue", url="https://github.com/mindsdb/cowork/issues/42",
        ))
