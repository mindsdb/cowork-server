from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from cowork.coding.project_models import (
    CodeProject,
    DeliveryRecord,
    IntegrationStatus,
    ProjectConnection,
    PublishRequest,
    PullRequestStatus,
    SourceContext,
    SourceContextRequest,
)
from cowork.coding.workspace import WorkspaceError
from cowork.db.scoped import TenantScope
from cowork.services.connectors.connections import ConnectionsService


@dataclass(frozen=True)
class SlackTarget:
    channel: str
    thread_ts: str | None


@dataclass(frozen=True)
class GitPushCredentials:
    remote_url: str
    environment: dict[str, str]


class DeveloperIntegrationService:
    """Agent-neutral reads and explicit, user-confirmed developer-tool writes."""

    def __init__(
        self,
        scope: TenantScope | None,
        client: httpx.Client | None = None,
    ) -> None:
        self.connections = ConnectionsService(scope)
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=20.0, follow_redirects=True)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def statuses(self, project: CodeProject) -> list[IntegrationStatus]:
        available = {(item.engine, item.name): item for item in self.connections.list()}
        result: list[IntegrationStatus] = []
        for configured in project.connections:
            connection = available.get((configured.provider, configured.name))
            if connection is None:
                status, detail = "missing", "Connection is no longer available"
            elif connection.status == "needs_reconnect":
                status, detail = "reconnect", "Reconnect to continue using this tool"
            else:
                status, detail = "connected", ""
            result.append(
                IntegrationStatus(
                    provider=configured.provider,
                    connection_name=configured.name,
                    label=configured.label or (connection.display_name if connection else None) or configured.name,
                    status=status,
                    detail=detail,
                )
            )
        return result

    def read(self, project: CodeProject, request: SourceContextRequest) -> SourceContext:
        connection, fields = self._connection(project, request.provider, request.connection_name)
        if request.provider == "github":
            return self._read_github(request, connection, fields)
        if request.provider == "linear":
            return self._read_linear(request, connection, fields)
        return self._read_slack(request, connection, fields)

    def publish(self, project: CodeProject, request: PublishRequest) -> DeliveryRecord:
        if not request.confirmed:
            raise WorkspaceError("Confirm this external update before publishing it")
        connection, fields = self._connection(project, request.provider, request.connection_name)
        if request.provider == "github":
            external_url = self._publish_github(request, fields)
        elif request.provider == "linear":
            external_url = self._publish_linear(request, fields)
        else:
            external_url = self._publish_slack(request, fields)
        return DeliveryRecord(
            provider=request.provider,
            action=request.action,
            target_url=request.target_url,
            status="published",
            external_url=external_url,
            detail=f"Published with {connection.label or connection.name}",
        )

    def create_draft_pull_request(
        self,
        project: CodeProject,
        *,
        repository_url: str,
        title: str,
        body: str,
        head: str,
        base_branch: str,
        connection_name: str | None,
    ) -> str:
        """Create a draft PR through a project-scoped GitHub connection."""
        _, fields = self._connection(project, "github", connection_name)
        api, token, host = self._github_credentials(fields)
        owner, repository = self._github_repository(repository_url, host)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        existing = self._request(
            "GET",
            f"{api}/repos/{owner}/{repository}/pulls",
            headers=headers,
            params={"state": "all", "head": f"{owner}:{head}", "per_page": 10},
        ).json()
        existing_items = existing if isinstance(existing, list) else []
        for item in existing_items:
            if str((item.get("head") or {}).get("ref") or "") == head and item.get("html_url"):
                return str(item["html_url"])
        response = self._request(
            "POST",
            f"{api}/repos/{owner}/{repository}/pulls",
            headers=headers,
            json={"title": title, "body": body, "head": head, "base": base_branch, "draft": True},
        )
        payload = response.json()
        external_url = payload.get("html_url")
        if not isinstance(external_url, str) or not external_url:
            raise WorkspaceError("GitHub created the draft pull request without returning its link")
        return external_url

    def git_push_credentials(
        self,
        project: CodeProject,
        repository_url: str,
        connection_name: str | None = None,
    ) -> GitPushCredentials:
        """Create ephemeral Git HTTP auth without putting a token in argv or persisted state."""
        _, fields = self._connection(project, "github", connection_name)
        _, token, host = self._github_credentials(fields)
        owner, repository = self._github_repository(repository_url, host)
        base = str(fields.get("base_url") or fields.get("url") or "https://github.com").rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme != "https":
            raise WorkspaceError("GitHub branch publishing requires an HTTPS connection URL")
        remote_url = f"{parsed.scheme}://{parsed.netloc}/{owner}/{repository}.git"
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        return GitPushCredentials(
            remote_url=remote_url,
            environment={
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": f"http.{remote_url}.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded}",
            },
        )

    def pull_request_status(
        self,
        project: CodeProject,
        url: str,
        connection_name: str | None = None,
    ) -> PullRequestStatus:
        """Return the small review/CI summary Code needs for a linked draft."""
        _, fields = self._connection(project, "github", connection_name)
        api, token, host = self._github_credentials(fields)
        owner, repository, resource, number = self._github_target(url, host)
        if resource != "pulls":
            raise WorkspaceError("The linked GitHub delivery is not a pull request")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        pull = self._request("GET", f"{api}/repos/{owner}/{repository}/pulls/{number}", headers=headers).json()
        head = str((pull.get("head") or {}).get("sha") or "")
        unavailable: list[str] = []
        try:
            reviews = self._request(
                "GET",
                f"{api}/repos/{owner}/{repository}/pulls/{number}/reviews",
                headers=headers,
                params={"per_page": 100},
            ).json()
        except WorkspaceError:
            reviews = []
            unavailable.append("review status")
        checks: dict[str, Any] = {}
        combined: dict[str, Any] = {}
        if head:
            try:
                checks = self._request(
                    "GET",
                    f"{api}/repos/{owner}/{repository}/commits/{head}/check-runs",
                    headers=headers,
                    params={"per_page": 100},
                ).json()
            except WorkspaceError:
                unavailable.append("check runs")
            try:
                combined = self._request(
                    "GET",
                    f"{api}/repos/{owner}/{repository}/commits/{head}/status",
                    headers=headers,
                ).json()
            except WorkspaceError:
                unavailable.append("commit status")
        return PullRequestStatus(
            state=self._pull_request_state(pull),
            review_state=self._review_state(reviews if isinstance(reviews, list) else []),
            ci_state=self._ci_state(checks, combined),
            detail=f"Could not load {', '.join(unavailable)}" if unavailable else "",
        )

    def _connection(
        self,
        project: CodeProject,
        provider: str,
        requested_name: str | None,
    ) -> tuple[ProjectConnection, dict[str, Any]]:
        candidates = [item for item in project.connections if item.provider == provider]
        connection = next((item for item in candidates if item.name == requested_name), None) if requested_name else None
        connection = connection or (candidates[0] if len(candidates) == 1 else None)
        provider_label = "GitHub" if provider == "github" else provider.title()
        if connection is None:
            if candidates:
                raise WorkspaceError(f"Choose which {provider_label} connection to use")
            raise WorkspaceError(f"Connect {provider_label} in Cowork, then add it to this Code Project")
        fields = self.connections.runtime_fields(provider, connection.name)
        if fields is None:
            raise WorkspaceError(f"The {provider_label} connection is unavailable")
        if fields.get("status") == "needs_reconnect":
            raise WorkspaceError(f"Reconnect {provider_label} before using this project source")
        return connection, fields

    def _read_github(
        self,
        request: SourceContextRequest,
        connection: ProjectConnection,
        fields: dict[str, Any],
    ) -> SourceContext:
        api, token, host = self._github_credentials(fields)
        owner, repository, resource, number = self._github_target(request.url, host)
        response = self._request(
            "GET",
            f"{api}/repos/{owner}/{repository}/{resource}/{number}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        payload = response.json()
        return SourceContext(
            provider="github",
            kind="pull_request" if resource == "pulls" else "issue",
            url=str(payload.get("html_url") or request.url),
            title=str(payload.get("title") or ""),
            external_id=f"{owner}/{repository}#{number}",
            connection_name=connection.name,
            body=str(payload.get("body") or ""),
        )

    def _read_linear(
        self,
        request: SourceContextRequest,
        connection: ProjectConnection,
        fields: dict[str, Any],
    ) -> SourceContext:
        identifier = urlparse(request.url).path.rstrip("/").split("/")[-1]
        token = self._secret(fields, "access_token", "api_key", "token")
        issue = self._linear_issue(identifier, token)
        return SourceContext(
            provider="linear",
            kind="issue",
            url=str(issue.get("url") or request.url),
            title=str(issue.get("title") or ""),
            external_id=str(issue.get("id") or issue.get("identifier") or identifier),
            connection_name=connection.name,
            body=str(issue.get("description") or ""),
        )

    def _linear_issue(self, identifier: str, token: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "query": "query CodeIssue($id: String!) { issue(id: $id) { id identifier title description url } }",
                "variables": {"id": identifier},
            },
        )
        payload = response.json()
        if payload.get("errors"):
            raise WorkspaceError(str(payload["errors"][0].get("message") or "Linear could not load this issue"))
        issue = (payload.get("data") or {}).get("issue") or {}
        if not issue.get("id"):
            raise WorkspaceError("Linear could not find this issue")
        return issue

    def _read_slack(
        self,
        request: SourceContextRequest,
        connection: ProjectConnection,
        fields: dict[str, Any],
    ) -> SourceContext:
        target = self._slack_target(request.url)
        token = self._secret(fields, "bot_token", "access_token", "token")
        method = "conversations.replies" if target.thread_ts else "conversations.history"
        params: dict[str, Any] = {"channel": target.channel, "limit": 100}
        if target.thread_ts:
            params["ts"] = target.thread_ts
        response = self._request(
            "GET",
            f"https://slack.com/api/{method}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        payload = response.json()
        if not payload.get("ok"):
            raise WorkspaceError(f"Slack could not load this conversation: {payload.get('error') or 'unknown error'}")
        messages = payload.get("messages") or []
        body = "\n\n".join(str(item.get("text") or "") for item in messages if item.get("text"))
        return SourceContext(
            provider="slack",
            kind="conversation",
            url=request.url,
            title="Slack conversation",
            external_id=f"{target.channel}:{target.thread_ts or ''}",
            connection_name=connection.name,
            body=body[:100_000],
        )

    def _publish_github(self, request: PublishRequest, fields: dict[str, Any]) -> str | None:
        api, token, host = self._github_credentials(fields)
        owner, repository, _, number = self._github_target(request.target_url, host)
        response = self._request(
            "POST",
            f"{api}/repos/{owner}/{repository}/issues/{number}/comments",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"body": request.text},
        )
        return response.json().get("html_url")

    def _publish_linear(self, request: PublishRequest, fields: dict[str, Any]) -> str | None:
        identifier = urlparse(request.target_url).path.rstrip("/").split("/")[-1]
        token = self._secret(fields, "access_token", "api_key", "token")
        issue_id = str(self._linear_issue(identifier, token)["id"])
        response = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "query": "mutation CodeComment($input: CommentCreateInput!) { commentCreate(input: $input) { success comment { url } } }",
                "variables": {"input": {"issueId": issue_id, "body": request.text}},
            },
        )
        payload = response.json()
        if payload.get("errors"):
            raise WorkspaceError(str(payload["errors"][0].get("message") or "Linear could not publish this update"))
        comment = (((payload.get("data") or {}).get("commentCreate") or {}).get("comment") or {})
        return comment.get("url")

    def _publish_slack(self, request: PublishRequest, fields: dict[str, Any]) -> str | None:
        target = self._slack_target(request.target_url)
        token = self._secret(fields, "bot_token", "access_token", "token")
        body: dict[str, Any] = {"channel": target.channel, "text": request.text}
        if target.thread_ts:
            body["thread_ts"] = target.thread_ts
        response = self._request(
            "POST",
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
        )
        payload = response.json()
        if not payload.get("ok"):
            raise WorkspaceError(f"Slack could not publish this update: {payload.get('error') or 'unknown error'}")
        return request.target_url

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise WorkspaceError("The connected developer tool is currently unavailable") from exc
        if response.status_code in {401, 403}:
            raise WorkspaceError("This connection has expired or lacks permission for that resource")
        if response.status_code == 404:
            raise WorkspaceError("The linked developer resource is unavailable or no longer accessible")
        if response.is_error:
            raise WorkspaceError(f"The connected developer tool returned HTTP {response.status_code}")
        return response

    @staticmethod
    def _pull_request_state(payload: dict[str, Any]) -> str:
        if payload.get("merged") or payload.get("merged_at"):
            return "merged"
        if payload.get("draft"):
            return "draft"
        return "closed" if payload.get("state") == "closed" else "open"

    @staticmethod
    def _review_state(reviews: list[dict[str, Any]]) -> str:
        latest: dict[str, str] = {}
        for review in reviews:
            actor = str((review.get("user") or {}).get("login") or review.get("id") or "")
            state = str(review.get("state") or "").upper()
            if actor and state in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
                latest[actor] = state
        if "CHANGES_REQUESTED" in latest.values():
            return "changes_requested"
        if "APPROVED" in latest.values():
            return "approved"
        return "pending" if reviews else "none"

    @staticmethod
    def _ci_state(checks: dict[str, Any], combined: dict[str, Any]) -> str:
        runs = checks.get("check_runs") if isinstance(checks, dict) else []
        runs = runs if isinstance(runs, list) else []
        combined_statuses = combined.get("statuses") if isinstance(combined, dict) else []
        has_combined_status = isinstance(combined_statuses, list) and bool(combined_statuses)
        combined_state = str(combined.get("state") or "").lower() if has_combined_status else ""
        conclusions = {str(item.get("conclusion") or "").lower() for item in runs if isinstance(item, dict)}
        statuses = {str(item.get("status") or "").lower() for item in runs if isinstance(item, dict)}
        if combined_state in {"failure", "error"} or conclusions & {"failure", "timed_out", "cancelled", "action_required", "stale"}:
            return "failing"
        if combined_state == "pending" or statuses & {"queued", "in_progress", "pending"}:
            return "pending"
        if runs or combined_state:
            return "passing"
        return "none"

    @staticmethod
    def _secret(fields: dict[str, Any], *names: str) -> str:
        for name in names:
            value = fields.get(name)
            if isinstance(value, str) and value:
                return value
        raise WorkspaceError("This connection is missing its authentication credential; reconnect it")

    def _github_credentials(self, fields: dict[str, Any]) -> tuple[str, str, str]:
        base = str(fields.get("base_url") or fields.get("url") or "https://github.com").rstrip("/")
        host = (urlparse(base).hostname or "").casefold()
        if not host:
            raise WorkspaceError("The GitHub connection URL is invalid; reconnect it")
        api = "https://api.github.com" if base == "https://github.com" else f"{base}/api/v3"
        token = self._secret(fields, "access_token", "personal_access_token", "token", "api_key")
        return api, token, host

    @staticmethod
    def _github_target(url: str, expected_host: str) -> tuple[str, str, str, str]:
        parsed = urlparse(url)
        if (parsed.hostname or "").casefold() != expected_host:
            raise WorkspaceError("Use a link from the GitHub host connected to this Code Project")
        path = parsed.path.strip("/").split("/")
        if len(path) < 4 or path[2] not in {"issues", "pull"} or not path[3].isdigit():
            raise WorkspaceError("Enter a GitHub issue or pull-request link")
        return path[0], path[1], "pulls" if path[2] == "pull" else "issues", path[3]

    @staticmethod
    def _github_repository(url: str, expected_host: str) -> tuple[str, str]:
        normalized = url.strip()
        if normalized.startswith("git@") and ":" in normalized:
            normalized = "ssh://" + normalized.replace(":", "/", 1)
        parsed = urlparse(normalized)
        if (parsed.hostname or "").casefold() != expected_host:
            raise WorkspaceError("The repository remote does not match this Code Project's GitHub connection")
        path = parsed.path.strip("/").removesuffix(".git").split("/")
        if len(path) < 2 or not path[-2] or not path[-1]:
            raise WorkspaceError("The Git remote does not identify a GitHub repository")
        return path[-2], path[-1]

    @staticmethod
    def _slack_target(url: str) -> SlackTarget:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        query = parse_qs(parsed.query)
        if "archives" in parts:
            index = parts.index("archives")
            channel = parts[index + 1] if len(parts) > index + 1 else ""
            compact = parts[index + 2].removeprefix("p") if len(parts) > index + 2 else ""
            thread_ts = query.get("thread_ts", [None])[0]
            if not thread_ts and len(compact) > 10 and compact.isdigit():
                thread_ts = f"{compact[:10]}.{compact[10:]}"
        elif "client" in parts and len(parts) >= parts.index("client") + 3:
            index = parts.index("client")
            channel = parts[index + 2]
            thread_ts = parts[index + 3] if len(parts) > index + 3 else None
        else:
            channel, thread_ts = "", None
        if not channel:
            raise WorkspaceError("Enter a Slack channel or thread link")
        return SlackTarget(channel=channel, thread_ts=thread_ts)
