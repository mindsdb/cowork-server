from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from cowork.coding.contracts import SourceAttachment, SourceComment
from cowork.coding.github_pull_requests import GitHubPullRequests
from cowork.coding.project_models import (
    CodeProject,
    DeliveryRecord,
    IntegrationStatus,
    ProjectConnection,
    PublishRequest,
    PullRequestActionRequest,
    PullRequestStatus,
    SourceActionRequest,
    SourceContext,
    SourceContextRequest,
    WorkItemPage,
    WorkItemSearchRequest,
)
from cowork.coding.work_discovery import DeveloperWorkDiscovery
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

    def search(self, project: CodeProject, request: WorkItemSearchRequest) -> WorkItemPage:
        # Discovery is account-scoped: opening the picker must not silently
        # mutate a project. The chosen account is added to the project only
        # when the user actually links one of its work items.
        connection, fields = self._account_connection(request.provider, request.connection_name)
        discovery = DeveloperWorkDiscovery(self._request)
        if request.provider == "github":
            api, token, _ = self._github_credentials(fields)
            return discovery.github(
                api=api,
                token=token,
                query=request.query,
                limit=request.limit,
                connection_name=connection.name,
            )
        return discovery.linear(
            token=self._secret(fields, "access_token", "api_key", "token"),
            query=request.query,
            limit=request.limit,
            connection_name=connection.name,
        )

    def _account_connection(
        self,
        provider: str,
        requested_name: str | None,
    ) -> tuple[ProjectConnection, dict[str, Any]]:
        candidates = [item for item in self.connections.list() if item.engine == provider]
        summary = next((item for item in candidates if item.name == requested_name), None) if requested_name else None
        summary = summary or (candidates[0] if len(candidates) == 1 else None)
        provider_label = "GitHub" if provider == "github" else provider.title()
        if summary is None:
            if candidates:
                raise WorkspaceError(f"Choose which {provider_label} connection to use")
            raise WorkspaceError(f"Connect {provider_label} in Cowork before searching its work")
        fields = self.connections.runtime_fields(provider, summary.name)
        if fields is None:
            raise WorkspaceError(f"The {provider_label} connection is unavailable")
        if fields.get("status") == "needs_reconnect":
            raise WorkspaceError(f"Reconnect {provider_label} before searching its work")
        return ProjectConnection(
            provider=provider,
            name=summary.name,
            label=summary.user_label or summary.display_name or summary.name,
        ), fields

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

    def complete_source(self, project: CodeProject, request: SourceActionRequest) -> DeliveryRecord:
        if not request.confirmed:
            raise WorkspaceError("Confirm this external work-item update before publishing it")
        connection, fields = self._connection(project, request.provider, request.connection_name)
        if request.provider == "github":
            self._complete_github_source(request, fields)
        else:
            self._complete_linear_source(request, fields)
        return DeliveryRecord(
            provider=request.provider,
            action="complete_source",
            target_url=request.target_url,
            status="published",
            external_url=request.target_url,
            detail=f"Marked complete with {connection.label or connection.name}",
            connection_name=connection.name,
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
        return GitHubPullRequests(
            self._request,
            api=api,
            token=token,
            host=host,
            owner=owner,
            repository=repository,
            number=number,
            url=url,
        ).status()

    def pull_request_action(
        self,
        project: CodeProject,
        request: PullRequestActionRequest,
    ) -> PullRequestStatus:
        if not request.confirmed:
            raise WorkspaceError("Confirm this GitHub action before publishing it")
        _, fields = self._connection(project, "github", request.connection_name)
        api, token, host = self._github_credentials(fields)
        owner, repository, resource, number = self._github_target(request.target_url, host)
        if resource != "pulls":
            raise WorkspaceError("The linked GitHub delivery is not a pull request")
        pull_requests = GitHubPullRequests(
            self._request,
            api=api,
            token=token,
            host=host,
            owner=owner,
            repository=repository,
            number=number,
            url=request.target_url,
        )
        if request.action == "ready":
            return pull_requests.mark_ready()
        if request.action == "merge":
            return pull_requests.merge()
        return pull_requests.resolve_thread(request.thread_id or "")

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
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        response = self._request(
            "GET",
            f"{api}/repos/{owner}/{repository}/{resource}/{number}",
            headers=headers,
        )
        payload = response.json()
        comments_payload = self._request(
            "GET",
            f"{api}/repos/{owner}/{repository}/issues/{number}/comments",
            headers=headers,
            params={"per_page": 50},
        ).json()
        comments = [
            SourceComment(
                id=str(item.get("id") or ""),
                author=str((item.get("user") or {}).get("login") or ""),
                body=str(item.get("body") or "")[:20_000],
                url=str(item.get("html_url") or ""),
                created_at=str(item.get("created_at") or ""),
            )
            for item in (comments_payload if isinstance(comments_payload, list) else [])[:50]
        ]
        attachment_texts = [str(payload.get("body") or ""), *(comment.body for comment in comments)]
        return SourceContext(
            provider="github",
            kind="pull_request" if resource == "pulls" else "issue",
            url=str(payload.get("html_url") or request.url),
            title=str(payload.get("title") or ""),
            external_id=f"{owner}/{repository}#{number}",
            connection_name=connection.name,
            body=str(payload.get("body") or ""),
            state=str(payload.get("state") or ""),
            author=str((payload.get("user") or {}).get("login") or ""),
            comments=comments,
            attachments=self._markdown_attachments(attachment_texts),
        )

    def _read_linear(
        self,
        request: SourceContextRequest,
        connection: ProjectConnection,
        fields: dict[str, Any],
    ) -> SourceContext:
        identifier = self._linear_identifier(request.url)
        token = self._secret(fields, "access_token", "api_key", "token")
        issue = self._linear_issue(identifier, token)
        return SourceContext(
            provider="linear",
            kind="issue",
            url=str(issue.get("url") or request.url),
            title=str(issue.get("title") or ""),
            external_id=str(issue.get("identifier") or issue.get("id") or identifier),
            connection_name=connection.name,
            body=str(issue.get("description") or ""),
            state=str((issue.get("state") or {}).get("name") or ""),
            author=str((issue.get("creator") or {}).get("name") or ""),
            comments=[
                SourceComment(
                    id=str(item.get("id") or ""),
                    author=str((item.get("user") or {}).get("name") or ""),
                    body=str(item.get("body") or "")[:20_000],
                    url=str(item.get("url") or ""),
                    created_at=str(item.get("createdAt") or ""),
                )
                for item in ((issue.get("comments") or {}).get("nodes") or [])[:50]
                if isinstance(item, dict)
            ],
            attachments=[
                SourceAttachment(
                    id=str(item.get("id") or ""),
                    title=str(item.get("title") or "Attachment"),
                    url=str(item.get("url") or ""),
                )
                for item in ((issue.get("attachments") or {}).get("nodes") or [])[:50]
                if isinstance(item, dict) and item.get("url")
            ],
        )

    def _linear_issue(self, identifier: str, token: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "query": "query CodeIssue($id: String!) { issue(id: $id) { id identifier title description url state { name } creator { name } comments(first: 50) { nodes { id body createdAt user { name } } } attachments(first: 50) { nodes { id title url } } } }",
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
        identifier = self._linear_identifier(request.target_url)
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

    def _complete_github_source(self, request: SourceActionRequest, fields: dict[str, Any]) -> None:
        api, token, host = self._github_credentials(fields)
        owner, repository, resource, number = self._github_target(request.target_url, host)
        if resource != "issues":
            raise WorkspaceError("Complete the originating GitHub issue, not its pull request")
        self._request(
            "PATCH",
            f"{api}/repos/{owner}/{repository}/issues/{number}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"state": "closed"},
        )

    def _complete_linear_source(self, request: SourceActionRequest, fields: dict[str, Any]) -> None:
        identifier = self._linear_identifier(request.target_url)
        token = self._secret(fields, "access_token", "api_key", "token")
        issue = self._linear_issue(identifier, token)
        response = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "query": (
                    "query CodeCompletion($id: String!) { issue(id: $id) { id team { states(first: 50) { "
                    "nodes { id name type } } } } }"
                ),
                "variables": {"id": str(issue.get("id") or identifier)},
            },
        ).json()
        if response.get("errors"):
            raise WorkspaceError(str(response["errors"][0].get("message") or "Linear could not load completion states"))
        loaded = (response.get("data") or {}).get("issue") or {}
        states = (((loaded.get("team") or {}).get("states") or {}).get("nodes") or [])
        completed = next(
            (item for item in states if str((item or {}).get("type") or "").casefold() == "completed"),
            None,
        )
        if not completed:
            raise WorkspaceError("This Linear team does not have a completed workflow state")
        update = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={
                "query": "mutation CodeComplete($id: String!, $stateId: String!) { issueUpdate(id: $id, input: {stateId: $stateId}) { success } }",
                "variables": {"id": str(loaded.get("id") or issue.get("id")), "stateId": str(completed.get("id") or "")},
            },
        ).json()
        if update.get("errors") or not (((update.get("data") or {}).get("issueUpdate") or {}).get("success")):
            message = ((update.get("errors") or [{}])[0].get("message") or "Linear could not complete this issue")
            raise WorkspaceError(str(message))

    @staticmethod
    def _linear_identifier(url: str) -> str:
        path_parts = urlparse(url).path.rstrip("/").split("/")
        # Canonical Linear URLs append a human-readable title slug after the
        # stable identifier: /issue/ENG-289/fix-login.
        return next(
            (part for part in path_parts if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-\d+", part)),
            path_parts[-1],
        )

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
    def _markdown_attachments(texts: list[str]) -> list[SourceAttachment]:
        pattern = re.compile(r"(?:!\[[^\]]*\]\(|\[[^\]]+\]\()(?P<url>https?://[^)\s]+)\)")
        result: list[SourceAttachment] = []
        seen: set[str] = set()
        for text in texts:
            for match in pattern.finditer(text):
                url = match.group("url")
                path = urlparse(url).path
                if url in seen or not ("/user-attachments/" in path or "." in path.rsplit("/", 1)[-1]):
                    continue
                seen.add(url)
                title = path.rsplit("/", 1)[-1] or "Attachment"
                result.append(SourceAttachment(id=url, title=title[:512], url=url))
                if len(result) >= 100:
                    return result
        return result

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
