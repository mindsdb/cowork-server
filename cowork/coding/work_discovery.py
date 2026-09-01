from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from cowork.coding.project_models import WorkItemPage, WorkItemSummary
from cowork.coding.workspace import WorkspaceError

Request = Callable[..., httpx.Response]


class DeveloperWorkDiscovery:
    """Provider-specific work search behind Code's normalized work-item contract."""

    def __init__(self, request: Request) -> None:
        self._request = request

    def github(
        self,
        *,
        api: str,
        token: str,
        query: str,
        limit: int,
        connection_name: str,
    ) -> WorkItemPage:
        search = f"{query.strip()} is:open" if query.strip() else "is:open assignee:@me"
        response = self._request(
            "GET",
            f"{api}/search/issues",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            params={"q": search, "sort": "updated", "order": "desc", "per_page": limit},
        ).json()
        items = response.get("items") if isinstance(response, dict) else []
        return WorkItemPage(
            items=[
                self._github_item(item, connection_name)
                for item in (items if isinstance(items, list) else [])[:limit]
                if isinstance(item, dict) and item.get("html_url")
            ],
            incomplete=bool(response.get("incomplete_results")) if isinstance(response, dict) else False,
        )

    def linear(
        self,
        *,
        token: str,
        query: str,
        limit: int,
        connection_name: str,
    ) -> WorkItemPage:
        term = query.strip()
        issue_fields = "id identifier title url updatedAt state { name } team { key name } assignee { name }"
        if term:
            graph_query = (
                "query CodeWorkSearch($first: Int!, $term: String!) { "
                f"searchIssues(first: $first, term: $term) {{ nodes {{ {issue_fields} }} }} }}"
            )
            variables: dict[str, Any] = {"first": limit, "term": term}
        else:
            graph_query = (
                "query CodeAssignedWork($first: Int!, $filter: IssueFilter) { viewer { "
                f"assignedIssues(first: $first, orderBy: updatedAt, filter: $filter) {{ nodes {{ {issue_fields} }} }} }} }}"
            )
            variables = {
                "first": limit,
                "filter": {"state": {"type": {"nin": ["completed", "canceled"]}}},
            }
        response = self._request(
            "POST",
            "https://api.linear.app/graphql",
            headers={"Authorization": token, "Content-Type": "application/json"},
            json={"query": graph_query, "variables": variables},
        ).json()
        if response.get("errors"):
            raise WorkspaceError(str(response["errors"][0].get("message") or "Linear could not search issues"))
        data = response.get("data") or {}
        collection = data.get("searchIssues") if term else (data.get("viewer") or {}).get("assignedIssues")
        items = (collection or {}).get("nodes") or []
        return WorkItemPage(
            items=[
                self._linear_item(item, connection_name)
                for item in items[:limit]
                if isinstance(item, dict) and item.get("url")
            ],
        )

    @staticmethod
    def _github_item(item: dict[str, Any], connection_name: str) -> WorkItemSummary:
        repository_api = str(item.get("repository_url") or "")
        repository_path = urlparse(repository_api).path.strip("/").split("/")
        scope = "/".join(repository_path[-2:]) if len(repository_path) >= 2 else ""
        number = str(item.get("number") or "")
        assignees = item.get("assignees") if isinstance(item.get("assignees"), list) else []
        return WorkItemSummary(
            provider="github",
            kind="pull_request" if item.get("pull_request") else "issue",
            url=str(item.get("html_url") or ""),
            title=str(item.get("title") or ""),
            external_id=f"{scope}#{number}" if scope else f"#{number}",
            state=str(item.get("state") or ""),
            scope=scope,
            assignee=", ".join(
                str((assignee or {}).get("login") or "")
                for assignee in assignees
                if isinstance(assignee, dict) and (assignee or {}).get("login")
            ),
            updated_at=str(item.get("updated_at") or ""),
            connection_name=connection_name,
        )

    @staticmethod
    def _linear_item(item: dict[str, Any], connection_name: str) -> WorkItemSummary:
        team = item.get("team") if isinstance(item.get("team"), dict) else {}
        assignee = item.get("assignee") if isinstance(item.get("assignee"), dict) else {}
        state = item.get("state") if isinstance(item.get("state"), dict) else {}
        return WorkItemSummary(
            provider="linear",
            kind="issue",
            url=str(item.get("url") or ""),
            title=str(item.get("title") or ""),
            external_id=str(item.get("identifier") or item.get("id") or ""),
            state=str(state.get("name") or ""),
            scope=str(team.get("name") or team.get("key") or ""),
            assignee=str(assignee.get("name") or ""),
            updated_at=str(item.get("updatedAt") or ""),
            connection_name=connection_name,
        )
