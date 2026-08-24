from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from cowork.coding.project_models import (
    PullRequestCheck,
    PullRequestFeedback,
    PullRequestStatus,
)
from cowork.coding.workspace import WorkspaceError

Request = Callable[..., httpx.Response]


class GitHubPullRequests:
    """A repository-scoped GitHub pull-request lifecycle client."""

    def __init__(
        self,
        request: Request,
        *,
        api: str,
        token: str,
        host: str,
        owner: str,
        repository: str,
        number: str,
        url: str,
    ) -> None:
        self._request = request
        self._api = api
        self._token = token
        self._host = host
        self._owner = owner
        self._repository = repository
        self._number = number
        self._url = url
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def status(self) -> PullRequestStatus:
        pull = self._request("GET", self._pull_url, headers=self._headers).json()
        head = str((pull.get("head") or {}).get("sha") or "")
        reviews, review_unavailable = self._optional_json(
            f"{self._pull_url}/reviews",
            params={"per_page": 100},
        )
        comments, comments_unavailable = self._optional_json(
            f"{self._pull_url}/comments",
            params={"per_page": 100},
        )
        checks: dict[str, Any] = {}
        combined: dict[str, Any] = {}
        unavailable = [
            label
            for label, missing in (
                ("review status", review_unavailable),
                ("review comments", comments_unavailable),
            )
            if missing
        ]
        if head:
            check_payload, missing = self._optional_json(
                f"{self._repo_url}/commits/{head}/check-runs",
                params={"per_page": 100},
            )
            checks = check_payload if isinstance(check_payload, dict) else {}
            if missing:
                unavailable.append("check runs")
            combined_payload, missing = self._optional_json(
                f"{self._repo_url}/commits/{head}/status",
            )
            combined = combined_payload if isinstance(combined_payload, dict) else {}
            if missing:
                unavailable.append("commit status")

        review_items = reviews if isinstance(reviews, list) else []
        comment_items = comments if isinstance(comments, list) else []
        return PullRequestStatus(
            state=self._state(pull),
            review_state=self._review_state(review_items),
            ci_state=self._ci_state(checks, combined),
            number=int(self._number),
            title=str(pull.get("title") or ""),
            url=str(pull.get("html_url") or self._url),
            updated_at=str(pull.get("updated_at") or datetime.now(UTC).isoformat()),
            checks=self._checks(checks, combined),
            feedback=self._feedback(review_items, comment_items),
            detail=f"Could not load {', '.join(unavailable)}" if unavailable else "",
        )

    def mark_ready(self) -> PullRequestStatus:
        pull = self._request("GET", self._pull_url, headers=self._headers).json()
        node_id = str(pull.get("node_id") or "")
        if not node_id:
            raise WorkspaceError("GitHub did not return the pull request identity")
        graph_url = (
            "https://api.github.com/graphql"
            if self._host == "github.com"
            else f"https://{self._host}/api/graphql"
        )
        payload = self._request(
            "POST",
            graph_url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json={
                "query": (
                    "mutation CodeReady($id: ID!) { "
                    "markPullRequestReadyForReview(input: {pullRequestId: $id}) "
                    "{ pullRequest { id } } }"
                ),
                "variables": {"id": node_id},
            },
        ).json()
        if payload.get("errors"):
            message = payload["errors"][0].get("message") or "GitHub could not mark this pull request ready"
            raise WorkspaceError(str(message))
        return self.status()

    def merge(self) -> PullRequestStatus:
        payload = self._request(
            "PUT",
            f"{self._pull_url}/merge",
            headers=self._headers,
            json={},
        ).json()
        if not payload.get("merged"):
            raise WorkspaceError(str(payload.get("message") or "GitHub could not merge this pull request"))
        return self.status()

    @property
    def _repo_url(self) -> str:
        return f"{self._api}/repos/{self._owner}/{self._repository}"

    @property
    def _pull_url(self) -> str:
        return f"{self._repo_url}/pulls/{self._number}"

    def _optional_json(self, url: str, **kwargs: Any) -> tuple[Any, bool]:
        try:
            return self._request("GET", url, headers=self._headers, **kwargs).json(), False
        except WorkspaceError:
            return {}, True

    @staticmethod
    def _state(payload: dict[str, Any]) -> str:
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
        statuses = combined.get("statuses") if isinstance(combined, dict) else []
        has_status = isinstance(statuses, list) and bool(statuses)
        combined_state = str(combined.get("state") or "").lower() if has_status else ""
        conclusions = {str(item.get("conclusion") or "").lower() for item in runs if isinstance(item, dict)}
        run_states = {str(item.get("status") or "").lower() for item in runs if isinstance(item, dict)}
        if combined_state in {"failure", "error"} or conclusions & {
            "failure", "timed_out", "cancelled", "action_required", "stale",
        }:
            return "failing"
        if combined_state == "pending" or run_states & {"queued", "in_progress", "pending"}:
            return "pending"
        if runs or combined_state:
            return "passing"
        return "none"

    @staticmethod
    def _checks(checks: dict[str, Any], combined: dict[str, Any]) -> list[PullRequestCheck]:
        result: list[PullRequestCheck] = []
        for item in checks.get("check_runs", []) if isinstance(checks, dict) else []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").lower()
            conclusion = str(item.get("conclusion") or "").lower()
            state = "pending" if status != "completed" else (
                "passing" if conclusion in {"success", "skipped"} else
                "neutral" if conclusion == "neutral" else "failing"
            )
            result.append(PullRequestCheck(
                name=str(item.get("name") or "GitHub check"),
                state=state,
                url=str(item.get("details_url") or item.get("html_url") or ""),
            ))
        for item in combined.get("statuses", []) if isinstance(combined, dict) else []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("state") or "").lower()
            state = "passing" if status == "success" else "pending" if status in {"pending", "queued"} else "failing"
            result.append(PullRequestCheck(
                name=str(item.get("context") or "Commit status"),
                state=state,
                url=str(item.get("target_url") or ""),
            ))
        return result[:200]

    @staticmethod
    def _feedback(
        reviews: list[dict[str, Any]],
        comments: list[dict[str, Any]],
    ) -> list[PullRequestFeedback]:
        result = [
            PullRequestFeedback(
                id=str(item.get("id") or ""),
                author=str((item.get("user") or {}).get("login") or ""),
                state=str(item.get("state") or "commented").lower(),
                body=str(item.get("body") or "")[:20_000],
                url=str(item.get("html_url") or ""),
                created_at=str(item.get("submitted_at") or ""),
            )
            for item in reviews
            if isinstance(item, dict) and (
                item.get("body") or str(item.get("state") or "").upper() == "CHANGES_REQUESTED"
            )
        ]
        result.extend(
            PullRequestFeedback(
                id=str(item.get("id") or ""),
                author=str((item.get("user") or {}).get("login") or ""),
                body=str(item.get("body") or "")[:20_000],
                url=str(item.get("html_url") or ""),
                path=str(item.get("path") or ""),
                created_at=str(item.get("created_at") or ""),
            )
            for item in comments
            if isinstance(item, dict) and item.get("body")
        )
        return result[:200]
