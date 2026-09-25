from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from cowork.coding.workspace import WorkspaceError


class GitHubRepository(BaseModel):
    full_name: str
    clone_url: str
    private: bool
    default_branch: str | None = None
    archived: bool = False
    connection_name: str


class GitHubRepositoryPage(BaseModel):
    items: list[GitHubRepository] = Field(default_factory=list)
    next_page: int | None = None


def github_repositories(
    request: Callable[..., httpx.Response], *, api: str, token: str, host: str, connection_name: str, page: int,
) -> GitHubRepositoryPage:
    """List only the connected user's repositories, including organisation memberships.

    Pagination stays on our fixed endpoint; never follow a provider-supplied
    next URL with credentials. Search filters these pages in the desktop.
    """
    response = request(
        "GET", f"{api}/user/repos",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        params={"affiliation": "owner,collaborator,organization_member", "sort": "updated", "per_page": 100, "page": page},
    )
    payload = response.json()
    if not isinstance(payload, list):
        raise WorkspaceError("GitHub could not list repositories. Try again.")
    items = []
    port = urlsplit(api).port
    origin = f"https://{host}" + (f":{port}" if port else "")
    for item in payload:
        if not isinstance(item, dict):
            continue
        full_name = item.get("full_name")
        if not isinstance(full_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9_.-]+", full_name):
            continue
        if full_name.split("/")[1] in {".", ".."}:
            continue
        # Derive clone URLs from the validated host and identity, not arbitrary
        # URLs returned in a repository object. No secrets cross this boundary.
        branch = item.get("default_branch")
        items.append(GitHubRepository(
            full_name=full_name,
            clone_url=f"{origin}/{full_name}.git",
            private=item.get("private") is True,
            default_branch=branch if isinstance(branch, str) and branch else None,
            archived=item.get("archived") is True,
            connection_name=connection_name,
        ))
    return GitHubRepositoryPage(items=items, next_page=page + 1 if "next" in response.links else None)
