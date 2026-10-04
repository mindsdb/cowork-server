"""Wait until a PR environment serves this PR's build.

mindsdb/deployer rolls each labeled PR's image into its environment on a
schedule, so the integration job can start while the environment still serves
an older build, or nothing at all. The job runs this module first, and the
suite starts only once the environment answers with this PR's code.

Every request through a PR host's ingress is authenticated, the health route
included. So this mints a throwaway user the same way the suite does
(``TEST_USER_MINT_URL``, which needs no secret) and reads ``/api/v1/health/``
with that user's key.

A build names its commit in ``server_version``: hatch-vcs appends
``+g<abbreviated sha>`` to the version of any untagged commit, and a PR build is
always the merge commit GitHub made for its run. The environment serves this
PR's build when that commit is this run's merge commit (``GITHUB_SHA``), or an
earlier merge of the same head. The second case happens when only the base
branch moved since the environment last rolled: mindsdb/deployer's
``pr-envs.yml`` deploys the ``development-head-<head sha>`` tag, which names the
same head both times, so it has nothing new to roll.

    COWORK_BASE_URL=https://cowork-pr-cowork-server-123.dev.mindshub.ai \\
    TEST_USER_MINT_URL=https://auth-pr-cowork-server-123.dev.mindshub.ai/dev/mint-test-user/ \\
    GITHUB_SHA=<merge sha> PR_HEAD_SHA=<head sha> \\
    GITHUB_REPOSITORY=mindsdb/cowork-server GH_TOKEN=<token> \\
    uv run python -m tests.integration.wait_for_build
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from tests.integration.test_post_deploy import BROWSER_UA

# The deployer's schedule, then an Argo CD sync and a rollout, with room for a
# late scheduled run.
DEADLINE_S = 50 * 60.0
POLL_INTERVAL_S = 20.0
HEALTH_PATH = "/api/v1/health/"
GITHUB_API_URL = "https://api.github.com"
# hatch-vcs's default local version: `+g<abbreviated sha>`, followed by
# `.d<date>` when the build context differed from the commit.
_BUILD_COMMIT = re.compile(r"\+g([0-9a-f]{7,40})(?![0-9a-f])")


@dataclass(frozen=True)
class PullRequestBuild:
    """This run's PR build, and the environment that should serve it."""

    base_url: str
    mint_url: str
    merge_sha: str
    head_sha: str
    repository: str
    github_token: str

    @classmethod
    def from_environment(cls) -> PullRequestBuild:
        return cls(
            base_url=_required("COWORK_BASE_URL").rstrip("/"),
            mint_url=_required("TEST_USER_MINT_URL"),
            merge_sha=_required("GITHUB_SHA"),
            head_sha=_required("PR_HEAD_SHA"),
            repository=_required("GITHUB_REPOSITORY"),
            github_token=_required("GH_TOKEN"),
        )


class _NotServing(Exception):
    """The environment, or GitHub, could not answer yet."""


class _KeyRefused(_NotServing):
    """The ingress refused the minted user's key, so the next attempt mints again."""


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def build_commit(server_version: object) -> str | None:
    """The abbreviated commit hatch-vcs wrote into a build's version.

    None for a release build, whose version carries no local segment, and for
    anything that is not a version string at all.
    """
    if not isinstance(server_version, str):
        return None
    match = _BUILD_COMMIT.search(server_version)
    return match.group(1) if match else None


def _mint_key(build: PullRequestBuild, client: httpx.Client) -> str:
    """A fresh throwaway user's API key, minted the way the suite mints one."""
    response = client.post(
        build.mint_url, json={}, headers={"User-Agent": BROWSER_UA}
    )
    if response.status_code != 201:
        raise _NotServing(f"minting a test user answered HTTP {response.status_code}")
    payload = response.json()
    api_key = payload.get("api_key") if isinstance(payload, dict) else None
    if not api_key:
        raise _NotServing("the minted test user came back without an API key")
    return api_key


def _served_commit(
    build: PullRequestBuild, client: httpx.Client, api_key: str
) -> str | None:
    response = client.get(
        f"{build.base_url}{HEALTH_PATH}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code == 401:
        raise _KeyRefused(f"{HEALTH_PATH} refused the minted user's key")
    if response.status_code != 200:
        raise _NotServing(f"{HEALTH_PATH} answered HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    return build_commit(payload.get("server_version"))


def _merges_head(commit: str, build: PullRequestBuild, client: httpx.Client) -> bool:
    """Whether ``commit`` is a merge GitHub made of this PR's head.

    GitHub's test merge commit has the base as its first parent and the PR head
    as its second. A commit GitHub does not know is an answer, not a failure;
    any other error raises, so a lookup that could not run is never read as
    "a different build".
    """
    response = client.get(
        f"{GITHUB_API_URL}/repos/{build.repository}/commits/{commit}",
        headers={
            "Authorization": f"Bearer {build.github_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    if response.status_code in (404, 422):
        return False
    if response.status_code != 200:
        raise _NotServing(
            f"GitHub answered HTTP {response.status_code} for commit {commit}"
        )
    parents = [parent["sha"] for parent in response.json()["parents"]]
    return parents[1:] == [build.head_sha]


def _is_this_prs_build(
    commit: str, build: PullRequestBuild, client: httpx.Client
) -> bool:
    return build.merge_sha.startswith(commit) or _merges_head(commit, build, client)


def wait_for_build(
    build: PullRequestBuild,
    client: httpx.Client,
    *,
    deadline_s: float = DEADLINE_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Return the served commit once the environment serves this PR's build.

    Raises TimeoutError, naming the last thing the environment answered, when
    the deadline passes first.
    """
    give_up_at = clock() + deadline_s
    api_key: str | None = None
    verdicts: dict[str, bool] = {}
    last_seen = ""
    while True:
        try:
            if api_key is None:
                api_key = _mint_key(build, client)
            commit = _served_commit(build, client, api_key)
            if commit is None:
                seen = "it serves a build whose version names no commit"
            else:
                if commit not in verdicts:
                    verdicts[commit] = _is_this_prs_build(commit, build, client)
                if verdicts[commit]:
                    return commit
                seen = f"it serves commit {commit}, a build of a different head"
        except _KeyRefused as refused:
            api_key = None
            seen = str(refused)
        except (_NotServing, httpx.HTTPError, ValueError) as error:
            seen = str(error) or type(error).__name__
        if seen != last_seen:
            print(f"Waiting for {build.base_url}: {seen}.", flush=True)
            last_seen = seen
        if clock() >= give_up_at:
            raise TimeoutError(
                f"{build.base_url} did not serve a build of PR head {build.head_sha} "
                f"within {deadline_s / 60:.0f} minutes. Last answer: {last_seen}."
            )
        sleep(POLL_INTERVAL_S)


def main() -> int:
    build = PullRequestBuild.from_environment()
    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
        try:
            commit = wait_for_build(build, client)
        except TimeoutError as error:
            print(f"::error::{error}")
            return 1
    print(f"{build.base_url} serves commit {commit}, a build of PR head {build.head_sha}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
