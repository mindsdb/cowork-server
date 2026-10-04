"""The PR-environment wait starts the suite only on this PR's build."""

from __future__ import annotations

import httpx
import pytest

from tests.integration import wait_for_build
from tests.integration.test_post_deploy import BROWSER_UA
from tests.integration.wait_for_build import PullRequestBuild, build_commit

MERGE_SHA = "4e8e4d581be742fec5f78ac3cfd81e7161a7b539"
HEAD_SHA = "ac65f317b5613015c92111a83ec026c8f0eaaf6d"
BASE_SHA = "0db386902de05451f6e2a4b288d0aba7179db2f5"
EARLIER_MERGE_SHA = "9c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d"
OTHER_HEAD_SHA = "5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c"

BUILD = PullRequestBuild(
    base_url="https://cowork-pr-cowork-server-7.dev.mindshub.ai",
    mint_url="https://auth-pr-cowork-server-7.dev.mindshub.ai/dev/mint-test-user/",
    merge_sha=MERGE_SHA,
    head_sha=HEAD_SHA,
    repository="mindsdb/cowork-server",
    github_token="ghs_example",
)


def _version(sha: str) -> str:
    return f"0.26.10.3.1.dev5+g{sha[:8]}.d20261003"


def _health(sha: str) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "server_version": _version(sha)})


def _minted(api_key: str) -> httpx.Response:
    return httpx.Response(201, json={"api_key": api_key, "user_id": "u", "organization_id": "o"})


def _commit(*parents: str) -> httpx.Response:
    return httpx.Response(200, json={"parents": [{"sha": sha} for sha in parents]})


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Hosts:
    """auth's mint route, cowork's health route and GitHub's commits API, scripted."""

    def __init__(
        self,
        *,
        mints: list[httpx.Response],
        healths: list[httpx.Response | Exception] | None = None,
        commits: dict[str, list[httpx.Response]] | None = None,
    ) -> None:
        self.mints = list(mints)
        self.healths = list(healths or [])
        self.commits = commits or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "api.github.com":
            return self.commits[request.url.path.rsplit("/", 1)[1]].pop(0)
        if request.url.host.startswith("auth-"):
            return self.mints.pop(0) if len(self.mints) > 1 else self.mints[0]
        answer = self.healths.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def to(self, host: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.host == host]


def _wait(hosts: _Hosts, *, deadline_s: float = 600.0) -> str:
    clock = _Clock()
    with httpx.Client(transport=httpx.MockTransport(hosts)) as client:
        return wait_for_build.wait_for_build(
            BUILD, client, deadline_s=deadline_s, sleep=clock.sleep, clock=clock
        )


@pytest.mark.parametrize(
    ("server_version", "commit"),
    [
        ("0.26.10.3.1.dev5+g4e8e4d58", "4e8e4d58"),
        ("0.26.10.3.1.dev5+g4e8e4d58.d20261003", "4e8e4d58"),
        ("0.26.10.3.1.dev5+g4e8e4d581be742fec5f78ac3cfd81e7161a7b539", MERGE_SHA),
        ("0.26.10.3.1", None),
        ("0.26.10.3.1rc2", None),
        ("0.26.10.3.1.dev5+g4e8e4d", None),
        ("0.26.10.3.1.dev5+gnothex1", None),
        (None, None),
        (42, None),
    ],
)
def test_build_commit_reads_the_hatch_vcs_local_segment(server_version, commit):
    assert build_commit(server_version) == commit


def test_waits_through_another_heads_build_until_this_merge_commit_serves():
    hosts = _Hosts(
        mints=[_minted("mdb_minted")],
        healths=[_health(EARLIER_MERGE_SHA), _health(MERGE_SHA)],
        commits={EARLIER_MERGE_SHA[:8]: [_commit(BASE_SHA, OTHER_HEAD_SHA)]},
    )

    assert _wait(hosts) == MERGE_SHA[:8]

    (mint,) = hosts.to("auth-pr-cowork-server-7.dev.mindshub.ai")
    assert mint.method == "POST"
    assert mint.headers["User-Agent"] == BROWSER_UA
    healths = hosts.to("cowork-pr-cowork-server-7.dev.mindshub.ai")
    assert [request.url.path for request in healths] == ["/api/v1/health/"] * 2
    assert {request.headers["Authorization"] for request in healths} == {
        "Bearer mdb_minted"
    }
    # This run's own merge commit needs no lookup; the other build needed one.
    (lookup,) = hosts.to("api.github.com")
    assert lookup.url.path == f"/repos/mindsdb/cowork-server/commits/{EARLIER_MERGE_SHA[:8]}"
    assert lookup.headers["Authorization"] == "Bearer ghs_example"


def test_accepts_an_earlier_merge_of_the_same_head():
    """Only the base moved, so the deployer kept the image it rolled for this head."""
    hosts = _Hosts(
        mints=[_minted("mdb_minted")],
        healths=[_health(EARLIER_MERGE_SHA)],
        commits={EARLIER_MERGE_SHA[:8]: [_commit(BASE_SHA, HEAD_SHA)]},
    )

    assert _wait(hosts) == EARLIER_MERGE_SHA[:8]


def test_a_failed_github_lookup_is_asked_again_rather_than_read_as_another_build():
    hosts = _Hosts(
        mints=[_minted("mdb_minted")],
        healths=[_health(EARLIER_MERGE_SHA), _health(EARLIER_MERGE_SHA)],
        commits={
            EARLIER_MERGE_SHA[:8]: [
                httpx.Response(503),
                _commit(BASE_SHA, HEAD_SHA),
            ]
        },
    )

    assert _wait(hosts) == EARLIER_MERGE_SHA[:8]
    assert len(hosts.to("api.github.com")) == 2


def test_mints_a_new_user_after_the_key_is_refused():
    hosts = _Hosts(
        mints=[_minted("mdb_first"), _minted("mdb_second")],
        healths=[httpx.Response(401), _health(MERGE_SHA)],
    )

    assert _wait(hosts) == MERGE_SHA[:8]

    assert len(hosts.to("auth-pr-cowork-server-7.dev.mindshub.ai")) == 2
    assert [
        request.headers["Authorization"]
        for request in hosts.to("cowork-pr-cowork-server-7.dev.mindshub.ai")
    ] == ["Bearer mdb_first", "Bearer mdb_second"]


def test_keeps_waiting_while_the_environment_is_not_up():
    hosts = _Hosts(
        mints=[httpx.Response(503), _minted("mdb_minted")],
        healths=[
            httpx.ConnectError("name does not resolve"),
            httpx.Response(502),
            _health(MERGE_SHA),
        ],
    )

    assert _wait(hosts) == MERGE_SHA[:8]


def test_gives_up_at_the_deadline_naming_the_last_answer():
    hosts = _Hosts(mints=[httpx.Response(403)])

    with pytest.raises(TimeoutError) as timeout:
        _wait(hosts, deadline_s=120.0)

    assert str(timeout.value) == (
        f"https://cowork-pr-cowork-server-7.dev.mindshub.ai did not serve a build "
        f"of PR head {HEAD_SHA} within 2 minutes. Last answer: minting a test "
        f"user answered HTTP 403."
    )
    # One attempt every 20 seconds from 0 to 120, then it stops.
    assert len(hosts.requests) == 7


def test_reads_its_inputs_from_the_job_environment(monkeypatch):
    for name, value in {
        "COWORK_BASE_URL": "https://cowork-pr-cowork-server-7.dev.mindshub.ai/",
        "TEST_USER_MINT_URL": BUILD.mint_url,
        "GITHUB_SHA": MERGE_SHA,
        "PR_HEAD_SHA": HEAD_SHA,
        "GITHUB_REPOSITORY": "mindsdb/cowork-server",
        "GH_TOKEN": "ghs_example",
    }.items():
        monkeypatch.setenv(name, value)

    assert PullRequestBuild.from_environment() == BUILD

    monkeypatch.delenv("PR_HEAD_SHA")
    with pytest.raises(SystemExit, match="PR_HEAD_SHA is not set"):
        PullRequestBuild.from_environment()
