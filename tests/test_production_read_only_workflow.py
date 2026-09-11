"""Contracts for the nightly production read-only smoke."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from tests.integration import test_production_read_only as production_smoke


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github/workflows/nightly-production-read-only.yml"
).read_text()
MAKEFILE = (ROOT / "Makefile").read_text()
SMOKE_SOURCE = (
    ROOT / "tests/integration/test_production_read_only.py"
).read_text()
README = (ROOT / "README.md").read_text()
EXPECTED_SMOKE_ENV = {
    "COWORK_BASE_URL": "https://cowork.mindshub.ai",
    "COWORK_REQUIRE_INTEGRATION": "true",
    "COWORK_TEST_IDENTITY_MODE": "standing",
    "COWORK_TEST_API_KEY": "${{ secrets.COWORK_PROD_READ_ONLY_API_KEY }}",
    "COWORK_TEST_USER_EMAIL": "${{ vars.COWORK_PROD_READ_ONLY_USER_EMAIL }}",
    "COWORK_TEST_ORG_ID": "${{ vars.COWORK_PROD_READ_ONLY_ORG_ID }}",
}


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


def test_workflow_is_nightly_manual_and_prod_scoped() -> None:
    workflow = yaml.safe_load(WORKFLOW)
    triggers = workflow.get("on", workflow.get(True))
    assert triggers == {
        "schedule": [{"cron": "43 7 * * *"}],
        "workflow_dispatch": None,
    }
    assert workflow["concurrency"] == {
        "group": "nightly-production-read-only",
        "cancel-in-progress": False,
    }
    assert set(workflow["jobs"]) == {"production-read-only", "notify"}
    production_job = workflow["jobs"]["production-read-only"]
    assert set(production_job) == {
        "if",
        "runs-on",
        "timeout-minutes",
        "environment",
        "steps",
    }
    assert production_job["runs-on"] == "mdb-prod"
    assert production_job["timeout-minutes"] == 10
    assert production_job["if"] == "github.ref == 'refs/heads/main'"
    assert production_job["environment"] == {"name": "prod-read-only"}


def test_workflow_uses_only_the_standing_identity_inputs() -> None:
    workflow = yaml.safe_load(WORKFLOW)
    production_job = workflow["jobs"]["production-read-only"]
    smoke_step = next(
        step
        for step in production_job["steps"]
        if step.get("name") == "Run production read-only smoke"
    )
    assert smoke_step["env"] == EXPECTED_SMOKE_ENV
    assert "env" not in production_job
    assert WORKFLOW.count("secrets.COWORK_PROD_READ_ONLY_API_KEY") == 1
    assert WORKFLOW.count("vars.COWORK_PROD_READ_ONLY_USER_EMAIL") == 1
    assert WORKFLOW.count("vars.COWORK_PROD_READ_ONLY_ORG_ID") == 1
    assert "secrets.COWORK_TEST_API_KEY" not in WORKFLOW
    assert "vars.COWORK_TEST_USER_EMAIL" not in WORKFLOW
    assert "vars.COWORK_TEST_ORG_ID" not in WORKFLOW
    assert "TEST_USER_PROVISION" not in WORKFLOW
    assert "TEST_USER_MINT" not in WORKFLOW
    assert "GH_PRIVATE_ACCESS_TOKEN" not in WORKFLOW


def test_workflow_runs_only_the_read_only_selection_and_notifies() -> None:
    workflow = yaml.safe_load(WORKFLOW)
    production_steps = workflow["jobs"]["production-read-only"]["steps"]
    smoke_step = next(
        step
        for step in production_steps
        if step.get("name") == "Run production read-only smoke"
    )
    assert smoke_step["run"] == "make test/integration-production-read-only"
    assert production_steps == [
        {
            "uses": "actions/checkout@v5",
            "with": {"persist-credentials": False},
        },
        {
            "name": "Setup uv",
            "uses": "astral-sh/setup-uv@v7",
            "with": {
                "version": "0.12.2",
                "cache-local-path": "/home/runner/_work/_tool/uv-local-cache",
                "prune-cache": False,
                "python-version": "3.12",
            },
        },
        {
            "name": "Install dependencies",
            "run": "uv sync --locked --group dev",
        },
        {
            "name": "Run production read-only smoke",
            "env": EXPECTED_SMOKE_ENV,
            "run": "make test/integration-production-read-only",
        },
    ]
    assert (
        "uses: mindsdb/github-actions/.github/workflows/"
        "notify-main-failure.yml@main"
    ) in WORKFLOW
    assert "actions: read" in WORKFLOW
    assert "secrets: inherit" in WORKFLOW
    assert "test/integration-production-read-only:" in MAKEFILE
    assert '-m "not production_read_only"' in MAKEFILE
    production_target = MAKEFILE.split(
        "test/integration-production-read-only:", 1
    )[1].split("\n\n", 1)[0]
    assert production_target == (
        " ## Run only the production GET-only smoke\n"
        "\t$(PYTEST) -v tests/integration/test_production_read_only.py"
    )
    dry_run = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-n",
            "test/integration-production-read-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert dry_run.stdout == (
        "uv run python -m pytest -v "
        "tests/integration/test_production_read_only.py\n"
    )


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def test_smoke_source_contains_only_the_reviewed_read_path() -> None:
    tree = ast.parse(SMOKE_SOURCE)
    call_names = Counter(
        _qualified_name(node.func).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )
    assert call_names == Counter(
        {
            "readonlyendpoint": 5,
            "isinstance": 2,
            "payload.get": 2,
            "pytest.fail": 2,
            "_assert_read_response": 1,
            "_headers": 1,
            "_production_identity": 1,
            "_verified_prod_standing_identity": 1,
            "dataclass": 1,
            "httpx.client": 1,
            "missing_prerequisite": 1,
            "os.environ.get": 1,
            "production_api.get": 1,
            "pytest.fixture": 1,
            "pytest.mark.parametrize": 1,
            "response.json": 1,
            "type": 1,
        }
    )
    assert SMOKE_SOURCE.count("production_api.get(endpoint.path)") == 1

    selected_tests = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    assert selected_tests == ["test_production_read_path"]

    post_deploy_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "tests.integration.test_post_deploy"
        for alias in node.names
    }
    assert post_deploy_imports == {
        "PROD_COWORK_BASE_URL",
        "_headers",
        "_verified_prod_standing_identity",
    }


def test_smoke_collects_only_the_five_reviewed_get_cases() -> None:
    collection = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/integration/test_production_read_only.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    node_ids = [
        line for line in collection.stdout.splitlines() if line.startswith("tests/")
    ]
    assert node_ids == [
        "tests/integration/test_production_read_only.py::"
        "test_production_read_path[service health]",
        "tests/integration/test_production_read_only.py::"
        "test_production_read_path[conversation listing]",
        "tests/integration/test_production_read_only.py::"
        "test_production_read_path[schedule listing]",
        "tests/integration/test_production_read_only.py::"
        "test_production_read_path[file listing]",
        "tests/integration/test_production_read_only.py::"
        "test_production_read_path[pin listing]",
    ]


def test_read_only_endpoints_are_an_explicit_reviewable_set() -> None:
    assert [(item.name, item.path) for item in production_smoke.READ_ONLY_ENDPOINTS] == [
        ("service health", "/api/v1/health/"),
        ("conversation listing", "/api/v1/conversations/?project=all&limit=1"),
        ("schedule listing", "/api/v1/schedules/"),
        ("file listing", "/api/v1/files/"),
        ("pin listing", "/api/v1/pins/"),
    ]


def test_smoke_requires_standing_mode_before_identity_network(monkeypatch) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.delenv("COWORK_TEST_IDENTITY_MODE", raising=False)
    monkeypatch.setattr(
        production_smoke,
        "_verified_prod_standing_identity",
        lambda: pytest.fail("identity lookup ran before mode validation"),
    )

    with pytest.raises(pytest.fail.Exception, match="requires.*standing"):
        production_smoke._production_identity()


def test_response_contract_accepts_expected_collection() -> None:
    endpoint = production_smoke.ReadOnlyEndpoint(
        name="example",
        path="/example",
        collection_key="items",
    )
    production_smoke._assert_read_response(_Response(payload={"items": []}), endpoint)


@pytest.mark.parametrize(
    "response",
    [
        _Response(status_code=403, payload={"items": []}, text="forbidden"),
        _Response(status_code=200, payload=[]),
        _Response(status_code=200, payload={"items": {}}),
    ],
)
def test_response_contract_rejects_failed_or_wrong_shaped_reads(response) -> None:
    endpoint = production_smoke.ReadOnlyEndpoint(
        name="example",
        path="/example",
        collection_key="items",
    )
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        production_smoke._assert_read_response(response, endpoint)


def test_readme_records_the_no_write_boundary() -> None:
    normalized = " ".join(README.split())
    for phrase in (
        "Nightly production read-only smoke",
        "GET-only",
        "never provisions an identity",
        "does not create conversations, schedules, files, artifacts, or model turns",
        "43 7 * * *",
        "cannot reference the existing `prod` Environment",
        "dedicated `prod-read-only` Environment",
        "no required-reviewer or wait-timer rule",
        "only entry is the exact `main` branch",
        "COWORK_PROD_READ_ONLY_API_KEY",
        "COWORK_PROD_READ_ONLY_USER_EMAIL",
        "COWORK_PROD_READ_ONLY_ORG_ID",
        "cowork-server#472 prerequisite has landed",
    ):
        assert phrase in normalized
