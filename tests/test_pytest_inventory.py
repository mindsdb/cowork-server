"""Exercise inventory hooks through isolated, real pytest subprocesses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pytest_inventory import source_identity

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inventory_suite(tmp_path):
    """Create an isolated Git source tree and external reporter for real runner probes."""
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.coverage*\n")
    (root / "pytest.ini").write_text("[pytest]\nmarkers =\n    feature: selected feature\n    variant: input variant\n")
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    shutil.copyfile(_ROOT / "scripts/pytest_inventory.py", plugins / "pytest_inventory.py")
    sequence = 0

    def run(*args: str, source: str | None = None, conftest: str | None = None):
        """Run the temporary suite, returning its process result and optional JSON evidence."""
        nonlocal sequence
        sequence += 1
        if source is not None:
            (root / "test_sample.py").write_text(source)
        if conftest is not None:
            (root / "conftest.py").write_text(conftest)
        report = tmp_path / f"result {sequence}.json"
        env = {key: value for key, value in os.environ.items() if key not in {
            "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "COWORK_REQUIRE_SERVICES", "COWORK_REQUIRE_INTEGRATION", "GITHUB_STEP_SUMMARY",
        }}
        env.update(PYTHONPATH=str(plugins), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "xdist.plugin", "-p", "pytest_inventory",
             "--test-cohort=probe", f"--test-inventory={report}", "-c", str(root / "pytest.ini"),
             "--rootdir", str(root), *args], cwd=root, env=env, capture_output=True, text=True, timeout=120,
        )
        document = json.loads(report.read_text()) if report.exists() else None
        return result, document, report

    return root, run


@pytest.mark.parametrize("workers", [0, 2])
def test_inventory_records_cases_markers_and_phase_costs(inventory_suite, workers):
    """Serial and distributed runs preserve each selected parameter and all three phases."""
    _, run = inventory_suite
    result, report, _ = run("-n", str(workers), source='''
import pytest
pytestmark = pytest.mark.feature
@pytest.mark.parametrize("value", [pytest.param(1, marks=pytest.mark.variant, id="one[quoted]"), 2])
def test_value(value):
    assert value > 0
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["completion"]["complete"]
    items = report["collection"]["items"]
    assert [item["nodeid"] for item in items] == ["test_sample.py::test_value[one[quoted]]", "test_sample.py::test_value[2]"]
    assert items[0]["markers"] == ["feature", "parametrize", "variant"]
    assert items[1]["markers"] == ["feature", "parametrize"]
    assert len(report["phases"]) == 6
    assert all(event["duration"] >= 0 and event["outcome"] == "passed" for event in report["phases"])
    assert len(report["workers"]) == workers
    assert all(worker["completed"] for worker in report["workers"])


def test_inventory_distinguishes_collection_and_runtime_skips(inventory_suite):
    """A skipped module never becomes a passed or runtime-skipped test case."""
    root, run = inventory_suite
    (root / "test_unavailable.py").write_text('import pytest\npytest.skip("platform prerequisite", allow_module_level=True)\n')
    result, report, _ = run("-n", "2", source='import pytest\ndef test_skip():\n    pytest.skip("runtime prerequisite")\n')
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["collection"]["reports"] == [{"nodeid": "test_unavailable.py", "outcome": "skipped", "reason": "platform prerequisite"}]
    assert len(report["collection"]["items"]) == 1
    calls = [phase for phase in report["phases"] if phase["phase"] == "call"]
    assert calls[0]["reason"] == "runtime prerequisite"


def test_collection_only_records_deselection_without_execution(inventory_suite):
    """Collection and marker deselection produce a manifest with no invented executions."""
    _, run = inventory_suite
    result, report, _ = run("--collect-only", "-k", "keep", source="def test_keep():\n    pass\ndef test_drop():\n    pass\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["mode"] == "collection"
    assert report["collection"]["deselected"] == ["test_sample.py::test_drop"]
    assert report["phases"] == []


@pytest.mark.parametrize("stage", ["setup", "call", "teardown"])
def test_inventory_preserves_failures_in_each_phase(inventory_suite, stage):
    """A failure remains visible in its actual phase and in the final exit status."""
    _, run = inventory_suite
    source = f'''
import pytest
@pytest.fixture
def resource():
    assert {stage != "setup"!r}
    yield
    assert {stage != "teardown"!r}
def test_sample(resource):
    assert {stage != "call"!r}
'''
    result, report, _ = run(source=source)
    assert result.returncode == report["completion"]["exitstatus"] == 1
    assert [event["phase"] for event in report["phases"] if event["outcome"] == "failed"] == [stage]
    assert report["completion"]["complete"]


def test_inventory_retains_xfail_and_xpass_classifications(inventory_suite):
    """Expected failures and unexpected passes remain distinct from skips and passes."""
    _, run = inventory_suite
    result, report, _ = run(source='''
import pytest
@pytest.mark.xfail(reason="known fault")
def test_expected():
    assert False
@pytest.mark.xfail(reason="known fault")
def test_unexpected():
    assert True
''')
    assert result.returncode == 0
    calls = [event for event in report["phases"] if event["phase"] == "call"]
    assert [(event["outcome"], event["xfail"]) for event in calls] == [("skipped", True), ("passed", True)]


@pytest.mark.parametrize("args,source", [
    (["-x"], "def test_a():\n    assert False\ndef test_b():\n    pass\n"),
    (["-n", "2", "--max-worker-restart=0"], "def test_crash():\n    import os\n    os._exit(9)\n"),
    ([], "raise RuntimeError('collection fault')\n"),
], ids=["early-stop", "worker-loss", "collection-error"])
def test_incomplete_runs_never_claim_complete_evidence(inventory_suite, args, source):
    """Early stop, worker loss and collection errors produce non-successful incomplete evidence."""
    _, run = inventory_suite
    result, report, _ = run(*args, source=source)
    assert result.returncode != 0
    assert report is not None, result.stdout + result.stderr
    assert not report["completion"]["complete"]
    assert report["completion"]["problems"]


def test_inventory_observes_status_changed_by_a_session_finish_guard(inventory_suite):
    """Controller finalization happens after ordinary hooks change the exit status."""
    _, run = inventory_suite
    result, report, _ = run(
        source="def test_passes():\n    pass\n",
        conftest="def pytest_sessionfinish(session):\n    session.exitstatus = 1\n",
    )
    assert result.returncode == report["completion"]["exitstatus"] == 1
    assert all(event["outcome"] == "passed" for event in report["phases"])


def test_source_changes_during_execution_invalidate_the_report(inventory_suite):
    """A body edit that leaves IDs unchanged cannot masquerade as a fixed-source run."""
    _, run = inventory_suite
    result, report, _ = run(source='''
from pathlib import Path
def test_changes_source():
    Path(__file__).write_text("def test_changes_source():\\n    assert True\\n")
''')
    assert result.returncode != 0
    assert not report["completion"]["complete"]
    assert report["revision"]["tree_digest"] != report["revision_end"]["tree_digest"]


def test_source_identity_covers_bodies_untracked_files_and_symlinks(inventory_suite):
    """Uncommitted source changes affect identity even without a changed commit or node ID."""
    root, _ = inventory_suite
    source = root / "test_sample.py"
    source.write_text("def test_sample():\n    assert 1\n")
    before = source_identity(root)
    source.write_text("def test_sample():\n    assert 2\n")
    changed = source_identity(root)
    assert before["sha"] == changed["sha"]
    assert before["tree_digest"] != changed["tree_digest"]
    (root / "new.py").write_text("value = 3\n")
    added = source_identity(root)
    assert added["tree_digest"] != changed["tree_digest"]
    (root / "link").symlink_to("new.py")
    linked = source_identity(root)
    (root / "link").unlink()
    (root / "link").symlink_to("test_sample.py")
    assert source_identity(root)["tree_digest"] != linked["tree_digest"]


def test_inventory_refuses_to_replace_an_existing_report(inventory_suite):
    """A conflicting output keeps the original evidence intact and fails visibly."""
    _, run = inventory_suite
    result, report, path = run(source="def test_ok():\n    pass\n")
    assert result.returncode == 0
    before = path.read_bytes()
    result, _, _ = run(f"--test-inventory={path}")
    assert result.returncode != 0
    assert path.read_bytes() == before


def test_reporter_does_not_copy_captured_output_or_marker_values(inventory_suite):
    """Metadata excludes test output and parameter values not already present in node IDs."""
    _, run = inventory_suite
    result, report, path = run(source='''
import pytest
@pytest.mark.variant("private-marker-payload")
def test_output():
    print("private-captured-payload")
''')
    assert result.returncode == 0
    assert report["collection"]["items"][0]["markers"] == ["variant"]
    assert "private-marker-payload" not in path.read_text()
    assert "private-captured-payload" not in path.read_text()


@pytest.mark.parametrize("failure", ["pytest.exit('finish failure', returncode=1)", "raise RuntimeError('finish failure')"],
                         ids=["pytest-exit", "exception"])
def test_finish_hook_exceptions_never_publish_successful_evidence(inventory_suite, failure):
    """An exceptional finish cannot be certified before pytest determines its exit code."""
    _, run = inventory_suite
    result, report, _ = run(source="def test_passes():\n    pass\n",
                            conftest=f"import pytest\ndef pytest_sessionfinish(session):\n    {failure}\n")
    assert result.returncode != 0
    assert report is None
    assert "Test inventory omitted: a session-finish hook failed" in result.stdout


def test_worker_finish_failure_leaves_incomplete_controller_evidence(inventory_suite):
    """A worker with a failed finish hook cannot send a successful inventory handoff."""
    _, run = inventory_suite
    result, report, _ = run("-n", "2", source="def test_passes():\n    pass\n", conftest='''
import pytest
def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        pytest.exit("worker finish failure", returncode=1)
''')
    assert result.returncode != 0
    assert report is not None, result.stdout + result.stderr
    assert not report["completion"]["complete"]
    assert all(not worker["completed"] for worker in report["workers"])


def test_junit_write_failure_cannot_leave_a_successful_inventory(inventory_suite):
    """A failure in the standard XML writer prevents publication of successful JSON."""
    root, run = inventory_suite
    invalid_target = root.parent / "xml-directory"
    result, report, _ = run(f"--junitxml={invalid_target}", source=f'''
from pathlib import Path
def test_passes():
    Path({str(invalid_target)!r}).mkdir()
''')
    assert invalid_target.is_dir()
    assert result.returncode != 0
    assert report is None


def test_late_output_collision_fails_without_replacing_evidence(inventory_suite):
    """Publishing after a competing writer fails without overwriting that writer's file."""
    _, run = inventory_suite
    result, report, path = run(source="def test_passes():\n    pass\n", conftest='''
from pathlib import Path
def pytest_sessionfinish(session):
    Path(session.config.option.test_inventory).write_text("{}")
''')
    assert result.returncode == 3
    assert report == {}
    assert path.read_text() == "{}"
