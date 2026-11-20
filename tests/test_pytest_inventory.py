"""Exercise inventory hooks through isolated, real pytest subprocesses."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from coverage import CoverageData
from coverage.exceptions import CoverageException

from scripts.blocking_guard_evidence import sidecar
from scripts.pytest_inventory import source_identity
from scripts.test_inventory import Report, compare, preserve, read_report, summary, validate, validate_junit
from scripts.test_shards import aggregate, latest_artifact, partition

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
            "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "COVERAGE_FILE", "COWORK_REQUIRE_SERVICES", "COWORK_REQUIRE_INTEGRATION", "GITHUB_STEP_SUMMARY",
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


def test_serial_parallel_reports_reconcile_without_timing_equality(inventory_suite):
    """The same cases and outcomes compare across workers while timings stay independent."""
    _, run = inventory_suite
    result, _, serial_path = run("-n", "0", source="def test_one():\n    pass\ndef test_two():\n    pass\n")
    assert result.returncode == 0
    result, _, parallel_path = run("-n", "2")
    assert result.returncode == 0
    serial, parallel = read_report(serial_path), read_report(parallel_path)
    compare(serial, parallel)
    assert "2 collected" in summary(parallel)
    assert "Cumulative phase work" in summary(parallel)


@pytest.mark.parametrize("fault", ["missing_case", "extra_case", "duplicate_case", "missing_phase", "duplicate_phase",
                                  "missing_worker", "marker", "source", "lock", "coverage", "skip", "exit"])
def test_report_comparison_refuses_corrupt_or_incompatible_evidence(inventory_suite, fault):
    """Independent case, phase, source and selection discrepancies cannot compare green."""
    _, run = inventory_suite
    result, raw, path = run("-n", "2", source="def test_one():\n    pass\ndef test_two():\n    pass\n")
    assert result.returncode == 0
    changed = deepcopy(raw)
    if fault == "missing_case":
        changed["collection"]["items"].pop()
    elif fault == "extra_case":
        changed["collection"]["items"].append({"nodeid": "test_sample.py::test_missing", "module": "test_sample.py", "markers": []})
    elif fault == "duplicate_case":
        changed["collection"]["items"].append(changed["collection"]["items"][0])
    elif fault == "missing_phase":
        changed["phases"].pop()
    elif fault == "duplicate_phase":
        changed["phases"].append(changed["phases"][0])
    elif fault == "missing_worker":
        changed["workers"].pop()
    elif fault == "marker":
        changed["collection"]["items"][0]["markers"] = ["feature"]
    elif fault == "source":
        changed["revision"]["tree_digest"] = changed["revision_end"]["tree_digest"] = "f" * 64
    elif fault == "lock":
        changed["revision"]["config"]["uv.lock"] = "f" * 64
        changed["revision_end"]["config"]["uv.lock"] = "f" * 64
    elif fault == "coverage":
        changed["invocation"]["coverage"]["branch"] = True
    elif fault == "skip":
        call = next(phase for phase in changed["phases"] if phase["phase"] == "call")
        call.update(outcome="skipped", reason="new skip")
    elif fault == "exit":
        changed["completion"]["exitstatus"] = 1
    with pytest.raises(ValueError):
        compare(read_report(path), Report.model_validate(changed))


def test_body_only_edits_fail_execution_comparison(inventory_suite):
    """Matching node IDs and commit IDs do not permit different uncommitted source bodies."""
    _, run = inventory_suite
    result, _, before_path = run(source="def test_same_id():\n    assert 1\n")
    assert result.returncode == 0
    result, _, after_path = run(source="def test_same_id():\n    assert 2\n")
    assert result.returncode == 0
    before, after = read_report(before_path), read_report(after_path)
    assert before.collection == after.collection
    with pytest.raises(ValueError, match="revision"):
        compare(before, after)


@pytest.mark.parametrize("fault", [None, "removed", "marker", "unapproved", "collector"])
def test_preservation_allows_only_declared_additions(inventory_suite, fault):
    """The separate collection audit permits new source but preserves old case ownership."""
    _, run = inventory_suite
    result, _, base_path = run("--collect-only", source="def test_old():\n    pass\n")
    assert result.returncode == 0
    source = "def test_old():\n    pass\ndef test_added():\n    pass\n"
    if fault == "removed":
        source = "def test_added():\n    pass\n"
    elif fault == "marker":
        source = "import pytest\n@pytest.mark.feature\n" + source
    result, raw, candidate_path = run("--collect-only", source=source)
    assert result.returncode == 0
    base, candidate = read_report(base_path), read_report(candidate_path)
    # Synthetic repositories have no commit; exercise declared revision matching explicitly.
    base.revision.sha = base.revision_end.sha = "a" * 40
    candidate.revision.sha = candidate.revision_end.sha = "b" * 40
    additions = ["test_sample.py::test_added"] if fault != "unapproved" else []
    if fault == "collector":
        candidate.reporter_digest = "c" * 64
    if fault is None:
        preserve(base, candidate, "a" * 40, "b" * 40, additions)
    else:
        with pytest.raises(ValueError):
            preserve(base, candidate, "a" * 40, "b" * 40, additions)


def test_junit_reconciles_call_and_teardown_failure_and_module_skip(inventory_suite):
    """Pytest's extra XML row for a double failure is not mistaken for another case."""
    root, run = inventory_suite
    (root / "test_skip.py").write_text('import pytest\npytest.skip("module prerequisite", allow_module_level=True)\n')
    xml = root.parent / "result.xml"
    result, _, path = run(f"--junitxml={xml}", source='''
import pytest
@pytest.fixture
def resource():
    yield
    assert False
def test_double_failure(resource):
    assert False
''')
    assert result.returncode == 1
    validate_junit(read_report(path), xml)
    xml.write_text("<testsuite/>")
    with pytest.raises(ValueError, match="count"):
        validate_junit(read_report(path), xml)


def test_invalid_report_values_are_rejected_without_echoing_payload(tmp_path):
    """Malformed artifact content is diagnosed without copying it to CLI output."""
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": 1, "private": "hidden-payload"}')
    result = subprocess.run([sys.executable, str(_ROOT / "scripts/test_inventory.py"), "check", str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "hidden-payload" not in result.stderr


def test_validator_accepts_setup_skips_and_expected_failures(inventory_suite):
    """Valid missing call phases and xfail classifications remain valid evidence."""
    _, run = inventory_suite
    result, _, path = run(source='''
import pytest
@pytest.mark.skip(reason="platform prerequisite")
def test_skip():
    pass
@pytest.mark.xfail(reason="expected fault")
def test_xfail():
    assert False
''')
    assert result.returncode == 0
    report = read_report(path)
    assert len(validate(report)) == 2
    assert "'skipped': 1" in summary(report)
    assert "'xfailed': 1" in summary(report)


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


@pytest.mark.parametrize("change", ["disabled", "append", "external-config"])
def test_coverage_comparison_checks_effective_measurement(inventory_suite, change):
    """Different instrumentation/configuration cannot be accepted as a matched timing run."""
    root, run = inventory_suite
    coverage_config = root.parent / "coverage.ini"
    coverage_config.write_text("[run]\nbranch = false\n")
    args = ["-p", "pytest_cov.plugin", "--cov=.", "--cov-report=", f"--cov-config={coverage_config}"]
    result, _, before_path = run(*args, source="def test_passes():\n    assert True\n")
    assert result.returncode == 0, result.stdout + result.stderr
    if change == "disabled":
        args.append("--no-cov")
    elif change == "append":
        args.append("--cov-append")
    else:
        coverage_config.write_text("[run]\nbranch = true\n")
    result, _, after_path = run(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    before, after = read_report(before_path), read_report(after_path)
    assert before.revision == after.revision
    assert before.invocation.coverage.enabled
    if change == "disabled":
        assert not after.invocation.coverage.enabled
    with pytest.raises(ValueError, match="coverage settings"):
        compare(before, after)


@pytest.mark.parametrize("workers", [0, 2])
def test_complete_evidence_requires_collection_timings(inventory_suite, workers):
    """Required serial and worker timing fields cannot silently disappear from a report."""
    _, run = inventory_suite
    result, raw, _ = run("-n", str(workers), source="def test_passes():\n    pass\n")
    assert result.returncode == 0
    if workers:
        raw["workers"][0].pop("collection_seconds")
    else:
        raw["collection_seconds"] = None
    with pytest.raises(ValueError, match="collection"):
        validate(Report.model_validate(raw))


def test_nested_coverage_keeps_parent_data(inventory_suite, monkeypatch):
    """A nested measured pytest run preserves the parent runner's coverage bytes."""
    root, run = inventory_suite
    source = root / "sentinel.py"
    source.write_text("value = 1\n")
    parent = root.parent / "parent-coverage.data"
    data = CoverageData(basename=str(parent))
    data.add_lines({str(source): {1}})
    data.write()
    before = parent.read_bytes()
    monkeypatch.setenv("COVERAGE_FILE", str(parent))
    result, _, _ = run("-p", "pytest_cov.plugin", "--cov=.", "--cov-report=",
                       source="def test_child():\n    assert True\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert parent.read_bytes() == before


@pytest.fixture
def shard_evidence(inventory_suite, monkeypatch):
    """Produce a full collection and four real shard reports in an isolated checkout."""
    return _shard_evidence(inventory_suite, monkeypatch)


def _shard_evidence(inventory_suite, monkeypatch, *, prepare=None):
    """Build the shared shard probe, optionally installing a real runtime guard before collection."""
    root, run = inventory_suite
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("test_shards.py", "test_inventory.py", "pytest_inventory.py", "blocking_guard_evidence.py"):
        shutil.copyfile(_ROOT / "scripts" / name, scripts / name)
    package = root / "cowork"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "sample.py").write_text(
        "def choose(value):\n    if value == 2:\n        return 2\n    if value == 3:\n        return 3\n"
        "    if value:\n        return 1\n    return 0\n"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_left é.py").write_text('''
import os
import pytest
from cowork.sample import choose
pytestmark = pytest.mark.feature
@pytest.mark.parametrize("value", [True, True], ids=['quoted["one"]', 'line\\nnext'])
def test_left(value):
    assert not os.environ.get("PROBE_FAIL")
    assert choose(value) == 1
''')
    (tests / "test_right.py").write_text('''
import pytest
from cowork.sample import choose
def test_right():
    assert choose(False) == 0
def test_runtime_skip():
    pytest.skip("runtime prerequisite")
''')
    (tests / "test_skip.py").write_text('import pytest\npytest.skip("module prerequisite", allow_module_level=True)\n')
    for name, value in (("third", 2), ("fourth", 3)):
        (tests / f"test_{name}.py").write_text(f'''
import os
from cowork.sample import choose
def test_{name}():
    assert not os.environ.get("PROBE_FAIL")
    assert choose({value}) == {value}
''')
    (tests / "integration").mkdir()
    (tests / "integration/test_ignored.py").write_text('raise RuntimeError("integration must stay excluded")\n')
    if prepare is not None:
        prepare(root)
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_JOB", "collect-tests")
    result, collection, path = run("tests/", "--ignore=tests/integration", "-n", "0", "--collect-only",
                                   "--test-cohort=broad")
    assert result.returncode == 0, result.stdout + result.stderr
    inputs = root.parent / "inputs"
    collected = inputs / "test-inventory-broad-collection-1234-1"
    collected.mkdir(parents=True)
    shutil.copyfile(path, collected / "inventory.json")
    if (root / "tests/_blocking_guard.py").is_file():
        shutil.copyfile(sidecar(path), sidecar(collected / "inventory.json"))

    def execute(index: int, attempt: int = 1, *, fail: bool = False):
        """Run the actual shard CLI with explicit producer context and optional test failure."""
        environment = {k: v for k, v in os.environ.items() if k not in {
            "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "COVERAGE_FILE", "GITHUB_STEP_SUMMARY",
        }}
        environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_PLUGINS="xdist.plugin,pytest_cov.plugin",
                           GITHUB_RUN_ATTEMPT=str(attempt), GITHUB_JOB="unit-test-shards")
        if fail:
            environment["PROBE_FAIL"] = "1"
        output = inputs / f"test-inventory-broad-shard-{index}-1234-{attempt}"
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.test_shards", "run", "--inputs", str(inputs),
             "--output", str(output), "--index", str(index)],
            cwd=root, env=environment, capture_output=True, text=True, timeout=120,
        )
        return completed, output

    for index in range(4):
        result, _ = execute(index)
        assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.setenv("GITHUB_JOB", "unit-tests")
    monkeypatch.chdir(root)
    return root, run, inputs, Report.model_validate(collection), execute


def test_shards_preserve_cases_skips_and_covered_lines(inventory_suite, monkeypatch):
    """Four real executions preserve the full selection and equal unsharded line coverage."""
    def observe_directories(root):
        """Record real collector visits outside the temporary source, separately per worker."""
        (root / "tests/conftest.py").write_text('''
from collections import Counter
import json
import pytest

VISITS = Counter()

def pytest_collectstart(collector):
    """Count visits to real directories without changing collection."""
    if isinstance(collector, pytest.Dir):
        VISITS[collector.path.relative_to(collector.config.rootpath).as_posix()] += 1

def pytest_sessionfinish(session):
    """Save this worker's counts outside the fingerprinted source tree."""
    config = session.config
    worker = getattr(config, "workerinput", {}).get("workerid", "serial")
    path = config.rootpath.parent / f"directories-{config.option.test_cohort}-{worker}.json"
    path.write_text(json.dumps(VISITS))
''')

    root, run, inputs, collection, _ = _shard_evidence(inventory_suite, monkeypatch, prepare=observe_directories)
    result, control, _ = run("tests/", "--ignore=tests/integration", "-n", "0", "-p", "pytest_cov.plugin",
                             "--cov=cowork", "--cov-report=")
    assert result.returncode == 0, result.stdout + result.stderr
    output = root.parent / "combined"
    combined = aggregate(inputs, output)
    assert combined["collected"] == len(collection.collection.items) == 6
    assert combined["collector_skips"] == 1
    actual = [read_report(output / f"shard-{i}/inventory.json") for i in range(4)]
    assert [shard["index"] for shard in combined["shards"]] == [0, 1, 2, 3]
    assert sorted(case.nodeid for report in actual for case in report.collection.items) == sorted(
        case["nodeid"] for case in control["collection"]["items"])
    skips = [phase.reason for report in actual for phase in report.phases if phase.outcome == "skipped"]
    assert skips == ["runtime prerequisite"]
    data = [CoverageData(basename=str(path)) for path in (root / ".coverage", output / "coverage.data")]
    for measurement in data:
        measurement.read()
    assert {name: set(data[0].lines(name) or []) for name in data[0].measured_files()} == {
        name: set(data[1].lines(name) or []) for name in data[1].measured_files()}
    for index in range(4):
        for worker in range(4):
            counts = json.loads((root.parent / f"directories-broad-shard-{index}-gw{worker}.json").read_text())
            assert counts["tests"] == 1
            assert all(visits == 1 for visits in counts.values())


def test_shards_accept_flat_single_collection_download(shard_evidence, monkeypatch):
    """A real singleton download runs and joins without losing producer identity checks."""
    root, _, inputs, _, execute = shard_evidence
    named = inputs / "test-inventory-broad-collection-1234-1"
    flat = inputs / "inventory.json"
    (named / "inventory.json").rename(flat)
    named.rmdir()
    original = flat.read_bytes()
    completed, _ = execute(0, 2)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    combined = aggregate(inputs, root.parent / "flat-combined")
    assert combined["collection_attempt"] == 1
    assert [shard["producer_attempt"] for shard in combined["shards"]] == [2, 1, 1, 1]
    assert flat.read_bytes() == original
    for key, value, message in (
        ("GITHUB_RUN_ID", "9999", "producer identity"),
        ("GITHUB_RUN_ATTEMPT", "3", "producer attempt"),
        ("GITHUB_RUN_ATTEMPT", "01", "producer attempt"),
        ("GITHUB_RUN_ATTEMPT", "", "producer attempt"),
    ):
        changed = json.loads(original)
        changed["environment"]["ci"][key] = value
        try:
            flat.write_text(json.dumps(changed))
            with pytest.raises(ValueError, match=message):
                latest_artifact(inputs, "test-inventory-broad-collection")
        finally:
            flat.write_bytes(original)
    named.mkdir()
    shutil.copyfile(flat, named / "inventory.json")
    with pytest.raises(ValueError, match="Duplicate producer attempt"):
        latest_artifact(inputs, "test-inventory-broad-collection")


def test_shard_aggregate_accepts_hosted_runner_differences(shard_evidence, monkeypatch, capsys):
    """Host metadata stays in the reports without blocking complete evidence."""
    root, _, inputs, _, _ = shard_evidence
    collection_path = inputs / "test-inventory-broad-collection-1234-1/inventory.json"
    shard_path = inputs / "test-inventory-broad-shard-0-1234-1/inventory.json"
    collection = json.loads(collection_path.read_bytes())
    collection["environment"]["ci"].update(ImageOS="ubuntu26", ImageVersion="collection-image", RUNNER_ARCH="X64")
    collection_path.write_text(json.dumps(collection))
    monkeypatch.setenv("ImageOS", "ubuntu26")
    monkeypatch.setenv("ImageVersion", "collection-image")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    for index in range(4):
        path = inputs / f"test-inventory-broad-shard-{index}-1234-1/inventory.json"
        raw = json.loads(path.read_bytes())
        raw["environment"]["ci"].update(ImageOS="ubuntu26", ImageVersion="collection-image", RUNNER_ARCH="X64")
        path.write_text(json.dumps(raw))

    shard = json.loads(shard_path.read_bytes())
    original_shard = shard_path.read_bytes()
    shard["environment"].update(release="different-kernel", logical_cpus=None)
    if collection["environment"]["logical_cpus"] is None:
        shard["environment"]["logical_cpus"] = 4
    shard["environment"]["ci"].update(ImageOS="different-image-os", ImageVersion="shard-image")
    shard_path.write_text(json.dumps(shard))
    output = root.parent / "mixed-shard"
    assert aggregate(inputs, output)["collected"] == 6
    assert read_report(output / "shard-0/inventory.json").environment.model_dump() == shard["environment"]
    diagnostic = capsys.readouterr().err
    assert "broad-shard-0: hosted runner metadata differs from collection (ImageOS, ImageVersion, logical_cpus, release)" in diagnostic
    assert "different-kernel" not in diagnostic and "shard-image" not in diagnostic
    assert "aggregate:" not in diagnostic

    shard_path.write_bytes(original_shard)
    monkeypatch.setenv("ImageVersion", "consumer-image")
    assert aggregate(inputs, root.parent / "mixed-consumer")["collected"] == 6
    diagnostic = capsys.readouterr().err
    assert "aggregate: hosted runner metadata differs from collection (ImageVersion)" in diagnostic
    assert "broad-shard-0:" not in diagnostic

    monkeypatch.delenv("ImageVersion")
    shard["environment"]["ci"].pop("ImageVersion")
    shard_path.write_text(json.dumps(shard))
    assert aggregate(inputs, root.parent / "missing-image")["collected"] == 6
    diagnostic = capsys.readouterr().err
    assert "aggregate: hosted runner metadata differs from collection (ImageVersion)" in diagnostic
    assert "broad-shard-0: hosted runner metadata differs from collection (ImageOS, ImageVersion, logical_cpus, release)" in diagnostic

    monkeypatch.setenv("RUNNER_ARCH", "ARM64")
    with pytest.raises(ValueError, match="Consumer runner/run metadata differs"):
        aggregate(inputs, root.parent / "wrong-consumer-arch")


def test_shard_aggregate_rejects_damaged_evidence(shard_evidence):
    """Missing inputs and altered source, runtime, coverage or membership cannot pass the join."""
    root, _, inputs, _, _ = shard_evidence
    shard = inputs / "test-inventory-broad-shard-0-1234-1"
    report_path = shard / "inventory.json"
    original = report_path.read_bytes()
    changes = (
        ("source", "Shard source"), ("runtime", "runtime"), ("package", "runtime"),
        ("runner-arch", "Runner/run metadata"), ("marker", "cases/markers"),
        ("selection", "selection"), ("coverage", "coverage"), ("worker", "worker"),
        ("run", "producer identity"),
        ("missing-integration", "selection"), ("missing-module", "selection"),
        ("extra-module", "selection"), ("duplicate-ignore", "selection"),
        ("reordered-ignore", "selection"), ("escaped-ignore", "selection"),
        ("directory-ignore", "selection"), ("narrowed-paths", "selection"),
    )
    for fault, message in changes:
        raw = json.loads(original)
        if fault == "source":
            raw["revision"]["tree_digest"] = raw["revision_end"]["tree_digest"] = "f" * 64
        elif fault == "runtime":
            raw["environment"]["python"] = "0.0.0"
        elif fault == "package":
            raw["environment"]["packages"]["pytest"] = "0.0.0"
        elif fault == "runner-arch":
            raw["environment"]["ci"]["RUNNER_ARCH"] = "different-arch"
        elif fault == "marker":
            raw["collection"]["items"][0]["markers"] = ["altered"]
            digest = hashlib.sha256(json.dumps(raw["collection"], sort_keys=True, ensure_ascii=True).encode()).hexdigest()
            for worker in raw["workers"]:
                worker["digest"] = digest
        elif fault == "selection":
            raw["invocation"]["selection"]["markers"] = "feature"
        elif fault == "coverage":
            raw["invocation"]["coverage"]["disabled"] = True
        elif fault == "worker":
            raw["workers"].pop()
        elif fault == "missing-integration":
            raw["invocation"]["selection"]["ignore"].remove("tests/integration")
        elif fault == "missing-module":
            raw["invocation"]["selection"]["ignore"].pop()
        elif fault == "extra-module":
            raw["invocation"]["selection"]["ignore"].append("tests/test_left é.py")
        elif fault == "duplicate-ignore":
            raw["invocation"]["selection"]["ignore"].append("tests/integration")
        elif fault == "reordered-ignore":
            raw["invocation"]["selection"]["ignore"].reverse()
        elif fault == "escaped-ignore":
            raw["invocation"]["selection"]["ignore"][-1] = "../outside.py"
        elif fault == "directory-ignore":
            raw["invocation"]["selection"]["ignore"][-1] = "tests"
        elif fault == "narrowed-paths":
            raw["invocation"]["selection"]["paths"] = ["tests/test_left é.py"]
        else:
            raw["environment"]["ci"]["GITHUB_RUN_ID"] = "9999"
        try:
            report_path.write_text(json.dumps(raw))
            with pytest.raises(ValueError, match=message):
                aggregate(inputs, root.parent / "must-not-pass")
        finally:
            report_path.write_bytes(original)
    coverage = shard / "coverage.data"
    original_data = coverage.read_bytes()
    for damaged in (b"", b"not a coverage database"):
        try:
            coverage.write_bytes(damaged)
            with pytest.raises((ValueError, CoverageException)):
                aggregate(inputs, root.parent / "must-not-pass")
        finally:
            coverage.write_bytes(original_data)
    for index in range(4):
        producer = inputs / f"test-inventory-broad-shard-{index}-1234-1"
        hidden = producer.with_name("withheld-shard")
        producer.rename(hidden)
        try:
            with pytest.raises(ValueError, match="Missing required artifact"):
                aggregate(inputs, root.parent / "must-not-pass")
        finally:
            hidden.rename(producer)
    for index in (2, 3):
        path = inputs / f"test-inventory-broad-shard-{index}-1234-1/inventory.json"
        original_report = path.read_bytes()
        altered = json.loads(original_report)
        altered["invocation"]["coverage"]["context"] = "different-context"
        try:
            path.write_text(json.dumps(altered))
            with pytest.raises(ValueError, match="Coverage configurations differ"):
                aggregate(inputs, root.parent / "must-not-pass")
        finally:
            path.write_bytes(original_report)
    fourth_coverage = inputs / "test-inventory-broad-shard-3-1234-1/coverage.data"
    original_fourth = fourth_coverage.read_bytes()
    try:
        fourth_coverage.unlink()
        arcs = CoverageData(basename=str(fourth_coverage))
        arcs.add_arcs({str(root / "cowork/sample.py"): {(1, 2)}})
        arcs.write()
        with pytest.raises(ValueError, match="Coverage line/arc modes differ"):
            aggregate(inputs, root.parent / "must-not-pass")
    finally:
        fourth_coverage.write_bytes(original_fourth)


def test_shards_retain_partial_rerun_producers(shard_evidence, monkeypatch):
    """Numerically latest successful retries combine with carried collection and sibling data."""
    root, _, inputs, _, execute = shard_evidence
    for attempt in (9, 10):
        completed, _ = execute(0, attempt)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "10")
    combined = aggregate(inputs, root.parent / "partial-rerun")
    assert combined["collection_attempt"] == 1
    assert combined["consumer_attempt"] == 10
    assert [shard["producer_attempt"] for shard in combined["shards"]] == [10, 1, 1, 1]
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "9")
    with pytest.raises(ValueError, match="future producer"):
        latest_artifact(inputs, "test-inventory-broad-shard-0")


def test_shard_failure_never_falls_back_to_old_success(shard_evidence, monkeypatch):
    """A real failed retry preserves its nonzero status and shadows the older passing input."""
    root, _, inputs, _, execute = shard_evidence
    for index in (0, 3):
        completed, output = execute(index, 2, fail=True)
        assert completed.returncode == 1
        failed = read_report(output / "inventory.json")
        assert failed.completion.exitstatus == 1
        assert any(phase.outcome == "failed" for phase in failed.phases)
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        try:
            with pytest.raises(ValueError, match="Newest producer did not succeed"):
                aggregate(inputs, root.parent / "must-not-pass")
        finally:
            output.rename(output.with_name(f"withheld-failed-{index}"))


def test_shards_reject_unsafe_modules_and_indices(shard_evidence):
    """Malformed collection paths or indices cannot expand what the runner executes."""
    root, _, inputs, collection, execute = shard_evidence
    outside = root.parent / "outside.py"
    outside.write_text("raise RuntimeError('must not execute')\n")
    (root / "tests/link.py").symlink_to(outside)
    for name in ("../outside.py", str(outside), "tests/../outside.py", "--collect-only", "tests/link.py",
                 "tests/missing.py", "tests/test_right.py::test_right", ""):
        changed = collection.model_copy(deep=True)
        changed.collection.items[0].module = name
        changed.collection.items[0].nodeid = name + "::test_bad"
        with pytest.raises(ValueError, match="module" if name else "metadata"):
            partition(changed)
    for index in (-1, 4):
        completed, output = execute(index)
        assert completed.returncode != 0
        assert not output.exists()
