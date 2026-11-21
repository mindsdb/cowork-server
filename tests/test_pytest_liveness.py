"""Exercise opt-in liveness evidence through real isolated pytest executions."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NAME = "test_liveness_answers_at_once_while_a_hostile_export_converts"
_BODY = """
import gc, time, pytest
from conftest import ORIGINAL
def {name}():
    assert (gc.isenabled(), gc.get_threshold()) == ORIGINAL
    private_value = 'PRIVATE_SENTINEL'
    started, elapsed, free_rate = time.monotonic(), 0.5, 10.0
    latencies, ticks = [0.01, 0.02], [3]
    gc.collect()
    {action}
"""


@pytest.fixture
def observation_suite(tmp_path):
    """Build a miniature regular scripts package and run copied plugins in fresh processes."""
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (root / "pytest.ini").write_text("[pytest]\n")
    for name in ("__init__.py", "pytest_inventory.py", "pytest_liveness.py"):
        shutil.copyfile(_ROOT / "scripts" / name, root / "scripts" / name)
    marker = tmp_path / "cleanup"
    (root / "conftest.py").write_text(
        "import gc\nfrom pathlib import Path\n"
        "def existing(phase, info):\n    pass\n"
        "gc.callbacks.append(existing)\n"
        "CALLBACKS = tuple(gc.callbacks)\nORIGINAL = (gc.isenabled(), gc.get_threshold())\n"
        "def pytest_sessionfinish(session):\n"
        "    assert tuple(gc.callbacks) == CALLBACKS\n"
        f"    Path({str(marker)!r}).write_text('clean')\n"
    )
    sequence = 0

    def run(action="pass", *, reverse=False, enabled=True, guard=None, source=None):
        """Return actual exit/evidence and an external callback-cleanup witness."""
        nonlocal sequence
        sequence += 1
        (root / "tests/test_export_child.py").write_text(source or _BODY.format(name=_NAME, action=action))
        report, inventory = tmp_path / f"observation-{sequence}.json", tmp_path / f"inventory-{sequence}.json"
        if guard == "inside":
            report = root / "observations.json"
        if guard == "existing":
            report.write_text("keep")
        if guard == "publication":
            parent = tmp_path / "not-a-directory"
            parent.write_text("keep")
            report = parent / "observations.json"
        plugins = ["scripts.pytest_inventory", "scripts.pytest_liveness"]
        if reverse:
            plugins.reverse()
        if guard == "inventory":
            plugins.remove("scripts.pytest_inventory")
        command = [sys.executable, "-m", "pytest", "-p", "xdist.plugin"]
        for plugin in plugins:
            command += ["-p", plugin]
        command += ["--rootdir", str(root), "-c", str(root / "pytest.ini"), "-q", "--tb=short"]
        if guard != "inventory":
            command += ["--test-cohort=probe", "--test-inventory=" + str(inventory)]
        if enabled:
            command += ["--test-liveness-observations=" + str(report)]
        if guard == "parallel":
            command += ["-n2"]
        env = {key: os.environ[key] for key in ("PATH", "HOME", "USER", "LOGNAME", "SYSTEMROOT", "TMPDIR") if key in os.environ}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
        document = json.loads(report.read_text()) if report.is_file() and guard != "existing" else None
        recorded = json.loads(inventory.read_text()) if inventory.exists() else None
        return result, document, recorded, marker.exists(), report

    return run


@pytest.mark.parametrize("reverse", [False, True], ids=["normal", "reversed"])
@pytest.mark.parametrize("failure", [False, True], ids=["passed", "failed"])
def test_observations_preserve_outcome_and_private_locals(observation_suite, reverse, failure):
    """Real GC events are captured without exposing private locals or changing outcomes."""
    result, data, inventory, clean, _ = observation_suite(
        "assert False, private_value" if failure else "pass", reverse=reverse,
    )
    assert result.returncode == int(failure), result.stdout + result.stderr
    assert clean and data["call_entered"] and data["target_selected"]
    assert data["source"] == data["source_end"] == inventory["revision"]
    assert data["phases"]["call"] == ("failed" if failure else "passed")
    assert data["gc_events"] and {event[1] for event in data["gc_events"]} == {"start", "stop"}
    assert data["state"]["gc_enabled"] == data["state"]["gc_enabled_end"]
    assert data["state"]["gc_threshold"] == data["state"]["gc_threshold_end"]
    assert data["call_start_monotonic"] <= data["call_end_monotonic"]
    assert "PRIVATE_SENTINEL" not in json.dumps(data)
    timings = data["failure_timings"]
    if failure:
        assert data["call_start_monotonic"] <= timings.pop("started") <= data["call_end_monotonic"]
        assert timings == {"elapsed": 0.5, "free_rate": 10.0, "latencies": [0.01, 0.02],
                           "latencies_truncated": False, "ticks": 3, "omitted_fields": []}
    else:
        assert timings == {}


@pytest.mark.parametrize("action,code", [("assert False", 1), ("pytest.skip('skip')", 0), ("raise KeyboardInterrupt", 2)], ids=["failure", "skip", "interrupt"])
def test_observer_releases_only_its_callback_on_every_exit(observation_suite, action, code):
    """The session-finish witness sees all original callbacks after each call exit."""
    result, data, _, clean, _ = observation_suite(action)
    assert result.returncode == code, result.stdout + result.stderr
    assert clean and data["call_entered"] and data["call_end_monotonic"] is not None


@pytest.mark.parametrize("kind", ["setup-skip", "absent"])
def test_unexecuted_target_never_claims_observations(observation_suite, kind):
    """Selection and call execution remain distinct for skipped and absent targets."""
    source = "import pytest\n@pytest.mark.skip(reason='skip')\ndef " + _NAME + "():\n    pass\n" if kind == "setup-skip" else "def test_other():\n    pass\n"
    result, data, _, clean, _ = observation_suite(source=source)
    assert result.returncode == 0 and clean
    assert data["target_selected"] == (kind == "setup-skip")
    assert not data["call_entered"] and data["gc_events"] == [] and data["failure_timings"] == {}


def test_loading_without_option_installs_no_observer(observation_suite):
    """Ordinary suites retain their callback list and produce no observation file."""
    result, data, _, clean, _ = observation_suite("from conftest import CALLBACKS; assert tuple(gc.callbacks) == CALLBACKS", enabled=False)
    assert result.returncode == 0 and clean and data is None


@pytest.mark.parametrize("guard", ["inside", "existing", "parallel", "inventory"])
def test_invalid_observer_configuration_stops_before_tests(observation_suite, guard):
    """Unsafe destinations and incompatible execution cannot start the observer."""
    result, data, _, clean, path = observation_suite(guard=guard)
    assert result.returncode == 4 and "Liveness observations require" in result.stderr
    assert data is None and not clean
    if guard == "existing":
        assert path.read_text() == "keep"


def test_publication_failure_cannot_leave_a_green_inventory(observation_suite):
    """An output failure is reflected before the source-aware inventory finalizes."""
    result, data, inventory, clean, _ = observation_suite(guard="publication")
    assert result.returncode == 3 and data is None and clean
    assert inventory["completion"]["exitstatus"] == 3 and not inventory["completion"]["complete"]
