"""Opt-in, bounded observations of the original export liveness check."""
from __future__ import annotations

import gc
import hashlib
import math
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.pytest_inventory import Inventory, _publish, source_identity

_TARGET = "tests/test_export_child.py::test_liveness_answers_at_once_while_a_hostile_export_converts"
_LIMIT = 256
_MODULES = (
    "test_artifact_export_route", "test_artifact_ownership", "test_artifact_export_service",
    "tests.support.artifacts.files", "tests.support.artifacts.fixtures",
)


def _number(value: object) -> bool:
    """Accept bounded finite builtin numbers without coercing arbitrary objects."""
    return (type(value) is int or type(value) is float) and abs(value) <= 2**53 and math.isfinite(value)


def _timings(values: dict[str, object]) -> dict[str, object]:
    """Copy only bounded numeric timing fields from an already-failed test frame."""
    result: dict[str, object] = {}
    omitted = []
    for name in ("started", "elapsed", "free_rate"):
        value = values.get(name)
        if _number(value):
            result[name] = value
        else:
            omitted.append(name)
    latencies = values.get("latencies")
    if type(latencies) is list and all(_number(value) for value in latencies[:_LIMIT]):
        result.update(latencies=latencies[:_LIMIT], latencies_truncated=len(latencies) > _LIMIT)
    else:
        omitted.append("latencies")
    ticks = values.get("ticks")
    if type(ticks) is list and len(ticks) == 1 and type(ticks[0]) is int and 0 <= ticks[0] < 2**63:
        result["ticks"] = ticks[0]
    else:
        omitted.append("ticks")
    result["omitted_fields"] = omitted
    return result


class LivenessObserver:
    """Observe one serial test call without changing its execution or outcome."""

    def __init__(self, inventory: Inventory, path: Path) -> None:
        """Bind a fresh external output to the existing source-aware inventory."""
        self.inventory, self.path = inventory, path
        self.start: float | None = None
        self.end: float | None = None
        self.events: list[tuple[float, str, int, int]] = []
        self.dropped = 0
        self.state: dict[str, object] = {}
        self.phases: dict[str, str] = {}
        self.metrics: dict[str, object] = {}

    def _gc_event(self, phase: str, info: dict[str, int]) -> None:
        """Record bounded timestamps only; never request or alter a collection."""
        if self.start is None:
            return
        if len(self.events) < _LIMIT:
            self.events.append((time.monotonic() - self.start, phase, info["generation"], threading.get_ident()))
        else:
            self.dropped += 1

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        """Attach during the target call and remove only this callback on every exit."""
        if item.nodeid != _TARGET:
            yield
            return
        self.state = {"gc_enabled": gc.isenabled(), "gc_threshold": gc.get_threshold(),
                      "gc_count": gc.get_count(), "modules": {name: name in sys.modules for name in _MODULES}}
        self.start = time.monotonic()
        callback = self._gc_event
        gc.callbacks.append(callback)
        try:
            yield
        finally:
            if callback in gc.callbacks:
                gc.callbacks.remove(callback)
            self.end = time.monotonic()
            self.state.update(gc_enabled_end=gc.isenabled(), gc_threshold_end=gc.get_threshold())

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> Generator[None, object, None]:
        """Read only numeric locals from the matching failure after callback removal."""
        yield
        if item.nodeid == _TARGET and call.when == "call" and call.excinfo is not None:
            for entry in call.excinfo.traceback:
                if entry.name == _TARGET.split("::")[1] and entry.frame.code.raw.co_filename == str(item.path):
                    self.metrics = _timings(entry.frame.f_locals)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Retain actual phase outcomes without copying their messages or output."""
        if report.nodeid == _TARGET:
            self.phases[report.when] = report.outcome

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Publish outside the measured call before the inventory finalizes status."""
        try:
            _publish(self.path, {
                "schema_version": 1, "target": _TARGET,
                "target_selected": any(item["nodeid"] == _TARGET for item in self.inventory.items),
                "call_entered": self.start is not None, "call_start_monotonic": self.start,
                "call_end_monotonic": self.end, "clock": "time.monotonic; default asyncio loop assumed",
                "gc_events": self.events, "gc_events_dropped": self.dropped,
                "state": self.state, "phases": self.phases, "failure_timings": self.metrics,
                "source": self.inventory.revision, "source_end": source_identity(self.inventory.root),
                "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            })
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if session.exitstatus == pytest.ExitCode.OK:
                session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
            reporter = session.config.pluginmanager.getplugin("terminalreporter")
            if reporter:
                reporter.write_line(f"Liveness observations omitted: {type(exc).__name__}", red=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register an explicit output path without enabling observations by default."""
    parser.getgroup("test inventory").addoption("--test-liveness-observations", metavar="PATH")


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Enable only beside an existing serial inventory with a fresh external path."""
    destination = config.getoption("test_liveness_observations")
    if not destination:
        return
    inventory = config.pluginmanager.getplugin("cowork-test-inventory")
    if inventory is None:
        raise pytest.UsageError("Liveness observations require --test-inventory")
    if getattr(config.option, "numprocesses", 0) or hasattr(config, "workerinput"):
        raise pytest.UsageError("Liveness observations require serial execution")
    path = Path(destination).resolve()
    if path.is_relative_to(inventory.root) or path.exists():
        raise pytest.UsageError("Liveness observations require a fresh external path")
    config.pluginmanager.register(LivenessObserver(inventory, path), "cowork-test-liveness")
