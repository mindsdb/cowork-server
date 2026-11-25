"""Name every test a run left out, and fail a run that promised it would run them.

A skip prints as an `s` among the dots and the job still goes green, so a
cross-replica test that never ran reads the same as one that passed.

A run that sets `COWORK_REQUIRE_INTEGRATION` or `COWORK_REQUIRE_SERVICES` to
`true` promised every prerequisite. In such a run, each skipped test and each
`--deselect` must appear in `skip_allowlist.yaml` with its reason, or the run
fails. A run that sets either flag, to any value, lists what it left out at the
end of the terminal report and in the GitHub job summary.

`tests/conftest.py` imports the two hooks below. A run outside `tests/` loads
them with `-p tests.skip_guard`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

REQUIREMENT_FLAGS = ("COWORK_REQUIRE_INTEGRATION", "COWORK_REQUIRE_SERVICES")
DEFAULT_ALLOWLIST = Path(__file__).with_name("skip_allowlist.yaml")


class AllowedSkip(BaseModel):
    test: str
    reason: str


class LeftOut(BaseModel):
    test: str
    reason: str
    allowed: bool


def load_allowlist(path: Path) -> list[AllowedSkip]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [AllowedSkip.model_validate(entry) for entry in entries]


def is_allowed(test: str, allowlist: list[AllowedSkip]) -> bool:
    """An entry names a test id; it also covers that test's parametrized cases."""
    return any(test == entry.test or test.startswith(f"{entry.test}[") for entry in allowlist)


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


class SkipGuard:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.flag_set = any(flag in os.environ for flag in REQUIREMENT_FLAGS)
        self.enforcing = any(os.environ.get(flag) == "true" for flag in REQUIREMENT_FLAGS)
        self.skips: dict[str, str] = {}
        self.left_out: list[LeftOut] = []

    @property
    def unexpected(self) -> list[LeftOut]:
        return [item for item in self.left_out if not item.allowed]

    def _record(self, report: pytest.TestReport | pytest.CollectReport) -> None:
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skips.setdefault(report.nodeid, _skip_reason(report))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._record(report)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self._record(report)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        allowlist = load_allowlist(Path(self.config.getoption("--skip-allowlist")))
        self.left_out = [
            LeftOut(test=test, reason=reason, allowed=is_allowed(test, allowlist))
            for test, reason in self.skips.items()
        ]
        self.left_out += [
            LeftOut(test=test, reason="deselected with --deselect", allowed=is_allowed(test, allowlist))
            for test in self.config.getoption("deselect") or []
        ]
        if self.enforcing and self.unexpected and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path and self.flag_set:
            with open(summary_path, "a", encoding="utf-8") as summary:
                summary.write(self._markdown())

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if not self.flag_set:
            return
        terminalreporter.section("tests left out of this run")
        if not self.left_out:
            terminalreporter.write_line("none")
        for item in self.left_out:
            status = "allowed" if item.allowed else "NOT ALLOWED"
            terminalreporter.write_line(f"{status}: {item.test}: {item.reason}")
        if self.enforcing and self.unexpected:
            terminalreporter.write_line(
                f"{len(self.unexpected)} test(s) left out with no entry in "
                f"{self.config.getoption('--skip-allowlist')}, in a run that requires "
                f"every prerequisite, so the run fails."
            )

    def _markdown(self) -> str:
        lines = ["### Tests left out of this run", ""]
        if not self.left_out:
            return "\n".join(lines + ["None.", "", ""])
        lines += ["| Test | Status | Reason |", "| --- | --- | --- |"]
        for item in self.left_out:
            status = "NOT RUN (allowed)" if item.allowed else "NOT RUN (not allowed)"
            reason = item.reason.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{item.test}` | {status} | {reason} |")
        if self.enforcing and self.unexpected:
            lines += ["", f"{len(self.unexpected)} not allowed, so the run fails."]
        return "\n".join(lines + ["", ""])


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="YAML list of tests that may be left out of a run that requires every prerequisite.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(SkipGuard(config), "cowork-skip-guard")
