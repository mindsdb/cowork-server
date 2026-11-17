"""Opt-in case and phase evidence for serial pytest and local xdist runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

_WORKER_KEY = "cowork_test_inventory"
_CONFIG_FILES = ("uv.lock", "pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "Makefile")
_CI_FIELDS = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion")


def _digest(value: object) -> str:
    """Return a stable digest of a JSON-compatible value without writing it."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    """Run a read-only Git query at root and return its stdout bytes."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=check, capture_output=True, timeout=30,
    ).stdout


def _file_digest(path: Path) -> str | None:
    """Hash an existing file's bytes, or return None for an absent file."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_identity(root: Path) -> dict[str, Any]:
    """Fingerprint current tracked/untracked source, including deletion and symlinks."""
    paths = sorted(set(_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")) - {b""})
    digest = hashlib.sha256()
    for raw_path in paths:
        path = root / os.fsdecode(raw_path)
        digest.update(len(raw_path).to_bytes(8, "big") + raw_path)
        if path.is_symlink():
            content = b"link\0" + os.fsencode(os.readlink(path))
        elif path.is_file():
            content = b"file\0" + path.read_bytes()
        elif not path.exists():
            content = b"deleted"
        else:
            raise ValueError(f"Cannot fingerprint source entry: {path.relative_to(root)}")
        digest.update(hashlib.sha256(content).digest())
    config_paths = [*_CONFIG_FILES, *sorted(p.relative_to(root).as_posix() for p in (root / ".github/workflows").glob("*.yml"))]
    return {
        "sha": _git(root, "rev-parse", "--verify", "HEAD", check=False).decode().strip() or None,
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=normal")),
        "tree_digest": digest.hexdigest(),
        "files": len(paths),
        "config": {name: _file_digest(root / name) for name in config_paths},
    }


def _environment() -> dict[str, Any]:
    """Describe the interpreter and runner using an explicit nonsecret allowlist."""
    versions = {}
    for package in ("pytest", "pytest-xdist", "pytest-cov", "coverage"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "packages": versions,
        "ci": {key: os.environ[key] for key in _CI_FIELDS if key in os.environ},
    }


def _coverage(config: pytest.Config) -> dict[str, Any]:
    """Record effective coverage settings and hashes of the configuration actually read."""
    option = config.option
    plugin = config.pluginmanager.getplugin("_cov")
    controller = getattr(plugin, "cov_controller", None)
    coverage = getattr(controller, "cov", None)
    effective = {}
    files = {}
    if coverage is not None:
        for name in ("branch", "source", "include", "omit", "context", "dynamic_context", "concurrency", "timid"):
            effective[name] = coverage.get_option(f"run:{name}")
        files = {str(Path(path).resolve()): _file_digest(Path(path)) for path in coverage.config.config_files_read}
    return {
        "enabled": coverage is not None, "disabled": bool(getattr(option, "no_cov", False)),
        "source": getattr(option, "cov_source", None), "branch": getattr(option, "cov_branch", None),
        "context": getattr(option, "cov_context", None), "append": bool(getattr(option, "cov_append", False)),
        "config": getattr(option, "cov_config", None), "config_files": files, "effective": effective,
        "reports": sorted(getattr(option, "cov_report", {}) or {}),
        "no_report_on_failure": bool(getattr(option, "no_cov_on_fail", False)),
    }


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str | None:
    """Read only pytest's skip reason; never copy tracebacks or captured output."""
    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        return str(report.longrepr[2]).removeprefix("Skipped: ")
    if report.skipped and hasattr(report, "wasxfail"):
        return str(report.wasxfail)
    return None


def _publish(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish JSON without replacing existing evidence; clean the temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(document, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Inventory:
    """Observe one pytest session without changing its selection or execution."""

    def __init__(self, config: pytest.Config, path: Path) -> None:
        """Initialize observation state and fingerprint the controller's source tree."""
        self.config = config
        self.path = path
        self.worker = hasattr(config, "workerinput")
        self.root = Path(_git(config.rootpath, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        if path.is_relative_to(self.root):
            raise pytest.UsageError("Write --test-inventory outside the source checkout")
        if path.exists():
            raise pytest.UsageError("The --test-inventory output already exists")
        self.started = time.perf_counter()
        self.collection_started = self.started
        self.collection_seconds = 0.0
        self.collected = False
        self.items: list[dict[str, Any]] = []
        self.deselected: list[str] = []
        self.collection_reports: list[dict[str, Any]] = []
        self.phases: list[dict[str, Any]] = []
        self.workers: dict[str, dict[str, Any]] = {}
        self.payloads: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.revision = source_identity(self.root) if not self.worker else None
        self.reporter_digest = _file_digest(Path(__file__))
        self.coverage = _coverage(config)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(self) -> None:
        """Snapshot coverage after the coverage plugin has started its controller."""
        self.coverage = _coverage(self.config)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection(self) -> Generator[None, None, None]:
        """Time collection locally, independently of test execution."""
        self.collection_started = time.perf_counter()
        yield
        self.collection_seconds = time.perf_counter() - self.collection_started

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Record the final selected items and inherited marker names in order."""
        self.items = [
            {"nodeid": item.nodeid, "module": item.path.relative_to(self.root).as_posix(),
             "markers": sorted({mark.name for mark in item.iter_markers()})}
            for item in session.items
        ]
        self.collected = True

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        """Retain actual deselected IDs instead of treating them as skipped executions."""
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """Record module skips and collection errors separately from executable cases."""
        if report.failed or report.skipped:
            self.collection_reports.append({"nodeid": report.nodeid, "outcome": report.outcome, "reason": _skip_reason(report)})

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Keep controller/serial phase events, preserving repeated reports as evidence."""
        if self.worker:
            return
        self.phases.append({
            "nodeid": report.nodeid, "worker": getattr(report, "worker_id", "serial"),
            "phase": report.when, "outcome": report.outcome, "duration": report.duration,
            "xfail": hasattr(report, "wasxfail"), "reason": _skip_reason(report),
        })

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """Track each requested local worker so missing completions remain visible."""
        self.workers[node.gateway.id] = {"id": node.gateway.id, "completed": False}

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object | None) -> None:
        """Accept collection evidence only from workers that complete their handoff."""
        worker = self.workers[node.gateway.id]
        payload = getattr(node, "workeroutput", {}).get(_WORKER_KEY)
        if error is not None or payload is None:
            self.problems.append(f"Worker {node.gateway.id} did not complete its evidence handoff")
            return
        worker.update(completed=True, collection_seconds=payload["seconds"], digest=_digest(payload["manifest"]))
        self.payloads.append(payload)

    def _manifest(self) -> dict[str, Any]:
        """Build deterministic collection evidence without volatile timing fields."""
        return {
            "items": self.items, "deselected": sorted(self.deselected),
            "reports": sorted(self.collection_reports, key=lambda item: (item["nodeid"], item["outcome"], item["reason"] or "")),
            "finished": self.collected,
        }

    @pytest.hookimpl(hookwrapper=True, trylast=True, specname="pytest_sessionfinish")
    def pytest_inventory_worker_finish(self) -> Generator[None, Any, None]:
        """Hand successful worker evidence over after hooks but before xdist sends it."""
        outcome = yield
        if self.worker and outcome.excinfo is None:
            self.config.workeroutput[_WORKER_KEY] = {"manifest": self._manifest(), "seconds": self.collection_seconds}

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> Generator[None, Any, None]:
        """Publish controller evidence only after finish hooks return without an exception."""
        outcome = yield
        if self.worker:
            return
        if outcome.excinfo is not None:
            terminal = self.config.pluginmanager.getplugin("terminalreporter")
            if terminal:
                terminal.write_line("Test inventory omitted: a session-finish hook failed", red=True)
            return
        try:
            self._finish(session)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if session.exitstatus == pytest.ExitCode.OK:
                session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
            terminal = self.config.pluginmanager.getplugin("terminalreporter")
            if terminal:
                terminal.write_line(f"Test inventory could not be written: {type(exc).__name__}", red=True)

    def _finish(self, session: pytest.Session) -> None:
        """Combine agreed worker manifests and publish the final observed exit status."""
        manifest = self._manifest()
        if self.workers:
            if len(self.payloads) != len(self.workers):
                self.problems.append("Not every worker supplied collection evidence")
            if self.payloads:
                manifest = self.payloads[0]["manifest"]
                if any(payload["manifest"] != manifest for payload in self.payloads[1:]):
                    self.problems.append("Worker collection manifests disagree")
        if not manifest["finished"]:
            self.problems.append("Collection did not finish")
        if session.shouldstop or session.shouldfail:
            self.problems.append("The session stopped early")
        if session.exitstatus not in (0, 1, 5):
            self.problems.append(f"The session ended with exit status {int(session.exitstatus)}")
        end_revision = source_identity(self.root)
        if end_revision != self.revision:
            self.problems.append("Source content changed during the run")
        if _coverage(self.config) != self.coverage:
            self.problems.append("Coverage configuration changed during the run")
        if self.problems and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
        option = self.config.option
        document = {
            "schema_version": 1, "cohort": option.test_cohort,
            "mode": "collection" if option.collectonly else "execution",
            "revision": self.revision, "revision_end": end_revision,
            "reporter_digest": self.reporter_digest, "environment": _environment(),
            "invocation": {
                "args": list(self.config.invocation_params.args),
                "selection": {"paths": list(self.config.args), "keyword": option.keyword, "markers": option.markexpr,
                              "ignore": option.ignore or [], "deselect": option.deselect or []},
                "workers": getattr(option, "numprocesses", 0) or 0,
                "coverage": self.coverage,
            },
            "collection": manifest, "collection_seconds": self.collection_seconds if not self.workers else None,
            "phases": self.phases, "workers": list(self.workers.values()),
            "completion": {"exitstatus": int(session.exitstatus), "elapsed_seconds": time.perf_counter() - self.started,
                           "complete": not self.problems, "problems": self.problems},
        }
        _publish(self.path, document)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit report arguments without enabling observation by default."""
    group = parser.getgroup("test inventory")
    group.addoption("--test-inventory", metavar="PATH", help="Write case/phase JSON outside the checkout")
    group.addoption("--test-cohort", help="Name the measured selection, such as broad or windows-export")


def pytest_configure(config: pytest.Config) -> None:
    """Enable the observer only for invocations requesting a named report."""
    path = config.getoption("test_inventory")
    if path:
        if not config.getoption("test_cohort"):
            raise pytest.UsageError("--test-inventory requires --test-cohort")
        config.pluginmanager.register(Inventory(config, Path(path).resolve()), "cowork-test-inventory")
