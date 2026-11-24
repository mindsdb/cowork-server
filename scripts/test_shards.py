"""Run four complete broad-suite partitions and verify their combined evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from scripts.blocking_guard_evidence import GuardEvidence, read_evidence, sidecar
from scripts.pytest_inventory import source_identity
from scripts.test_inventory import Report, read_report, validate, validate_junit

_COLLECTION = "test-inventory-broad-collection"
_SHARD = "test-inventory-broad-shard-"
_SHARD_COUNT = 4
_VARIABLE_CI_FIELDS = {"GITHUB_JOB", "GITHUB_RUN_ATTEMPT"}
_HOST_RUNTIME_FIELDS = {"release", "logical_cpus"}
_HOST_IMAGE_FIELDS = {"ImageOS", "ImageVersion"}


def _require(condition: bool, message: str) -> None:
    """Reject invalid evidence without copying its contents into diagnostics."""
    if not condition:
        raise ValueError(message)


def _context() -> tuple[str, int]:
    """Read the consuming workflow identity from its explicit CI fields."""
    run = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    _require(run.isascii() and run.isdecimal() and int(run) > 0 and attempt.isascii() and
             attempt.isdecimal() and int(attempt) > 0, "Missing workflow run/attempt identity")
    return run, int(attempt)


def _artifact_file(directory: Path, name: str) -> Path:
    """Require a regular artifact input without following a substituted symlink."""
    path = directory / name
    _require(path.is_file() and not path.is_symlink(), "Missing or invalid artifact input")
    return path


def latest_artifact(inputs: Path, cohort: str) -> tuple[Path, Report, int]:
    """Read the newest same-run producer, retaining failure instead of falling back."""
    run, consumer = _context()
    prefix = f"{cohort}-{run}-"
    candidates = []
    flat = inputs / "inventory.json"
    if cohort == _COLLECTION and (flat.exists() or flat.is_symlink()):
        report = read_report(_artifact_file(inputs, "inventory.json"))
        validate(report)
        _require(report.environment.ci.get("GITHUB_RUN_ID") == run, "Artifact producer identity differs")
        producer = report.environment.ci.get("GITHUB_RUN_ATTEMPT", "")
        _require(producer.isascii() and producer.isdecimal() and str(int(producer)) == producer and
                 0 < int(producer) <= consumer, "Invalid or future producer attempt")
        candidates.append((int(producer), inputs))
    for path in inputs.glob(prefix + "*"):
        suffix = path.name.removeprefix(prefix)
        _require(suffix.isdecimal() and str(int(suffix)) == suffix and 0 < int(suffix) <= consumer,
                 "Invalid or future producer attempt")
        _require(path.is_dir() and not path.is_symlink(), "Invalid artifact directory")
        candidates.append((int(suffix), path))
    _require(bool(candidates), "Missing required artifact")
    _require(len({attempt for attempt, _ in candidates}) == len(candidates), "Duplicate producer attempt")
    attempt, path = max(candidates)
    report = read_report(_artifact_file(path, "inventory.json"))
    validate(report)
    _require(report.environment.ci.get("GITHUB_RUN_ID") == run and
             report.environment.ci.get("GITHUB_RUN_ATTEMPT") == str(attempt), "Artifact producer identity differs")
    _require(report.completion.exitstatus == 0, "Newest producer did not succeed")
    return path, report, attempt


def _module_path(name: str) -> str:
    """Accept an existing Python module contained in this checkout's test tree."""
    path = PurePosixPath(name)
    root = Path.cwd().resolve() / "tests"
    _require(not path.is_absolute() and len(path.parts) > 1 and path.parts[0] == "tests" and
             ".." not in path.parts and path.as_posix() == name and path.suffix == ".py" and "::" not in name,
             "Invalid collected module path")
    resolved = Path(name).resolve()
    _require(resolved.is_relative_to(root) and resolved.is_file(), "Collected module escapes or is absent")
    return name


def partition(collection: Report, modules: Iterable[str] = ()) -> list[list[str]]:
    """Assign every collected module once using descending case counts and stable ties."""
    validate(collection)
    selection = collection.invocation.selection
    _require(collection.mode == "collection" and collection.cohort == "broad" and
             collection.completion.exitstatus == 0, "Expected successful full broad collection")
    _require(selection.paths == ["tests/"] and selection.ignore == ["tests/integration"] and
             not selection.keyword and not selection.markers and not selection.deselect and
             not collection.collection.deselected, "Full collection selection differs")
    counts = Counter()
    for case in collection.collection.items:
        module = _module_path(case.module)
        _require(case.nodeid.split("::", 1)[0] == module, "Case module ownership differs")
        counts[module] += 1
    for skipped in collection.collection.reports:
        counts.setdefault(_module_path(skipped.nodeid), 0)
    for module in modules:
        counts.setdefault(_module_path(module), 0)
    groups: list[list[str]] = [[] for _ in range(_SHARD_COUNT)]
    totals = [0] * _SHARD_COUNT
    for module in sorted(counts, key=lambda name: (-counts[name], name)):
        index = min(range(_SHARD_COUNT), key=lambda candidate: (totals[candidate], candidate))
        groups[index].append(module)
        totals[index] += counts[module]
    _require(all(totals), "Every shard requires executable cases")
    return [sorted(group) for group in groups]


def _collection(inputs: Path) -> tuple[Path, Report, int]:
    """Validate the full collection against the actual checked-out source."""
    path, report, attempt = latest_artifact(inputs, _COLLECTION)
    _require(report.revision.model_dump() == source_identity(Path.cwd()), "Collection source differs from checkout")
    declaration = _guard_evidence(path, report)
    if declaration is not None:
        required_files = sorted(path.relative_to(Path.cwd()).as_posix() for path in (Path.cwd() / "tests").rglob("test_*.py")
                                if "integration" not in path.relative_to(Path.cwd() / "tests").parts)
        _require(declaration.required_files == required_files and
                 set(required_files) <= set(declaration.collected_files), "Blocking guard full collection differs")
        _require(declaration.ineligible_because == "collection only", "Expected blocking guard declaration")
    partition(report, declaration.collected_files if declaration else ())
    return path, report, attempt


def _guard_evidence(path: Path, report: Report) -> GuardEvidence | None:
    """Require source-bound guard evidence when the checkout defines the runtime guard."""
    if not (Path.cwd() / "tests/_blocking_guard.py").is_file():
        return None
    evidence = read_evidence(_artifact_file(path, sidecar(Path("inventory.json")).name))
    _require(evidence.revision == evidence.revision_end == report.revision and
             evidence.mode == report.mode and evidence.cohort == report.cohort,
             "Blocking guard source/cohort differs")
    _require(evidence.run_id == report.environment.ci.get("GITHUB_RUN_ID") == _context()[0] and
             evidence.run_attempt == report.environment.ci.get("GITHUB_RUN_ATTEMPT"),
             "Blocking guard producer identity differs")
    for name in [*evidence.required_files, *evidence.collected_files]:
        _module_path(name)
    return evidence


def _global_guard(declaration: GuardEvidence | None, paths: list[Path], reports: list[Report],
                  groups: list[list[str]]) -> dict[str, object] | None:
    """Compose complete real worker observations before making the original stale decision."""
    if declaration is None:
        return None
    reached = set()
    collected = set()
    for path, report, modules in zip(paths, reports, groups):
        evidence = _guard_evidence(path, report)
        _require(evidence.required == declaration.required and evidence.required_files == declaration.required_files,
                 "Blocking guard declarations differ")
        _require(evidence.ineligible_because is None and not evidence.workers_lost and
                 evidence.workers == sorted(worker.id for worker in report.workers) and
                 len(evidence.workers) == 4, "Blocking guard execution is incomplete")
        _require(evidence.collected_files == modules, "Blocking guard module membership differs")
        reached.update(leaf.key() for leaf in evidence.reached)
        collected.update(evidence.collected_files)
    _require(collected == set(declaration.collected_files), "Blocking guard collected-file union differs")
    required = {leaf.key() for leaf in declaration.required}
    _require(reached == required, "Blocking guard ALLOWLIST names sites no test reached")
    return {"complete": True, "required_leaves": len(required), "reached_leaves": len(reached),
            "collected_files": len(collected)}


def _output(path: Path) -> Path:
    """Create fresh output outside the checkout, refusing to replace evidence."""
    path = path.resolve()
    _require(not path.is_relative_to(Path.cwd().resolve()), "Write shard evidence outside the checkout")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_shard(inputs: Path, output: Path, index: int) -> int:
    """Execute one complete module partition and propagate pytest's exit status."""
    _require(index in range(_SHARD_COUNT), "Invalid shard index")
    collection_path, collection, _ = _collection(inputs)
    declaration = _guard_evidence(collection_path, collection)
    groups = partition(collection, declaration.collected_files if declaration else ())
    ignored = ["tests/integration", *sorted(name for i, group in enumerate(groups) if i != index for name in group)]
    output = _output(output)
    environment = dict(os.environ, COVERAGE_FILE=str(output / "coverage.data"))
    command = [sys.executable, "-m", "pytest", "-q", "-o", "faulthandler_timeout=120",
               "-n", "4", "--cov=cowork", "--cov-report=",
               *[f"--ignore={name}" for name in ignored], "-p", "scripts.pytest_inventory",
               f"--test-cohort=broad-shard-{index}", f"--test-inventory={output / 'inventory.json'}",
               f"--junitxml={output / 'junit.xml'}", "tests/"]
    return subprocess.run(command, env=environment, check=False).returncode


def _runner_differences(cohort: str, fields: Iterable[str]) -> None:
    """Report host differences without printing metadata values."""
    changed = sorted(fields)
    if changed:
        print(f"{cohort}: hosted runner metadata differs from collection ({', '.join(changed)})", file=sys.stderr)


def _runner_identity(report: Report) -> dict[str, str]:
    """Return CI fields that identify the runner platform and workflow run."""
    return {key: value for key, value in report.environment.ci.items()
            if key not in _VARIABLE_CI_FIELDS | _HOST_IMAGE_FIELDS}


def _compatible(collection: Report, shard: Report) -> None:
    """Require compatible source and runtime while allowing carried producer attempts."""
    _require(shard.revision == shard.revision_end == collection.revision, "Shard source/configuration differs")
    _require(shard.reporter_digest == collection.reporter_digest, "Reporter differs")
    excluded = {"ci"} | _HOST_RUNTIME_FIELDS
    _require(shard.environment.model_dump(exclude=excluded) == collection.environment.model_dump(exclude=excluded),
             "Shard runtime differs; rerun collection and its dependent jobs")
    _require(_runner_identity(collection) == _runner_identity(shard),
             "Runner/run metadata differs; rerun collection and its dependent jobs")
    _runner_differences(shard.cohort, [
        *(key for key in _HOST_RUNTIME_FIELDS if getattr(shard.environment, key) != getattr(collection.environment, key)),
        *(key for key in _HOST_IMAGE_FIELDS if shard.environment.ci.get(key) != collection.environment.ci.get(key)),
    ])


def _coverage_data(path: Path) -> CoverageData:
    """Read a nonempty coverage input and constrain measured paths before reporting."""
    _require(path.is_file() and path.stat().st_size > 0, "Missing coverage input")
    data = CoverageData(basename=str(path))
    data.read()
    _require(bool(data.measured_files()), "Empty coverage measurement")
    package = Path.cwd().resolve() / "cowork"
    _require(all(Path(name).resolve().is_relative_to(package) for name in data.measured_files()),
             "Coverage source escapes the package")
    return data


def _coverage_sets(data: CoverageData) -> dict[str, set[int | tuple[int, int]]]:
    """Read measured lines or arcs without converting absence into measured coverage."""
    read = data.arcs if data.has_arcs() else data.lines
    return {name: set(read(name) or []) for name in data.measured_files()}


def aggregate(inputs: Path, output: Path) -> dict[str, object]:
    """Audit every real shard execution, combine their coverage and retain provenance."""
    collection_path, collection, collection_attempt = _collection(inputs)
    declaration = _guard_evidence(collection_path, collection)
    groups = partition(collection, declaration.collected_files if declaration else ())
    reports, paths, attempts = [], [], []
    for index in range(_SHARD_COUNT):
        path, report, attempt = latest_artifact(inputs, _SHARD + str(index))
        _require(report.mode == "execution" and report.cohort == f"broad-shard-{index}", "Wrong shard cohort/mode")
        _compatible(collection, report)
        validate_junit(report, _artifact_file(path, "junit.xml"))
        selection = report.invocation.selection
        ignored = ["tests/integration", *sorted(name for i, group in enumerate(groups) if i != index for name in group)]
        _require(selection.paths == ["tests/"] and selection.ignore == ignored and
                 not selection.keyword and not selection.markers and not selection.deselect and
                 not report.collection.deselected, "Shard selection differs")
        _require(report.invocation.workers == 4, "Shard worker count differs")
        expected_cases = [case for case in collection.collection.items if case.module in groups[index]]
        _require({case.nodeid: case for case in report.collection.items} == {case.nodeid: case for case in expected_cases},
                 "Shard cases/markers differ from full collection")
        expected_skips = [skip for skip in collection.collection.reports if skip.nodeid in groups[index]]
        _require(report.collection.reports == expected_skips, "Shard collector skips differ")
        coverage = report.invocation.coverage
        _require(coverage.enabled and not coverage.disabled and not coverage.append and coverage.source == ["cowork"] and
                 not coverage.no_report_on_failure, "Shard coverage is not the full measurement")
        _require(all(Path(name).resolve().is_relative_to(Path.cwd().resolve()) for name in coverage.config_files),
                 "Coverage configuration is outside the source")
        reports.append(report)
        paths.append(path)
        attempts.append(attempt)
    _require(all(os.environ.get(k) == v for k, v in _runner_identity(collection).items()),
             "Consumer runner/run metadata differs")
    _runner_differences("aggregate", (key for key in _HOST_IMAGE_FIELDS
                                      if os.environ.get(key) != collection.environment.ci.get(key)))
    _require(all(report.invocation.coverage == reports[0].invocation.coverage for report in reports),
             "Coverage configurations differ")
    items = [case for report in reports for case in report.collection.items]
    _require(len(items) == len({case.nodeid for case in items}) == len(collection.collection.items), "Case union is not exact")
    guard_result = _global_guard(declaration, paths, reports, groups)
    coverage_paths = [_artifact_file(path, "coverage.data") for path in paths]
    data = [_coverage_data(path) for path in coverage_paths]
    _require(all(measurement.has_arcs() == data[0].has_arcs() for measurement in data), "Coverage line/arc modes differ")
    _require(data[0].has_arcs() == bool(reports[0].invocation.coverage.effective.get("branch")), "Coverage mode differs from report")
    expected = {}
    for measurement in data:
        for name, lines in _coverage_sets(measurement).items():
            expected.setdefault(name, set()).update(lines)
    output = _output(output)
    coverage = Coverage(data_file=str(output / "coverage.data"))
    coverage.combine(data_paths=[str(path) for path in coverage_paths], strict=True, keep=True)
    _require(_coverage_sets(coverage.get_data()) == expected, "Combined coverage is not the exact input union")
    coverage.save()
    coverage.report()
    percent = coverage.json_report(outfile=str(output / "coverage.json"))
    shutil.copyfile(collection_path / "inventory.json", output / "collection.json")
    if declaration is not None:
        shutil.copyfile(sidecar(collection_path / "inventory.json"), sidecar(output / "collection.json"))
    producers = []
    for index, (path, report, attempt) in enumerate(zip(paths, reports, attempts)):
        target = output / f"shard-{index}"
        target.mkdir()
        for name in ("inventory.json", "junit.xml", "coverage.data"):
            shutil.copyfile(path / name, target / name)
        if declaration is not None:
            shutil.copyfile(sidecar(path / "inventory.json"), sidecar(target / "inventory.json"))
        producers.append({"index": index, "producer_attempt": attempt, "artifact": path.name,
                          "cases": len(report.collection.items),
                          "inventory_sha256": hashlib.sha256((path / "inventory.json").read_bytes()).hexdigest()})
    summary = {"schema_version": 1, "kind": "broad-shard-composition", "run_id": _context()[0],
               "consumer_attempt": _context()[1], "collection_attempt": collection_attempt,
               "revision": collection.revision.model_dump(), "collected": len(items),
               "collector_skips": len(collection.collection.reports), "coverage_percent": percent,
               "shards": producers}
    if guard_result is not None:
        summary["blocking_guard"] = guard_result
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified complete broad selection: {len(items)} cases across {_SHARD_COUNT} shards.")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run or verify explicit CI inputs, reporting failures without raw artifact payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--index", type=int, choices=range(_SHARD_COUNT), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_shard(args.inputs, args.output, args.index)
        aggregate(args.inputs, args.output)
    except (OSError, ValueError, CoverageException) as error:
        reason = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Shard evidence rejected: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
