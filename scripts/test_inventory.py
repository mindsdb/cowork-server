"""Validate and compare recorded test cases; timings never stand in for coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class Case(_Record):
    nodeid: str
    module: str
    markers: list[str]


class CollectionReport(_Record):
    nodeid: str
    outcome: Literal["skipped", "failed"]
    reason: str | None


class Collection(_Record):
    items: list[Case]
    deselected: list[str]
    reports: list[CollectionReport]
    finished: bool


class Phase(_Record):
    nodeid: str
    worker: str
    phase: Literal["setup", "call", "teardown"]
    outcome: Literal["passed", "failed", "skipped"]
    duration: float = Field(ge=0)
    xfail: bool
    reason: str | None


class Revision(_Record):
    sha: str | None
    dirty: bool
    tree_digest: str
    files: int = Field(ge=0)
    config: dict[str, str | None]


class Environment(_Record):
    python: str
    implementation: str
    system: str
    release: str
    machine: str
    logical_cpus: int | None
    packages: dict[str, str | None]
    ci: dict[str, str]


class Selection(_Record):
    paths: list[str]
    keyword: str
    markers: str
    ignore: list[str]
    deselect: list[str]


class Coverage(_Record):
    enabled: bool
    disabled: bool
    source: list[str] | None
    branch: bool | None
    context: str | None
    append: bool
    config: str | None
    config_files: dict[str, str | None]
    effective: dict[str, str | bool | list[str] | None]
    reports: list[str]
    no_report_on_failure: bool


class Invocation(_Record):
    args: list[str]
    selection: Selection
    workers: int = Field(ge=0)
    coverage: Coverage


class Worker(_Record):
    id: str
    completed: bool
    collection_seconds: float | None = Field(default=None, ge=0)
    digest: str | None = None


class Completion(_Record):
    exitstatus: int = Field(ge=0, le=5)
    elapsed_seconds: float = Field(ge=0)
    complete: bool
    problems: list[str]


class Report(_Record):
    schema_version: Literal[1]
    cohort: str
    mode: Literal["collection", "execution"]
    revision: Revision
    revision_end: Revision
    reporter_digest: str
    environment: Environment
    invocation: Invocation
    collection: Collection
    collection_seconds: float | None = Field(ge=0)
    phases: list[Phase]
    workers: list[Worker]
    completion: Completion


def read_report(path: Path) -> Report:
    """Read and strictly validate report structure without echoing invalid payloads."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if type(raw.get("schema_version")) is not int:
            raise ValueError("Report schema_version must be an integer")
        return Report.model_validate(raw)
    except (ValidationError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Invalid test inventory structure") from exc


def _require(condition: bool, message: str) -> None:
    """Reject invalid evidence with a concise diagnostic that excludes raw report contents."""
    if not condition:
        raise ValueError(message)


def validate(report: Report) -> dict[str, list[Phase]]:
    """Reconcile collection, workers and phases; return events grouped by case ID."""
    completion = report.completion
    _require(completion.complete and not completion.problems, "Report is incomplete")
    _require(report.collection.finished, "Collection did not finish")
    _require(report.revision == report.revision_end, "Source changed during the run")
    ids = [case.nodeid for case in report.collection.items]
    id_set = set(ids)
    _require(len(ids) == len(id_set), "Duplicate collected case")
    _require(all(case.nodeid and case.module and case.markers == sorted(set(case.markers)) for case in report.collection.items), "Invalid case or marker metadata")
    _require(not id_set & set(report.collection.deselected), "Selected case also appears as deselected")
    _require(len(report.collection.deselected) == len(set(report.collection.deselected)), "Duplicate deselected case")
    collection_ids = [entry.nodeid for entry in report.collection.reports]
    _require(len(collection_ids) == len(set(collection_ids)), "Duplicate collection report")
    _require(not any(entry.outcome == "failed" for entry in report.collection.reports), "Collection contains errors")
    worker_ids = [worker.id for worker in report.workers]
    _require(len(worker_ids) == len(set(worker_ids)), "Duplicate worker evidence")
    manifest_digest = hashlib.sha256(json.dumps(report.collection.model_dump(), sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    _require(all(worker.completed and worker.digest == manifest_digest and worker.collection_seconds is not None
                 for worker in report.workers), "Missing or mismatched worker collection")
    if report.mode == "execution" and report.invocation.workers:
        _require(len(report.workers) == report.invocation.workers, "Missing or replaced worker evidence")
    else:
        _require(not report.workers, "Unexpected worker evidence")
        _require(report.collection_seconds is not None, "Missing serial collection duration")
    events: dict[str, list[Phase]] = defaultdict(list)
    for phase in report.phases:
        _require(phase.nodeid in id_set, "Execution reported an uncollected case")
        _require(phase.worker in (worker_ids or ["serial"]), "Execution reported an unknown worker")
        events[phase.nodeid].append(phase)
    if report.mode == "collection":
        _require(not events, "Collection-only report contains executions")
        _require(completion.exitstatus in (0, 5), "Collection did not succeed")
        return events
    _require(set(events) == id_set, "Collected cases are missing executions")
    for phases in events.values():
        _require(len({phase.worker for phase in phases}) == 1, "Case executed on more than one worker")
        names = [phase.phase for phase in phases]
        _require(names in (["setup", "call", "teardown"], ["setup", "teardown"]), "Missing, duplicate or out-of-order phase")
        _require((phases[0].outcome == "passed") == ("call" in names), "Call phase does not match setup outcome")
    has_failures = any(phase.outcome == "failed" for phase in report.phases)
    _require(not has_failures or completion.exitstatus != 0, "Failing phases have a successful exit status")
    _require(completion.exitstatus in (0, 1, 5), "Execution did not finish normally")
    _require(completion.exitstatus != 5 or not ids, "No-tests exit status with collected cases")
    return events


def compare(left: Report, right: Report) -> None:
    """Require matching source, selection and outcomes for a serial/parallel comparison."""
    left_events, right_events = validate(left), validate(right)
    for field in ("cohort", "mode", "revision", "reporter_digest", "environment", "collection"):
        _require(getattr(left, field) == getattr(right, field), f"Reports differ in {field}")
    _require(left.invocation.selection == right.invocation.selection, "Reports differ in selection")
    _require(left.invocation.coverage == right.invocation.coverage, "Reports differ in coverage settings")
    _require(left.completion.exitstatus == right.completion.exitstatus, "Reports differ in final exit status")
    for nodeid in left_events:
        first = [(p.phase, p.outcome, p.xfail, p.reason) for p in left_events[nodeid]]
        second = [(p.phase, p.outcome, p.xfail, p.reason) for p in right_events[nodeid]]
        _require(first == second, f"Case outcomes differ: {nodeid!r}")


def preserve(base: Report, candidate: Report, base_sha: str, candidate_sha: str, additions: list[str]) -> None:
    """Audit collection across declared revisions, permitting only reviewed additions."""
    validate(base)
    validate(candidate)
    _require(base.mode == candidate.mode == "collection", "Preservation requires collection-only reports")
    _require(base.revision.sha == base_sha and candidate.revision.sha == candidate_sha, "Declared revisions do not match")
    for field in ("cohort", "reporter_digest", "environment"):
        _require(getattr(base, field) == getattr(candidate, field), f"Preservation reports differ in {field}")
    _require(base.revision.config.get("uv.lock") == candidate.revision.config.get("uv.lock"), "Dependency locks differ")
    _require(base.invocation.selection == candidate.invocation.selection, "Preservation selections differ")
    _require(base.collection.reports == candidate.collection.reports, "Collection skips changed")
    _require(base.collection.deselected == candidate.collection.deselected, "Deselection changed")
    old = {case.nodeid: case for case in base.collection.items}
    new = {case.nodeid: case for case in candidate.collection.items}
    _require(old.keys() <= new.keys(), "Pre-existing cases were removed")
    _require(all(new[nodeid] == case for nodeid, case in old.items()), "Pre-existing case metadata changed")
    _require(len(additions) == len(set(additions)), "Duplicate entry in additions ledger")
    _require(new.keys() - old.keys() == set(additions), "Added cases do not match the reviewed ledger")


def validate_junit(report: Report, path: Path) -> None:
    """Reconcile pytest XML totals, including its extra row for call plus teardown failure."""
    events = validate(report)
    _require(report.mode == "execution", "JUnit comparison requires an execution report")
    root = ET.parse(path).getroot()
    rows = list(root.iter("testcase"))
    extra = sum(any(p.phase == "call" and p.outcome == "failed" for p in phases)
                and any(p.phase == "teardown" and p.outcome == "failed" for p in phases)
                for phases in events.values())
    expected_rows = len(events) + extra + len(report.collection.reports)
    _require(len(rows) == expected_rows, "JUnit testcase count does not reconcile")
    expected_failures = sum(p.phase == "call" and p.outcome == "failed" for p in report.phases)
    expected_errors = sum(p.phase != "call" and p.outcome == "failed" for p in report.phases)
    expected_skips = sum(p.outcome == "skipped" for p in report.phases) + len(report.collection.reports)
    for tag, expected in (("failure", expected_failures), ("error", expected_errors), ("skipped", expected_skips)):
        _require(sum(len(row.findall(tag)) for row in rows) == expected, f"JUnit {tag} count does not reconcile")


def summary(report: Report) -> str:
    """Summarize case outcomes and phase work without interpreting sums as wall time."""
    events = validate(report)
    outcomes: Counter[str] = Counter()
    for phases in events.values():
        if any(phase.outcome == "failed" for phase in phases):
            outcome = "failed"
        elif any(phase.outcome == "skipped" for phase in phases):
            outcome = "xfailed" if any(p.xfail for p in phases) else "skipped"
        else:
            outcome = "xpassed" if any(p.xfail for p in phases) else "passed"
        outcomes[outcome] += 1
    lines = [f"{report.cohort}: {len(report.collection.items)} collected; {len(report.collection.reports)} collection skips",
             f"Mode: {report.mode}; pytest exit: {report.completion.exitstatus}; wall time: {report.completion.elapsed_seconds:.3f}s",
             f"Outcomes: {dict(sorted(outcomes.items()))}"]
    totals: Counter[str] = Counter()
    modules: Counter[str] = Counter()
    cases: Counter[str] = Counter()
    paths = {case.nodeid: case.module for case in report.collection.items}
    for phase in report.phases:
        totals[phase.phase] += phase.duration
        modules[paths[phase.nodeid]] += phase.duration
        cases[phase.nodeid] += phase.duration
    lines.append("Cumulative phase work: " + ", ".join(f"{key}={value:.3f}s" for key, value in sorted(totals.items())))
    for label, values in (("Slowest cases", cases), ("Slowest modules", modules)):
        lines.append(label + ":")
        lines.extend(f"  {duration:.3f}s {name!r}" for name, duration in values.most_common(20))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Validate, compare or audit preservation; print a concise result and return status."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("report", type=Path)
    check.add_argument("--junit", type=Path)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("base", type=Path)
    comparison.add_argument("candidate", type=Path)
    preservation = subparsers.add_parser("preserve")
    preservation.add_argument("base", type=Path)
    preservation.add_argument("candidate", type=Path)
    preservation.add_argument("--base-sha", required=True)
    preservation.add_argument("--candidate-sha", required=True)
    preservation.add_argument("--allowed-additions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report = read_report(args.report)
            validate(report)
            if args.junit:
                validate_junit(report, args.junit)
            print(summary(report))
        else:
            base, candidate = read_report(args.base), read_report(args.candidate)
            if args.command == "compare":
                compare(base, candidate)
                print("Collection and outcomes match at the same source and environment.")
            else:
                additions = json.loads(args.allowed_additions.read_text(encoding="utf-8"))
                _require(isinstance(additions, list) and all(isinstance(item, str) for item in additions), "Invalid additions ledger")
                preserve(base, candidate, args.base_sha, args.candidate_sha, additions)
                print(f"Preserved {len(base.collection.items)} cases; added {len(additions)} reviewed cases.")
                print(f"Source fingerprints: {base.revision.tree_digest} -> {candidate.revision.tree_digest}")
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"Test inventory rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
