"""Measure server-process CPU and capture py-spy flamegraphs around API turns.

The staging mode is for the opt-in CI workflow. Local mode measures an already
running loopback server; neither mode installs packages or changes deployment.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import httpx

from scripts.performance.workload import (
    BROWSER_UA, MeasurementFailure, STAGING_URL, Workload, WorkloadConfig,
    summarize, verify_staging_identity, workload_identity,
)

# Run with the target's own Python so package provenance comes from the same
# interpreter as the serving process. It returns no environment or argv values.
PROCESS_PROBE = r"""
import importlib.metadata as m, json, os, pathlib, sys
requested = int(sys.argv[1])
root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path('/proc')
found = []
for directory in root.iterdir():
    if not directory.name.isdigit() or (requested and int(directory.name) != requested):
        continue
    try:
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        executable = pathlib.Path(argv[0].decode(errors='replace')).name
        if executable not in {'python', 'python3', 'python3.12', 'python3.13', 'uvicorn', 'cowork-server'}:
            continue
        if b'cowork.server:app' not in argv and b'spa_wrapper:app' not in argv and executable != 'cowork-server':
            continue
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        found.append({'pid': int(directory.name), 'start_ticks': int(fields[19]),
                      'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
if len(found) != 1:
    raise SystemExit('expected exactly one verified cowork-server process')
result = found[0]
result['boot_id'] = (root / 'sys/kernel/random/boot_id').read_text().strip()
result['versions'] = {name: m.version(name) for name in ('cowork-server', 'anton-agent')}
result['versions']['python'] = sys.version.split()[0]
raw = m.distribution('anton-agent').read_text('direct_url.json')
result['anton_commit'] = json.loads(raw).get('vcs_info', {}).get('commit_id') if raw else None
print(json.dumps(result))
"""

STOP_PROFILE = r"""
import os, pathlib, signal, sys
output = sys.argv[1].encode()
for directory in pathlib.Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        if output in argv and b'record' in argv and any(pathlib.Path(a.decode(errors='replace')).name == 'py-spy' for a in argv):
            os.kill(int(directory.name), signal.SIGINT)
    except (FileNotFoundError, ProcessLookupError):
        continue
"""

READ_PROFILE = r"""
import pathlib, sys
path = pathlib.Path(sys.argv[1])
try:
    if path.stat().st_size > 50_000_000:
        raise SystemExit('profile exceeds artifact limit')
    sys.stdout.buffer.write(path.read_bytes())
finally:
    path.unlink(missing_ok=True)
"""


@dataclass
class Target:
    name: str
    prefix: list[str]
    identity: dict
    pid: int = 0
    profiler_prefix: list[str] | None = None

    def command(self, *args: str, timeout: int = 30, profiler: bool = False) -> bytes:
        try:
            prefix = self.profiler_prefix if profiler and self.profiler_prefix is not None else self.prefix
            result = subprocess.run([*prefix, *args], capture_output=True, check=True, timeout=timeout)
        except (subprocess.SubprocessError, OSError) as exc:
            raise MeasurementFailure(f"target_command_failed:{self.name}:{type(exc).__name__}") from exc
        return result.stdout

    def sample(self) -> dict:
        data = json.loads(self.command("python", "-c", PROCESS_PROBE, str(self.pid)))
        self.pid = data["pid"]
        return {**self.identity, **data}


@dataclass
class Profile:
    target: Target
    path: str
    process: subprocess.Popen


def select_pods(data: dict, expected_sha: str) -> list[dict]:
    if not re.fullmatch(r"[a-f0-9]{40}", expected_sha):
        raise MeasurementFailure("expected_server_sha_must_be_full_sha")
    pods = []
    for pod in data.get("items", []):
        metadata = pod["metadata"]
        if not metadata["name"].startswith("cowork-server-") or metadata.get("deletionTimestamp"):
            continue
        spec = next((container for container in pod["spec"]["containers"] if container["name"] == "cowork-server"), None)
        statuses = pod.get("status", {}).get("containerStatuses", [])
        status = next((container for container in statuses if container["name"] == "cowork-server"), None)
        if not spec or not status or not status.get("ready") or not status.get("containerID"):
            raise MeasurementFailure("all_server_pods_must_be_ready")
        if not spec["image"].endswith(f":staging-{expected_sha}"):
            raise MeasurementFailure("staging_image_does_not_match_expected_sha")
        pods.append({"name": metadata["name"], "pod_uid": metadata["uid"],
                     "container_id": status["containerID"], "image": spec["image"],
                     "image_id": status.get("imageID"), "resources": spec.get("resources", {})})
    if len(pods) != 2:
        raise MeasurementFailure("expected_exactly_two_ready_staging_server_pods")
    return sorted(pods, key=lambda pod: pod["name"])


def staging_pods(expected_sha: str) -> list[dict]:
    try:
        data = subprocess.run(["kubectl", "-n", "staging", "get", "pods", "-o", "json", "--request-timeout=20s"],
                              capture_output=True, check=True, timeout=30)
        return select_pods(json.loads(data.stdout), expected_sha)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise MeasurementFailure("cannot_read_staging_pod_metadata") from exc


def cpu_delta(before: list[dict], after: list[dict]) -> float:
    first = {sample["name"]: sample for sample in before}
    last = {sample["name"]: sample for sample in after}
    if first.keys() != last.keys():
        raise MeasurementFailure("server_pod_set_changed")
    total = 0.0
    for name, start in first.items():
        end = last[name]
        for key in ("pod_uid", "container_id", "pid", "start_ticks", "boot_id"):
            if start.get(key) != end.get(key):
                raise MeasurementFailure("server_process_lifetime_changed")
        delta = end["cpu_seconds"] - start["cpu_seconds"]
        if delta < 0:
            raise MeasurementFailure("server_cpu_counter_went_backwards")
        total += delta
    return total


def start_profile(target: Target, run_id: str, duration: int) -> Profile:
    target.command("py-spy", "--version", profiler=True)
    # Fails before load if ptrace access or the Python version is unsupported.
    target.command("py-spy", "dump", "--pid", str(target.pid), profiler=True)
    path = f"/tmp/cowork-performance-{run_id}-{target.name}.svg"
    process = subprocess.Popen([
        *(target.profiler_prefix if target.profiler_prefix is not None else target.prefix), "py-spy", "record", "--pid", str(target.pid),
        "--duration", str(duration), "--rate", "99", "--format", "flamegraph", "--output", path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Profile(target, path, process)


def finish_profile(profile: Profile, directory: Path) -> None:
    # The unique output argument identifies only this run's recorder. The
    # duration remains an upper bound if the CI runner loses its connection.
    profile.target.command("python", "-c", STOP_PROFILE, profile.path, profiler=True)
    try:
        code = profile.process.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        profile.process.terminate()
        raise MeasurementFailure("profiler_did_not_stop") from exc
    if code != 0:
        raise MeasurementFailure("profiler_failed")
    content = profile.target.command("python", "-c", READ_PROFILE, profile.path, profiler=True)
    if b"<svg" not in content or b"</svg>" not in content:
        raise MeasurementFailure("profiler_did_not_produce_svg")
    (directory / f"{profile.target.name}.svg").write_bytes(content)


def compare_reports(baseline: dict, candidate: dict, *, scope: str = "server") -> dict:
    for report in (baseline, candidate):
        if report.get("status") != "complete" or report.get("cleanup_errors"):
            raise MeasurementFailure("cannot_compare_incomplete_run")
        if report["summary"]["failed"] or not report["summary"]["completed"]:
            raise MeasurementFailure("cannot_compare_failed_answers")
    for report in (baseline, candidate):
        provenance = report.get("remote_worker_provenance", "").strip()
        if not provenance or re.search(r"\b(unverified|unknown|tbd)\b", provenance, re.IGNORECASE):
            raise MeasurementFailure("remote_worker_provenance_is_unverified")
    if scope not in {"server", "full-stack"}:
        raise MeasurementFailure("invalid_comparison_scope")
    if scope == "server" and baseline["remote_worker_provenance"] != candidate["remote_worker_provenance"]:
        raise MeasurementFailure("remote_workers_differ_use_explicit_full_stack_comparison")
    if baseline.get("profiler_versions") != candidate.get("profiler_versions"):
        raise MeasurementFailure("profiler_versions_differ")
    if baseline["workload_id"] != candidate["workload_id"]:
        raise MeasurementFailure("workloads_differ")
    # Names, UIDs and versions change on a candidate rollout; capacity must not.
    base_resources = [target.get("resources") for target in baseline["cpu_before"]]
    next_resources = [target.get("resources") for target in candidate["cpu_before"]]
    if base_resources != next_resources:
        raise MeasurementFailure("server_capacity_differs")
    before = baseline["cpu_seconds_per_completed_answer"]
    after = candidate["cpu_seconds_per_completed_answer"]
    return {"scope": scope, "baseline_run": baseline["run_id"], "candidate_run": candidate["run_id"],
            "cpu_seconds_per_answer_before": before, "cpu_seconds_per_answer_after": after,
            "cpu_reduction_fraction": 1 - after / before if before > 0 else None,
            "baseline_summary": baseline["summary"], "candidate_summary": candidate["summary"]}


async def run(config: WorkloadConfig, args) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "started_at": datetime.now(timezone.utc).isoformat(),
              "config": config.model_dump(), "workload_id": workload_identity(config),
              "driver_sha": os.environ.get("GITHUB_SHA"), "expected_server_sha": args.server_sha,
              "remote_worker_provenance": args.remote_worker_provenance,
              "debugger_image": args.debugger_image,
              "cpu_scope": "sum of cowork-server process user+system CPU; excludes remote workers, setup and cleanup; includes other traffic and profiler overhead"}
    profiles: list[Profile] = []
    debuggers = []
    workload = None
    try:
        staging = config.base_url == STAGING_URL
        if staging:
            if os.environ.get("GITHUB_ACTIONS") != "true":
                raise MeasurementFailure("staging_profiling_runs_only_in_the_reviewed_CI_workflow")
            if config.total_timeout_seconds > 1800:
                raise MeasurementFailure("staging_workload_deadline_must_not_exceed_1800_seconds")
            required = ["COWORK_PERF_API_KEY", "COWORK_PERF_ORG_ID", "COWORK_PERF_USER_ID"]
            if any(not os.environ.get(name) for name in required):
                raise MeasurementFailure("dedicated_performance_identity_is_missing")
            if not args.remote_worker_provenance.strip() or re.search(r"\b(unverified|unknown|tbd)\b", args.remote_worker_provenance, re.IGNORECASE):
                raise MeasurementFailure("remote_worker_provenance_is_unverified")
            pods = await asyncio.to_thread(staging_pods, args.server_sha)
            targets = [Target(pod["name"], ["kubectl", "-n", "staging", "exec", pod["name"], "-c", "cowork-server", "--"], pod) for pod in pods]
        else:
            if not args.pid:
                raise MeasurementFailure("local_mode_requires_explicit_server_pid")
            if sys.platform != "linux":
                raise MeasurementFailure("process_counter_collection_requires_linux_procfs")
            targets = [Target("local", [], {"name": "local"}, args.pid)]
        headers = {"User-Agent": BROWSER_UA}
        key = os.environ.get("COWORK_PERF_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(base_url=config.base_url, headers=headers, follow_redirects=False, timeout=30) as client:
            if staging:
                await verify_staging_identity(client, expected_org=os.environ["COWORK_PERF_ORG_ID"], expected_user=os.environ["COWORK_PERF_USER_ID"])
            workload = Workload(config, client)
            report["run_id"] = workload.run_id
            try:
                initial = [await asyncio.to_thread(target.sample) for target in targets]
                report["preflight_processes"] = initial
                if staging:
                    from scripts.performance.debugger import Debugger
                    for target, process in zip(targets, initial, strict=True):
                        debugger = Debugger(pod=target.identity, image=args.debugger_image, run_id=workload.run_id)
                        debuggers.append(debugger)
                        await asyncio.to_thread(debugger.start, process)
                        target.profiler_prefix = debugger.prefix
                report["profiler_versions"] = []
                for target in targets:
                    version = (await asyncio.to_thread(target.command, "py-spy", "--version", profiler=True)).decode().strip()
                    if not re.fullmatch(r"py-spy [0-9.]+", version):
                        raise MeasurementFailure("unrecognized_profiler_version")
                    report["profiler_versions"].append(version)
                    await asyncio.to_thread(target.command, "py-spy", "dump", "--pid", str(target.pid), profiler=True)
                async with asyncio.timeout(config.total_timeout_seconds):
                    await workload.prepare()
                    report["history_before_measurement"] = [history.model_dump() for history in workload.history]
                    for target in targets:
                        profiles.append(await asyncio.to_thread(start_profile, target, workload.run_id, config.total_timeout_seconds))
                    await asyncio.sleep(1)
                    if any(profile.process.poll() is not None for profile in profiles):
                        raise MeasurementFailure("profiler_exited_before_workload")
                    report["cpu_before"] = [await asyncio.to_thread(target.sample) for target in targets]
                    samples, elapsed = await workload.measure()
                    report["cpu_after"] = [await asyncio.to_thread(target.sample) for target in targets]
                    if staging and pods != await asyncio.to_thread(staging_pods, args.server_sha):
                        raise MeasurementFailure("server_pod_set_changed")
                    report["cpu_seconds"] = cpu_delta(report["cpu_before"], report["cpu_after"])
                    report["summary"] = summarize(samples, elapsed)
                    completed = report["summary"]["completed"]
                    report["cpu_seconds_per_completed_answer"] = report["cpu_seconds"] / completed if completed else None
                    report["answers"] = [sample.model_dump() for sample in samples]
                    report["history_after_measurement"] = [(await workload.inspect_history(cid)).model_dump() for cid in workload.conversations]
                    report["status"] = "complete" if report["summary"]["failed"] == 0 else "failed"
            finally:
                for profile in profiles:
                    try:
                        await asyncio.to_thread(finish_profile, profile, output)
                    except MeasurementFailure as exc:
                        report.setdefault("profile_errors", []).append(str(exc))
                        report["status"] = "failed"
                await workload.cleanup()
                report["cleanup_errors"] = workload.cleanup_errors
                if workload.cleanup_errors:
                    report["status"] = "failed"
    except (MeasurementFailure, httpx.HTTPError, TimeoutError, ValueError, OSError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc) if isinstance(exc, MeasurementFailure) else type(exc).__name__
    finally:
        for debugger in debuggers:
            try:
                await asyncio.to_thread(debugger.stop)
            except MeasurementFailure as exc:
                report.setdefault("debugger_errors", []).append(str(exc))
                report["status"] = "failed"
        report["debuggers"] = [debugger.provenance() for debugger in debuggers]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-sha", default="")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--debugger-image", default="", help="CI-built profiler image pinned by digest; required for staging")
    parser.add_argument("--remote-worker-provenance", required=True, help="Exact controller/worker image and Anton revision; 'unverified' is not evidence of a matched deployment")
    args = parser.parse_args()
    config = WorkloadConfig.model_validate_json(args.config.read_text())
    report = asyncio.run(run(config, args))
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "error": report.get("error"), "output": args.output}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
