"""Opt-in proof that an HTTP disconnect kills a real deployed worker cell.

Run with the standard post-deploy target/identity inputs plus
``COWORK_RUN_DISCONNECT_WORKER_TEST=true`` and ``COWORK_DISCONNECT_NAMESPACE``.
``COWORK_DISCONNECT_CONTEXT`` optionally selects kubectl's context. The runner
needs pods/list and pods/exec access in that namespace; the observer executes only
reads of the test marker and /proc. It changes no deployment or credentials.
The controller normally deletes the whole worker pod, taking that observer
with it. In that case Kubernetes must confirm the captured pod UID disappeared
or its captured container terminated; a deletion request alone is insufficient.

The model must execute the supplied cell unchanged. A missing cell, observer,
or native run_id fails an opted-in run; none becomes a successful skip. Normal
local and release suites skip before requesting an identity unless opted in.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import BaseModel

from tests.integration.test_post_deploy import (
    TURN_TIMEOUT_S,
    _base_url,
    _headers,
    _provision_identity,
    _sse_events,
)

pytestmark = pytest.mark.postdeploy
STOP_BOUND_SECONDS = 10.0

# A separate process in the SAME scratchpad container reads the real cell's
# heartbeat and checks Linux process identity. PID reuse and a zombie both
# mean the original executing process is gone. Access errors are errors,
# never interpreted as proof that a cell stopped.
_OBSERVE = """\
import json, pathlib, sys, time
prefix, token = sys.argv[1:]
identity = json.loads(pathlib.Path(prefix + '.identity.json').read_text())
assert identity['token'] == token
pid = identity['pid']
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        fields = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        running = fields[0] != 'Z' and fields[19] == identity['start_ticks']
    except FileNotFoundError:
        running = False
    print(json.dumps({
        'pid': pid,
        'running': running,
        'heartbeat_bytes': pathlib.Path(prefix + '.heartbeat').stat().st_size,
        'cell_age_seconds': (time.monotonic_ns() - identity['started_ns']) / 1e9,
    }), flush=True)
    time.sleep(0.1)
"""


class WorkerSample(BaseModel):
    pid: int
    running: bool
    heartbeat_bytes: int
    cell_age_seconds: float


@dataclass(frozen=True)
class WorkerTarget:
    base_url: str
    kubectl: tuple[str, ...]


@dataclass(frozen=True)
class WorkerIdentity:
    pod: str
    uid: str
    container_id: str


@pytest.fixture(scope="module")
def worker_target() -> WorkerTarget:
    if os.environ.get("COWORK_RUN_DISCONNECT_WORKER_TEST") != "true":
        pytest.skip("real worker disconnect proof requires COWORK_RUN_DISCONNECT_WORKER_TEST=true")
    namespace = os.environ.get("COWORK_DISCONNECT_NAMESPACE")
    if not namespace or not os.environ.get("COWORK_BASE_URL"):
        pytest.fail("opted-in worker proof requires COWORK_DISCONNECT_NAMESPACE and COWORK_BASE_URL")
    binary = shutil.which("kubectl")
    if binary is None:
        pytest.fail("opted-in worker proof requires kubectl with pods/exec access")
    command = [binary, "--namespace", namespace]
    if context := os.environ.get("COWORK_DISCONNECT_CONTEXT"):
        command += ["--context", context]
    return WorkerTarget(base_url=_base_url(), kubectl=tuple(command))


@pytest.fixture
def worker_api(worker_target):
    # Resolve only after the explicit execution gate, preserving the existing
    # post-deploy suite's production standing-identity checks.
    try:
        identity = _provision_identity()
    except pytest.skip.Exception as exc:
        pytest.fail(f"opted-in worker proof requires a usable deployment identity: {exc}")
    with httpx.Client(
        base_url=worker_target.base_url, headers=_headers(identity), timeout=30,
    ) as client:
        yield client


def _cell_code(prefix: str, token: str) -> str:
    return (
        "import json, os, pathlib, time\n"
        f"prefix = {prefix!r}\n"
        "pid = os.getpid()\n"
        "started_ns = time.monotonic_ns()\n"
        "start_ticks = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]\n"
        "pathlib.Path(prefix + '.identity.json').write_text(json.dumps({\n"
        f"    'token': {token!r}, 'pid': pid, 'start_ticks': start_ticks, 'started_ns': started_ns,\n"
        "}))\n"
        "heartbeat = pathlib.Path(prefix + '.heartbeat')\n"
        "heartbeat.write_text('started\\n')\n"
        "while time.monotonic_ns() - started_ns < 180_000_000_000:\n"
        "    with heartbeat.open('a') as output:\n"
        "        output.write('alive\\n')\n"
        "    time.sleep(0.1)\n"
    )


def _read_pod(target: WorkerTarget, name: str, *, timeout: float = 5) -> dict | None:
    result = subprocess.run(
        [*target.kubectl, "get", "pod", name, "--ignore-not-found", "-o", "json"],
        check=True, capture_output=True, text=True, timeout=timeout,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _worker_termination(target: WorkerTarget, worker: WorkerIdentity, *, timeout: float) -> str | None:
    pod = _read_pod(target, worker.pod, timeout=timeout)
    if pod is None or pod["metadata"]["uid"] != worker.uid:
        return "captured pod UID deleted"
    for container in pod.get("status", {}).get("containerStatuses", []):
        if container["name"] != "scratchpad":
            continue
        if container.get("containerID") == worker.container_id and container.get("state", {}).get("terminated"):
            return "captured container terminated"
        if container.get("lastState", {}).get("terminated", {}).get("containerID") == worker.container_id:
            return "captured container terminated before restart"
    # Neither deletionTimestamp nor a dropped exec channel proves the running
    # container stopped. Keep waiting for its actual status or UID disappearance.
    return None


def _find_worker(target: WorkerTarget, prefix: str) -> WorkerIdentity:
    """Locate this test's marker without duplicating the controller's pod hash.

    Scratchpad stdout is retained for its eventual tool result, so SSE cannot
    identify the worker while its cell is running. Each probe reads only the
    existence of our random marker, never another conversation's files.
    """
    probe = "import pathlib, sys; print(pathlib.Path(sys.argv[1]).is_file())"

    def has_marker(worker: WorkerIdentity) -> bool:
        result = subprocess.run(
            [*target.kubectl, "exec", worker.pod, "-c", "scratchpad", "--",
             "python", "-c", probe, prefix + ".identity.json"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if result.returncode:
            current = _read_pod(target, worker.pod)
            if (current is None or current["metadata"]["uid"] != worker.uid
                    or current["metadata"].get("deletionTimestamp")):
                # Another conversation's idle worker can disappear during the
                # scan. This only skips a candidate; it is not stop evidence.
                return False
            result.check_returncode()
        assert result.stdout.strip() in {"True", "False"}, "unexpected marker-probe output"
        return result.stdout.strip() == "True"

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = subprocess.run(
            [*target.kubectl, "get", "pods", "-l", "scratchpad.mindsdb.com/type=live",
             "--field-selector=status.phase=Running", "-o", "json"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        workers = []
        for item in json.loads(result.stdout)["items"]:
            if item["metadata"].get("deletionTimestamp"):
                continue
            for container in item.get("status", {}).get("containerStatuses", []):
                if container["name"] == "scratchpad" and "running" in container.get("state", {}):
                    workers.append(WorkerIdentity(
                        item["metadata"]["name"], item["metadata"]["uid"], container["containerID"],
                    ))
        with ThreadPoolExecutor(max_workers=8) as pool:
            matches = [worker for worker, found in zip(workers, pool.map(has_marker, workers), strict=True) if found]
        assert len(matches) <= 1, "the test marker appeared in multiple worker containers"
        if matches:
            assert _worker_termination(target, matches[0], timeout=5) is None, "worker ended during discovery"
            return matches[0]
        time.sleep(0.25)
    pytest.fail("the requested cell never wrote its marker in a real scratchpad worker")


@contextmanager
def _observe_worker(target: WorkerTarget, pod: str, prefix: str, token: str):
    command = [*target.kubectl, "exec", pod, "-c", "scratchpad", "--",
               "python", "-u", "-c", _OBSERVE, prefix, token]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    samples: queue.Queue[tuple[float, WorkerSample] | BaseException] = queue.Queue()

    def read_samples():
        try:
            for line in process.stdout:
                samples.put((time.monotonic(), WorkerSample.model_validate_json(line)))
            status = process.wait(timeout=5)
            error = process.stderr.read().decode(errors="replace")[:500]
            samples.put(RuntimeError(f"worker observer exited ({status}): {error}"))
        except Exception as exc:
            samples.put(exc)

    reader = threading.Thread(target=read_samples, daemon=True)
    reader.start()
    try:
        yield samples
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=2)
        process.stdout.close()
        process.stderr.close()


def _next_sample(samples, *, deadline: float) -> tuple[float, WorkerSample]:
    try:
        item = samples.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
        pytest.fail("worker process observer did not report within the required bound")
    if isinstance(item, BaseException):
        raise AssertionError("could not observe the real worker process") from item
    return item


def _await_worker_stop(samples, target: WorkerTarget, worker: WorkerIdentity, *, pid: int, deadline: float) -> str:
    """Require execution evidence independent of the server's terminal record."""
    stopped: WorkerSample | None = None
    stopped_at: float | None = None
    next_pod_check = time.monotonic()
    while time.monotonic() < deadline:
        try:
            item = samples.get(timeout=min(0.1, max(0, deadline - time.monotonic())))
        except queue.Empty:
            item = None
        if isinstance(item, tuple):
            received_at, sample = item
            assert sample.pid == pid
            if stopped is not None:
                assert not sample.running and sample.heartbeat_bytes == stopped.heartbeat_bytes
                if received_at - stopped_at >= 0.5:
                    assert received_at <= deadline
                    return "process exit and stable heartbeat"
            elif not sample.running:
                stopped, stopped_at = sample, received_at
        # An observer EOF/error is not itself evidence. The controller kills
        # this observer together with the pod, so resolve the captured identity
        # through the independent Kubernetes API instead of failing on EOF.
        if isinstance(item, BaseException) or time.monotonic() >= next_pod_check:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            evidence = _worker_termination(target, worker, timeout=min(2, remaining))
            assert time.monotonic() <= deadline, "worker termination was not observed within ten seconds"
            if evidence is not None:
                return evidence
            next_pod_check = time.monotonic() + 0.5
    pytest.fail("neither process exit nor captured worker termination was confirmed within ten seconds")


def _events(response: httpx.Response) -> Iterator[dict]:
    for line in response.iter_lines():
        if line.startswith("data:"):
            yield json.loads(line.removeprefix("data:").strip())


def test_opted_in_disconnect_stops_the_executing_remote_worker_cell(worker_target, worker_api, record_property):
    api = worker_api
    conversation_id, token = str(uuid4()), uuid4().hex
    prefix = f"/tmp/cowork-disconnect-{token}"
    code = _cell_code(prefix, token)
    prompt = (
        "This is an integration test of cancelling a running scratchpad cell. "
        "Call scratchpad exec exactly once with the Python code below unchanged. "
        "Run it in the foreground; do not launch a backend, a thread, or a background job. "
        "Do not shorten the loop, add a timeout, or run another cell. "
        "The test client will cancel it after observing its heartbeat.\n\n```python\n"
        + code + "```"
    )
    run_id = None
    proved = False
    try:
        with api.stream(
            "POST", "/api/v1/responses/",
            json={"input": prompt, "conversation": conversation_id, "stream": True,
                  "cancel_on_disconnect": True},
            timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10),
        ) as response:
            assert response.status_code == 200, f"question rejected: HTTP {response.status_code}"
            exact_cell = False
            events = _events(response)
            for event in events:
                if event.get("type") == "response.created":
                    run_id = str(UUID(event["run_id"]))
                if event.get("thought_role") == "thought.scratchpad.end":
                    arguments = json.loads(event["content"])
                    if arguments.get("action") == "exec":
                        assert arguments.get("code", "").strip() == code.strip(), (
                            "the model changed the requested instrumentation cell"
                        )
                        exact_cell = True
                        break
                if event.get("type") in {"response.completed", "response.cancelled", "response.failed"}:
                    pytest.fail("turn ended before the real worker cell could be observed executing")
            assert exact_cell, "the model did not execute the requested instrumentation cell unchanged"
            assert run_id is not None, "no native run_id was emitted"
            worker = _find_worker(worker_target, prefix)

            with _observe_worker(worker_target, worker.pod, prefix, token) as samples:
                premise_deadline = time.monotonic() + 10
                _, first = _next_sample(samples, deadline=premise_deadline)
                assert first.running, "the cell was already stopped before the test disconnected"
                while True:
                    _, live = _next_sample(samples, deadline=premise_deadline)
                    assert live.pid == first.pid and live.running
                    if live.heartbeat_bytes > first.heartbeat_bytes:
                        break
                # The bounded cell cannot finish naturally during the ten
                # seconds being measured: at least 150 seconds remain.
                assert live.cell_age_seconds < 30, "cell startup observation arrived too late"
                assert _worker_termination(worker_target, worker, timeout=5) is None
                disconnected_at = time.monotonic()
                response.close()
                deadline = disconnected_at + STOP_BOUND_SECONDS
                evidence = _await_worker_stop(samples, worker_target, worker, pid=first.pid, deadline=deadline)
                record_property("worker_stop_evidence", evidence)
                record_property("worker_pod_uid", worker.uid)
                record_property("worker_container_id", worker.container_id)

        replay = api.get(
            "/api/v1/responses/tail", params={"conversation_id": conversation_id}, timeout=30,
        )
        assert replay.status_code == 200
        terminal_events = _sse_events(replay.text)
        assert "response.cancelled" in terminal_events
        assert "response.completed" not in terminal_events
        proved = True
    finally:
        # Failure cleanup follows all process assertions and cannot make them
        # pass. The immutable ID confines it to this test's original turn.
        if not proved and run_id is not None:
            try:
                cleanup = api.post("/api/v1/responses/cancel", json={
                    "conversation_id": conversation_id, "expected_run_id": run_id,
                }, timeout=15)
                if cleanup.status_code not in {200, 404}:
                    warnings.warn(f"guarded cell cleanup returned HTTP {cleanup.status_code}", RuntimeWarning)
            except httpx.HTTPError:
                warnings.warn("guarded cell cleanup was unreachable; the cell has a 180-second bound", RuntimeWarning)
