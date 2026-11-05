"""The live worker proof must distinguish a delete request from a stopped worker."""
from __future__ import annotations

import json
import queue
import subprocess
import time
from copy import deepcopy

import pytest

from tests.integration import test_disconnect_worker as proof

TARGET = proof.WorkerTarget("https://cowork.example", ("kubectl", "-n", "test"))
WORKER = proof.WorkerIdentity("sp-live-test", "original-uid", "containerd://original")
RUNNING = {
    "metadata": {"name": WORKER.pod, "uid": WORKER.uid},
    "status": {"containerStatuses": [{
        "name": "scratchpad", "containerID": WORKER.container_id,
        "state": {"running": {"startedAt": "2026-10-14T00:00:00Z"}},
    }]},
}


@pytest.mark.parametrize("state", ["running", "delete_requested", "other_container_terminated"])
def test_live_or_merely_terminating_pod_is_not_stop_evidence(monkeypatch, state):
    pod = deepcopy(RUNNING)
    if state == "delete_requested":
        pod["metadata"]["deletionTimestamp"] = "2026-10-14T00:01:00Z"
    elif state == "other_container_terminated":
        pod["status"]["containerStatuses"][0]["lastState"] = {
            "terminated": {"containerID": "containerd://unrelated", "exitCode": 137},
        }
    monkeypatch.setattr(proof, "_read_pod", lambda *_args, **_kwargs: pod)
    assert proof._worker_termination(TARGET, WORKER, timeout=1) is None


@pytest.mark.parametrize("outcome", ["deleted", "replaced", "terminated", "restarted"])
def test_confirmed_original_identity_termination_is_stop_evidence(monkeypatch, outcome):
    pod = deepcopy(RUNNING)
    if outcome == "deleted":
        pod = None
    elif outcome == "replaced":
        pod["metadata"]["uid"] = "replacement-uid"
    elif outcome == "terminated":
        pod["status"]["containerStatuses"][0]["state"] = {"terminated": {"exitCode": 137}}
    else:
        container = pod["status"]["containerStatuses"][0]
        container["containerID"] = "containerd://replacement"
        container["lastState"] = {"terminated": {"containerID": WORKER.container_id, "exitCode": 137}}
    monkeypatch.setattr(proof, "_read_pod", lambda *_args, **_kwargs: pod)
    assert proof._worker_termination(TARGET, WORKER, timeout=1) is not None


def test_observer_eof_requires_independent_termination_evidence(monkeypatch):
    samples = queue.Queue()
    samples.put(RuntimeError("exec channel closed"))
    terminating = deepcopy(RUNNING)
    terminating["metadata"]["deletionTimestamp"] = "2026-10-14T00:01:00Z"
    monkeypatch.setattr(proof, "_read_pod", lambda *_args, **_kwargs: terminating)
    with pytest.raises(pytest.fail.Exception, match="neither process exit"):
        proof._await_worker_stop(samples, TARGET, WORKER, pid=123, deadline=time.monotonic() + 0.05)


def test_observer_eof_accepts_confirmed_pod_uid_deletion(monkeypatch):
    samples = queue.Queue()
    samples.put(RuntimeError("exec channel closed"))
    monkeypatch.setattr(proof, "_read_pod", lambda *_args, **_kwargs: None)
    evidence = proof._await_worker_stop(samples, TARGET, WORKER, pid=123, deadline=time.monotonic() + 1)
    assert evidence == "captured pod UID deleted"


def test_discovery_retries_when_an_unrelated_pod_disappears_during_exec(monkeypatch):
    listed = 0

    def run(command, **kwargs):
        nonlocal listed
        if "get" in command and "pods" in command:
            listed += 1
            return subprocess.CompletedProcess(command, 0, json.dumps({"items": [RUNNING]}), "")
        assert "exec" in command
        if listed == 1:
            return subprocess.CompletedProcess(command, 1, "", "pod disappeared")
        return subprocess.CompletedProcess(command, 0, "True\n", "")

    monkeypatch.setattr(proof.subprocess, "run", run)
    monkeypatch.setattr(proof, "_read_pod", lambda *_args, **_kwargs: None if listed == 1 else RUNNING)
    monkeypatch.setattr(proof.time, "sleep", lambda _delay: None)
    assert proof._find_worker(TARGET, "/tmp/test-marker") == WORKER
    assert listed == 2
