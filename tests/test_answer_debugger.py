"""The staging profiler only modifies its selected ephemeral container."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest
import yaml

from scripts.performance import debugger
from scripts.performance.debugger import Debugger, IMAGE_REPOSITORY
from scripts.performance.workload import MeasurementFailure


SERVER_SHA = "a" * 40
IMAGE = IMAGE_REPOSITORY + "@sha256:" + "b" * 64
PROCESS = {"pid": 1, "start_ticks": 100, "boot_id": "boot-one"}
POD = {
    "name": "cowork-server-76ccd8b8bf-abcde",
    "pod_uid": "pod-one",
    "container_id": "containerd://server-one",
    "image": IMAGE_REPOSITORY + ":staging-" + SERVER_SHA,
}


class Cluster:
    def __init__(self):
        self.pod = {
            "metadata": {"name": POD["name"], "namespace": "staging", "uid": POD["pod_uid"], "resourceVersion": "42"},
            "spec": {"containers": [{"name": "cowork-server", "image": POD["image"]}]},
            "status": {"containerStatuses": [{"name": "cowork-server", "containerID": POD["container_id"], "ready": True}]},
        }
        self.commands = []
        self.patch = None
        self.identity = deepcopy(PROCESS)
        self.image_id = "docker-pullable://" + IMAGE
        self.patch_error = False
        self.waiting = False
        self.stop_sent = False
        self.stop_error = False

    def run(self, args, **kwargs):
        assert kwargs == {"capture_output": True, "check": True, "timeout": 30}
        assert args[:3] == ["kubectl", "-n", "staging"]
        self.commands.append(args)
        if args[3] == "get":
            result = json.dumps(self.pod).encode() if self.pod else b""
        elif args[3] == "patch":
            assert args[4:6] == ["pod", POD["name"]]
            assert "--subresource=ephemeralcontainers" in args
            self.patch = json.loads(args[args.index("--patch") + 1])
            for item in self.patch:
                if item["op"] == "test":
                    key = item["path"].split("/")[-1]
                    assert self.pod["metadata"][key] == item["value"]
            last = self.patch[-1]
            if last["path"].endswith("/-"):
                self.pod["spec"]["ephemeralContainers"].append(last["value"])
            else:
                self.pod["spec"]["ephemeralContainers"] = last["value"]
            self.pod["status"]["ephemeralContainerStatuses"] = [{
                "name": self.pod["spec"]["ephemeralContainers"][-1]["name"],
                "imageID": self.image_id,
                "state": {"waiting" if self.waiting else "running": {}},
            }]
            if self.patch_error:
                raise subprocess.TimeoutExpired(args, 30)
            result = b""
        elif args[3] == "exec":
            assert args[4] == POD["name"]
            assert args[5] == "-c" and args[6].startswith("answer-profiler-")
            assert args[7:10] == ["--", "python", "-c"]
            if args[10] == debugger.PROCESS_IDENTITY:
                assert args[11] == str(PROCESS["pid"])
                result = json.dumps(self.identity).encode()
            else:
                assert args[10] == "from pathlib import Path; Path('/tmp/cowork-profiler-stop').touch()"
                self.stop_sent = True
                self.pod["status"]["ephemeralContainerStatuses"][0]["state"] = {"terminated": {"exitCode": 0}}
                if self.stop_error:
                    raise subprocess.CalledProcessError(137, args)
                result = b""
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, result, b"")


@pytest.fixture
def cluster(monkeypatch):
    cluster = Cluster()
    monkeypatch.setattr(debugger.subprocess, "run", cluster.run)
    monkeypatch.setattr(debugger.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/staging")
    return cluster


def make_debugger(**kwargs):
    return Debugger(kwargs.pop("pod", deepcopy(POD)), kwargs.pop("image", IMAGE),
                    kwargs.pop("run_id", str(uuid4())), **kwargs)


def test_provisions_only_pinned_target_and_terminates_it(cluster):
    recorder = make_debugger()
    recorder.start(PROCESS)
    assert cluster.patch[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": POD["pod_uid"]},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "42"},
    ]
    spec = cluster.pod["spec"]["ephemeralContainers"][0]
    assert spec["image"] == IMAGE
    assert spec["targetContainerName"] == "cowork-server"
    assert set(spec) == {"name", "image", "imagePullPolicy", "targetContainerName", "command", "securityContext"}
    assert spec["securityContext"] == {
        "runAsUser": 0, "runAsNonRoot": False, "privileged": False,
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"], "add": ["SYS_PTRACE"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert spec["command"] == ["python", "-c", debugger.WATCHDOG, str(recorder.expires_at)]
    assert recorder.prefix == ["kubectl", "-n", "staging", "exec", POD["name"], "-c", recorder.container_name, "--"]
    recorder.stop()
    assert cluster.stop_sent
    assert recorder.provenance()["cleanup_state"] == "terminated"
    assert recorder.provenance()["image_id"] == "docker-pullable://" + IMAGE


def test_preserves_other_ephemeral_containers(cluster):
    other = {"name": "another-debugger", "image": "unrelated"}
    cluster.pod["spec"]["ephemeralContainers"] = [other]
    recorder = make_debugger()
    recorder.start(PROCESS)
    assert cluster.patch[-1]["path"] == "/spec/ephemeralContainers/-"
    assert cluster.pod["spec"]["ephemeralContainers"][0] == other
    recorder.stop()


@pytest.mark.parametrize("change", [
    {"name": "kube-proxy-abcde"}, {"name": "auth-76ccd8b8bf-abcde"},
    {"image": IMAGE_REPOSITORY + ":production-" + SERVER_SHA},
    {"image": IMAGE_REPOSITORY + ":staging"},
    {"image": "untrusted.invalid/cowork-server:staging-" + SERVER_SHA},
])
def test_rejects_non_server_or_unpinned_target_without_commands(cluster, change):
    with pytest.raises(MeasurementFailure):
        make_debugger(pod={**POD, **change})
    assert cluster.commands == []


@pytest.mark.parametrize("image", [IMAGE_REPOSITORY + ":profiler-latest", "untrusted.invalid/image@sha256:" + "a" * 64])
def test_rejects_unverified_profiler_image(cluster, image):
    with pytest.raises(MeasurementFailure, match="pinned_repository_digest"):
        make_debugger(image=image)
    assert cluster.commands == []


@pytest.mark.parametrize("lifetime", [0, 59, 2701, True, 1.5])
def test_rejects_unbounded_or_invalid_lifetime(cluster, lifetime):
    with pytest.raises(MeasurementFailure, match="invalid_profiler_lifetime"):
        make_debugger(lifetime_seconds=lifetime)
    assert cluster.commands == []


@pytest.mark.parametrize("variable,value", [("GITHUB_ACTIONS", "false"), ("GITHUB_REF", "refs/heads/main"),
                                           ("GITHUB_REF", "refs/heads/other-feature")])
def test_only_staging_ci_may_provision(cluster, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    recorder = make_debugger()
    with pytest.raises(MeasurementFailure, match="requires_staging_CI"):
        recorder.start(PROCESS)
    recorder.stop()
    assert cluster.commands == []


@pytest.mark.parametrize("field,value", [("namespace", "prod"), ("uid", "replacement-pod")])
def test_changed_pod_is_not_patched(cluster, field, value):
    cluster.pod["metadata"][field] = value
    with pytest.raises(MeasurementFailure):
        make_debugger().start(PROCESS)
    assert all(command[3] == "get" for command in cluster.commands)


@pytest.mark.parametrize("key,value", [("ready", False), ("containerID", "containerd://replacement")])
def test_restarted_or_unready_server_is_not_patched(cluster, key, value):
    cluster.pod["status"]["containerStatuses"][0][key] = value
    with pytest.raises(MeasurementFailure, match="target_container_changed"):
        make_debugger().start(PROCESS)
    assert all(command[3] == "get" for command in cluster.commands)


def test_bad_process_input_is_rejected_before_patch(cluster):
    with pytest.raises(MeasurementFailure, match="invalid_profiler_process_identity"):
        make_debugger().start({**PROCESS, "pid": 0})
    assert cluster.commands == []


@pytest.mark.parametrize("field,value", [("pid", 2), ("start_ticks", 101), ("boot_id", "other-node")])
def test_isolated_or_reused_pid_namespace_fails_and_stops(cluster, field, value):
    cluster.identity[field] = value
    recorder = make_debugger()
    with pytest.raises(MeasurementFailure, match="process_namespace_mismatch"):
        recorder.start(PROCESS)
    recorder.stop()
    assert cluster.stop_sent


def test_wrong_running_digest_fails_and_stops(cluster):
    cluster.image_id = IMAGE_REPOSITORY + "@sha256:" + "c" * 64
    recorder = make_debugger()
    with pytest.raises(MeasurementFailure, match="digest_mismatch"):
        recorder.start(PROCESS)
    recorder.stop()
    assert cluster.stop_sent


def test_patch_timeout_still_cleans_up_owned_container(cluster):
    cluster.patch_error = True
    recorder = make_debugger()
    with pytest.raises(MeasurementFailure, match="kubernetes_command_failed"):
        recorder.start(PROCESS)
    recorder.stop()
    assert cluster.stop_sent


@pytest.mark.parametrize("state", ["gone", "replaced", "absent"])
def test_failed_start_cleanup_does_not_target_replacement_or_missing_container(cluster, state):
    recorder = make_debugger()
    recorder.start(PROCESS)
    cluster.commands.clear()
    if state == "gone":
        cluster.pod = None
    elif state == "replaced":
        cluster.pod["metadata"]["uid"] = "new-pod"
    else:
        cluster.pod["spec"].pop("ephemeralContainers")
    recorder.stop()
    assert all(command[3] == "get" for command in cluster.commands)
    assert recorder.provenance()["cleanup_state"] in {"pod_gone", "not_created"}


def test_cleanup_refuses_name_reused_by_another_container(cluster):
    recorder = make_debugger()
    recorder.start(PROCESS)
    cluster.pod["spec"]["ephemeralContainers"][0]["image"] = "other-image"
    cluster.commands.clear()
    with pytest.raises(MeasurementFailure, match="ownership_changed"):
        recorder.stop()
    assert all(command[3] == "get" for command in cluster.commands)


def test_pending_image_cleanup_is_reported_and_keeps_absolute_expiry(cluster, monkeypatch):
    cluster.waiting = True
    cluster.patch_error = True
    recorder = make_debugger()
    with pytest.raises(MeasurementFailure):
        recorder.start(PROCESS)
    moments = iter([0, 0, 31])
    monkeypatch.setattr(debugger.time, "monotonic", lambda: next(moments))
    with pytest.raises(MeasurementFailure, match="termination_unconfirmed"):
        recorder.stop()
    assert recorder.provenance()["cleanup_state"] == "termination_unconfirmed"
    assert cluster.pod["spec"]["ephemeralContainers"][0]["command"][-1] == str(recorder.expires_at)
    assert not cluster.stop_sent


def test_watchdog_exits_immediately_when_image_starts_after_its_deadline():
    result = subprocess.run(["python3", "-c", debugger.WATCHDOG, "1"], capture_output=True, timeout=2)
    assert result.returncode == 0
    assert result.stdout == b""


def test_standalone_rbac_has_only_staging_subresources_and_no_default_chart_hook():
    root = Path(__file__).resolve().parents[1]
    path = root / "deployment/performance/staging-profiler-rbac.yaml"
    role, binding = list(yaml.safe_load_all(path.read_text()))
    assert role["kind"] == "Role" and binding["kind"] == "RoleBinding"
    assert role["metadata"]["namespace"] == binding["metadata"]["namespace"] == "staging"
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
        {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["create"]},
        {"apiGroups": [""], "resources": ["pods/ephemeralcontainers"], "verbs": ["patch"]},
    ]
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "newdev-gha-runner", "namespace": "infrastructure"}]
    assert binding["roleRef"] == {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": role["metadata"]["name"]}


def test_reviewed_baseline_branch_may_provision(cluster, monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/perf/eng-3362-cpu-per-answer")
    recorder = make_debugger()
    recorder.start(PROCESS)
    recorder.stop()
    assert recorder.provenance()["cleanup_state"] == "terminated"


def test_cleanup_accepts_exit_racing_with_sentinel_exec(cluster):
    recorder = make_debugger()
    recorder.start(PROCESS)
    cluster.stop_error = True
    recorder.stop()
    assert recorder.provenance()["cleanup_state"] == "terminated"
