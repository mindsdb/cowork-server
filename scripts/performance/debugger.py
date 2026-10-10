"""Provision bounded, digest-pinned profilers for existing staging API pods."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from scripts.performance.workload import MeasurementFailure


IMAGE_REPOSITORY = "168681354662.dkr.ecr.us-east-1.amazonaws.com/mindsdb-cowork-server"

# A late image pull must not start a new lifetime after the CI run has ended.
# Exiting the container terminates its exec processes too; no signal is sent to
# PID 1 because the debugger shares the serving container's PID namespace.
WATCHDOG = """
import pathlib, sys, time
expires_at = int(sys.argv[1])
stop = pathlib.Path('/tmp/cowork-profiler-stop')
deadline = time.monotonic() + max(0, min(2700, expires_at - time.time()))
while time.monotonic() < deadline and not stop.exists():
    time.sleep(0.2)
"""

PROCESS_IDENTITY = """
import json, pathlib, sys
pid = int(sys.argv[1])
root = pathlib.Path('/proc')
fields = (root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
print(json.dumps({'pid': pid, 'start_ticks': int(fields[19]),
                  'boot_id': (root / 'sys/kernel/random/boot_id').read_text().strip()}))
"""


class PodIdentity(BaseModel):
    name: str = Field(pattern=r"^cowork-server-[a-z0-9]+-[a-z0-9]{5}$")
    pod_uid: str = Field(min_length=1)
    container_id: str = Field(min_length=1)
    image: str


class ProcessIdentity(BaseModel):
    pid: int = Field(strict=True, ge=1)
    start_ticks: int = Field(strict=True, ge=1)
    boot_id: str = Field(min_length=1)


class Debugger:
    """Append to the cleanup list before calling start, including failed starts."""

    def __init__(self, pod: dict, image: str, run_id: str, lifetime_seconds: int = 2700):
        try:
            self.pod = PodIdentity.model_validate(pod)
            suffix = UUID(run_id).hex
        except (ValueError, ValidationError) as exc:
            raise MeasurementFailure("invalid_profiler_target") from exc
        if not re.fullmatch(re.escape(IMAGE_REPOSITORY) + r":staging-[a-f0-9]{40}", self.pod.image):
            raise MeasurementFailure("profiler_target_is_not_a_pinned_staging_image")
        if not re.fullmatch(re.escape(IMAGE_REPOSITORY) + r"@sha256:[a-f0-9]{64}", image):
            raise MeasurementFailure("profiler_image_must_be_a_pinned_repository_digest")
        if type(lifetime_seconds) is not int or not 60 <= lifetime_seconds <= 2700:
            raise MeasurementFailure("invalid_profiler_lifetime")
        self.image = image
        self.container_name = f"answer-profiler-{suffix}"
        self.expires_at = int(time.time()) + lifetime_seconds
        self._requested = False
        self._cleanup_state = "not_started"
        self._image_id = None
        self._spec = {
            "name": self.container_name,
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "targetContainerName": "cowork-server",
            "command": ["python", "-c", WATCHDOG, str(self.expires_at)],
            "securityContext": {
                "runAsUser": 0,
                "runAsNonRoot": False,
                "privileged": False,
                "allowPrivilegeEscalation": False,
                "capabilities": {"drop": ["ALL"], "add": ["SYS_PTRACE"]},
                "seccompProfile": {"type": "RuntimeDefault"},
            },
        }

    @property
    def prefix(self) -> list[str]:
        return ["kubectl", "-n", "staging", "exec", self.pod.name, "-c", self.container_name, "--"]

    def provenance(self) -> dict:
        return {
            "pod_name": self.pod.name,
            "pod_uid": self.pod.pod_uid,
            "container_name": self.container_name,
            "image": self.image,
            "image_id": self._image_id,
            "expires_at": self.expires_at,
            "cleanup_state": self._cleanup_state,
        }

    @staticmethod
    def _command(command: list[str]) -> bytes:
        try:
            return subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError) as exc:
            raise MeasurementFailure("profiler_kubernetes_command_failed") from exc

    def _read_pod(self) -> dict | None:
        raw = self._command([
            "kubectl", "-n", "staging", "get", "pod", self.pod.name,
            "--ignore-not-found", "-o", "json", "--request-timeout=20s",
        ])
        if not raw.strip():
            return None
        try:
            pod = json.loads(raw)
            metadata = pod["metadata"]
            if metadata["namespace"] != "staging" or metadata["name"] != self.pod.name:
                raise MeasurementFailure("profiler_target_metadata_changed")
            return pod
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementFailure("invalid_profiler_pod_response") from exc

    def _owned_status(self, pod: dict) -> dict | None:
        container = next((item for item in pod.get("spec", {}).get("ephemeralContainers", [])
                          if item.get("name") == self.container_name), None)
        if container is None:
            return None
        if any(container.get(key) != value for key, value in self._spec.items()):
            raise MeasurementFailure("profiler_container_ownership_changed")
        return next((item for item in pod.get("status", {}).get("ephemeralContainerStatuses", [])
                     if item.get("name") == self.container_name), {})

    def start(self, expected_process: dict) -> None:
        if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REF") not in {
            "refs/heads/staging", "refs/heads/perf/eng-3362-cpu-per-answer",
        }:
            raise MeasurementFailure("profiler_provisioning_requires_staging_CI")
        try:
            expected = ProcessIdentity.model_validate(expected_process)
        except ValidationError as exc:
            raise MeasurementFailure("invalid_profiler_process_identity") from exc
        pod = self._read_pod()
        if not pod or pod["metadata"].get("uid") != self.pod.pod_uid or pod["metadata"].get("deletionTimestamp"):
            raise MeasurementFailure("profiler_target_pod_changed")
        spec = next((item for item in pod["spec"]["containers"] if item["name"] == "cowork-server"), {})
        status = next((item for item in pod.get("status", {}).get("containerStatuses", [])
                       if item["name"] == "cowork-server"), {})
        if spec.get("image") != self.pod.image or status.get("containerID") != self.pod.container_id or not status.get("ready"):
            raise MeasurementFailure("profiler_target_container_changed")
        if self._owned_status(pod) is not None:
            raise MeasurementFailure("profiler_container_already_exists")
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": self.pod.pod_uid},
            {"op": "test", "path": "/metadata/resourceVersion", "value": pod["metadata"]["resourceVersion"]},
        ]
        if "ephemeralContainers" in pod["spec"]:
            patch.append({"op": "add", "path": "/spec/ephemeralContainers/-", "value": self._spec})
        else:
            patch.append({"op": "add", "path": "/spec/ephemeralContainers", "value": [self._spec]})
        # A timed-out patch may still have reached the API server.
        self._requested = True
        self._cleanup_state = "pending"
        self._command([
            "kubectl", "-n", "staging", "patch", "pod", self.pod.name,
            "--subresource=ephemeralcontainers", "--type=json", "--patch", json.dumps(patch),
            "--request-timeout=20s",
        ])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            pod = self._read_pod()
            if not pod or pod["metadata"].get("uid") != self.pod.pod_uid:
                raise MeasurementFailure("profiler_target_pod_changed")
            status = self._owned_status(pod)
            if status and "running" in status.get("state", {}):
                self._image_id = status.get("imageID", "")
                if self._image_id.removeprefix("docker-pullable://") != self.image:
                    raise MeasurementFailure("profiler_running_image_digest_mismatch")
                break
            if status and "terminated" in status.get("state", {}):
                raise MeasurementFailure("profiler_exited_before_measurement")
            time.sleep(0.5)
        else:
            raise MeasurementFailure("profiler_did_not_start")
        try:
            identity = json.loads(self._command([
                *self.prefix, "python", "-c", PROCESS_IDENTITY, str(expected.pid),
            ]))
            if any(identity[key] != getattr(expected, key) for key in ("pid", "start_ticks", "boot_id")):
                raise MeasurementFailure("profiler_process_namespace_mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementFailure("invalid_profiler_process_identity") from exc

    def stop(self) -> None:
        if not self._requested:
            self._cleanup_state = "not_requested"
            return
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            pod = self._read_pod()
            if not pod or pod["metadata"].get("uid") != self.pod.pod_uid:
                self._cleanup_state = "pod_gone"
                return
            status = self._owned_status(pod)
            if status is None:
                self._cleanup_state = "not_created"
                return
            if "terminated" in status.get("state", {}):
                self._cleanup_state = "terminated"
                return
            if "running" in status.get("state", {}):
                try:
                    self._command([
                        *self.prefix, "python", "-c",
                        "from pathlib import Path; Path('/tmp/cowork-profiler-stop').touch()",
                    ])
                except MeasurementFailure:
                    # The watchdog may exit while exec returns. Only a later
                    # pod read confirming termination makes cleanup succeed.
                    pass
            time.sleep(0.5)
        self._cleanup_state = "termination_unconfirmed"
        raise MeasurementFailure("profiler_termination_unconfirmed")
