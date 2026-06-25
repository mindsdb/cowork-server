"""Optional shadow backends for artifact versions.

Cowork's database rows and content-addressed blobs remain authoritative.
Shadow backends are best-effort mirrors used for experiments such as Lix:
they must never be required for restore, publish, or preview safety.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowBackendSnapshot:
    artifact_id: UUID
    version_id: UUID
    version_number: int
    artifact_dir: Path
    store_root: Path
    manifest_path: Path
    files_hash: str
    manifest_hash: str


class ArtifactVersionShadowBackend(Protocol):
    name: str

    def snapshot(self, context: ShadowBackendSnapshot) -> dict:
        """Mirror a committed Cowork snapshot and return diagnostics."""


class ManifestShadowBackend:
    """Small deterministic backend used to verify the extension boundary."""

    name = "manifest"

    def snapshot(self, context: ShadowBackendSnapshot) -> dict:
        target = (
            context.store_root
            / "shadow"
            / self.name
            / str(context.artifact_id)
            / f"{context.version_number:06d}-{context.version_id}.json"
        )
        payload = {
            "backend": self.name,
            "artifactId": str(context.artifact_id),
            "versionId": str(context.version_id),
            "versionNumber": context.version_number,
            "filesHash": context.files_hash,
            "manifestHash": context.manifest_hash,
            "manifestPath": context.manifest_path.relative_to(context.store_root).as_posix(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "ok", "path": str(target)}


class LixShadowBackend:
    """Best-effort wrapper for a future Lix file-tree mirror.

    The current Cowork HTML flow still uses Cowork blobs for restore,
    publish, preview, visual diffs, and dataset diffs. This backend only
    calls an optional Node runner when explicitly enabled, so packaged
    builds without Lix remain unaffected.
    """

    name = "lix"

    def __init__(self, runner: Path | None = None) -> None:
        self.runner = runner or Path(__file__).with_name("lix_adapter_runner.mjs")

    def snapshot(self, context: ShadowBackendSnapshot) -> dict:
        if not self.runner.is_file():
            return {"status": "unavailable", "reason": "runner-missing"}
        command = [
            os.environ.get("COWORK_LIX_NODE", "node"),
            str(self.runner),
            "snapshot",
            "--artifact-dir",
            str(context.artifact_dir),
            "--store-root",
            str(context.store_root),
            "--manifest",
            str(context.manifest_path),
            "--artifact-id",
            str(context.artifact_id),
            "--version-id",
            str(context.version_id),
            "--version-number",
            str(context.version_number),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("COWORK_LIX_TIMEOUT_SECONDS", "10")),
            )
        except Exception as exc:
            logger.debug("Lix shadow snapshot failed to start", exc_info=True)
            return {"status": "failed", "reason": str(exc)}

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            return {
                "status": "failed",
                "exitCode": completed.returncode,
                "stderr": stderr[-1000:],
            }
        if not stdout:
            return {"status": "ok"}
        try:
            payload = json.loads(stdout)
            return payload if isinstance(payload, dict) else {"status": "ok", "output": payload}
        except json.JSONDecodeError:
            return {"status": "ok", "stdout": stdout[-1000:]}


def configured_shadow_backends() -> list[ArtifactVersionShadowBackend]:
    raw = os.environ.get("COWORK_ARTIFACT_VERSION_SHADOW_BACKENDS", "").strip()
    if not raw:
        return []
    backends: list[ArtifactVersionShadowBackend] = []
    for name in [part.strip().lower() for part in raw.split(",") if part.strip()]:
        if name == "manifest":
            backends.append(ManifestShadowBackend())
        elif name == "lix":
            backends.append(LixShadowBackend())
        else:
            logger.warning("Unknown artifact version shadow backend: %s", name)
    return backends


def run_shadow_snapshots(context: ShadowBackendSnapshot) -> list[dict]:
    results: list[dict] = []
    for backend in configured_shadow_backends():
        try:
            result = backend.snapshot(context)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("Artifact shadow backend %s failed", backend.name, exc_info=True)
            result = {"status": "failed", "reason": str(exc)}
        results.append({"backend": backend.name, **(result if isinstance(result, dict) else {})})
    return results
