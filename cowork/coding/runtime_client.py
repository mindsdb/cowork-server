from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from cowork.coding.contracts import TaskCapability
from cowork.coding.control_models import (
    RUNTIME_PROTOCOL_VERSION,
    ComputerCapabilities,
    RuntimeCommand,
    RuntimeEvent,
)
from cowork.coding.engines.registry import CodingEngineRegistry, engine_registry
from cowork.coding.runtime_protocol import (
    ComputerRegistrationRequest,
    ComputerRegistrationResponse,
    RuntimeLease,
)
from cowork.coding.shells import shell_inventory


_EVENT_DELIVERY_ATTEMPTS = 3
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class RuntimeClientError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RuntimeIdentity:
    computer_id: str
    runtime_token: str
    name: str


@dataclass(frozen=True)
class StoredRuntimeIdentity:
    server_url: str
    identity: RuntimeIdentity


class RemoteRuntimeClient:
    """Authenticated outbound client for the versioned execution protocol."""

    def __init__(
        self,
        server_url: str,
        identity: RuntimeIdentity,
        client: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.identity = identity
        self.client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._sleep = sleep
        self._sequence_lock = threading.Lock()
        self._sequences: dict[str, int] = {}

    @classmethod
    def register(
        cls,
        server_url: str,
        registration_token: str,
        name: str,
        registry: CodingEngineRegistry = engine_registry,
    ) -> RemoteRuntimeClient:
        shells = [item.id.value for item in shell_inventory().items]
        system = platform.system().lower()
        capabilities = ComputerCapabilities(
            platform="darwin" if system == "darwin" else "windows" if system == "windows" else "linux",
            architecture=platform.machine() or "unknown",
            runtime_version="cowork-code-runtime-1",
            agent_engines=registry.available_ids(),
            shells=shells,
            has_git=True,
            has_terminal=True,
            supports_local_folders=True,
            task_capabilities=[
                TaskCapability.review,
                TaskCapability.terminal,
                TaskCapability.project_actions,
                TaskCapability.slash_commands,
            ],
            # One worker process retains one engine/workspace loop at a time.
            # Advertise that truthfully until the runtime supervisor fans out.
            max_concurrent_runs=1,
        )
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{server_url.rstrip('/')}/api/v1/coding/runtime/register",
                json=ComputerRegistrationRequest(
                    registration_token=registration_token,
                    name=name,
                    capabilities=capabilities,
                ).model_dump(mode="json"),
            )
            cls._raise(response)
            registered = ComputerRegistrationResponse.model_validate(response.json())
        return cls(
            server_url,
            RuntimeIdentity(registered.computer.id, registered.runtime_token, registered.computer.name),
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def heartbeat(self, active_run_count: int = 0) -> None:
        response = self.client.post(
            self._url(f"computers/{self.identity.computer_id}/heartbeat"),
            headers=self._headers(),
            json={"protocol_version": RUNTIME_PROTOCOL_VERSION, "active_run_count": active_run_count},
        )
        self._raise(response)

    def lease(self, wait_seconds: float = 0) -> RuntimeLease | None:
        response = self.client.post(
            self._url(f"computers/{self.identity.computer_id}/lease"),
            headers=self._headers(),
            json={"protocol_version": RUNTIME_PROTOCOL_VERSION, "wait_seconds": wait_seconds},
        )
        self._raise(response)
        return RuntimeLease.model_validate(response.json()) if response.json() is not None else None

    def event(
        self,
        lease: RuntimeLease,
        kind: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        with self._sequence_lock:
            sequence = self._sequences.get(lease.run.id, lease.run.last_event_seq) + 1
            event = RuntimeEvent(
                run_id=lease.run.id,
                computer_id=self.identity.computer_id,
                lease_id=lease.lease_id,
                epoch=lease.run.epoch,
                seq=sequence,
                kind=kind,
                payload=payload or {},
            )
            self._deliver(event)
            self._sequences[lease.run.id] = sequence

    def _deliver(self, event: RuntimeEvent) -> None:
        """Redeliver the same event id on transient failures; the control plane deduplicates it."""

        for attempt in range(1, _EVENT_DELIVERY_ATTEMPTS + 1):
            try:
                response = self.client.post(
                    self._url(f"runs/{event.run_id}/events"),
                    headers=self._headers(),
                    json=event.model_dump(mode="json"),
                )
            except httpx.TransportError:
                if attempt == _EVENT_DELIVERY_ATTEMPTS:
                    raise
            else:
                if response.status_code not in _RETRYABLE_STATUSES or attempt == _EVENT_DELIVERY_ATTEMPTS:
                    self._raise(response)
                    return
            self._sleep(0.5 * attempt)

    def commands(self, lease: RuntimeLease) -> list[RuntimeCommand]:
        response = self.client.post(
            self._url(f"runs/{lease.run.id}/commands/claim"),
            params={"computer_id": self.identity.computer_id},
            headers=self._headers(),
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "lease_id": lease.lease_id,
                "epoch": lease.run.epoch,
            },
        )
        self._raise(response)
        return [RuntimeCommand.model_validate(item) for item in response.json().get("items", [])]

    def acknowledge(
        self,
        lease: RuntimeLease,
        command: RuntimeCommand,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        response = self.client.post(
            self._url(f"runs/{lease.run.id}/commands/ack"),
            params={"computer_id": self.identity.computer_id},
            headers=self._headers(),
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "lease_id": lease.lease_id,
                "epoch": lease.run.epoch,
                "command_id": command.id,
                "result": result,
                "error": error,
            },
        )
        self._raise(response)

    def inference_endpoint(self, lease: RuntimeLease) -> str:
        return self._url(f"computers/{self.identity.computer_id}/runs/{lease.run.id}/inference")

    def _url(self, path: str) -> str:
        return f"{self.server_url}/api/v1/coding/runtime/{path}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.identity.runtime_token}"}

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = str(response.json().get("detail") or "")
            except (ValueError, AttributeError):
                pass
            raise RuntimeClientError(
                detail or f"Runtime request failed ({response.status_code})",
                response.status_code,
            ) from exc

def atomic_write(target: Path, contents: str, mode: int = 0o666) -> None:
    """Replace ``target`` through a uniquely named temporary file created with ``mode``."""

    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_runtime_identity(root: Path) -> StoredRuntimeIdentity | None:
    try:
        payload = json.loads((root / "runtime-identity.json").read_text(encoding="utf-8"))
        return StoredRuntimeIdentity(
            server_url=str(payload["server_url"]),
            identity=RuntimeIdentity(
                computer_id=str(payload["computer_id"]),
                runtime_token=str(payload["runtime_token"]),
                name=str(payload["name"]),
            ),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_runtime_identity(root: Path, server_url: str, identity: RuntimeIdentity) -> None:
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "runtime-identity.json", json.dumps({
        "server_url": server_url.rstrip("/"),
        "computer_id": identity.computer_id,
        "runtime_token": identity.runtime_token,
        "name": identity.name,
    }) + "\n", mode=0o600)


class RuntimeLoop(Protocol):
    def run_once(self, wait_seconds: float = 0) -> bool: ...


def run_runtime_forever(
    runtime: RuntimeLoop,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Keep an outbound runtime alive through transient control-plane outages."""

    retry_delay = 1.0
    while True:
        try:
            runtime.run_once(wait_seconds=20)
            retry_delay = 1.0
        except (httpx.TransportError, RuntimeClientError) as exc:
            if isinstance(exc, RuntimeClientError) and exc.status_code not in {429, 500, 502, 503, 504}:
                raise
            print(
                f"MindsHub Code is temporarily unreachable; reconnecting in {retry_delay:g}s "
                f"({exc})",
                file=sys.stderr,
            )
            sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)
