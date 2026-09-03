from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol, TypeVar

from pydantic import BaseModel

from cowork.coding.contracts import utc_now
from cowork.coding.control_errors import StateConflict
from cowork.coding.control_models import (
    CodeTask,
    Computer,
    ConnectorGrant,
    ExecutionWorkspace,
    RunStatus,
    RuntimeCommand,
    RuntimeCredential,
    RuntimeRegistrationCredential,
    SecurityAuditEvent,
    TaskRun,
    TaskRunCredential,
)
from cowork.coding.run_state import transition_run

Document = TypeVar("Document", bound=BaseModel)


class ControlPlaneStore(Protocol):
    def save_computer(self, computer: Computer) -> Computer: ...
    def get_computer(self, computer_id: str) -> Computer: ...
    def list_computers(self) -> list[Computer]: ...
    def save_task(self, task: CodeTask) -> CodeTask: ...
    def get_task(self, task_id: str) -> CodeTask: ...
    def list_tasks(self) -> list[CodeTask]: ...
    def save_run(self, run: TaskRun) -> TaskRun: ...
    def get_run(self, run_id: str) -> TaskRun: ...
    def list_runs(self) -> list[TaskRun]: ...
    def save_workspace(self, workspace: ExecutionWorkspace) -> ExecutionWorkspace: ...
    def list_workspaces(self, run_id: str | None = None) -> list[ExecutionWorkspace]: ...
    def save_grant(self, grant: ConnectorGrant) -> ConnectorGrant: ...
    def get_grant(self, grant_id: str) -> ConnectorGrant: ...
    def list_grants(self, run_id: str | None = None) -> list[ConnectorGrant]: ...
    def update_grant(
        self,
        grant_id: str,
        operation: Callable[[ConnectorGrant, TaskRun], None],
    ) -> ConnectorGrant: ...
    def save_command(self, command: RuntimeCommand) -> RuntimeCommand: ...
    def get_command(self, command_id: str) -> RuntimeCommand: ...
    def list_commands(self, run_id: str | None = None) -> list[RuntimeCommand]: ...
    def save_runtime_credential(self, credential: RuntimeCredential) -> RuntimeCredential: ...
    def get_runtime_credential(self, computer_id: str) -> RuntimeCredential: ...
    def save_registration_credential(self, credential: RuntimeRegistrationCredential) -> RuntimeRegistrationCredential: ...
    def consume_registration_credential(self, token_hash: str, now: datetime) -> RuntimeRegistrationCredential | None: ...
    def list_registration_credentials(self) -> list[RuntimeRegistrationCredential]: ...
    def delete_registration_credential(self, credential_id: str) -> None: ...
    def save_run_credential(self, credential: TaskRunCredential) -> TaskRunCredential: ...
    def get_run_credential(self, run_id: str) -> TaskRunCredential: ...
    def save_audit_event(self, event: SecurityAuditEvent) -> SecurityAuditEvent: ...
    def list_audit_events(self, run_id: str | None = None) -> list[SecurityAuditEvent]: ...
    def create_task_run(
        self,
        task: CodeTask,
        run: TaskRun,
        workspaces: list[ExecutionWorkspace],
    ) -> tuple[CodeTask, TaskRun, list[ExecutionWorkspace]]: ...
    def claim_run(self, computer_id: str, lease_id: str, lease_expires_at: datetime) -> TaskRun | None: ...
    def update_run(self, run_id: str, operation: Callable[[TaskRun], None]) -> TaskRun: ...
    def delete_task(self, task_id: str) -> None: ...
    def prune(self, auxiliary_older_than: datetime, audit_older_than: datetime) -> int: ...


class LocalControlPlaneStore:
    """Atomic local implementation of the control-plane storage contract.

    The interface deliberately mirrors records a future shared database owns;
    filesystem paths and processes stay in ExecutionWorkspace records instead.
    """

    _MODELS: ClassVar[dict[str, type[BaseModel]]] = {
        "computers": Computer,
        "tasks": CodeTask,
        "runs": TaskRun,
        "workspaces": ExecutionWorkspace,
        "grants": ConnectorGrant,
        "commands": RuntimeCommand,
        "runtime_credentials": RuntimeCredential,
        "registration_credentials": RuntimeRegistrationCredential,
        "run_credentials": TaskRunCredential,
        "audit": SecurityAuditEvent,
    }

    def __init__(self, root: Path) -> None:
        self.root = root / "control"
        self._lock = threading.RLock()
        for collection in self._MODELS:
            (self.root / collection).mkdir(parents=True, exist_ok=True)
        self._recover_transaction()
        self._parent_index: dict[str, dict[str, set[str]]] = {
            collection: {} for collection in ("runs", "workspaces", "grants", "commands", "run_credentials", "audit")
        }
        self._rebuild_parent_index()

    def save_computer(self, computer: Computer) -> Computer:
        computer.updated_at = utc_now()
        return self._save("computers", computer)

    def get_computer(self, computer_id: str) -> Computer:
        return self._get("computers", computer_id, Computer)

    def list_computers(self) -> list[Computer]:
        return self._list("computers", Computer)

    def save_task(self, task: CodeTask) -> CodeTask:
        task.updated_at = utc_now()
        return self._save("tasks", task)

    def get_task(self, task_id: str) -> CodeTask:
        return self._get("tasks", task_id, CodeTask)

    def list_tasks(self) -> list[CodeTask]:
        return self._list("tasks", CodeTask)

    def save_run(self, run: TaskRun) -> TaskRun:
        run.updated_at = utc_now()
        return self._save("runs", run)

    def get_run(self, run_id: str) -> TaskRun:
        return self._get("runs", run_id, TaskRun)

    def list_runs(self) -> list[TaskRun]:
        return self._list("runs", TaskRun)

    def save_workspace(self, workspace: ExecutionWorkspace) -> ExecutionWorkspace:
        workspace.updated_at = utc_now()
        return self._save("workspaces", workspace)

    def list_workspaces(self, run_id: str | None = None) -> list[ExecutionWorkspace]:
        if run_id is not None:
            return self._list_for_parent("workspaces", run_id, ExecutionWorkspace)
        return self._list("workspaces", ExecutionWorkspace)

    def save_grant(self, grant: ConnectorGrant) -> ConnectorGrant:
        return self._save("grants", grant)

    def get_grant(self, grant_id: str) -> ConnectorGrant:
        return self._get("grants", grant_id, ConnectorGrant)

    def list_grants(self, run_id: str | None = None) -> list[ConnectorGrant]:
        if run_id is not None:
            return self._list_for_parent("grants", run_id, ConnectorGrant)
        return self._list("grants", ConnectorGrant)

    def update_grant(
        self,
        grant_id: str,
        operation: Callable[[ConnectorGrant, TaskRun], None],
    ) -> ConnectorGrant:
        """Validate the grant and its current run under one local lock."""

        with self._lock:
            grant = self.get_grant(grant_id)
            run = self.get_run(grant.run_id)
            operation(grant, run)
            return self.save_grant(grant)

    def save_command(self, command: RuntimeCommand) -> RuntimeCommand:
        return self._save("commands", command)

    def get_command(self, command_id: str) -> RuntimeCommand:
        return self._get("commands", command_id, RuntimeCommand)

    def list_commands(self, run_id: str | None = None) -> list[RuntimeCommand]:
        if run_id is not None:
            return self._list_for_parent("commands", run_id, RuntimeCommand)
        return self._list("commands", RuntimeCommand)

    def save_runtime_credential(self, credential: RuntimeCredential) -> RuntimeCredential:
        return self._save("runtime_credentials", credential)

    def get_runtime_credential(self, computer_id: str) -> RuntimeCredential:
        return self._get("runtime_credentials", computer_id, RuntimeCredential)

    def list_registration_credentials(self) -> list[RuntimeRegistrationCredential]:
        return self._list("registration_credentials", RuntimeRegistrationCredential)

    def delete_registration_credential(self, credential_id: str) -> None:
        self._delete("registration_credentials", credential_id)

    def save_registration_credential(self, credential: RuntimeRegistrationCredential) -> RuntimeRegistrationCredential:
        return self._save("registration_credentials", credential)

    def consume_registration_credential(self, token_hash: str, now: datetime) -> RuntimeRegistrationCredential | None:
        with self._lock:
            try:
                credential = self._get("registration_credentials", token_hash, RuntimeRegistrationCredential)
            except KeyError:
                return None
            if credential.consumed_at is not None or credential.expires_at <= now:
                return None
            credential.consumed_at = now
            self._save("registration_credentials", credential)
            return credential

    def save_run_credential(self, credential: TaskRunCredential) -> TaskRunCredential:
        return self._save("run_credentials", credential)

    def get_run_credential(self, run_id: str) -> TaskRunCredential:
        return self._get("run_credentials", run_id, TaskRunCredential)

    def save_audit_event(self, event: SecurityAuditEvent) -> SecurityAuditEvent:
        with self._lock:
            try:
                self._get("audit", event.id, SecurityAuditEvent)
            except KeyError:
                return self._save("audit", event)
            raise ValueError("security audit events are append-only")

    def list_audit_events(self, run_id: str | None = None) -> list[SecurityAuditEvent]:
        if run_id is not None:
            return self._list_for_parent("audit", run_id, SecurityAuditEvent)
        return self._list("audit", SecurityAuditEvent)

    def create_task_run(
        self,
        task: CodeTask,
        run: TaskRun,
        workspaces: list[ExecutionWorkspace],
    ) -> tuple[CodeTask, TaskRun, list[ExecutionWorkspace]]:
        """Persist a task, its run, and workspace claims as one crash-safe unit."""

        documents: list[tuple[str, BaseModel]] = [
            ("tasks", task),
            ("runs", run),
            *(("workspaces", workspace) for workspace in workspaces),
        ]
        self._save_many(documents)
        return task, run, workspaces

    def delete_task(self, task_id: str) -> None:
        """Delete a task and every execution-plane record owned by its runs."""

        with self._lock:
            run_ids = [run.id for run in self._list_for_parent("runs", task_id, TaskRun)]
            for run_id in run_ids:
                for collection, model in (
                    ("workspaces", ExecutionWorkspace),
                    ("grants", ConnectorGrant),
                    ("commands", RuntimeCommand),
                    ("run_credentials", TaskRunCredential),
                    ("audit", SecurityAuditEvent),
                ):
                    for document in self._list_for_parent(collection, run_id, model):
                        self._delete(collection, str(document.id))
                self._delete("runs", run_id)
            self._delete("tasks", task_id)

    def prune(self, auxiliary_older_than: datetime, audit_older_than: datetime) -> int:
        """Bound non-user-visible protocol history without deleting task history."""

        removed = 0
        with self._lock:
            for command in self._list("commands", RuntimeCommand):
                if command.acked_at is not None and command.acked_at < auxiliary_older_than:
                    self._delete("commands", command.id)
                    removed += 1
            for grant in self._list("grants", ConnectorGrant):
                retired_at = grant.revoked_at or grant.expires_at
                if retired_at < auxiliary_older_than:
                    self._delete("grants", grant.id)
                    removed += 1
            for credential in self._list("registration_credentials", RuntimeRegistrationCredential):
                retired_at = credential.consumed_at or credential.expires_at
                if retired_at < auxiliary_older_than:
                    self._delete("registration_credentials", credential.id)
                    removed += 1
            for event in self._list("audit", SecurityAuditEvent):
                if event.created_at < audit_older_than:
                    self._delete("audit", event.id)
                    removed += 1
        return removed

    def claim_run(self, computer_id: str, lease_id: str, lease_expires_at: datetime) -> TaskRun | None:
        with self._lock:
            candidates = [
                run for run in self.list_runs()
                if run.computer_id == computer_id and run.status in {RunStatus.queued, RunStatus.recovering}
            ]
            if not candidates:
                return None
            run = min(candidates, key=lambda item: item.created_at)
            run.lease_id = lease_id
            run.lease_expires_at = lease_expires_at
            transition_run(run, RunStatus.preparing)
            return self.save_run(run)

    def update_run(self, run_id: str, operation: Callable[[TaskRun], None]) -> TaskRun:
        """Read, mutate, and persist one run under the store lock; unchanged runs are not rewritten."""

        with self._lock:
            run = self.get_run(run_id)
            before = run.model_dump(mode="json")
            operation(run)
            if run.model_dump(mode="json") == before:
                return run
            return self.save_run(run)

    def _save(self, collection: str, document: Document) -> Document:
        with self._lock:
            target = self._path(collection, str(document.id))
            # Each store instance owns its own lock, while API requests may
            # construct separate instances concurrently. A unique staging
            # name keeps their atomic replacements from stealing one another's
            # temporary file.
            temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
            try:
                temp.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
                os.replace(temp, target)
                self._index_document(collection, document)
            finally:
                temp.unlink(missing_ok=True)
            return document

    def _delete(self, collection: str, document_id: str) -> None:
        self._path(collection, document_id).unlink(missing_ok=True)
        index = self._parent_index.get(collection, {})
        empty_parents = []
        for parent_id, ids in index.items():
            ids.discard(document_id)
            if not ids:
                empty_parents.append(parent_id)
        for parent_id in empty_parents:
            del index[parent_id]

    def _get(self, collection: str, document_id: str, model: type[Document]) -> Document:
        with self._lock:
            try:
                return model.model_validate_json(self._path(collection, document_id).read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise KeyError(f"{model.__name__} not found") from exc

    def _list(self, collection: str, model: type[Document]) -> list[Document]:
        with self._lock:
            items: list[Document] = []
            for path in (self.root / collection).glob("*.json"):
                try:
                    items.append(model.model_validate(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            epoch = datetime.min.replace(tzinfo=UTC)
            return sorted(
                items,
                key=lambda item: getattr(item, "updated_at", None)
                or getattr(item, "created_at", None)
                or epoch,
                reverse=True,
            )

    def _list_for_parent(
        self,
        collection: str,
        parent_id: str,
        model: type[Document],
    ) -> list[Document]:
        with self._lock:
            items = []
            for document_id in self._parent_index.get(collection, {}).get(parent_id, set()):
                try:
                    items.append(self._get(collection, document_id, model))
                except KeyError:
                    continue
            epoch = datetime.min.replace(tzinfo=UTC)
            return sorted(
                items,
                key=lambda item: getattr(item, "updated_at", None)
                or getattr(item, "created_at", None)
                or epoch,
                reverse=True,
            )

    def _rebuild_parent_index(self) -> None:
        for collection, model in self._MODELS.items():
            if collection not in self._parent_index:
                continue
            for document in self._list(collection, model):
                self._index_document(collection, document)

    def _index_document(self, collection: str, document: BaseModel) -> None:
        index = self._parent_index.get(collection)
        if index is None:
            return
        document_id = str(document.id)
        for ids in index.values():
            ids.discard(document_id)
        parent_id = self._parent_id(collection, document)
        if parent_id:
            index.setdefault(parent_id, set()).add(document_id)

    @staticmethod
    def _parent_id(collection: str, document: BaseModel) -> str | None:
        if collection == "runs" and isinstance(document, TaskRun):
            return document.task_id
        if collection in {"workspaces", "grants", "commands", "run_credentials", "audit"}:
            value = getattr(document, "run_id", None)
            return str(value) if value else None
        return None

    def _path(self, collection: str, document_id: str) -> Path:
        if collection not in self._MODELS:
            raise ValueError("unknown control-plane collection")
        if not document_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in document_id):
            raise ValueError("invalid control-plane document id")
        return self.root / collection / f"{document_id}.json"

    def _save_many(self, documents: list[tuple[str, BaseModel]]) -> None:
        transaction_id = uuid.uuid4().hex
        journal = self.root / ".transaction.json"
        journal_temp = self.root / ".transaction.tmp"
        entries: list[dict[str, object]] = []
        staged: list[tuple[Path, Path]] = []
        with self._lock:
            try:
                for collection, document in documents:
                    target = self._path(collection, str(document.id))
                    if target.exists():
                        raise StateConflict(f"{collection.rstrip('s')} already exists")
                    stage = target.parent / f".{document.id}.{transaction_id}.tmp"
                    stage.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
                    entries.append({
                        "collection": collection,
                        "document_id": str(document.id),
                        "stage": stage.name,
                    })
                    staged.append((stage, target))
                journal_temp.write_text(
                    json.dumps({"schema_version": 1, "entries": entries}, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(journal_temp, journal)
                for stage, target in staged:
                    os.replace(stage, target)
                for collection, document in documents:
                    self._index_document(collection, document)
                journal.unlink()
            except Exception:
                if journal.exists():
                    self._recover_transaction()
                raise
            finally:
                journal_temp.unlink(missing_ok=True)
                for stage, _ in staged:
                    stage.unlink(missing_ok=True)

    def _recover_transaction(self) -> None:
        journal = self.root / ".transaction.json"
        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        entries = payload.get("entries")
        if payload.get("schema_version") != 1 or not isinstance(entries, list):
            raise ValueError("unsupported control-plane transaction journal")
        for raw in entries:
            if not isinstance(raw, dict):
                raise ValueError("invalid control-plane transaction journal")
            collection = raw.get("collection")
            document_id = raw.get("document_id")
            stage = raw.get("stage")
            if (
                not isinstance(collection, str)
                or collection not in self._MODELS
                or not isinstance(document_id, str)
                or not isinstance(stage, str)
                or Path(stage).name != stage
            ):
                raise ValueError("invalid control-plane transaction journal")
            self._path(collection, document_id).unlink(missing_ok=True)
            (self.root / collection / stage).unlink(missing_ok=True)
        journal.unlink()
