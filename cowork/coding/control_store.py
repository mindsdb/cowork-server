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
from cowork.coding.control_models import (
    CodeTask,
    Computer,
    ConnectorGrant,
    ExecutionWorkspace,
    RuntimeCommand,
    RuntimeCredential,
    TaskRun,
    TaskRunCredential,
)

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
    def save_command(self, command: RuntimeCommand) -> RuntimeCommand: ...
    def list_commands(self, run_id: str | None = None) -> list[RuntimeCommand]: ...
    def save_runtime_credential(self, credential: RuntimeCredential) -> RuntimeCredential: ...
    def get_runtime_credential(self, computer_id: str) -> RuntimeCredential: ...
    def save_run_credential(self, credential: TaskRunCredential) -> TaskRunCredential: ...
    def get_run_credential(self, run_id: str) -> TaskRunCredential: ...
    def create_task_run(
        self,
        task: CodeTask,
        run: TaskRun,
        workspaces: list[ExecutionWorkspace],
    ) -> tuple[CodeTask, TaskRun, list[ExecutionWorkspace]]: ...


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
        "run_credentials": TaskRunCredential,
    }

    def __init__(self, root: Path) -> None:
        self.root = root / "control"
        self._lock = threading.RLock()
        for collection in self._MODELS:
            (self.root / collection).mkdir(parents=True, exist_ok=True)
        self._recover_transaction()

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
        items = self._list("workspaces", ExecutionWorkspace)
        return [item for item in items if run_id is None or item.run_id == run_id]

    def save_grant(self, grant: ConnectorGrant) -> ConnectorGrant:
        return self._save("grants", grant)

    def get_grant(self, grant_id: str) -> ConnectorGrant:
        return self._get("grants", grant_id, ConnectorGrant)

    def list_grants(self, run_id: str | None = None) -> list[ConnectorGrant]:
        items = self._list("grants", ConnectorGrant)
        return [item for item in items if run_id is None or item.run_id == run_id]

    def save_command(self, command: RuntimeCommand) -> RuntimeCommand:
        return self._save("commands", command)

    def list_commands(self, run_id: str | None = None) -> list[RuntimeCommand]:
        items = self._list("commands", RuntimeCommand)
        return [item for item in items if run_id is None or item.run_id == run_id]

    def save_runtime_credential(self, credential: RuntimeCredential) -> RuntimeCredential:
        return self._save("runtime_credentials", credential)

    def get_runtime_credential(self, computer_id: str) -> RuntimeCredential:
        return self._get("runtime_credentials", computer_id, RuntimeCredential)

    def save_run_credential(self, credential: TaskRunCredential) -> TaskRunCredential:
        return self._save("run_credentials", credential)

    def get_run_credential(self, run_id: str) -> TaskRunCredential:
        return self._get("run_credentials", run_id, TaskRunCredential)

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

    def update_run(self, run_id: str, operation: Callable[[TaskRun], None]) -> TaskRun:
        with self._lock:
            run = self.get_run(run_id)
            operation(run)
            return self.save_run(run)

    def _save(self, collection: str, document: Document) -> Document:
        with self._lock:
            target = self._path(collection, str(document.id))
            temp = target.with_suffix(".tmp")
            temp.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(temp, target)
            return document

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
                        raise ValueError(f"{collection.rstrip('s')} already exists")
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
