from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, TypeVar

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, delete, select

from cowork.coding.contracts import utc_now
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
from cowork.models.code_control import CodeControlRecord

Document = TypeVar("Document", bound=BaseModel)


class SqlControlPlaneStore:
    """Transactional, tenant-isolated control-plane persistence.

    A store instance is permanently bound to one namespace.  Every read and
    write includes that namespace structurally; callers cannot accidentally
    issue an unscoped query.  A short-lived SQLAlchemy session per operation
    makes it safe to use from API and scheduler threads.
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

    def __init__(self, session_factory: sessionmaker[Session], namespace_id: str) -> None:
        normalized = namespace_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("a valid control-plane namespace is required")
        self._session_factory = session_factory
        self.namespace_id = normalized

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
        return self._list("workspaces", ExecutionWorkspace, parent_id=run_id)

    def save_grant(self, grant: ConnectorGrant) -> ConnectorGrant:
        return self._save("grants", grant)

    def get_grant(self, grant_id: str) -> ConnectorGrant:
        return self._get("grants", grant_id, ConnectorGrant)

    def list_grants(self, run_id: str | None = None) -> list[ConnectorGrant]:
        return self._list("grants", ConnectorGrant, parent_id=run_id)

    def update_grant(
        self,
        grant_id: str,
        operation: Callable[[ConnectorGrant, TaskRun], None],
    ) -> ConnectorGrant:
        """Serialize grant and Task Run validation across API replicas."""

        with self._session_factory.begin() as session:
            row = self._locked_row(session, "grants", grant_id)
            grant = ConnectorGrant.model_validate(row.payload)
            run_row = self._locked_row(session, "runs", grant.run_id)
            run = TaskRun.model_validate(run_row.payload)
            operation(grant, run)
            row.payload = grant.model_dump(mode="json")
            row.revision += 1
            row.updated_at = utc_now()
            session.add(row)
            return grant

    def save_command(self, command: RuntimeCommand) -> RuntimeCommand:
        return self._save("commands", command)

    def get_command(self, command_id: str) -> RuntimeCommand:
        return self._get("commands", command_id, RuntimeCommand)

    def list_commands(self, run_id: str | None = None) -> list[RuntimeCommand]:
        return self._list("commands", RuntimeCommand, parent_id=run_id)

    def save_runtime_credential(self, credential: RuntimeCredential) -> RuntimeCredential:
        return self._save("runtime_credentials", credential)

    def get_runtime_credential(self, computer_id: str) -> RuntimeCredential:
        return self._get("runtime_credentials", computer_id, RuntimeCredential)

    def save_registration_credential(self, credential: RuntimeRegistrationCredential) -> RuntimeRegistrationCredential:
        return self._save("registration_credentials", credential)

    def consume_registration_credential(self, token_hash: str, now: datetime) -> bool:
        with self._session_factory.begin() as session:
            try:
                row = self._locked_row(session, "registration_credentials", token_hash)
            except KeyError:
                return False
            credential = RuntimeRegistrationCredential.model_validate(row.payload)
            if credential.consumed_at is not None or credential.expires_at <= now:
                return False
            credential.consumed_at = now
            row.payload = credential.model_dump(mode="json")
            row.revision += 1
            row.updated_at = now
            session.add(row)
            return True

    def save_run_credential(self, credential: TaskRunCredential) -> TaskRunCredential:
        return self._save("run_credentials", credential)

    def get_run_credential(self, run_id: str) -> TaskRunCredential:
        return self._get("run_credentials", run_id, TaskRunCredential)

    def save_audit_event(self, event: SecurityAuditEvent) -> SecurityAuditEvent:
        with self._session_factory.begin() as session:
            key = (self.namespace_id, "audit", event.id)
            if session.get(CodeControlRecord, key) is not None:
                raise ValueError("security audit events are append-only")
            session.add(self._record("audit", event))
        return event

    def list_audit_events(self, run_id: str | None = None) -> list[SecurityAuditEvent]:
        return self._list("audit", SecurityAuditEvent, parent_id=run_id)

    def create_task_run(
        self,
        task: CodeTask,
        run: TaskRun,
        workspaces: list[ExecutionWorkspace],
    ) -> tuple[CodeTask, TaskRun, list[ExecutionWorkspace]]:
        with self._session_factory.begin() as session:
            for collection, document in [
                ("tasks", task),
                ("runs", run),
                *(("workspaces", workspace) for workspace in workspaces),
            ]:
                key = (self.namespace_id, collection, str(document.id))
                if session.get(CodeControlRecord, key) is not None:
                    raise ValueError(f"{collection.rstrip('s')} already exists")
                session.add(self._record(collection, document))
            try:
                session.flush()
            except IntegrityError as exc:
                raise ValueError("task run already exists") from exc
        return task, run, workspaces

    def delete_task(self, task_id: str) -> None:
        """Transactionally remove one task and all records owned by its runs."""

        with self._session_factory.begin() as session:
            run_ids = list(session.exec(
                select(CodeControlRecord.document_id)
                .where(CodeControlRecord.namespace_id == self.namespace_id)
                .where(CodeControlRecord.collection == "runs")
                .where(CodeControlRecord.parent_id == task_id)
            ).all())
            if run_ids:
                session.exec(
                    delete(CodeControlRecord)
                    .where(CodeControlRecord.namespace_id == self.namespace_id)
                    .where(CodeControlRecord.parent_id.in_(run_ids))
                )
                session.exec(
                    delete(CodeControlRecord)
                    .where(CodeControlRecord.namespace_id == self.namespace_id)
                    .where(CodeControlRecord.collection == "runs")
                    .where(CodeControlRecord.document_id.in_(run_ids))
                )
            session.exec(
                delete(CodeControlRecord)
                .where(CodeControlRecord.namespace_id == self.namespace_id)
                .where(CodeControlRecord.collection == "tasks")
                .where(CodeControlRecord.document_id == task_id)
            )

    def prune(self, auxiliary_older_than: datetime, audit_older_than: datetime) -> int:
        """Bound auxiliary protocol records while retaining tasks and runs."""

        removed = 0
        with self._session_factory.begin() as session:
            rows = session.exec(
                select(CodeControlRecord)
                .where(CodeControlRecord.namespace_id == self.namespace_id)
                .where(CodeControlRecord.collection.in_([
                    "commands", "grants", "registration_credentials", "audit",
                ]))
            ).all()
            for row in rows:
                should_remove = False
                if row.collection == "commands":
                    command = RuntimeCommand.model_validate(row.payload)
                    should_remove = command.acked_at is not None and command.acked_at < auxiliary_older_than
                elif row.collection == "grants":
                    grant = ConnectorGrant.model_validate(row.payload)
                    should_remove = (grant.revoked_at or grant.expires_at) < auxiliary_older_than
                elif row.collection == "registration_credentials":
                    credential = RuntimeRegistrationCredential.model_validate(row.payload)
                    should_remove = (credential.consumed_at or credential.expires_at) < auxiliary_older_than
                elif row.collection == "audit":
                    should_remove = SecurityAuditEvent.model_validate(row.payload).created_at < audit_older_than
                if should_remove:
                    session.delete(row)
                    removed += 1
        return removed

    def claim_run(self, computer_id: str, lease_id: str, lease_expires_at: datetime) -> TaskRun | None:
        """Claim the oldest queued run under a database row lock.

        ``SKIP LOCKED`` lets multiple scheduler replicas make progress without
        assigning the same run twice. SQLite ignores row locks, but desktop
        uses the local store; this path is intended for PostgreSQL deployments.
        """

        with self._session_factory.begin() as session:
            row = session.exec(
                select(CodeControlRecord)
                .where(CodeControlRecord.namespace_id == self.namespace_id)
                .where(CodeControlRecord.collection == "runs")
                .where(CodeControlRecord.assigned_computer_id == computer_id)
                .where(CodeControlRecord.lifecycle_status.in_([
                    RunStatus.queued.value,
                    RunStatus.recovering.value,
                ]))
                .order_by(CodeControlRecord.created_at.asc())
                .with_for_update(skip_locked=True)
                .limit(1)
            ).one_or_none()
            if row is None:
                return None
            run = TaskRun.model_validate(row.payload)
            run.lease_id = lease_id
            run.lease_expires_at = lease_expires_at
            transition_run(run, RunStatus.preparing)
            run.updated_at = utc_now()
            row.payload = run.model_dump(mode="json")
            self._apply_projection(row, "runs", run)
            row.revision += 1
            row.updated_at = run.updated_at
            session.add(row)
            return run

    def update_run(self, run_id: str, operation: Callable[[TaskRun], None]) -> TaskRun:
        """Mutate one run under a row lock so replicas serialize on the same record."""

        with self._session_factory.begin() as session:
            row = self._locked_row(session, "runs", run_id)
            run = TaskRun.model_validate(row.payload)
            operation(run)
            if run.model_dump(mode="json") == row.payload:
                return run
            run.updated_at = utc_now()
            row.payload = run.model_dump(mode="json")
            self._apply_projection(row, "runs", run)
            row.revision += 1
            row.updated_at = utc_now()
            session.add(row)
            return run

    def _save(self, collection: str, document: Document) -> Document:
        self._require_collection(collection)
        with self._session_factory.begin() as session:
            key = (self.namespace_id, collection, str(document.id))
            row = session.get(CodeControlRecord, key)
            if row is None:
                session.add(self._record(collection, document))
            else:
                row.payload = document.model_dump(mode="json")
                self._apply_projection(row, collection, document)
                row.revision += 1
                row.updated_at = utc_now()
                session.add(row)
        return document

    def _get(self, collection: str, document_id: str, model: type[Document]) -> Document:
        self._require_collection(collection)
        with self._session_factory() as session:
            row = session.get(CodeControlRecord, (self.namespace_id, collection, document_id))
            if row is None:
                raise KeyError(f"{model.__name__} not found")
            return model.model_validate(row.payload)

    def _list(
        self,
        collection: str,
        model: type[Document],
        *,
        parent_id: str | None = None,
    ) -> list[Document]:
        self._require_collection(collection)
        with self._session_factory() as session:
            statement = (
                select(CodeControlRecord)
                .where(CodeControlRecord.namespace_id == self.namespace_id)
                .where(CodeControlRecord.collection == collection)
                .order_by(CodeControlRecord.updated_at.desc())
            )
            if parent_id is not None:
                statement = statement.where(CodeControlRecord.parent_id == parent_id)
            rows = session.exec(statement).all()
            return [model.model_validate(row.payload) for row in rows]

    def _locked_row(self, session: Session, collection: str, document_id: str) -> CodeControlRecord:
        row = session.exec(
            select(CodeControlRecord)
            .where(CodeControlRecord.namespace_id == self.namespace_id)
            .where(CodeControlRecord.collection == collection)
            .where(CodeControlRecord.document_id == document_id)
            .with_for_update()
        ).one_or_none()
        if row is None:
            raise KeyError(f"{self._MODELS[collection].__name__} not found")
        return row

    def _record(self, collection: str, document: BaseModel) -> CodeControlRecord:
        self._require_collection(collection)
        record = CodeControlRecord(
            namespace_id=self.namespace_id,
            collection=collection,
            document_id=str(document.id),
            payload=document.model_dump(mode="json"),
        )
        self._apply_projection(record, collection, document)
        return record

    @staticmethod
    def _apply_projection(
        record: CodeControlRecord,
        collection: str,
        document: BaseModel,
    ) -> None:
        if collection == "runs" and isinstance(document, TaskRun):
            record.assigned_computer_id = document.computer_id
            record.lifecycle_status = document.status.value
            record.parent_id = document.task_id
        else:
            record.assigned_computer_id = None
            record.lifecycle_status = None
            if collection in {"workspaces", "grants", "commands", "run_credentials", "audit"}:
                value = getattr(document, "run_id", None)
                record.parent_id = str(value) if value else None
            else:
                record.parent_id = None

    def _require_collection(self, collection: str) -> None:
        if collection not in self._MODELS:
            raise ValueError("unknown control-plane collection")
