from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cowork.coding.contracts import utc_now
from cowork.coding.project_models import CodeProject


@dataclass(frozen=True)
class _JournalEntry:
    project_id: str
    before: dict[str, object]
    stage: str | None


class CodeProjectStore:
    """Crash-safe local persistence for the durable Code Project catalogue."""

    def __init__(self, root: Path, migration_computer_id: str = "local") -> None:
        self.root = root / "projects"
        self.migration_computer_id = migration_computer_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._recover_transaction()

    def save(self, project: CodeProject) -> CodeProject:
        with self._lock:
            project.updated_at = utc_now()
            target = self._path(project.id)
            temp = target.with_suffix(".tmp")
            temp.write_text(project.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(temp, target)
            return project

    def create(self, project: CodeProject) -> CodeProject:
        with self._lock:
            if self._path(project.id).exists():
                raise ValueError("Code Project already exists")
            return self.save(project)

    def get(self, project_id: str) -> CodeProject:
        with self._lock:
            try:
                raw = json.loads(self._path(project_id).read_text(encoding="utf-8"))
                return self._load(raw)
            except FileNotFoundError as exc:
                raise KeyError("Code Project not found") from exc

    def list(self) -> list[CodeProject]:
        with self._lock:
            projects: list[CodeProject] = []
            for path in self.root.glob("*.json"):
                try:
                    projects.append(self._load(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def update(self, project_id: str, operation: Callable[[CodeProject], None]) -> CodeProject:
        with self._lock:
            project = self.get(project_id)
            operation(project)
            return self.save(project)

    def update_many(self, operations: dict[str, Callable[[CodeProject], None]]) -> list[CodeProject]:
        """Validate and persist a related set of project edits as one recoverable unit."""
        with self._lock:
            projects: list[CodeProject] = []
            for project_id, operation in operations.items():
                project = self.get(project_id)
                operation(project)
                # Revalidate mutations performed by callers before any file is staged.
                projects.append(CodeProject.model_validate(project.model_dump(mode="python")))
            return self._save_many(projects)

    def delete(self, project_id: str) -> None:
        with self._lock:
            try:
                self._path(project_id).unlink()
            except FileNotFoundError as exc:
                raise KeyError("Code Project not found") from exc

    def _path(self, project_id: str) -> Path:
        if not project_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in project_id
        ):
            raise ValueError("invalid Code Project id")
        return self.root / f"{project_id}.json"

    def _save_many(self, projects: list[CodeProject]) -> list[CodeProject]:
        if not projects:
            return []
        transaction_id = uuid.uuid4().hex
        journal = self.root / ".transaction.json"
        journal_temp = self.root / ".transaction.tmp"
        entries: list[dict[str, object]] = []
        staged: list[tuple[Path, Path]] = []
        timestamp = utc_now()
        try:
            for project in projects:
                project.updated_at = timestamp
                target = self._path(project.id)
                before = json.loads(target.read_text(encoding="utf-8"))
                stage = self.root / f".{project.id}.{transaction_id}.tmp"
                stage.write_text(project.model_dump_json(indent=2) + "\n", encoding="utf-8")
                entries.append({"project_id": project.id, "before": before, "stage": stage.name})
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
        return projects

    def _recover_transaction(self) -> None:
        journal = self.root / ".transaction.json"
        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
            raise ValueError("unsupported Code Project transaction journal")
        entries = [self._journal_entry(entry) for entry in payload["entries"]]
        for entry in entries:
            target = self._path(entry.project_id)
            restore = self.root / f".{entry.project_id}.restore.tmp"
            restore.write_text(json.dumps(entry.before, indent=2) + "\n", encoding="utf-8")
            os.replace(restore, target)
            if entry.stage:
                (self.root / entry.stage).unlink(missing_ok=True)
        journal.unlink()

    @staticmethod
    def _journal_entry(raw: object) -> _JournalEntry:
        if not isinstance(raw, dict):
            raise ValueError("invalid Code Project transaction journal")
        project_id = raw.get("project_id")
        before = raw.get("before")
        stage = raw.get("stage")
        if (
            not isinstance(project_id, str)
            or not isinstance(before, dict)
            or (stage is not None and (not isinstance(stage, str) or Path(stage).name != stage))
        ):
            raise ValueError("invalid Code Project transaction journal")
        return _JournalEntry(project_id=project_id, before=before, stage=stage)

    def _load(self, raw: dict) -> CodeProject:
        version = raw.get("schema_version", 1)
        if version not in {1, 2}:
            raise ValueError(f"unsupported Code Project schema {version}")
        if version == 1:
            raw = {**raw, "_migration_computer_id": self.migration_computer_id}
        return CodeProject.model_validate(raw)
