from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

from cowork.coding.contracts import utc_now
from cowork.coding.project_models import CodeProject


class CodeProjectStore:
    """Crash-safe local persistence for the durable Code Project catalogue."""

    def __init__(self, root: Path) -> None:
        self.root = root / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

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

    @staticmethod
    def _load(raw: dict) -> CodeProject:
        version = raw.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"unsupported Code Project schema {version}")
        return CodeProject.model_validate(raw)
