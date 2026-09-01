from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

from cowork.coding.contracts import utc_now
from cowork.coding.skill_models import TeamSkillSource


class SkillSourceStore:
    """Crash-safe catalogue for organisation-wide Git skill sources."""

    def __init__(self, root: Path) -> None:
        self.root = root / "skill-library"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "sources.json"
        self._lock = threading.RLock()

    def list(self) -> list[TeamSkillSource]:
        with self._lock:
            return sorted(self._read(), key=lambda item: (item.name.casefold(), item.id))

    def get(self, source_id: str) -> TeamSkillSource:
        with self._lock:
            source = next((item for item in self._read() if item.id == source_id), None)
            if source is None:
                raise KeyError("Skill source not found")
            return source

    def create(self, source: TeamSkillSource) -> TeamSkillSource:
        with self._lock:
            items = self._read()
            if any(item.id == source.id for item in items):
                raise ValueError("Skill source already exists")
            repository = source.repository.rstrip("/\\").casefold()
            branch = source.branch.casefold()
            if any(
                item.repository.rstrip("/\\").casefold() == repository
                and item.branch.casefold() == branch
                for item in items
            ):
                raise ValueError("That repository branch is already in the Skills Library")
            items.append(source)
            self._write(items)
            return source

    def update(self, source_id: str, operation: Callable[[TeamSkillSource], None]) -> TeamSkillSource:
        with self._lock:
            items = self._read()
            source = next((item for item in items if item.id == source_id), None)
            if source is None:
                raise KeyError("Skill source not found")
            operation(source)
            source.updated_at = utc_now()
            self._write(items)
            return source

    def delete(self, source_id: str) -> None:
        with self._lock:
            items = self._read()
            filtered = [item for item in items if item.id != source_id]
            if len(filtered) == len(items):
                raise KeyError("Skill source not found")
            self._write(filtered)

    def _read(self) -> list[TeamSkillSource]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if raw.get("schema_version") != 1 or not isinstance(raw.get("items"), list):
            raise ValueError("unsupported skill source catalogue")
        return [TeamSkillSource.model_validate(item) for item in raw["items"]]

    def _write(self, items: list[TeamSkillSource]) -> None:
        temp = self.path.with_suffix(".tmp")
        payload = {
            "schema_version": 1,
            "items": [item.model_dump(mode="json") for item in items],
        }
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)
