from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore, PROJECT_SLOTS, ProjectMemoryStore
from cowork.models.project import Project
from cowork.schemas.memory import MemoryResponse, MemoryScope


def build_turn_memory(
    scope: TenantScope, project_path: str | None = None
) -> dict[str, dict[str, str]]:
    """Memory slots for a remote turn: ``{"global": {slot: text}, "project": {...}}``.

    Org-keyed through the passed `scope` — the remote producer binds no ambient
    scope, so reading it from there would hit the unkeyed root.
    """
    out: dict[str, dict[str, str]] = {}

    global_store = GlobalMemoryStore(scope=scope)
    if global_slots := {
        s.value: text for s in MemorySlot if (text := global_store.read(s).strip())
    }:
        out["global"] = global_slots

    if project_path:
        store = ProjectMemoryStore(Path(project_path))
        if slots := {s.value: text for s in PROJECT_SLOTS if (text := store.read(s).strip())}:
            out["project"] = slots

    return out


class _WireEngram(BaseModel):
    """One memory entry as a pod may send it. A pod runs model-authored code, so
    entries are untrusted and anything unparseable is skipped.

    The literals mirror anton's ``Engram`` (pinned by
    test_engram_vocabulary_matches_anton) and each is load-bearing: ``scope``
    decides personal vs shared-with-the-team, so an unrecognised value would
    quietly publish a personal note to the whole org, and ``source``/``confidence``
    are interpolated raw into the ``<!-- ... -->`` metadata tail.

    ``max_length`` on text is a sanity bound on what reaches disk, not a content
    limit: a stored slot is re-read into every later turn's payload, but anton
    decides what actually enters a prompt. There is deliberately no cap on entries
    per turn — it would lose real memories while an abusive pod just sends the
    maximum every turn.
    """

    model_config = ConfigDict(extra="ignore")

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
    kind: Literal["always", "never", "when", "lesson", "profile"]
    scope: Literal["global", "project"] = "global"
    topic: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    source: Literal["user", "llm", "consolidation"] = "llm"

    @field_validator("topic")
    @classmethod
    def _bound_topic(cls, value: str) -> str:
        # Truncated, not rejected: an over-long topic is a mislabelled memory, and
        # the memory is the part worth keeping. Hippocampus slugifies the charset.
        return value[:64]


def _one_line(text: str) -> str:
    """Neutralize forged structure before it reaches the slot files, which store
    one entry per line with an HTML-comment metadata tail. Spaces and tabs inside
    a line are the user's formatting and are preserved.

    Mirrors Hippocampus._entry_text deliberately: this must hold whichever anton
    version is pinned, since the wire data is untrusted and the pin can lag.
    """
    text = text.replace("<!--", "<!-").replace("-->", "->")
    return " ".join(line.strip() for line in text.splitlines()).strip()


def apply_turn_memory(
    scope: TenantScope, project_path: str | None, entries: list[dict]
) -> int:
    """Persist engrams a remote turn asked to remember; returns how many passed
    validation (Hippocampus dedupes, so a write may still be a no-op).

    The pod reports intent — this is the only place a cloud turn's memory is
    written, and it goes through the caller's `scope`, so a compromised pod can't
    reach another org. Writes via anton's own Hippocampus rather than reproducing
    its markdown format.
    """
    from anton.core.memory.hippocampus import Hippocampus

    global_hc = Hippocampus(GlobalMemoryStore(scope=scope).root)
    project_hc = Hippocampus(ProjectMemoryStore(Path(project_path)).root) if project_path else None

    applied = 0
    for raw in entries:
        try:
            entry = _WireEngram.model_validate(raw)
        except ValidationError:
            continue

        text = _one_line(entry.text)
        # Identity is global-only; everything else honours the declared scope.
        target = global_hc if (entry.kind == "profile" or entry.scope == "global") else project_hc
        if not text or target is None:
            continue

        if entry.kind == "profile":
            target.rewrite_identity([text])
        elif entry.kind == "lesson":
            target.encode_lesson(text, topic=entry.topic, source=entry.source)
        else:
            target.encode_rule(
                text, kind=entry.kind, confidence=entry.confidence, source=entry.source,
            )
        applied += 1

    return applied


class MemoryService:
    def __init__(self, session: ScopedSession) -> None:
        self.session = session
        # Org mode → per-(org, user); the store keys itself (see GlobalMemoryStore).
        self._global_store = GlobalMemoryStore(scope=session.scope)

    async def get_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        content = self._read(scope, category, project_id)
        return MemoryResponse(
            scope=scope,
            category=category,
            content=content,
            project_id=project_id if scope == MemoryScope.project else None,
        )

    async def update_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None = None,
    ) -> MemoryResponse:
        self._write(scope, category, content, project_id)
        return MemoryResponse(
            scope=scope,
            category=category,
            content=content,
            project_id=project_id if scope == MemoryScope.project else None,
        )

    async def delete_memory(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None = None,
    ) -> None:
        self._delete(scope, category, project_id)

    async def list_memory(self, project_id: UUID | None = None) -> list[MemoryResponse]:
        items: list[MemoryResponse] = []

        for category in MemorySlot:
            items.append(
                MemoryResponse(
                    scope=MemoryScope.global_,
                    category=category,
                    content=self._global_store.read(category),
                    project_id=None,
                )
            )

        projects = list(self.session.exec(self.session.select(Project)).all())
        if project_id is not None:
            projects = [p for p in projects if p.id == project_id]

        for project in projects:
            store = ProjectMemoryStore(Path(project.path))
            for category in PROJECT_SLOTS:
                items.append(
                    MemoryResponse(
                        scope=MemoryScope.project,
                        category=category,
                        content=store.read(category),
                        project_id=project.id,
                    )
                )

        if project_id is not None:
            items = [
                item
                for item in items
                if item.scope == MemoryScope.global_ or item.project_id == project_id
            ]

        return items

    def _read(self, scope: MemoryScope, category: MemorySlot, project_id: UUID | None) -> str:
        if scope == MemoryScope.global_:
            return self._global_store.read(category)
        project = self._resolve_project(project_id)
        return ProjectMemoryStore(Path(project.path)).read(category)

    def _write(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        content: str,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.write(category, content)
            return
        project = self._resolve_project(project_id)
        ProjectMemoryStore(Path(project.path)).write(category, content)

    def _delete(
        self,
        scope: MemoryScope,
        category: MemorySlot,
        project_id: UUID | None,
    ) -> None:
        if scope == MemoryScope.global_:
            self._global_store.delete(category)
            return
        project = self._resolve_project(project_id)
        ProjectMemoryStore(Path(project.path)).delete(category)

    def _resolve_project(self, project_id: UUID | None) -> Project:
        if project_id is None:
            raise ValueError("project_id is required for project-scoped memory.")
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        return project
