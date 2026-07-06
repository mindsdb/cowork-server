from typing import AsyncIterator, Literal, Protocol
from typing_extensions import TypedDict
from uuid import UUID

from pydantic import BaseModel

from cowork.models.conversation import Conversation
from cowork.schemas.memory import MemoryScope
from cowork.models.project import Project
from cowork.models.skill import Skill


class TextInputBlock(TypedDict):
    type: Literal["text"]
    text: str


class FileInputBlock(TypedDict):
    type: Literal["file"]
    path: str
    filename: str


class MemoryItem(TypedDict):
    scope: MemoryScope
    category: str
    content: str
    project: Project | None


class HarnessProvider(Protocol):
    id: str
    label: str
    formatter: AsyncIterator[str]

    # ── Coworker descriptor (schema-driven Settings/composer picker) ──
    #
    # The UI never special-cases a coworker by id — SettingsView and the
    # composer's coworker picker only read these class-level fields and
    # `configuration_schema()`. Grouping is metadata (category/tags), not
    # a hardcoded "Agents vs CLI Coworkers" split, so a future non-CLI
    # coworker (a remote API agent, an MCP worker) fits the same contract.
    category: str = "General"
    priority: int = 100
    tags: tuple[str, ...] = ()

    @classmethod
    def configuration_schema(cls) -> list[dict]:
        """Declarative UI controls for this coworker's execution profile
        (e.g. [{"type": "model-picker", "id": "model"}]). Empty list means
        there's nothing to configure beyond picking this coworker."""
        return []

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        model: str | None = None,
        disabled_connections: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        ...

    async def sync_skills(self, skills: list[Skill]) -> None:
        ...

    # Requests will be made to overwrite memory content in the harness,
    # by including both new and existing content.
    async def overwrite_memory(
        self,
        scope: MemoryScope,
        category: str,
        content: str,
        project: Project | None = None
    ) -> None:
        ...

    async def retrieve_memory(
        self,
        scope: MemoryScope,
        category: str,  # Each harness will define the categories it supports.
        project: Project | None = None
    ) -> str:
        ...
        
    async def delete_memory(
        self,
        scope: MemoryScope,
        category: str,
        project: Project | None = None
    ) -> None:
        ...

    async def list_memory(
        self,
        projects: list[Project],
    ) -> list[MemoryItem]:
        ...


_registry: dict[str, type[HarnessProvider]] = {}


def register(cls: type[HarnessProvider]) -> type[HarnessProvider]:
    _registry[cls.id] = cls
    return cls


def get_harness(name: str) -> HarnessProvider:
    cls = _registry.get(name)
    if cls is None:
        available = ", ".join(_registry) or "none"
        raise ValueError(f"Unknown harness {name!r}. Available: {available}")
    return cls()


def list_descriptors() -> list[dict]:
    """Every registered coworker's descriptor + configuration_schema — the
    single source the frontend's schema-driven picker/Settings panel reads."""
    return [
        {
            "id": cls.id,
            "label": cls.label,
            "category": getattr(cls, "category", "General"),
            "priority": getattr(cls, "priority", 100),
            "tags": list(getattr(cls, "tags", ())),
            "configurationSchema": cls.configuration_schema(),
        }
        for cls in sorted(_registry.values(), key=lambda c: getattr(c, "priority", 100))
    ]