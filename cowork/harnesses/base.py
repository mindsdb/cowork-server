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

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
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