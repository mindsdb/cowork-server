from __future__ import annotations

from cowork.coding.contracts import EngineCapabilities
from cowork.coding.engines.base import CodingEngine


class CodingEngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, CodingEngine] = {}

    def register(self, engine: CodingEngine) -> None:
        if engine.id in self._engines:
            raise ValueError(f"coding engine already registered: {engine.id}")
        self._engines[engine.id] = engine

    def get(self, engine_id: str) -> CodingEngine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise KeyError(f"unknown coding engine: {engine_id}") from exc

    def available_ids(self) -> list[str]:
        return [engine_id for engine_id, engine in self._engines.items() if engine.capabilities().available]

    def ids(self) -> list[str]:
        return list(self._engines)

    def capabilities(self) -> list[EngineCapabilities]:
        return [engine.capabilities() for engine in self._engines.values()]


def _build_registry() -> CodingEngineRegistry:
    from cowork.coding.engines.codex import CodexEngine

    registry = CodingEngineRegistry()
    registry.register(CodexEngine())
    return registry


engine_registry = _build_registry()
