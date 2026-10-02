"""Cowork-owned agentic coding domain.

This package is intentionally independent from the desktop's parked Claude
terminal prototype.  It exposes product concepts (sessions, turns, approvals,
events and workspaces); engine-specific wire objects stay inside adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cowork.coding.service import CodingService, get_coding_service

__all__ = ["CodingService", "get_coding_service"]


def __getattr__(name: str) -> Any:
    # Resolved on first use. An eager import made every ``python -m
    # cowork.coding.<module>`` subprocess, such as the integration MCP server
    # Codex waits on before a turn starts, load the whole service first.
    if name in __all__:
        from cowork.coding import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
