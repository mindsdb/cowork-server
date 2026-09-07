"""Reasoning effort levels belong to the model gateway, not to this server.

MindsHub's ``/v1/models`` advertises ``reasoning_efforts`` per model and the
vocabularies differ by family: GPT 5.6 Sol lists none, low, medium, high,
xhigh and max; Claude models list low through max; Gemini lists low, medium
and high; a few models offer no levels at all. Codex hands the level to the
gateway unchanged, so the only correct check is against the list the chosen
model advertises, and the only correct names are the gateway's own.

``levels`` below is that listing (:class:`ModelLevels`): the ids the gateway
knows and, for each model that has them, its levels. ``None`` means the
listing isn't known right now (nothing fetched yet, or the gateway unreachable);
the gateway then stays the judge and an unknown level fails the first turn
with its own error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class ModelLevels(Protocol):
    """The slice of ``cowork.services.providers.MindsModelListing`` this needs."""

    @property
    def ids(self) -> list[str] | None: ...

    @property
    def efforts(self) -> Mapping[str, Mapping]: ...


def advertised_levels(model: str, levels: ModelLevels | None) -> list[str] | None:
    """The levels ``model`` advertises: ``[]`` for a known model without any, None when unknown."""
    if levels is None or not levels.ids or model not in levels.ids:
        return None
    entry = levels.efforts.get(model)
    offered = entry.get("efforts") if isinstance(entry, Mapping) else None
    if not isinstance(offered, (list, tuple)):
        return []
    return [str(level) for level in offered]


def check_reasoning_effort(model: str, effort: str | None, levels: ModelLevels | None) -> None:
    """Raise ValueError when ``effort`` isn't a level ``model`` advertises."""
    if effort is None:
        return
    offered = advertised_levels(model, levels)
    if offered is None or effort in offered:
        return
    if not offered:
        raise ValueError(f"{model} doesn't take a reasoning effort setting.")
    raise ValueError(
        f'Reasoning effort "{effort}" isn\'t available for {model}. It offers: {", ".join(offered)}.'
    )


def resolve_reasoning_effort(
    model: str,
    requested: str | None,
    project_default: str | None,
    levels: ModelLevels | None,
) -> str | None:
    """The level a new task runs at.

    A level the task asked for must be one its model advertises (ValueError
    otherwise). A project default is inherited only when the task's model
    advertises it: a project may set "max" for GPT 5.6 Sol and still start a
    Gemini task, which then runs at Gemini's own default instead of failing.
    """
    check_reasoning_effort(model, requested, levels)
    if requested:
        return requested
    if project_default is None:
        return None
    offered = advertised_levels(model, levels)
    if offered is not None and project_default not in offered:
        return None
    return project_default
