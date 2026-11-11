"""Connector specs a scope can see: the static registry plus its custom connectors.

Built per request (or per tool call) from a ScopedSession, so every reader that
needs "is this a connector?" answers the same way for built-in and custom ids.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from anton.utils.datasources import scrub_output

from cowork.db.scoped import ScopedSession
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorSpecResponse
from cowork.services.connectors.custom_connectors import scope_definitions, scope_is_one_persons
from cowork.services.connectors.specs._registry import ConnectorSpecRegistry, registry

logger = logging.getLogger(__name__)


def custom_connector_spec(row: CustomConnector) -> dict[str, Any]:
    """Return a stored custom connector in the registry's raw spec shape.

    Args:
        row: The stored definition.

    Returns:
        A dict shaped like a registry JSON spec, flagged ``custom: True``, so
        the registry's listing, lookup and matching code reads it unchanged.
    """
    return {
        "id": row.connector_id,
        "label": row.label,
        "description": row.description,
        "category": row.category or "other",
        "logo_color": row.logo_color,
        "aliases": [],
        "keywords": [],
        "featured": row.featured,
        "custom": True,
        "usage_notes": row.usage_notes,
        "form": dict(row.spec),
    }


class ConnectorCatalog(ConnectorSpecRegistry):
    """The static registry merged with one scope's custom connectors.

    Reuses the registry's listing, lookup and matching over the merged set.
    A built-in id always wins: a custom row whose id a later release added to
    the registry is skipped and logged rather than shadowing the built-in.
    """

    def __init__(self, session: ScopedSession) -> None:
        """Bind the catalog to a scoped session.

        Args:
            session: The request's or tool call's tenant-scoped session.
        """
        super().__init__()
        self._session = session
        self._base = registry

    def _custom_rows(self) -> list[CustomConnector]:
        """The scope's stored definitions."""
        return list(self._session.exec(scope_definitions(self._session)).all())

    def get_connector(self, connector_id: str) -> ConnectorSpecResponse | None:
        """Return one connector's spec, the built-in one when the id is both.

        Args:
            connector_id: The connector id to look up.

        Returns:
            The base registry's spec when it has the id, else this scope's
            custom spec, else None.
        """
        return self._base.get_connector(connector_id) or super().get_connector(connector_id)

    def usage_notes_for(self, engines: Iterable[str]) -> dict[str, str]:
        """Return engine -> usage notes for the turn context and the probe.

        A custom connector's notes are model-written and, there, read with the
        caller's credentials in reach, so they are used only in one person's
        scope (scope_is_one_persons), never in org mode, and pass anton's
        output scrubber first. Otherwise only the built-ins' notes.
        """
        wanted = list(engines)
        notes = self._base.usage_notes_for(wanted)
        if not scope_is_one_persons(self._session.scope):
            return notes
        for engine, text in super().usage_notes_for(wanted).items():
            notes.setdefault(engine, scrub_output(text))
        return notes

    def get_connectors(self) -> dict[str, dict]:
        """Return built-in and custom specs keyed by connector id.

        Returns:
            The registry's specs plus this scope's custom ones, computed once
            per catalog instance. Outside one person's scope a custom spec is
            served without its usage notes, rows stored before that rule
            included.
        """
        if self._cache is None:
            merged = dict(self._base.get_connectors())
            own = scope_is_one_persons(self._session.scope)
            for row in self._custom_rows():
                if row.connector_id in merged:
                    logger.warning(
                        "custom connector %r shares an id with a built-in connector; "
                        "the built-in one is served",
                        row.connector_id,
                    )
                    continue
                spec = custom_connector_spec(row)
                if not own:
                    spec["usage_notes"] = None
                merged[row.connector_id] = spec
            self._cache = merged
        return self._cache
