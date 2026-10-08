"""Connector specs a scope can see: the static registry plus its custom connectors.

Built per request (or per tool call) from a ScopedSession, so every reader that
needs "is this a connector?" answers the same way for built-in and custom ids.
"""
from __future__ import annotations

import logging
from typing import Any

from cowork.db.scoped import ScopedSession
from cowork.models.custom_connector import CustomConnector
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
        "form": dict(row.spec),
    }


class ConnectorCatalog(ConnectorSpecRegistry):
    """The static registry merged with one scope's custom connectors.

    Reuses the registry's listing, lookup and matching over the merged set.
    A built-in id always wins: a custom row whose id a later release added to
    the registry is skipped and logged rather than shadowing the built-in.
    """

    def __init__(self, session: ScopedSession, base: ConnectorSpecRegistry = registry) -> None:
        """Bind the catalog to a scoped session.

        Args:
            session: The request's or tool call's tenant-scoped session.
            base: The static registry to merge onto; the module singleton by default.
        """
        super().__init__()
        self._session = session
        self._base = base

    def _custom_rows(self) -> list[CustomConnector]:
        stmt = self._session.select(CustomConnector)
        # ScopedSession only filters in org mode; a local scope must not read
        # rows that belong to an org.
        if not self._session.scope.org_mode:
            stmt = stmt.where(CustomConnector.org_id.is_(None))
        return list(self._session.exec(stmt).all())

    def get_connectors(self) -> dict[str, dict]:
        """Return built-in and custom specs keyed by connector id.

        Returns:
            The registry's specs plus this scope's custom ones, computed once
            per catalog instance.
        """
        if self._cache is None:
            merged = dict(self._base.get_connectors())
            for row in self._custom_rows():
                if row.connector_id in merged:
                    logger.warning(
                        "custom connector %r shares an id with a built-in connector; "
                        "the built-in one is served",
                        row.connector_id,
                    )
                    continue
                merged[row.connector_id] = custom_connector_spec(row)
            self._cache = merged
        return self._cache
