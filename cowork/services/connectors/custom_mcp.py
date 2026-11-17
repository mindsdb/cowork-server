"""Custom MCP servers a user adds by URL on the desktop app.

A custom MCP connection's settings (server URL, endpoints, allowed tools) and
its token decide where a token goes and what Anton may do, so none of them is
taken from the vault file, which agent code running as the same OS user can
read and edit. The desktop app keeps them in its keychain and hands them to
this server's memory (``app_held``) after every start; turns read only that.
The vault file holds a record with ``_method`` and nothing else of use.

``custom_mcp`` is a reserved engine id: no custom connector definition can
take it, and every generic connection writer refuses it. The one writer is
the app-secret route in ``api/v1/endpoints/connectors/custom_mcp.py``.
"""
from __future__ import annotations

import re

#: The same id as anton's ``anton.core.mcp.wiring.CUSTOM_MCP_ENGINE``.
CUSTOM_MCP_ENGINE = "custom_mcp"

#: A connection id the desktop app draws at connect: never derived from the
#: label and never reused, so nothing a removed connection left behind can
#: attach to a new one.
_CONNECTION_ID = re.compile(r"cm-[0-9a-f]{32}")


class ReservedEngineError(ValueError):
    """A write to ``custom_mcp`` through a path that is not its own writer."""


def is_custom_mcp(engine: str) -> bool:
    """Whether ``engine`` is the reserved custom MCP engine."""
    return engine == CUSTOM_MCP_ENGINE


def is_connection_id(name: str) -> bool:
    """Whether ``name`` has the shape of a custom MCP connection id."""
    return isinstance(name, str) and _CONNECTION_ID.fullmatch(name) is not None


def refuse_generic_write(engine: str) -> None:
    """Raise ``ReservedEngineError`` for a generic write to ``custom_mcp``."""
    if is_custom_mcp(engine):
        raise ReservedEngineError("custom MCP connections are written only through their own route")
