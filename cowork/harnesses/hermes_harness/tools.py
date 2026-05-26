from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def _hermes_lookup_connector(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector
    return await _cowork_lookup_connector(None, args)


async def _hermes_request_credentials(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
    return await _cowork_request_credentials(None, args)


def register_connector_tools() -> None:
    from tools.registry import registry
    from cowork.harnesses.anton_harness.tools import (
        _LOOKUP_CONNECTOR_SCHEMA,
        _LOOKUP_CONNECTOR_PROMPT,
        _REQUEST_CREDENTIALS_SCHEMA,
        _REQUEST_CREDENTIALS_PROMPT,
    )

    if registry.get_entry("lookup_connector") is None:
        registry.register(
            name="lookup_connector",
            toolset="connectors",
            schema={
                "name": "lookup_connector",
                "description": (
                    "Look up the canonical connector spec for a service by id or "
                    "natural-language query. Returns the same form blob the "
                    "in-app Connector Picker uses — pass it straight to "
                    "`request_credentials`.\n\n"
                    + _LOOKUP_CONNECTOR_PROMPT
                ),
                "parameters": _LOOKUP_CONNECTOR_SCHEMA,
            },
            handler=_hermes_lookup_connector,
            is_async=True,
            description="Look up a connector spec by id or natural-language query.",
            emoji="🔌",
        )

    if registry.get_entry("request_credentials") is None:
        registry.register(
            name="request_credentials",
            toolset="connectors",
            schema={
                "name": "request_credentials",
                "description": (
                    "Request credentials / configuration from the user via an interactive "
                    "form rendered in the side panel. Returns a markdown block you must "
                    "include verbatim in your next assistant message so the form appears.\n\n"
                    + _REQUEST_CREDENTIALS_PROMPT
                ),
                "parameters": _REQUEST_CREDENTIALS_SCHEMA,
            },
            handler=_hermes_request_credentials,
            is_async=True,
            description="Render a credential form in the side panel.",
            emoji="🔐",
        )
