"""Slack connections that call Slack's Web API with the user's own token.

Slack lets only Marketplace and internal apps use its MCP server, so a
distributed app signs in with the same user-token flow and hands the token to
the agent for REST calls. Slack grants such a token every scope the app is
configured with, whatever the sign-in asks for, so the connection always
reads and writes; there is no mode to change.
"""
from __future__ import annotations

from typing import Any

ENGINE = "slack"
METHOD = "browser_oauth_builtin"
ACCESS_MODE = "write"


def is_slack_rest(engine: str, fields: dict[str, Any]) -> bool:
    return engine == ENGINE and fields.get("_method") == METHOD
