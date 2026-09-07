"""Wire shapes for the MindsHub workspace selector."""

from typing import Optional

from pydantic import Field

from cowork.schemas.base import CamelRequest, CamelResponse


class HubWorkspaceRow(CamelResponse):
    """One row in the selector."""

    id: str
    slug: str = ""
    display_name: str = ""
    is_default: bool = False
    archived_at: Optional[str] = None
    role: str = ""


class HubWorkspaceView(CamelResponse):
    """Everything the selector needs from one request.

    One response rather than three so a menu open costs one round trip, and so
    the client cannot render a list against a gate answer from a different
    moment.

    ``enabled`` and ``reachable`` are separate answers to separate questions, and
    collapsing them loses the distinction that matters. ``enabled`` false means
    the surface is switched off, so render nothing. ``reachable`` false means the
    surface is on but auth could not be asked, so say so rather than showing an
    empty list, which reads as an organization with one workspace.
    """

    enabled: bool = False
    reachable: bool = False
    workspaces: list[HubWorkspaceRow] = Field(default_factory=list)
    active_workspace_id: Optional[str] = None


class HubWorkspaceActivateRequest(CamelRequest):
    """Which workspace to switch to."""

    workspace_id: str
