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

    One response rather than three so a menu open costs one round trip.

    ``enabled`` always answers true now: the selector's own kill switch,
    `authorization_ui`, was retired once its surfaces had been live in
    production long enough to trust. Kept on the wire rather than dropped, so an
    older client build reading it does not need to change on the same day.
    ``reachable`` false means auth could not be asked, so say so rather than
    showing an empty list, which reads as an organization with one workspace.
    """

    enabled: bool = False
    reachable: bool = False
    workspaces: list[HubWorkspaceRow] = Field(default_factory=list)
    active_workspace_id: Optional[str] = None


class HubWorkspaceActivateRequest(CamelRequest):
    """Which workspace to switch to."""

    workspace_id: str
