"""Wire shapes for the caller's MindsHub teams."""

from pydantic import Field

from cowork.schemas.base import CamelResponse


class HubTeamRow(CamelResponse):
    id: str
    display_name: str = ""
    # The org's default team: its members are everyone in the organization,
    # so sharing with it is "Everyone at <org>".
    is_default: bool = False


class HubTeamsView(CamelResponse):
    """The caller's teams. ``reachable`` false means auth could not be asked,
    which a client must not read as "in no team"."""

    reachable: bool = False
    teams: list[HubTeamRow] = Field(default_factory=list)
