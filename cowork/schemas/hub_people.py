"""Wire shapes for the people in the caller's MindsHub organization."""

from pydantic import Field

from cowork.schemas.base import CamelResponse


class HubPersonRow(CamelResponse):
    id: str
    name: str = ""
    email: str = ""


class HubPeopleView(CamelResponse):
    """The org directory, or the members among some ids. ``reachable`` false
    means the auth service could not be asked, which a client must not read
    as "nobody"."""

    reachable: bool = False
    people: list[HubPersonRow] = Field(default_factory=list)
    # The directory only: the org has more members than it returns.
    truncated: bool = False
