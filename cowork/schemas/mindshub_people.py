"""Wire shapes for the people in the caller's MindsHub organization."""

from pydantic import Field

from cowork.schemas.base import CamelResponse


class MindsHubPersonRow(CamelResponse):
    id: str
    name: str = ""
    email: str = ""


class MindsHubPeopleView(CamelResponse):
    """The org directory, or the members among some ids. ``reachable`` false
    means MindsHub could not be asked, which a client must not read as
    "nobody"."""

    reachable: bool = False
    people: list[MindsHubPersonRow] = Field(default_factory=list)
    # The directory only: the org has more members than it returns.
    truncated: bool = False
