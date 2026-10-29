"""Wire shapes for the people search in the caller's MindsHub organization."""

from pydantic import Field

from cowork.schemas.base import CamelResponse


class HubPersonRow(CamelResponse):
    id: str
    name: str = ""
    email: str = ""


class HubPeopleView(CamelResponse):
    """Members matching a search. ``reachable`` false means auth could not be
    asked, which a client must not read as "nobody matches"."""

    reachable: bool = False
    people: list[HubPersonRow] = Field(default_factory=list)
