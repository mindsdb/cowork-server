"""Wire shapes for the icons of the caller's MindsHub organizations."""

from pydantic import Field

from cowork.schemas.base import CamelResponse


class MindsHubOrgIconRow(CamelResponse):
    """One organization's icon. ``id`` is its Keycloak id, the one the org
    switcher lists it by; ``icon_url`` an absolute https URL, or null when
    the org has no icon."""

    id: str
    icon_url: str | None = None


class MindsHubOrgIconsView(CamelResponse):
    """The caller's organizations' icons. ``reachable`` false means auth
    couldn't be asked, which a client must not read as "no icons": it keeps
    the icons it has."""

    reachable: bool = False
    orgs: list[MindsHubOrgIconRow] = Field(default_factory=list)
