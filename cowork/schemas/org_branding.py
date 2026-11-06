"""Wire shapes for an organization's branding (ENG-4020).

``GET /api/v1/org-branding`` answers OrgBrandingView in camelCase. ``PUT
/api/v1/org-branding/appearance`` takes OrgAppearanceUpdate in snake_case
(camelCase accepted too), which the route passes to auth's ``PATCH
/v1/organizations/current/branding/appearance/``. auth owns the look and
validates every value (the app's colors, fonts, skins and radius range); a
value it refuses comes back as auth's own 422.
"""

from __future__ import annotations

from typing import Any

from cowork.schemas.base import CamelRequest, CamelResponse


class OrgAppearanceView(CamelResponse):
    """The org's look as auth keeps it; null fields are the app's stock
    values."""

    accent: str | None = None
    # For clients from before the light/dark split: backgroundLight ??
    # backgroundDark.
    background: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: str | None = None
    radius: int | None = None
    skin: str | None = None
    default_theme: str | None = None
    greeting: str | None = None
    # The admin's own title, in 'custom' mode; OrgBrandingView's
    # sidebar_title is the one to show.
    sidebar_title: str | None = None
    sidebar_title_mode: str = "company"
    members_can_choose: bool = True
    guidelines: str | None = None


class OrgBrandingView(CamelResponse):
    """What the app reads on startup, org switch, sign-in and when Settings
    opens. ``enabled`` is false, and the rest null, without a MindsHub org:
    no credential, or auth refused (no current org). The image URLs are
    auth's (absolute, with ``?v=<hash>``). ``sidebar_title`` is auth's
    resolved title: the company name, the custom text, or null for no title
    (only the logo). ``can_edit`` is auth's ``can_edit``. When auth couldn't
    be asked the look is stock and the title "MindsHub"."""

    enabled: bool = False
    org_name: str | None = None
    sidebar_title: str | None = None
    icon_url: str | None = None
    logo_url: str | None = None
    logo_dark_url: str | None = None
    appearance: OrgAppearanceView | None = None
    can_edit: bool = False


# The fields a PUT passes to auth, as auth names them.
APPEARANCE_FIELDS: tuple[str, ...] = (
    "accent",
    "background_light",
    "background_dark",
    "font",
    "radius",
    "skin",
    "default_theme",
    "greeting",
    "sidebar_title_mode",
    "sidebar_title",
    "members_can_choose",
    "guidelines",
)


class OrgAppearanceUpdate(CamelRequest):
    """A change to the org's look, passed to auth as it is: a field left out
    is left out, and an explicit null resets it there. A client from before
    the light/dark split may send ``background`` alone, which sets both."""

    accent: str | None = None
    background: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: str | None = None
    radius: int | None = None
    skin: str | None = None
    default_theme: str | None = None
    greeting: str | None = None
    sidebar_title_mode: str | None = None
    sidebar_title: str | None = None
    members_can_choose: bool | None = None
    guidelines: str | None = None

    def auth_changes(self) -> dict[str, Any]:
        """The body for auth's PATCH: the fields sent, under auth's names."""
        sent = self.model_fields_set
        changes = {name: getattr(self, name) for name in APPEARANCE_FIELDS if name in sent}
        if "background" in sent and not ({"background_light", "background_dark"} & sent):
            changes["background_light"] = changes["background_dark"] = self.background
        return changes
