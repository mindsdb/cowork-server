"""Wire shapes for an organization's branding (ENG-4020).

``GET /api/v1/org-branding`` answers OrgBrandingView in camelCase. ``PUT
/api/v1/org-branding/appearance`` takes OrgAppearanceUpdate in snake_case
(camelCase accepted too).

The allowed values mirror what the Cowork app can apply, so an org look never
names something the app would ignore. Source: the cowork app (mindsdb/cowork)
src/renderer/lib/customTheme.ts (CustomTheme: ``accent`` and the backgrounds
are #rrggbb hex, ``font`` is 'standard' | 'mono', ``radius`` is clamped to
0..16 px) and src/renderer/lib/skins.ts (STOCK_SKINS: normal, 8bit, custom).
A deployment's own brand skin is not offered here: the deployment brand
always wins over the org's look anyway. Keep these in step with the app.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cowork.schemas.base import CamelRequest, CamelResponse

APP_FONTS: tuple[str, ...] = ("standard", "mono")
APP_SKINS: tuple[str, ...] = ("normal", "8bit", "custom")
RADIUS_MIN = 0
RADIUS_MAX = 16
GREETING_MAX_CHARS = 200
SIDEBAR_TITLE_MAX_CHARS = 40
GUIDELINES_MAX_CHARS = 8000

SidebarTitleMode = Literal["company", "custom", "none"]
SIDEBAR_CUSTOM_NEEDS_TITLE = "A custom sidebar title needs sidebar_title: 1 to 40 characters."

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class OrgAppearanceView(CamelResponse):
    """The org's look; every field null (stock) until an admin sets it."""

    accent: str | None = None
    # For clients from before the light/dark split: backgroundLight ??
    # backgroundDark.
    background: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: str | None = None
    radius: int | None = None
    skin: str | None = None
    default_theme: Literal["light", "dark"] | None = None
    greeting: str | None = None
    # The admin's own title, set only in 'custom' mode; OrgBrandingView's
    # sidebar_title is the one to show.
    sidebar_title: str | None = None
    sidebar_title_mode: SidebarTitleMode = "company"
    members_can_choose: bool | None = None
    guidelines: str | None = None


class OrgBrandingView(CamelResponse):
    """What every member's app reads on startup, org switch, sign-in and
    when Settings opens. Outside hosted org mode ``enabled`` is false and the
    rest is null. The image URLs are auth's (absolute, with ``?v=<hash>``),
    null when unset or when auth couldn't be asked. ``sidebar_title`` is the
    title to show, by the appearance's ``sidebar_title_mode``: the org's name
    when it is short enough, else the product's ('company'); the admin's
    text ('custom'); null for no title, only the logo ('none')
    (cowork.services.org_branding.resolve_sidebar_title).
    ``org_name`` is the org's display name from auth, null when unknown."""

    enabled: bool = False
    org_name: str | None = None
    sidebar_title: str | None = None
    icon_url: str | None = None
    logo_url: str | None = None
    logo_dark_url: str | None = None
    appearance: OrgAppearanceView | None = None
    can_edit: bool = False


def _blank_to_none(value: object) -> object:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


class OrgAppearanceUpdate(CamelRequest):
    """An admin's change to the org's look. A field the body leaves out
    keeps its stored value; an explicit null puts it back to stock."""

    accent: str | None = None
    # A client from before the light/dark split sends one background: it
    # sets both, unless the body names background_light or background_dark.
    background: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: str | None = None
    radius: int | None = Field(default=None, ge=RADIUS_MIN, le=RADIUS_MAX)
    skin: str | None = None
    default_theme: Literal["light", "dark"] | None = None
    greeting: str | None = Field(default=None, max_length=GREETING_MAX_CHARS)
    sidebar_title: str | None = Field(default=None, max_length=SIDEBAR_TITLE_MAX_CHARS)
    # 'custom' needs a non-blank sidebar_title; 'company' and 'none' clear
    # it. Left out, a sidebar_title in the body picks the mode as before: a
    # title means 'custom', null means 'company'.
    sidebar_title_mode: SidebarTitleMode | None = None
    members_can_choose: bool | None = None
    guidelines: str | None = Field(default=None, max_length=GUIDELINES_MAX_CHARS)

    @model_validator(mode="after")
    def _custom_needs_a_title(self) -> OrgAppearanceUpdate:
        if self.sidebar_title_mode == "custom" and not self.sidebar_title:
            raise ValueError(SIDEBAR_CUSTOM_NEEDS_TITLE)
        return self

    @field_validator("accent", "background", "background_light", "background_dark", mode="before")
    @classmethod
    def _hex(cls, value: object) -> object:
        value = _blank_to_none(value)
        if value is None:
            return None
        if not isinstance(value, str) or not _HEX.match(value):
            raise ValueError("must be a hex color like #1a2b3c")
        return value.lower()

    @field_validator("font", mode="before")
    @classmethod
    def _font(cls, value: object) -> object:
        value = _blank_to_none(value)
        if value is not None and value not in APP_FONTS:
            raise ValueError(f"must be one of: {', '.join(APP_FONTS)}")
        return value

    @field_validator("skin", mode="before")
    @classmethod
    def _skin(cls, value: object) -> object:
        value = _blank_to_none(value)
        if value is not None and value not in APP_SKINS:
            raise ValueError(f"must be one of: {', '.join(APP_SKINS)}")
        return value

    @field_validator("default_theme", mode="before")
    @classmethod
    def _theme(cls, value: object) -> object:
        return _blank_to_none(value)

    @field_validator("greeting", "sidebar_title", "guidelines", mode="before")
    @classmethod
    def _text(cls, value: object) -> object:
        return _blank_to_none(value)
