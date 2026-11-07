"""Wire shapes for an organization's branding (ENG-4020).

``GET /api/v1/org-branding`` answers OrgBrandingView in camelCase. ``PUT
/api/v1/org-branding/appearance`` takes OrgAppearanceUpdate in snake_case
(camelCase accepted too). auth owns the look and validates it again, but the
body is never forwarded as it came: it is parsed here into a strict model
(unknown fields are a 422) checked against what the Cowork app can apply,
and only its allowed fields are sent to auth's ``PATCH
/v1/organizations/current/branding/appearance/`` (auth_changes).

The allowed values mirror the Cowork app (mindsdb/cowork):
src/renderer/lib/customTheme.ts (CustomTheme: the accent and backgrounds are
#rrggbb hex, ``font`` is 'standard' | 'mono', ``radius`` is clamped to 0..16
px) and src/renderer/lib/skins.ts (STOCK_SKINS: normal, 8bit, custom). Keep
these in step with the app and with auth.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from cowork.schemas.base import CamelRequest, CamelResponse

APP_FONTS: tuple[str, ...] = ("standard", "mono")
APP_SKINS: tuple[str, ...] = ("normal", "8bit", "custom")
RADIUS_MIN = 0
RADIUS_MAX = 16
GREETING_MAX_CHARS = 200
SIDEBAR_TITLE_MAX_CHARS = 40
GUIDELINES_MAX_CHARS = 8000
SIDEBAR_CUSTOM_NEEDS_TITLE = "A custom sidebar title needs sidebar_title: 1 to 40 characters."

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


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


def _blank_to_none(value: object) -> object:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


class OrgAppearanceUpdate(CamelRequest):
    """A change to the org's look. A field left out is left out; an explicit
    null resets it in auth. A client from before the light/dark split may
    send ``background`` alone, which sets both. Any other field is a 422."""

    model_config = {**CamelRequest.model_config, "extra": "forbid"}

    accent: str | None = None
    background: str | None = None
    background_light: str | None = None
    background_dark: str | None = None
    font: Literal["standard", "mono"] | None = None
    radius: int | None = Field(default=None, ge=RADIUS_MIN, le=RADIUS_MAX, strict=True)
    skin: Literal["normal", "8bit", "custom"] | None = None
    default_theme: Literal["light", "dark"] | None = None
    greeting: str | None = Field(default=None, max_length=GREETING_MAX_CHARS)
    sidebar_title_mode: Literal["company", "custom", "none"] | None = None
    sidebar_title: str | None = Field(default=None, max_length=SIDEBAR_TITLE_MAX_CHARS)
    members_can_choose: bool | None = Field(default=None, strict=True)
    guidelines: str | None = Field(default=None, max_length=GUIDELINES_MAX_CHARS)

    @field_validator("accent", "background", "background_light", "background_dark", mode="before")
    @classmethod
    def _hex(cls, value: object) -> object:
        value = _blank_to_none(value)
        if value is None:
            return None
        if not isinstance(value, str) or not _HEX.match(value):
            raise ValueError("must be a hex color like #1a2b3c")
        return value.lower()

    @field_validator("font", "skin", "default_theme", "sidebar_title_mode", "greeting", "sidebar_title",
                     "guidelines", mode="before")
    @classmethod
    def _text(cls, value: object) -> object:
        return _blank_to_none(value)

    @model_validator(mode="after")
    def _custom_needs_a_title(self) -> OrgAppearanceUpdate:
        if self.sidebar_title_mode == "custom" and not self.sidebar_title:
            raise ValueError(SIDEBAR_CUSTOM_NEEDS_TITLE)
        return self

    def auth_changes(self) -> dict[str, Any]:
        """A new dict for auth's PATCH, built from the validated fields the
        body set, under auth's names, and nothing else."""
        sent = self.model_dump(exclude_unset=True)
        changes: dict[str, Any] = {name: sent[name] for name in APPEARANCE_FIELDS if name in sent}
        if "background" in sent and "background_light" not in sent and "background_dark" not in sent:
            changes["background_light"] = changes["background_dark"] = sent["background"]
        return changes
