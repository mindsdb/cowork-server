"""An organization's look in the Cowork app, set by its admins (ENG-4020).

Hosted org mode only: one row per org, keyed by the org's UUID like
skill_access, read by every member (GET /api/v1/org-branding) and written by
an org admin (PUT /api/v1/org-branding/appearance). Every field is optional:
a NULL means the app's stock value. The org's images (icon, logo, dark logo)
are not here; auth stores them and serves them at fixed public URLs.

``guidelines`` is the admin's own markdown for the org's Brand guidelines
skill (cowork.services.org_branding), the only way that skill is edited.
"""

from __future__ import annotations

from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class OrgAppearance(BaseSQLModel, table=True):
    __tablename__ = "org_appearance"
    __table_args__ = (sa.UniqueConstraint("org_id", name="uq_org_appearance_org"),)

    org_id: UUID = Field(index=True, description="Owning organization")
    accent: str | None = Field(default=None, max_length=7, description="Accent color, #rrggbb")
    # Legacy single background: mirrored on every write as
    # background_light ?? background_dark, so a server or client from before
    # the split still reads one; read only as a fallback when neither of
    # the two below is set (cowork.services.org_branding.backgrounds).
    background: str | None = Field(default=None, max_length=7, description="Background color, #rrggbb")
    background_light: str | None = Field(
        default=None, max_length=7, description="Background in the light theme, #rrggbb"
    )
    background_dark: str | None = Field(
        default=None, max_length=7, description="Background in the dark theme, #rrggbb"
    )
    font: str | None = Field(default=None, max_length=32, description="One of the app's font keys")
    radius: int | None = Field(default=None, description="Base corner radius in px")
    skin: str | None = Field(default=None, max_length=32, description="One of the app's skin keys")
    default_theme: str | None = Field(default=None, max_length=8, description="'light' or 'dark'")
    greeting: str | None = Field(default=None, max_length=200, description="The new-chat greeting")
    sidebar_title: str | None = Field(
        default=None, max_length=40, description="The admin's own sidebar title, in 'custom' mode"
    )
    # 'company' (the org's name), 'custom' (sidebar_title) or 'none' (no
    # title, only the logo).
    sidebar_title_mode: str = Field(
        default="company",
        max_length=16,
        sa_column_kwargs={"server_default": "company"},
        description="Which sidebar title the app shows",
    )
    members_can_choose: bool = Field(
        default=True, description="False: every member gets the org's look"
    )
    guidelines: str | None = Field(
        default=None,
        sa_type=sa.Text(),  # type: ignore[call-overload]
        description="The admin's markdown for the Brand guidelines skill",
    )
