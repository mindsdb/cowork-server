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
    background: str | None = Field(default=None, max_length=7, description="Background color, #rrggbb")
    font: str | None = Field(default=None, max_length=32, description="One of the app's font keys")
    radius: int | None = Field(default=None, description="Base corner radius in px")
    skin: str | None = Field(default=None, max_length=32, description="One of the app's skin keys")
    default_theme: str | None = Field(default=None, max_length=8, description="'light' or 'dark'")
    greeting: str | None = Field(default=None, max_length=200, description="The new-chat greeting")
    sidebar_title: str | None = Field(
        default=None, max_length=40, description="The title at the top of the app's sidebar"
    )
    members_can_choose: bool = Field(
        default=True, description="False: every member gets the org's look"
    )
    guidelines: str | None = Field(
        default=None,
        sa_type=sa.Text(),  # type: ignore[call-overload]
        description="The admin's markdown for the Brand guidelines skill",
    )
