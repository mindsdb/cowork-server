import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Setting(BaseSQLModel, table=True):
    __tablename__ = "settings"

    # Per-scope uniqueness (migration c8e1a4f7b2d9): one global row (scope NULL),
    # one per org, one per (org, user) — the same user in two orgs must not share
    # a row. The CHECK pins the valid row shapes. Mirrors the migration.
    __table_args__ = (
        sa.Index(
            "uq_settings_key_global",
            "key",
            unique=True,
            sqlite_where=sa.text("scope IS NULL"),
            postgresql_where=sa.text("scope IS NULL"),
        ),
        sa.Index(
            "uq_settings_key_org",
            "key",
            "org_id",
            unique=True,
            sqlite_where=sa.text("scope = 'org'"),
            postgresql_where=sa.text("scope = 'org'"),
        ),
        sa.Index(
            "uq_settings_key_user",
            "key",
            "org_id",
            "user_id",
            unique=True,
            sqlite_where=sa.text("scope = 'user'"),
            postgresql_where=sa.text("scope = 'user'"),
        ),
        sa.CheckConstraint(
            "(scope IS NULL AND org_id IS NULL AND user_id IS NULL) OR "
            "(scope = 'org' AND org_id IS NOT NULL AND user_id IS NULL) OR "
            "(scope = 'user' AND org_id IS NOT NULL AND user_id IS NOT NULL)",
            name="ck_settings_scope_shape",
        ),
    )

    # No longer globally unique — uniqueness is per-scope (above); the plain
    # index backs the by-key fallback lookups.
    key: str = Field(max_length=128, index=True)
    value: str
    scope: str | None = Field(default=None, max_length=16, description="'org' | 'user'; NULL = legacy/global row")
    user_id: str | None = Field(default=None, max_length=36, description="Owning user for user-scoped rows")
    org_id: str | None = Field(default=None, max_length=36, description="Owning org for org-scoped rows")
