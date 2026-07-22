from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Setting(BaseSQLModel, table=True):
    __tablename__ = "settings"

    # Scope columns below are inert until the settings split lands: the
    # unique `key` and the hot read path can't be org-scoped before the
    # composite-key schema. The scoped layer neither filters nor stamps this
    # table — see _TENANCY_DEFERRED_TABLES in cowork/db/scoped.py.

    key: str = Field(max_length=128, unique=True, index=True)
    value: str
    # No indexes yet — the only live query path is by `key`.
    scope: str | None = Field(default=None, max_length=16, description="'org' | 'user'; NULL = legacy/global row")
    user_id: str | None = Field(default=None, max_length=36, description="Owning user for user-scoped rows")
    org_id: str | None = Field(default=None, max_length=36, description="Owning org for org-scoped rows")
