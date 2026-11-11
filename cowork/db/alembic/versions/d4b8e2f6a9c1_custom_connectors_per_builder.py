"""custom_connectors: one definition per connector id per builder in an org

An org-mode custom connector belongs to the member who built it, so two
members of one org may each keep a definition under the same connector id.
Local rows (org_id NULL) keep their one-per-id rule.

Revision ID: d4b8e2f6a9c1
Revises: e2a7c4f9b613
Create Date: 2026-11-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4b8e2f6a9c1"
down_revision: Union[str, Sequence[str], None] = "e2a7c4f9b613"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "custom_connectors"


def _index_names() -> set[str]:
    """Index names on the table. A database created from the models before
    Alembic already has this revision's indexes and not the old one."""
    return {ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes(_TABLE)}


def upgrade() -> None:
    """Upgrade schema."""
    names = _index_names()
    if "uq_custom_connectors_connector_id_org" in names:
        op.drop_index("uq_custom_connectors_connector_id_org", table_name=_TABLE)
    if "uq_custom_connectors_connector_id_org_builder" not in names:
        op.create_index(
            "uq_custom_connectors_connector_id_org_builder",
            _TABLE,
            ["connector_id", "org_id", "created_by"],
            unique=True,
            sqlite_where=sa.text("org_id IS NOT NULL AND created_by IS NOT NULL"),
            postgresql_where=sa.text("org_id IS NOT NULL AND created_by IS NOT NULL"),
        )


def downgrade() -> None:
    """Downgrade schema. Aborts with the schema intact when two rows in one
    org share a connector id, which the one-per-org index cannot hold."""
    dupes = op.get_bind().exec_driver_sql(
        "SELECT connector_id, org_id FROM custom_connectors WHERE org_id IS NOT NULL "
        "GROUP BY connector_id, org_id HAVING COUNT(*) > 1"
    ).fetchall()
    if dupes:
        raise RuntimeError(
            f"cannot downgrade custom connectors per builder: {len(dupes)} connector id(s) "
            "are kept by more than one member of an org, which the one-per-org index "
            "cannot hold; remove the extra rows first. Nothing was changed."
        )
    op.drop_index("uq_custom_connectors_connector_id_org_builder", table_name=_TABLE)
    op.create_index(
        "uq_custom_connectors_connector_id_org",
        _TABLE,
        ["connector_id", "org_id"],
        unique=True,
        sqlite_where=sa.text("org_id IS NOT NULL"),
        postgresql_where=sa.text("org_id IS NOT NULL"),
    )
