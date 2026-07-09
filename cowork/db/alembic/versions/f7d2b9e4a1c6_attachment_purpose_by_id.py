"""re-key attachment purposes by conversation id only

Old format: "attachment:{project_name}:{session_id}" — coupling the tag to
the mutable project name stranded every existing attachment the moment a
project was renamed (ENG-338). New format: "attachment:{session_id}".

Rewrites every old-format row by keeping only the segment after the LAST
colon (project names may themselves contain colons; session ids never do —
they are UUIDs or client-allocated ids minted without colons). Rows already
in the new format (exactly one colon) and non-attachment purposes are left
untouched, so the migration is idempotent.

Downgrade is a no-op: the project-name segment is not recoverable from the
tag alone, and the old code can still relink by conversation id, so nothing
is lost by leaving new-format tags in place.

Revision ID: f7d2b9e4a1c6
Revises: e8b3c5d7a9f1
Create Date: 2026-07-09 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7d2b9e4a1c6"
down_revision: Union[str, Sequence[str], None] = "e8b3c5d7a9f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def rekeyed_purpose(purpose: str) -> str | None:
    """New-format tag for an old-format attachment purpose, or None if the
    row needs no rewrite. Pure so it can be unit-tested directly."""
    if not purpose.startswith("attachment:"):
        return None
    rest = purpose[len("attachment:"):]
    if ":" not in rest:
        return None  # already new-format
    session_id = rest.rsplit(":", 1)[1]
    return f"attachment:{session_id}"


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT id, purpose FROM files WHERE purpose LIKE 'attachment:%'")
    ).fetchall()
    for row_id, purpose in rows:
        new = rekeyed_purpose(purpose)
        if new is not None:
            bind.execute(
                sa.text("UPDATE files SET purpose = :new WHERE id = :id"),
                {"new": new, "id": row_id},
            )


def downgrade() -> None:
    """Downgrade schema — intentionally a no-op (see module docstring)."""
