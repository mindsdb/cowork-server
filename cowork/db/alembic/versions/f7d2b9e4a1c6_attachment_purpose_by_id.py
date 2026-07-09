"""re-key attachment purposes by conversation id only

Old format: "attachment:{project_name}:{session_id}" — coupling the tag to
the mutable project name stranded every existing attachment the moment a
project was renamed (ENG-338). New format: "attachment:{session_id}".

Upgrade rewrites every old-format row by keeping only the segment after the
LAST colon (project names may themselves contain colons; session ids from
real clients — UUIDs or the legacy timestamp allocator — never do, and the
upload route now rejects colon-bearing ids). Rows already in the new format
and non-attachment purposes are untouched, so re-applying is a no-op.
`cowork.db.migrations` also runs the same rewrite on every boot as a safety
net for old-format rows written by an older build after this migration ran.

Downgrade is BEST-EFFORT: the project-name segment can usually be
reconstructed by resolving the tag's conversation id to its project
(files.purpose → conversations.project_id → projects.name). Rows whose
conversation no longer exists (or was never adopted) stay in the new
format — old code cannot see those either way, so nothing further is lost —
which makes rolling the app back after upgrading lossy for exactly that
subset. Prefer roll-forward.

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
    row needs no rewrite. Frozen twin of
    cowork.db.migrations.rekey_legacy_attachment_purpose (migrations stay
    self-contained; the live copy backs the every-boot safety net)."""
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
    """Best-effort restore of "attachment:{project}:{session}" tags.

    Resolves each new-format tag's session id against conversations (ids are
    stored as 32-char hex on SQLite, while tags carry the dashed string form,
    so both spellings are tried) and prepends the owning project's name.
    Unresolvable rows are left as-is (see module docstring).
    """
    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT id, purpose FROM files WHERE purpose LIKE 'attachment:%'")
    ).fetchall()
    for row_id, purpose in rows:
        rest = purpose[len("attachment:"):]
        if ":" in rest:
            continue  # already old-format (or colon-bearing legacy id) — leave
        project_name = bind.execute(
            sa.text(
                "SELECT p.name FROM conversations c "
                "JOIN projects p ON p.id = c.project_id "
                "WHERE c.id IN (:sid, :sid_hex)"
            ),
            {"sid": rest, "sid_hex": rest.replace("-", "")},
        ).scalar()
        if project_name is not None:
            bind.execute(
                sa.text("UPDATE files SET purpose = :old WHERE id = :id"),
                {"old": f"attachment:{project_name}:{rest}", "id": row_id},
            )
