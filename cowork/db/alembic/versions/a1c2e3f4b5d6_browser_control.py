"""browser_control: content-free read-only Browser Control tables

Adds the three tables backing Browser Control Milestone 1 (read-only):

- ``browser_sessions`` — one per conversation; holds control/bridge state
  and the single approved active domain.
- ``browser_tab_grants`` — per-domain, per-action-class permission grant.
- ``browser_actions`` — ordered history of brokered read-only actions with
  a content-free ``observed_result`` digest (JSON).

All columns are content-free by construction: host-only ``domain``, action
type/class, timing, and typed codes only. Guarded with ``_has_table`` so a
re-run (or a database that already carries the tables) is a no-op, matching
the ``d5f3a8c1e6b2_task_objects`` template.

Revision ID: a1c2e3f4b5d6
Revises: f7d2b9e4a1c6
Create Date: 2026-07-15 09:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c2e3f4b5d6"
down_revision: Union[str, Sequence[str], None] = "f7d2b9e4a1c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_table("browser_sessions"):
        op.create_table(
            "browser_sessions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("conversation_id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("control_state", sa.String(length=16), nullable=False),
            sa.Column("bridge_state", sa.String(length=24), nullable=False),
            sa.Column("active_domain", sa.String(length=255), nullable=True),
            sa.Column("available", sa.Boolean(), nullable=False),
            sa.Column("requires_reapproval", sa.Boolean(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "modified_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(
                ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "conversation_id", name="uq_browser_sessions_conversation"
            ),
        )
        op.create_index(
            "ix_browser_sessions_conversation_id",
            "browser_sessions",
            ["conversation_id"],
        )
        op.create_index(
            "ix_browser_sessions_project_id",
            "browser_sessions",
            ["project_id"],
        )

    if not _has_table("browser_tab_grants"):
        op.create_table(
            "browser_tab_grants",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("session_id", sa.Uuid(), nullable=False),
            sa.Column("domain", sa.String(length=255), nullable=False),
            sa.Column("action_class", sa.String(length=16), nullable=False),
            sa.Column("decision", sa.String(length=16), nullable=False),
            sa.Column("granted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "modified_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(
                ["session_id"], ["browser_sessions.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "session_id",
                "domain",
                "action_class",
                name="uq_browser_tab_grants_scope",
            ),
        )
        op.create_index(
            "ix_browser_tab_grants_session_id",
            "browser_tab_grants",
            ["session_id"],
        )
        op.create_index(
            "ix_browser_tab_grants_domain",
            "browser_tab_grants",
            ["domain"],
        )

    if not _has_table("browser_actions"):
        op.create_table(
            "browser_actions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("session_id", sa.Uuid(), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("command_id", sa.String(length=64), nullable=False),
            sa.Column("idempotency_key", sa.String(length=128), nullable=False),
            sa.Column("action_type", sa.String(length=16), nullable=False),
            sa.Column("action_class", sa.String(length=16), nullable=False),
            sa.Column("domain", sa.String(length=255), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("result_code", sa.String(length=24), nullable=True),
            sa.Column("observed_result", sa.JSON(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "modified_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(
                ["session_id"], ["browser_sessions.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "session_id", "sequence", name="uq_browser_actions_sequence"
            ),
            sa.UniqueConstraint(
                "command_id", name="uq_browser_actions_command_id"
            ),
        )
        op.create_index(
            "ix_browser_actions_session_id",
            "browser_actions",
            ["session_id"],
        )
        op.create_index(
            "ix_browser_actions_command_id",
            "browser_actions",
            ["command_id"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("browser_actions"):
        op.drop_index("ix_browser_actions_command_id", table_name="browser_actions")
        op.drop_index("ix_browser_actions_session_id", table_name="browser_actions")
        op.drop_table("browser_actions")
    if _has_table("browser_tab_grants"):
        op.drop_index("ix_browser_tab_grants_domain", table_name="browser_tab_grants")
        op.drop_index(
            "ix_browser_tab_grants_session_id", table_name="browser_tab_grants"
        )
        op.drop_table("browser_tab_grants")
    if _has_table("browser_sessions"):
        op.drop_index(
            "ix_browser_sessions_project_id", table_name="browser_sessions"
        )
        op.drop_index(
            "ix_browser_sessions_conversation_id", table_name="browser_sessions"
        )
        op.drop_table("browser_sessions")
