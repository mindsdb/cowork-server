"""Upgrade proof for clearing the comparison usage snapshots (c33e840ac409).

The history list serves each side's stored ``usage_snapshot`` as it is. A
snapshot stored before the usage service answered with paid usage only holds
the list price and tokens of calls a plan covered, and it stays until a read of
its side succeeds, or for good when that read now fails.

Builds a real SQLite database at the previous head, seeds a side with such a
snapshot, upgrades, and checks the history list serves no figure for it.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlmodel import Session

from cowork.api.v1.endpoints.comparisons import list_comparisons
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession

PREVIOUS_HEAD = "a7c3e9f1b5d2"

_ALEMBIC_DIR = Path(__file__).resolve().parent.parent / "cowork" / "db" / "alembic"

_STORED_SNAPSHOT = {"estimated_cost_usd": 0.25, "tokens": 1050, "partial": False}


def _alembic_config() -> Config:
    """Alembic config built without alembic.ini, so env.py leaves the test
    run's logging alone (see tests/test_message_seq_backfill_migration.py)."""
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    return cfg


@pytest.fixture()
def stored_snapshot_db(tmp_path, monkeypatch):
    """A SQLite file at the previous head holding one comparison whose side a
    has a stored snapshot and whose side b has none."""
    db_path = tmp_path / "cowork.db"
    monkeypatch.setenv("DATABASE_URI", f"sqlite:///{db_path}")
    get_app_settings.cache_clear()

    cfg = _alembic_config()
    command.upgrade(cfg, PREVIOUS_HEAD)

    comparison_id = uuid4().hex
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO comparisons (id, title) VALUES (:id, 'stored')"), {"id": comparison_id})
        for label, snapshot in (("a", json.dumps(_STORED_SNAPSHOT)), ("b", None)):
            conn.execute(text(
                "INSERT INTO comparison_sides "
                "(id, comparison_id, label, model, project_id, conversation_id, usage_snapshot) "
                "VALUES (:id, :cid, :label, 'kimi', :pid, :conv, :snapshot)"
            ), {
                "id": uuid4().hex,
                "cid": comparison_id,
                "label": label,
                "pid": uuid4().hex,
                "conv": uuid4().hex,
                "snapshot": snapshot,
            })

    yield cfg, engine, comparison_id
    engine.dispose()
    get_app_settings.cache_clear()


def _listed_usage(engine, comparison_id: str) -> list:
    """Each side's ``usage`` as the history list serves it."""
    with Session(engine) as session:
        page = list_comparisons(ScopedSession(session, LOCAL_SCOPE))
    comparison = next(c for c in page["comparisons"] if c["id"] == UUID(comparison_id))
    return [side["usage"] for side in comparison["sides"]]


def test_upgrade_clears_a_snapshot_stored_before_paid_usage_only(stored_snapshot_db):
    cfg, engine, comparison_id = stored_snapshot_db
    # Before the upgrade the history serves the stored figure.
    assert _listed_usage(engine, comparison_id) == [
        {"estimatedCostUsd": 0.25, "tokens": 1050, "partial": False, "includedCalls": 0},
        None,
    ]

    command.upgrade(cfg, "head")

    assert _listed_usage(engine, comparison_id) == [None, None]
    with engine.begin() as conn:
        stored = conn.execute(text("SELECT usage_snapshot FROM comparison_sides ORDER BY label")).scalars().all()
    assert stored == [None, None]


def test_downgrade_leaves_the_cleared_snapshots_cleared(stored_snapshot_db):
    cfg, engine, comparison_id = stored_snapshot_db
    command.upgrade(cfg, "head")

    command.downgrade(cfg, PREVIOUS_HEAD)

    assert _listed_usage(engine, comparison_id) == [None, None]
