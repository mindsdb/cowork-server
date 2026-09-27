"""Upgrade proof for the message seq backfill (3e4b5f7586d3).

`seq` was added with server_default 0 and no backfill, so every row written
before that migration shares seq 0. Cursor pagination keys on seq, so until
those rows are renumbered a paginated read of a legacy conversation comes
back grouped by role in UUID order while the unbounded read of the same rows
comes back in history order — the same history rendering two different ways
depending on which branch the client hits.

Builds a real SQLite database at the previous head, seeds a legacy
conversation the way one actually looks (seq 0 everywhere, distinct
second-precision created_at), then upgrades and checks the two read paths
agree.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.services.conversations import ConversationService

PREVIOUS_HEAD = "e2262a14c001"

_ALEMBIC_DIR = Path(__file__).resolve().parent.parent / "cowork" / "db" / "alembic"


def _alembic_config() -> Config:
    """Alembic config built without alembic.ini on purpose.

    env.py calls `fileConfig(config.config_file_name)` whenever a file is
    attached, and that tears down and re-creates the root logger's handlers,
    disabling every logger configured before it. Any test relying on caplog
    that happens to run after this file would then fail. Passing the options
    directly leaves config_file_name None, so env.py skips the logging setup
    and only the migration runs.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    return cfg

# Interleaved roles, in real history order. The bug groups every user row
# ahead of every assistant row, so an interleaved fixture is what exposes it.
_LEGACY_ROWS = [
    ("user", "first question"),
    ("assistant", "first answer"),
    ("user", "second question"),
    ("assistant", "second answer"),
    ("user", "third question"),
    ("assistant", "third answer"),
]


@pytest.fixture()
def legacy_db(tmp_path, monkeypatch):
    """A SQLite file at the previous head holding one pre-seq conversation."""
    db_path = tmp_path / "cowork.db"
    monkeypatch.setenv("DATABASE_URI", f"sqlite:///{db_path}")
    get_app_settings.cache_clear()

    cfg = _alembic_config()
    command.upgrade(cfg, PREVIOUS_HEAD)

    project_id = uuid4().hex
    conversation_id = uuid4().hex
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO projects (id, name, path, is_active) "
            "VALUES (:id, 'legacy', '/tmp/legacy', 1)"
        ), {"id": project_id})
        conn.execute(text(
            "INSERT INTO conversations (id, topic, project_id) "
            "VALUES (:id, 'legacy chat', :pid)"
        ), {"id": conversation_id, "pid": project_id})
        for minute, (role, text_content) in enumerate(_LEGACY_ROWS):
            conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, seq, created_at) "
                "VALUES (:id, :cid, :role, :content, 0, :created_at)"
            ), {
                "id": uuid4().hex,
                "cid": conversation_id,
                "role": role,
                "content": f'"{text_content}"',
                "created_at": f"2026-01-01 00:{minute:02d}:00",
            })

    yield cfg, db_path, conversation_id
    get_app_settings.cache_clear()


def _service(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    return ConversationService(ScopedSession(Session(engine), LOCAL_SCOPE))


def test_upgrade_renumbers_legacy_rows_into_a_dense_ordinal(legacy_db):
    cfg, db_path, conversation_id = legacy_db
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        seqs = conn.execute(text(
            "SELECT seq FROM messages WHERE conversation_id = :cid ORDER BY created_at"
        ), {"cid": conversation_id}).scalars().all()
    assert seqs == list(range(len(_LEGACY_ROWS)))


def test_paginated_and_unbounded_reads_agree_after_upgrade(legacy_db):
    cfg, db_path, conversation_id = legacy_db
    command.upgrade(cfg, "head")

    svc = _service(db_path)
    cid = UUID(conversation_id)
    unbounded = [m["content"] for m in svc.get_messages(cid)]
    # Small limit so this walks several pages rather than returning everything
    # in one, which is where a degenerate cursor duplicates or skips rows.
    walked: list[str] = []
    before = None
    while True:
        page = svc.get_messages_page(cid, limit=2, before=before)
        walked = [m["content"] for m in page.items] + walked
        if not page.has_more:
            break
        before = page.next_before

    assert unbounded == [content for _, content in _LEGACY_ROWS]
    assert walked == unbounded


def test_conversations_written_after_the_seq_migration_are_untouched(legacy_db, tmp_path):
    """The backfill only renumbers conversations whose rows are actually tied."""
    cfg, db_path, _ = legacy_db
    engine = create_engine(f"sqlite:///{db_path}")
    project_id = uuid4().hex
    modern_id = uuid4().hex
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO projects (id, name, path, is_active) "
            "VALUES (:id, 'modern', '/tmp/modern', 1)"
        ), {"id": project_id})
        conn.execute(text(
            "INSERT INTO conversations (id, topic, project_id) VALUES (:id, 'new chat', :pid)"
        ), {"id": modern_id, "pid": project_id})
        for i, (role, text_content) in enumerate(_LEGACY_ROWS):
            conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, seq, created_at) "
                "VALUES (:id, :cid, :role, :content, :seq, :created_at)"
            ), {
                "id": uuid4().hex, "cid": modern_id, "role": role,
                "content": f'"{text_content}"', "seq": i * 10,
                "created_at": "2026-02-01 00:00:00",
            })

    command.upgrade(cfg, "head")

    with engine.begin() as conn:
        seqs = conn.execute(text(
            "SELECT seq FROM messages WHERE conversation_id = :cid ORDER BY seq"
        ), {"cid": modern_id}).scalars().all()
    assert seqs == [i * 10 for i in range(len(_LEGACY_ROWS))]


def test_a_conversation_spanning_the_seq_migration_keeps_its_modern_turns_paired(legacy_db):
    """The hard case: legacy rows tied at seq 0, then modern rows with real
    seq and the two created_at formats the live write path produces.

    Ordering the backfill by created_at first would swap each modern
    question/answer pair, because SQLite compares
    '...:00.500000' against '...:00' lexicographically — and it would be
    permanent, since afterwards the conversation's seqs are distinct and the
    HAVING filter never selects it again.
    """
    cfg, db_path, conversation_id = legacy_db
    engine = create_engine(f"sqlite:///{db_path}")
    modern = [
        ("user", "modern q1", 1, "2026-08-10 11:00:00.500000"),
        ("assistant", "modern a1", 2, "2026-08-10 11:00:00"),
        ("user", "modern q2", 3, "2026-08-10 11:00:09.900000"),
        ("assistant", "modern a2", 4, "2026-08-10 11:00:09"),
    ]
    with engine.begin() as conn:
        for role, content, seq, created_at in modern:
            conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, seq, created_at) "
                "VALUES (:id, :cid, :role, :content, :seq, :created_at)"
            ), {
                "id": uuid4().hex, "cid": conversation_id, "role": role,
                "content": f'"{content}"', "seq": seq, "created_at": created_at,
            })

    command.upgrade(cfg, "head")

    svc = _service(db_path)
    cid = UUID(conversation_id)
    expected = [content for _, content in _LEGACY_ROWS] + [c for _, c, _, _ in modern]
    assert [m["content"] for m in svc.get_messages(cid)] == expected
    assert [m["content"] for m in svc.get_messages_page(cid, limit=50).items] == expected


def test_a_legacy_turn_completed_inside_one_second_keeps_question_before_answer(legacy_db):
    """Legacy rows can share a created_at to the second, so the role tiebreak
    is what orders them — the backfill must not renumber the answer first.

    The fixture above gives every legacy row its own minute, which is the one
    shape where this cannot go wrong. Pre-seq rows all came from the column
    default (the explicit-datetime write path arrived three weeks after the seq
    column), so they share one format and tie exactly rather than comparing as
    unequal-length strings.
    """
    cfg, db_path, _ = legacy_db
    engine = create_engine(f"sqlite:///{db_path}")
    project_id = uuid4().hex
    same_second_id = uuid4().hex
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO projects (id, name, path, is_active) VALUES (:id, 'ss', '/tmp/ss', 1)"
        ), {"id": project_id})
        conn.execute(text(
            "INSERT INTO conversations (id, topic, project_id) VALUES (:id, 'same second', :pid)"
        ), {"id": same_second_id, "pid": project_id})
        for role, content in [
            ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"),
        ]:
            conn.execute(text(
                "INSERT INTO messages (id, conversation_id, role, content, seq, created_at) "
                "VALUES (:id, :cid, :role, :content, 0, '2026-07-01 10:00:00')"
            ), {"id": uuid4().hex, "cid": same_second_id, "role": role, "content": f'"{content}"'})

    command.upgrade(cfg, "head")

    svc = _service(db_path)
    cid = UUID(same_second_id)
    rendered = [m["content"] for m in svc.get_messages(cid)]
    paged = [m["content"] for m in svc.get_messages_page(cid, limit=50).items]
    # Both questions sort ahead of both answers here, because every row shares
    # one timestamp and only the role tiebreak separates them. What must not
    # happen is an answer taking a lower seq than the question it replies to.
    assert rendered.index("q1") < rendered.index("a1")
    assert rendered.index("q2") < rendered.index("a2")
    assert paged == rendered
