"""Existing rows require explicit setup; their routing data remains stored."""
from uuid import uuid4

from alembic import command
from sqlalchemy import create_engine, inspect, text

from cowork.db.migrations import _alembic_config


def test_binding_upgrade_and_downgrade_preserve_legacy_rows(tmp_path):
    uri = f"sqlite:///{tmp_path / 'bindings.db'}"
    engine = create_engine(uri)
    config = _alembic_config(uri)
    ids = [uuid4().hex, uuid4().hex]
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "3e4b5f7586d3")
        for ident, partition in zip(ids, [None, "private"]):
            connection.execute(text(
                "INSERT INTO channel_bindings "
                "(id, channel_type, external_group_id, external_thread_key, trigger_rule, instructions, org_id) "
                "VALUES (:id, 'telegram', :id, '__default__', 'always', 'old instructions', :partition)"
            ), {"id": ident, "partition": partition})
        command.upgrade(config, "8c2d6f4a9b10")
        rows = connection.execute(text(
            "SELECT id, instructions, org_id, operator_configured FROM channel_bindings ORDER BY id"
        )).all()
        assert {row.id for row in rows} == set(ids)
        assert all(row.instructions == "old instructions" and not row.operator_configured for row in rows)
        assert {row.org_id for row in rows} == {None, "private"}
        command.downgrade(config, "3e4b5f7586d3")
        assert "operator_configured" not in {column["name"] for column in inspect(connection).get_columns("channel_bindings")}
        assert connection.execute(text("SELECT COUNT(*) FROM channel_bindings")).scalar_one() == 2
        command.upgrade(config, "8c2d6f4a9b10")
        assert connection.execute(text("SELECT COUNT(*) FROM channel_bindings WHERE operator_configured")).scalar_one() == 0
    engine.dispose()
