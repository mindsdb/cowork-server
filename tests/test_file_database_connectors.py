"""SQLite and DuckDB connectors: a database file the agent opens read-only."""

import pytest

from cowork.services.connectors.specs._registry import ConnectorSpecRegistry


@pytest.fixture(scope="module")
def registry():
    return ConnectorSpecRegistry()


@pytest.mark.parametrize("connector_id", ["sqlite", "duckdb"])
def test_file_database_connector_takes_one_path(registry, connector_id):
    spec = registry.get_connector(connector_id)
    assert spec is not None
    assert spec.category == "database"
    [method] = spec.form.methods
    assert method.id == "file"
    assert [(f.name, f.required) for f in method.fields] == [("database_path", True)]


def test_usage_notes_open_both_files_read_only(registry):
    notes = registry.usage_notes_for(["sqlite", "duckdb"])
    assert "mode=ro" in notes["sqlite"]
    assert "read_only=True" in notes["duckdb"]
