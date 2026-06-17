import ast
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlmodel import SQLModel

from alembic.script import ScriptDirectory

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import _alembic_config, run_schema_migrations

VERSIONS_DIR = Path(__file__).resolve().parent.parent / "cowork" / "db" / "alembic" / "versions"

# Alembic op calls that are destructive and violate additive-only policy.
_FORBIDDEN_OPS = frozenset({
    "drop_column",
    "drop_table",
    "drop_index",
    "rename_table",
    "alter_column",   # can rename or change type
})

# Import models so SQLModel.metadata can create a pre-Alembic legacy schema.
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.skill  # noqa: F401


def _sqlite_uri(path) -> str:
    return f"sqlite:///{path}"


def _message_columns(path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {row[1] for row in connection.execute("pragma table_info(messages)")}


def _alembic_version(path) -> str:
    with sqlite3.connect(path) as connection:
        return connection.execute("select version_num from alembic_version").fetchone()[0]


def expected_head() -> str:
    # Resolve the head from the script directory so new migrations don't
    # require updating a hardcoded revision here.
    return ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()


def test_schema_migrations_create_new_database(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "new.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)

    run_schema_migrations(engine, uri)

    assert "harness" in _message_columns(db_path)
    assert _alembic_version(db_path) == expected_head()


def test_schema_migrations_upgrade_pre_alembic_database(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "legacy.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)
    SQLModel.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE messages DROP COLUMN harness"))
        # A real pre-Alembic database predates the channel tables too; drop them
        # so the upgrade path recreates them via the channels migration.
        for table in ("channel_events", "channel_sessions", "channel_bindings", "channel_installations"):
            connection.execute(text(f"DROP TABLE IF EXISTS {table}"))

    run_schema_migrations(engine, uri)

    assert "harness" in _message_columns(db_path)
    assert _alembic_version(db_path) == expected_head()


def test_schema_migrations_skip_on_future_revision(tmp_path, monkeypatch):
    """A DB stamped at a revision we don't know (from a newer release) should
    not crash — migrations are skipped and the existing schema is left intact."""
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()

    db_path = tmp_path / "future.db"
    uri = _sqlite_uri(db_path)
    engine = create_engine(uri)

    # First, create a normal DB at head.
    run_schema_migrations(engine, uri)
    assert _alembic_version(db_path) == expected_head()

    # Simulate a newer release having stamped a revision we don't recognise.
    fake_future = "ffffffffface"
    with engine.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": fake_future})

    assert _alembic_version(db_path) == fake_future

    # Running migrations again should NOT crash and should leave the stamp alone.
    run_schema_migrations(engine, uri)
    assert _alembic_version(db_path) == fake_future


# ---------------------------------------------------------------------------
# Additive-only migration policy enforcement
# ---------------------------------------------------------------------------

def _upgrade_calls_in_file(path: Path) -> list[str]:
    """Parse a migration file and return op.* call names inside upgrade()."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "upgrade":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "op"
            ):
                calls.append(child.func.attr)
    return calls


def test_migrations_are_additive_only():
    """Every upgrade() must only use additive ops (add_column, create_table, etc.).

    Destructive ops (drop_column, drop_table, rename_table, alter_column) break
    the local-first downgrade-safety guarantee.  See cowork/db/migrations.py
    module docstring for the full policy.
    """
    violations: list[str] = []
    for migration in sorted(VERSIONS_DIR.glob("*.py")):
        if migration.name == "__init__.py":
            continue
        ops = _upgrade_calls_in_file(migration)
        bad = [op for op in ops if op in _FORBIDDEN_OPS]
        if bad:
            violations.append(f"{migration.name}: {', '.join(bad)}")

    assert not violations, (
        "Migrations must be additive-only (no drops, renames, or alter_column "
        "in upgrade()).  Violations:\n  " + "\n  ".join(violations)
    )
