"""A per-account data root must not inherit the shared root's stores.

The desktop shell points the sidecar at a per-account root when a second
account signs in on the same machine. Both one-time store migrations key their
"already ran" sentinel on a DB row while reading sources that stay under the
shared COWORK_HOME, so a fresh per-account DB arrives sentinel-free and would
re-run them against the previous account's files — seeding that account's
provider keys and profile. Each test has a default-root control so a passing
skip cannot be mistaken for a migration that never did anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import (
    AppSettings,
    MemorySettings,
    get_app_settings,
)
from cowork.harnesses.memory.registry import MemorySlot
from cowork.models.setting import Setting

ACCOUNT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'account.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session


@pytest.fixture
def account_scoped(monkeypatch):
    monkeypatch.setenv("COWORK_ACCOUNT_ID", ACCOUNT_ID)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def default_root(monkeypatch):
    monkeypatch.delenv("COWORK_ACCOUNT_ID", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def shared_env_file(tmp_path, monkeypatch):
    """A populated `.env` at the shared root, as a previous account would leave."""
    env_path = tmp_path / "shared.env"
    env_path.write_text(
        "ANTON_ANTHROPIC_API_KEY=not-a-real-key-belonging-to-the-other-account\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("cowork.migrations._ENV_PATH", env_path)
    return env_path


@pytest.fixture
def shared_memory_file(tmp_path, monkeypatch):
    """A legacy anton profile at the shared root, as a previous account would leave."""
    profile = tmp_path / "anton" / "memory" / "profile.md"
    profile.parent.mkdir(parents=True)
    profile.write_text("The other account works at ExampleCorp\n", encoding="utf-8")
    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._MIGRATION_SOURCES",
        [(profile, MemorySlot.PROFILE)],
    )
    return profile


@pytest.fixture
def account_memory_root(tmp_path, monkeypatch):
    """This account's own (empty) canonical memory store."""
    root = tmp_path / "account-memory"
    monkeypatch.setattr(
        "cowork.harnesses.memory.store.get_app_settings",
        lambda: AppSettings(memory=MemorySettings(root_dir=str(root))),
    )
    return root


def test_env_seed_skipped_on_an_account_scoped_root(session, shared_env_file, account_scoped):
    from cowork.migrations import migrate_env_to_db

    assert migrate_env_to_db(session) is False
    # No settings at all, sentinel included: nothing was read and nothing claims
    # the migration ran, so the default root keeps owning it.
    assert session.exec(select(Setting)).all() == []


def test_env_seed_still_runs_on_the_default_root(session, shared_env_file, default_root):
    from cowork.migrations import migrate_env_to_db

    assert migrate_env_to_db(session) is True
    assert "anthropic_api_key" in {row.key for row in session.exec(select(Setting)).all()}


def test_memory_migration_skipped_on_an_account_scoped_root(
    session, shared_memory_file, account_memory_root, account_scoped
):
    from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared

    assert migrate_harness_memory_to_shared(session) is False
    assert session.exec(select(Setting)).all() == []
    assert shared_memory_file.exists(), "the shared source must not be consumed"
    assert not account_memory_root.exists(), "nothing may be written into this account's store"


def test_memory_migration_still_runs_on_the_default_root(
    session, shared_memory_file, account_memory_root, default_root
):
    from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared
    from cowork.harnesses.memory.store import GlobalMemoryStore

    assert migrate_harness_memory_to_shared(session) is True
    store = GlobalMemoryStore(root=account_memory_root)
    assert "ExampleCorp" in store.read(MemorySlot.PROFILE)


def test_coding_root_follows_its_env_override(tmp_path, monkeypatch):
    """The coding store had no override at all, so a second account inherited the
    first account's sessions and cloned repos."""
    from cowork.coding import service as coding_service

    root = tmp_path / "account-coding"
    captured: list[Path] = []

    def _capture(passed_root):
        captured.append(passed_root)
        return object()

    monkeypatch.setenv("COWORK_CODING_DIR", str(root))
    monkeypatch.setattr(coding_service, "CodingService", _capture)
    get_app_settings.cache_clear()
    coding_service.get_coding_service.cache_clear()
    try:
        coding_service.get_coding_service()
    finally:
        coding_service.get_coding_service.cache_clear()
        get_app_settings.cache_clear()

    assert captured == [root]


def test_coding_root_defaults_to_the_shared_home(monkeypatch):
    """Unset, the path must stay exactly where it has always been, so org
    deployments and single-account desktops are untouched."""
    monkeypatch.delenv("COWORK_CODING_DIR", raising=False)
    monkeypatch.delenv("CODING_ROOT_DIR", raising=False)
    get_app_settings.cache_clear()
    try:
        assert Path(get_app_settings().coding.root_dir) == cowork_home() / "coding"
    finally:
        get_app_settings.cache_clear()


def test_every_store_rooted_at_cowork_home_is_overridable(default_root):
    """A store whose default hangs off cowork_home() is shared by every account
    on the machine unless the desktop can point it elsewhere. This enumerates
    them so ADDING one fails here instead of silently leaking, and so each new
    one forces a decision about whether it is per-account.

    The desktop's own map lives in cowork/src/main/account-data.ts; the values
    below are the env aliases it sets.
    """
    from pydantic_settings import BaseSettings

    import cowork.harnesses.anton_harness.settings as anton_settings
    import cowork.harnesses.hermes_harness.settings as hermes_settings
    from cowork.common.settings import app_settings as app

    # group/field -> the env alias the desktop sets, or None when the store is
    # deliberately left shared (with the reason).
    overridable: dict[tuple[str, str], str | None] = {
        ("DatabaseSettings", "uri"): "DATABASE_URI",
        ("ProjectSettings", "root_dir"): "COWORK_PROJECTS_DIR",
        ("FileSettings", "root_dir"): "COWORK_FILES_DIR",
        ("MemorySettings", "root_dir"): "COWORK_MEMORY_DIR",
        ("ConnectorSettings", "vault_dir"): "COWORK_VAULT_DIR",
        ("StreamSettings", "dir"): "COWORK_STREAMS_DIR",
        ("CodingSettings", "root_dir"): "COWORK_CODING_DIR",
        ("SkillSettings", "root_dir"): "COWORK_SKILLS_DIR",
        ("OAuthSettings", "state_path"): "COWORK_OAUTH_STATE_PATH",
        ("HermesHarnessSettings", "root_dir"): "HERMES_ROOT_DIR",
        ("AntonHarnessSettings", "skills_root_dir"): "ANTON_SKILLS_ROOT_DIR",
        # Shared on purpose: one key per machine. A per-account key would orphan
        # vault contents on any path that crosses roots, and the vault directory
        # itself is per-account above.
        ("AppSettings", "master_key_path"): None,
        # Org mode only; local mode never reads it.
        ("StorageSettings", "shared_root"): None,
    }

    groups = [
        getattr(app, name)
        for name in dir(app)
        if isinstance(getattr(app, name), type)
        and issubclass(getattr(app, name), BaseSettings)
    ]
    groups += [
        hermes_settings.HermesHarnessSettings,
        anton_settings.AntonHarnessSettings,
    ]

    home = str(cowork_home())
    found: set[tuple[str, str]] = set()
    for group in groups:
        for field_name, field in getattr(group, "model_fields", {}).items():
            factory = getattr(field, "default_factory", None)
            if factory is None:
                continue
            try:
                value = factory()
            except Exception:
                continue
            # The aggregate fields on AppSettings build a whole settings group;
            # only their own leaf paths matter, and those are reached directly.
            if isinstance(value, BaseSettings):
                continue
            if home in str(value):
                found.add((group.__name__, field_name))

    unexpected = found - set(overridable)
    assert not unexpected, (
        "store(s) rooted at cowork_home() with no per-account decision: "
        f"{sorted(unexpected)}. Add an override in "
        "cowork/src/main/account-data.ts and list it here, or record why it "
        "stays shared."
    )

    # Every alias we claim to use must really be accepted by that field.
    by_name = {g.__name__: g for g in groups}
    for (group_name, field_name), alias in overridable.items():
        if alias is None or group_name == "DatabaseSettings":
            continue  # nested: DATABASE_URI resolves via env_nested_delimiter
        field = by_name[group_name].model_fields[field_name]
        aliases = getattr(field, "validation_alias", None)
        names = {str(c) for c in getattr(aliases, "choices", [aliases] if aliases else [])}
        assert alias in names, f"{group_name}.{field_name} does not accept {alias}"
