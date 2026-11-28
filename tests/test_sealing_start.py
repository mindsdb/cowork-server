"""The key file and the start checks for sealing under COWORK_KEY_FILE.

With a key file, its keys alone seal and open the server's secrets, a key file
that won't read stops the start naming the cause and line, and every sealed
setting, channel credential and saved connection must open before the server
serves. Without one, a deployment sealed earlier doesn't start, and no legacy
key is made beside its data. Org tenancy reads no vault either way, and a
deployment that never turned sealing on starts as it always has.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

import cowork.server as server
from cowork.channels.plugin import ChannelPlugin, CredentialField, CredentialSchema
from cowork.channels.registry import PluginRegistry
from cowork.common.encryption import (
    KeyFileError,
    UnopenableSettingError,
    decrypt,
    encrypt,
    get_fernet,
    sealing_keys,
    warn_ignored_key_settings,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.setting import Setting
from cowork.services import sealing
from cowork.services.channels import ChannelConfigService
from cowork.services.connectors.sealed_vault import SealedDataVault
from cowork.services.settings import SettingService
from tests._fakes import units_on_engine

CANARY = "zx-canary-7784"
# A person's partition folder under the shared root, which holds their vault.
PARTITION = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"


def _write_key_file(path: Path, *lines: bytes) -> None:
    path.write_bytes(b"\n".join(lines) + b"\n")
    sealing_keys.cache_clear()
    get_fernet.cache_clear()


@pytest.fixture(autouse=True)
def own_database(monkeypatch, tmp_path):
    """A database of the test's own: the start checks read every settings row, and other tests'
    rows in the shared one are sealed under their own keys."""
    uri = f"sqlite:///{tmp_path / 'sealing.db'}"
    monkeypatch.setenv("DATABASE_URI", uri)
    get_app_settings.cache_clear()
    engine = get_engine(uri)
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


@pytest.fixture
def stored_rows(own_database):
    """Settings rows a test adds, in its own database."""

    def add(key: str, value: str) -> None:
        with Session(own_database) as db:
            db.add(Setting(key=key, value=value))
            db.commit()

    return add


@pytest.fixture
def base_vault(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    yield tmp_path / "vault"
    get_app_settings.cache_clear()


@pytest.fixture
def legacy_key(monkeypatch, tmp_path) -> Path:
    """A legacy key path with no key there yet, and no COWORK_MASTER_KEY."""
    path = tmp_path / "home" / ".master_key"
    monkeypatch.setenv("MASTER_KEY_PATH", str(path))
    monkeypatch.delenv("COWORK_MASTER_KEY", raising=False)
    monkeypatch.delenv("COWORK_KEY_FILE", raising=False)
    get_app_settings.cache_clear()
    sealing_keys.cache_clear()
    get_fernet.cache_clear()
    yield path
    get_app_settings.cache_clear()
    sealing_keys.cache_clear()
    get_fernet.cache_clear()


# ── the key file ────────────────────────────────────────────────────────────


def test_a_missing_key_file_stops_the_start(key_file):
    key_file.unlink()

    with pytest.raises(KeyFileError, match="doesn't exist"):
        sealing_keys()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_key_file_stops_the_start(key_file):
    key_file.chmod(0)
    try:
        with pytest.raises(KeyFileError, match="can't read"):
            sealing_keys()
    finally:
        key_file.chmod(0o400)


def test_a_malformed_line_is_named_by_its_number_and_no_key_is_printed(key_file):
    good = Fernet.generate_key()
    _write_key_file(key_file, good, b"not-a-key")

    with pytest.raises(KeyFileError) as raised:
        sealing_keys()

    assert "line 2" in str(raised.value)
    assert good.decode() not in str(raised.value)


def test_a_key_file_with_no_key_stops_the_start(key_file):
    _write_key_file(key_file, b"")

    with pytest.raises(KeyFileError, match="holds no key"):
        sealing_keys()


def test_the_first_key_seals_and_every_key_opens(key_file):
    new, old = Fernet.generate_key(), Fernet.generate_key()
    _write_key_file(key_file, new, old)

    assert Fernet(new).decrypt(encrypt("sealed-now").encode()) == b"sealed-now"
    assert decrypt(Fernet(old).encrypt(b"sealed-before").decode()) == "sealed-before"


def test_a_key_file_overrides_the_legacy_key_lines_in_cowork_homes_env(monkeypatch, caplog, key_file, tmp_path):
    """The legacy key settings as lines of cowork-home's .env, which is where a
    desktop install keeps them, not as process variables."""
    from cowork.common.settings import app_settings

    legacy_key = tmp_path / "home" / ".master_key"
    legacy_value = Fernet.generate_key().decode()
    env_file = tmp_path / ".env"
    env_file.write_text(f"COWORK_MASTER_KEY={legacy_value}\nMASTER_KEY_PATH={legacy_key}\n")
    monkeypatch.delenv("COWORK_MASTER_KEY", raising=False)
    monkeypatch.delenv("MASTER_KEY_PATH", raising=False)
    settings = app_settings.AppSettings(_env_file=env_file)
    monkeypatch.setattr(app_settings, "get_app_settings", lambda: settings)
    get_fernet.cache_clear()
    assert (settings.master_key, settings.master_key_path) == (legacy_value, str(legacy_key))

    token = encrypt(CANARY)

    assert Fernet(key_file.read_bytes().strip()).decrypt(token.encode()) == CANARY.encode()
    with pytest.raises(InvalidToken):
        Fernet(legacy_value).decrypt(token.encode())
    assert not legacy_key.exists()
    with caplog.at_level(logging.WARNING):
        warn_ignored_key_settings(
            master_key=settings.master_key,
            master_key_path_set="master_key_path" in settings.model_fields_set,
            code_sandbox="off",
        )
    logged = [r.getMessage() for r in caplog.records if r.name == "cowork.common.encryption"]
    assert logged == [
        "COWORK_MASTER_KEY is ignored while COWORK_KEY_FILE is set",
        "MASTER_KEY_PATH is ignored while COWORK_KEY_FILE is set",
        "COWORK_CODE_SANDBOX is off, so agent code can read the key file COWORK_KEY_FILE names",
    ]
    assert settings.master_key not in caplog.text


# ── settings and channel credentials ────────────────────────────────────────


def test_a_setting_the_key_file_doesnt_open_raises_instead_of_reading_unset(key_file):
    foreign = Fernet(Fernet.generate_key())
    row = Setting(key="anthropic_api_key", value=foreign.encrypt(CANARY.encode()).decode())

    with pytest.raises(UnopenableSettingError) as raised:
        SettingService._load([row])

    assert "anthropic_api_key" in str(raised.value) and CANARY not in str(raised.value)


@pytest.fixture
def channel_service(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    units_on_engine(monkeypatch, engine)
    registry = PluginRegistry()
    registry.register(
        ChannelPlugin(
            channel_type="slack",
            display_name="Slack",
            factory=lambda **_kwargs: None,
            credentials=CredentialSchema(fields=(CredentialField(name="bot_token", label="Bot token"),)),
        )
    )
    session = Session(engine)
    yield ChannelConfigService(ScopedSession(session, LOCAL_SCOPE), registry=registry), session
    session.close()


def test_a_channel_secret_is_sealed_under_the_key_file(key_file, channel_service):
    service, session = channel_service

    service.set_config("slack", {"bot_token": "xoxb-real"})

    row = session.query(Setting).filter(Setting.key == "channel.slack.bot_token").one()
    assert Fernet(key_file.read_bytes().strip()).decrypt(row.value.encode()) == b"xoxb-real"
    assert service.load_credentials("slack") == {"bot_token": "xoxb-real"}
    row.value = Fernet(Fernet.generate_key()).encrypt(b"xoxb-real").decode()
    session.commit()
    with pytest.raises(UnopenableSettingError, match="channel.slack.bot_token"):
        service.load_credentials("slack")


# ── with a key file: every sealed thing must open ───────────────────────────


def test_the_start_names_a_setting_the_key_file_doesnt_open(key_file, base_vault, stored_rows):
    stored_rows("anthropic_api_key", Fernet(Fernet.generate_key()).encrypt(CANARY.encode()).decode())

    with pytest.raises(sealing.SealedDataError) as raised:
        sealing.refuse_unopenable_sealed_data()

    assert "setting anthropic_api_key (global)" in str(raised.value)
    assert CANARY not in str(raised.value)


def test_the_start_names_a_connection_that_isnt_sealed(key_file, base_vault):
    LocalDataVault(base_vault).save("postgres", "sales", {"password": CANARY})

    with pytest.raises(sealing.SealedDataError) as raised:
        sealing.refuse_unopenable_sealed_data()

    assert "connection postgres/sales" in str(raised.value) and "isn't sealed" in str(raised.value)
    assert CANARY not in str(raised.value)


def test_the_start_passes_when_everything_sealed_opens(key_file, base_vault, stored_rows):
    SealedDataVault(base_vault, keys=sealing_keys()).save("postgres", "sales", {"password": CANARY})
    stored_rows("anthropic_api_key", encrypt(CANARY))

    sealing.refuse_unopenable_sealed_data()


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any folder")
@pytest.mark.parametrize("unsearchable", ["partition", "shared root"])
def test_the_start_names_a_vault_the_server_cant_search(key_file, base_vault, tmp_path, unsearchable):
    shared_root = tmp_path / "shared"
    partition = shared_root / PARTITION
    (partition / "data-vault").mkdir(parents=True)
    if unsearchable == "partition":
        locked, named = partition, partition / "data-vault"
    else:
        locked, named = shared_root, shared_root
    locked.chmod(0)
    try:
        with pytest.raises(sealing.SealedDataError) as raised:
            sealing.refuse_unopenable_sealed_data()
    finally:
        locked.chmod(0o700)

    assert f"vault {named}: can't be read (EACCES)" in str(raised.value)


def test_an_org_start_with_a_key_file_reads_no_vault(monkeypatch, org_deployment, key_file):
    monkeypatch.setattr(sealing, "vault_roots", lambda: pytest.fail("an org start must not read a vault"))

    sealing.refuse_unopenable_sealed_data()


async def test_the_start_stops_after_migrations_when_sealed_data_wont_open(monkeypatch, key_file, base_vault):
    LocalDataVault(base_vault).save("postgres", "sales", {"password": CANARY})
    ran: list[str] = []
    monkeypatch.setattr(server, "run_dev_setup", lambda: ran.append("dev_setup"))

    with pytest.raises(sealing.SealedDataError):
        async with server.lifespan(FastAPI()):
            pass

    assert ran == ["dev_setup"]


# ── without a key file: a sealed deployment doesn't start ───────────────────


def test_no_key_file_and_the_marker_refuses_the_start(legacy_key, base_vault, stored_rows):
    stored_rows(sealing.SEALING_MARKER_KEY, "1")

    with pytest.raises(KeyFileError, match="Sealing was turned on"):
        sealing.refuse_sealed_state_without_key_file()


_ROOT = Path(__file__).resolve().parents[1]
# What an operator's script runs in the server's container: it imports only what it names, so
# no model a relationship names by string, such as a project's Schedule, is loaded.
_LEGACY_KEY_IN_A_FRESH_INTERPRETER = "from cowork.common.encryption import get_fernet\nget_fernet()\n"


def _in_a_fresh_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=dict(os.environ),
                          capture_output=True, text=True, timeout=60)


def test_no_key_file_and_a_fresh_interpreter_reaches_the_legacy_key_and_can_query_after_it(legacy_key, base_vault):
    """The check loads no model of its own, so the script's own query configures only what it loaded."""
    ran = _in_a_fresh_interpreter(
        _LEGACY_KEY_IN_A_FRESH_INTERPRETER
        + "from sqlmodel import Session, select\n"
        "from cowork.common.settings.app_settings import get_app_settings\n"
        "from cowork.db.session import get_engine\n"
        "from cowork.models.setting import Setting\n"
        "with Session(get_engine(get_app_settings().database.uri)) as session:\n"
        "    session.exec(select(Setting)).all()\n"
    )

    assert ran.returncode == 0, ran.stderr
    assert legacy_key.is_file()


def test_no_key_file_and_the_marker_refuse_a_fresh_interpreter_with_the_reason(legacy_key, base_vault, stored_rows):
    stored_rows(sealing.SEALING_MARKER_KEY, "1")

    ran = _in_a_fresh_interpreter(_LEGACY_KEY_IN_A_FRESH_INTERPRETER)

    assert ran.returncode != 0
    assert "KeyFileError: Sealing was turned on" in ran.stderr, ran.stderr
    assert not legacy_key.exists()


def test_no_key_file_and_a_sealed_record_refuses_the_start(legacy_key, base_vault):
    SealedDataVault(base_vault, keys=MultiFernet([Fernet(Fernet.generate_key())])).save("postgres", "sales", {"password": CANARY})

    with pytest.raises(KeyFileError, match="postgres-sales"):
        sealing.refuse_sealed_state_without_key_file()


def test_no_key_file_and_plain_or_broken_records_start_as_today(legacy_key, base_vault):
    LocalDataVault(base_vault).save("postgres", "sales", {"password": CANARY})
    (base_vault / "postgres-broken").write_text("{not json")

    sealing.refuse_sealed_state_without_key_file()


@pytest.mark.parametrize("mark", ["1", True, 1.0, 1], ids=["string", "bool", "float", "int"])
def test_no_key_file_and_a_plain_record_with_a_field_named_like_the_mark_start_as_today(legacy_key, base_vault, mark):
    """A person can ask the agent to save a custom connection with a variable
    named _sealed. Its record stays plain, whatever that variable holds, so
    the start and the first legacy key go ahead as before."""
    LocalDataVault(base_vault).save("acme_api", "acme", {"_sealed": mark, "api_token": CANARY})

    sealing.refuse_sealed_state_without_key_file()
    encrypt(CANARY)

    assert legacy_key.exists()


@pytest.mark.parametrize(
    "changed",
    [{"host": "db.example"}, {"host": 5432}, {"_sealed": "1"}, {"_sealed": True}],
    ids=["mark-beside-text", "mark-beside-number", "string-mark-beside-tokens", "bool-mark-beside-tokens"],
)
def test_no_key_file_and_a_record_with_tokens_that_isnt_wholly_sealed_start_as_today(legacy_key, base_vault, changed):
    """Only the integer mark beside values that are all tokens reads as sealed. A sealed record
    with its host put back as plain text or a number, or with its mark made a string or a bool,
    still holds a token, and it starts as a plain record does: the start goes ahead and the first
    legacy key is made."""
    keys = MultiFernet([Fernet(Fernet.generate_key())])
    SealedDataVault(base_vault, keys=keys).save("postgres", "mixed", {"host": "db.example", "password": CANARY})
    record_file = base_vault / "postgres-mixed"
    record = json.loads(record_file.read_text())
    record["fields"].update(changed)
    record_file.write_text(json.dumps(record))

    sealing.refuse_sealed_state_without_key_file()
    encrypt(CANARY)

    assert legacy_key.exists()


@pytest.mark.parametrize(
    "rewrite",
    [lambda token: token + "\n", lambda token: token[:-10]],
    ids=["trailing-newline", "truncated"],
)
def test_no_key_file_and_a_token_a_hand_edit_changed_start_as_today(legacy_key, base_vault, rewrite):
    """With no key file the check has no key to open a value with, so it reads a token's shape
    alone, and a token a hand edit left a newline after, or cut short, isn't one: the start goes
    ahead and the first legacy key is made. The re-seal reads such a value as sealed, by the text a
    token starts with, only given its keys, so a plain value that starts that way never stops a
    start."""
    keys = MultiFernet([Fernet(Fernet.generate_key())])
    SealedDataVault(base_vault, keys=keys).save("postgres", "padded", {"password": CANARY})
    record_file = base_vault / "postgres-padded"
    record = json.loads(record_file.read_text())
    record["fields"]["password"] = rewrite(record["fields"]["password"])
    record_file.write_text(json.dumps(record))

    sealing.refuse_sealed_state_without_key_file()
    encrypt(CANARY)

    assert legacy_key.exists()


def test_no_key_file_and_a_sealed_record_with_no_values_refuses_the_start(legacy_key, base_vault):
    """SealedDataVault writes only the mark beside the bookkeeping fields for a connection with no
    values, such as an empty custom MCP connection. A deployment sealed from its first start has no
    marker row, so the vault scan is its only guard, and such a record counts as sealed."""
    SealedDataVault(base_vault, keys=MultiFernet([Fernet(Fernet.generate_key())])).save(
        "custom_mcp", "empty", {"_label": "Empty"}
    )

    with pytest.raises(KeyFileError, match="custom_mcp-empty"):
        sealing.refuse_sealed_state_without_key_file()

    assert not legacy_key.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any folder")
@pytest.mark.parametrize("unsearchable", ["partition", "shared root"])
def test_no_key_file_and_a_folder_the_server_cant_search_starts_with_a_warning(
    legacy_key, base_vault, tmp_path, owned_logger, unsearchable
):
    """A start that never turned sealing on reads no vault it can't search,
    and says which folder it skipped."""
    from tests._logging_assertions import assert_log_fields

    shared_root = tmp_path / "shared"
    partition = shared_root / PARTITION
    (partition / "data-vault").mkdir(parents=True)
    if unsearchable == "partition":
        locked, skipped = partition, partition / "data-vault"
    else:
        locked, skipped = shared_root, shared_root
    logged = owned_logger(sealing.logger.name, level=logging.WARNING)
    locked.chmod(0)
    try:
        sealing.refuse_sealed_state_without_key_file()
    finally:
        locked.chmod(0o700)

    assert_log_fields(
        logged.output(), log_level="WARNING", event="vault_unreadable", vault_folder=str(skipped), reason="EACCES"
    )


def test_the_legacy_key_is_never_made_beside_a_sealed_deployment(legacy_key, base_vault, stored_rows):
    stored_rows(sealing.SEALING_MARKER_KEY, "1")

    with pytest.raises(KeyFileError):
        encrypt(CANARY)

    assert not legacy_key.exists()


def test_the_legacy_key_is_made_as_today_where_sealing_was_never_turned_on(legacy_key, base_vault):
    encrypt(CANARY)

    assert legacy_key.exists()


def test_an_org_start_without_a_key_file_reads_no_vault(monkeypatch, org_deployment, stored_rows):
    monkeypatch.delenv("COWORK_KEY_FILE", raising=False)
    sealing_keys.cache_clear()
    stored_rows(sealing.SEALING_MARKER_KEY, "1")
    monkeypatch.setattr(sealing, "vault_roots", lambda: pytest.fail("an org start must not read a vault"))

    sealing.refuse_sealed_state_without_key_file()


async def test_the_start_stops_before_migrations_without_a_key_file(monkeypatch, legacy_key, base_vault, stored_rows):
    stored_rows(sealing.SEALING_MARKER_KEY, "1")
    monkeypatch.setattr(server, "run_dev_setup", lambda: pytest.fail("the check runs before migrations"))

    with pytest.raises(KeyFileError):
        async with server.lifespan(FastAPI()):
            pass

    assert not legacy_key.exists()


# ── what a refused start logs ───────────────────────────────────────────────


def _start_refusal_lines(output: str) -> list[dict]:
    return [line for line in map(json.loads, output.splitlines()) if line.get("event") == "sealing_start_refused"]


@pytest.fixture
def start_log(monkeypatch, owned_logger):
    """The lifespan's and the sealing checks' lines, as the server's JSON console handler writes them."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    logs = [owned_logger(name, level=logging.WARNING) for name in (server.logger.name, sealing.logger.name)]
    return lambda: "".join(log.output() for log in logs)


def _refusal_record(caplog) -> logging.LogRecord:
    [record] = [
        record
        for record in caplog.records
        if record.name == server.logger.name and getattr(record, "event", None) == "sealing_start_refused"
    ]
    return record


@pytest.mark.parametrize("sealed", ["marker", "record"])
async def test_a_start_refused_without_a_key_file_logs_the_reason_once(
    monkeypatch, caplog, start_log, legacy_key, base_vault, stored_rows, sealed
):
    """Uvicorn's line for a failed start keeps only its frames and exception classes, so the
    lifespan's own line is the only one that says why Cowork won't start."""
    foreign = Fernet.generate_key()
    SealedDataVault(base_vault, keys=MultiFernet([Fernet(foreign)])).save("postgres", "sales", {"password": CANARY})
    token = json.loads((base_vault / "postgres-sales").read_text())["fields"]["password"]
    if sealed == "marker":
        stored_rows(sealing.SEALING_MARKER_KEY, "1")
    named = "Sealing was turned on" if sealed == "marker" else "postgres-sales is sealed"
    monkeypatch.setattr(server, "run_dev_setup", lambda: pytest.fail("the check runs before migrations"))

    with pytest.raises(KeyFileError, match=named) as raised:
        async with server.lifespan(FastAPI()):
            pass

    record = _refusal_record(caplog)
    assert record.levelno == logging.ERROR and record.exc_info is None
    output = start_log()
    [line] = _start_refusal_lines(output)
    assert line["log_level"] == "ERROR"
    assert line["reason"] == str(raised.value)
    assert named in line["reason"]
    for secret in (CANARY, token, foreign.decode()):
        assert secret not in output
    assert not legacy_key.exists()


async def test_a_start_refused_for_a_record_the_key_file_doesnt_open_logs_the_reason_once(
    monkeypatch, caplog, start_log, key_file, base_vault
):
    foreign = Fernet.generate_key()
    SealedDataVault(base_vault, keys=MultiFernet([Fernet(foreign)])).save("postgres", "sales", {"password": CANARY})
    token = json.loads((base_vault / "postgres-sales").read_text())["fields"]["password"]
    monkeypatch.setattr(server, "run_dev_setup", lambda: None)

    with pytest.raises(sealing.SealedDataError) as raised:
        async with server.lifespan(FastAPI()):
            pass

    record = _refusal_record(caplog)
    assert record.levelno == logging.ERROR and record.exc_info is None
    output = start_log()
    [line] = _start_refusal_lines(output)
    assert line["log_level"] == "ERROR"
    assert line["reason"] == str(raised.value)
    assert "connection postgres/sales" in line["reason"]
    assert "its password field doesn't open with the server's key" in line["reason"]
    for secret in (CANARY, token, foreign.decode(), key_file.read_text().strip()):
        assert secret not in output


# ── without a key file: an older or not yet made database starts as today ───

# The revision before the settings table gained its scope column
# (a3f9c2e8b1d4), as a desktop last started before then has it.
BEFORE_SETTINGS_SCOPE = "f7d2b9e4a1c6"


class _PastTheChecks(Exception):
    """Stops a start at the step after the sealing checks and migrations."""


@pytest.fixture
def start_past_the_checks(monkeypatch, tmp_path):
    """The lifespan through its real migrations and both sealing checks, on a
    home of the test's own, stopped at the step after them. Returns the steps
    that ran."""
    for name, folder in (
        ("COWORK_HOME", "home"),
        ("COWORK_PROJECTS_DIR", "projects"),
        ("COWORK_SKILLS_DIR", "skills"),
        ("COWORK_MEMORY_DIR", "memory"),
    ):
        monkeypatch.setenv(name, str(tmp_path / folder))
    get_app_settings.cache_clear()
    ran: list[str] = []
    migrate = server.run_dev_setup

    def run_dev_setup() -> None:
        ran.append("dev_setup")
        migrate()

    def stop(*_args) -> None:
        raise _PastTheChecks

    monkeypatch.setattr(server, "run_dev_setup", run_dev_setup)
    monkeypatch.setattr(server, "consume_pending_credential_clear", stop)

    async def start() -> list[str]:
        with pytest.raises(_PastTheChecks):
            async with server.lifespan(FastAPI()):
                pass
        return ran

    return start


def _database_at(monkeypatch, *, path: Path, revision: str) -> str:
    """A SQLite database at ``path``, migrated to ``revision``, that the server reads."""
    from alembic import command

    from cowork.db.migrations import _alembic_config

    uri = f"sqlite:///{path}"
    monkeypatch.setenv("DATABASE_URI", uri)
    get_app_settings.cache_clear()
    config = _alembic_config(uri)
    with get_engine(uri).begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)
    return uri


def _settings_columns(uri: str) -> set[str]:
    return {column["name"] for column in sa.inspect(get_engine(uri)).get_columns(Setting.__tablename__)}


async def test_no_key_file_and_a_database_from_before_the_scope_column_migrates(
    monkeypatch, legacy_key, base_vault, tmp_path, start_past_the_checks
):
    uri = _database_at(monkeypatch, path=tmp_path / "older.db", revision=BEFORE_SETTINGS_SCOPE)
    assert "scope" not in _settings_columns(uri)

    assert await start_past_the_checks() == ["dev_setup"]

    assert "scope" in _settings_columns(uri)


async def test_no_key_file_and_a_database_folder_not_made_yet_migrates(
    monkeypatch, legacy_key, base_vault, tmp_path, start_past_the_checks
):
    """A desktop organization's first start: run_dev_setup makes the folder."""
    database = tmp_path / "orgs" / "org-123" / "cowork.db"
    monkeypatch.setenv("DATABASE_URI", f"sqlite:///{database}")
    get_app_settings.cache_clear()

    assert await start_past_the_checks() == ["dev_setup"]

    assert database.is_file()


async def test_no_key_file_and_the_marker_in_a_database_from_before_the_scope_column_refuses(
    monkeypatch, legacy_key, base_vault, tmp_path
):
    uri = _database_at(monkeypatch, path=tmp_path / "older.db", revision=BEFORE_SETTINGS_SCOPE)
    with get_engine(uri).begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO settings (id, key, value, created_at, modified_at) "
                "VALUES (lower(hex(randomblob(16))), :key, '1', datetime('now'), datetime('now'))"
            ),
            {"key": sealing.SEALING_MARKER_KEY},
        )
    monkeypatch.setattr(server, "run_dev_setup", lambda: pytest.fail("the check runs before migrations"))

    with pytest.raises(KeyFileError, match="Sealing was turned on"):
        async with server.lifespan(FastAPI()):
            pass

    assert not legacy_key.exists()
