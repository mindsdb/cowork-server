"""The key file and the start checks for sealing under COWORK_KEY_FILE.

With a key file, its keys alone seal and open the server's secrets, a key file
that won't read stops the start naming the cause and line, and every sealed
setting, channel credential and saved connection must open before the server
serves. Without one, a deployment sealed earlier doesn't start, and no legacy
key is made beside its data. Org tenancy reads no vault either way, and a
deployment that never turned sealing on starts as it always has.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, MultiFernet
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


def test_a_key_file_overrides_the_legacy_key_and_never_writes_one(monkeypatch, caplog, key_file, legacy_key):
    monkeypatch.setenv("COWORK_KEY_FILE", str(key_file))
    monkeypatch.setenv("COWORK_MASTER_KEY", Fernet.generate_key().decode())
    get_app_settings.cache_clear()
    sealing_keys.cache_clear()
    get_fernet.cache_clear()
    settings = get_app_settings()

    token = encrypt(CANARY)

    assert Fernet(key_file.read_bytes().strip()).decrypt(token.encode()) == CANARY.encode()
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


def test_no_key_file_and_a_sealed_record_refuses_the_start(legacy_key, base_vault):
    SealedDataVault(base_vault, keys=MultiFernet([Fernet(Fernet.generate_key())])).save("postgres", "sales", {"password": CANARY})

    with pytest.raises(KeyFileError, match="postgres-sales"):
        sealing.refuse_sealed_state_without_key_file()


def test_no_key_file_and_plain_or_broken_records_start_as_today(legacy_key, base_vault):
    LocalDataVault(base_vault).save("postgres", "sales", {"password": CANARY})
    (base_vault / "postgres-broken").write_text("{not json")

    sealing.refuse_sealed_state_without_key_file()


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
