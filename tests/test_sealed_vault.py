"""Saved connections sealed under COWORK_KEY_FILE.

Every vault the server builds goes through persist.data_vault_at: anton's
plain LocalDataVault with no key file, a SealedDataVault with one. A sealed
record keeps anton's layout and its bookkeeping fields, and each other value
opens with the type and text it was saved with. A record that exists but
won't open names the connection, and never lets a same-named connection from
another layer answer in its place, in a turn or an app backend.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, MultiFernet

from cowork.common.encryption import sealing_keys
from cowork.common.settings.app_settings import get_app_settings
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, friendly_turn_error
from cowork.services.connectors.layered_vault import LayeredDataVault
from cowork.services.connectors.persist import data_vault_at
from cowork.services.connectors.sealed_vault import (
    SEALED_FIELD,
    SealedDataVault,
    SealedRecordError,
    refuse_unopenable_connections,
)

CANARY = "zx-canary-7781"


def _keys(*keys: bytes) -> MultiFernet:
    return MultiFernet([Fernet(key) for key in keys])


def _record_file(vault: LocalDataVault, engine: str, name: str) -> Path:
    return vault._path_for(engine, name)


# ── values and layout ───────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [True, 5432, "true", "1e3", "null", "plain text"])
def test_each_value_opens_as_a_plain_vault_gives_it(tmp_path, value):
    plain = LocalDataVault(tmp_path / "plain")
    sealed = SealedDataVault(tmp_path / "sealed", keys=_keys(Fernet.generate_key()))
    for vault in (plain, sealed):
        vault.save("postgres", "sales", {"value": value, "_label": "Sales"})

    assert sealed.load("postgres", "sales") == plain.load("postgres", "sales")
    assert sealed.read_record("postgres", "sales")["fields"] == plain.read_record("postgres", "sales")["fields"]
    assert sealed.env_for("postgres", "sales") == plain.env_for("postgres", "sales")


def test_values_are_sealed_on_disk_and_bookkeeping_stays_readable(tmp_path):
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save(
        "postgres",
        "sales",
        {"host": "db.example", "password": CANARY, "_label": "Sales", "_created_by": "a-person"},
        secure_keys=["password"],
    )

    text = _record_file(vault, "postgres", "sales").read_text()
    stored = json.loads(text)
    assert CANARY not in text and "db.example" not in text
    assert stored["engine"] == "postgres" and stored["name"] == "sales"
    assert stored["fields"]["_label"] == "Sales"
    assert stored["fields"]["_created_by"] == "a-person"
    assert stored["fields"][SEALED_FIELD] == 1
    assert stored["secure_keys"] == ["password"]
    # The listing and the opened record carry no trace of the mark.
    assert vault.list_connections()[0]["name"] == "sales"
    assert SEALED_FIELD not in vault.load("postgres", "sales")


def test_a_missing_record_reads_as_missing(tmp_path):
    vault = SealedDataVault(tmp_path / "never-made", keys=_keys(Fernet.generate_key()))

    assert vault.read_record("postgres", "sales") is None
    assert vault.load("postgres", "sales") is None
    assert vault.env_for("postgres", "sales") is None
    assert vault.list_connections() == []


def test_a_record_sealed_under_an_older_key_opens_only_while_the_file_lists_it(tmp_path):
    old, new = Fernet.generate_key(), Fernet.generate_key()
    SealedDataVault(tmp_path, keys=_keys(old)).save("postgres", "sales", {"password": CANARY})

    assert SealedDataVault(tmp_path, keys=_keys(new, old)).load("postgres", "sales") == {"password": CANARY}
    with pytest.raises(SealedRecordError):
        SealedDataVault(tmp_path, keys=_keys(new)).load("postgres", "sales")


# ── records that won't open ─────────────────────────────────────────────────


def test_a_truncated_managed_record_never_lets_an_own_same_named_one_answer(tmp_path):
    keys = _keys(Fernet.generate_key())
    managed = SealedDataVault(tmp_path / "managed", keys=keys)
    own = SealedDataVault(tmp_path / "own", keys=keys)
    managed.save("postgres", "managed-canary", {"password": CANARY})
    own.save("postgres", "managed-canary", {"password": "the-own-one"})
    path = _record_file(managed, "postgres", "managed-canary")
    path.write_bytes(path.read_bytes()[:100])
    layered = LayeredDataVault(own=own, managed=managed)

    for read in (layered.read_record, layered.load, layered.env_for):
        with pytest.raises(SealedRecordError) as raised:
            read("postgres", "managed-canary")
        assert "postgres/managed-canary" in str(raised.value)
        assert "the-own-one" not in str(raised.value)


def test_a_changed_sealed_value_names_the_connection_and_field_never_a_value(tmp_path):
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("postgres", "sales", {"password": CANARY})
    path = _record_file(vault, "postgres", "sales")
    stored = json.loads(path.read_text())
    token = stored["fields"]["password"]
    stored["fields"]["password"] = token[:20] + ("A" if token[20] != "A" else "B") + token[21:]
    path.write_text(json.dumps(stored))

    with pytest.raises(SealedRecordError) as raised:
        vault.load("postgres", "sales")

    assert str(raised.value) == (
        "The saved connection postgres/sales can't be opened: "
        "its password field doesn't open with the server's key."
    )


def test_an_unsealed_record_is_refused_while_a_key_is_set(tmp_path):
    LocalDataVault(tmp_path).save("postgres", "sales", {"password": CANARY})

    with pytest.raises(SealedRecordError) as raised:
        SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key())).load("postgres", "sales")

    assert "isn't sealed" in str(raised.value)
    assert CANARY not in str(raised.value)


def test_a_linked_record_and_a_record_with_a_second_name_are_refused(tmp_path):
    keys = _keys(Fernet.generate_key())
    elsewhere = SealedDataVault(tmp_path / "elsewhere", keys=keys)
    elsewhere.save("postgres", "sales", {"password": CANARY})
    target = _record_file(elsewhere, "postgres", "sales")
    linked = SealedDataVault(tmp_path / "linked", keys=keys)
    linked._ensure_dir()
    _record_file(linked, "postgres", "sales").symlink_to(target)
    second = SealedDataVault(tmp_path / "second", keys=keys)
    second._ensure_dir()
    os.link(target, _record_file(second, "postgres", "sales"))

    with pytest.raises(SealedRecordError, match="is a link"):
        linked.load("postgres", "sales")
    with pytest.raises(SealedRecordError, match="one name"):
        second.load("postgres", "sales")


def test_a_record_file_that_wont_parse_fails_the_listing(tmp_path):
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("postgres", "sales", {"password": CANARY})
    (tmp_path / "postgres-broken").write_text("{not json")

    with pytest.raises(SealedRecordError, match="postgres-broken"):
        vault.list_connections()
    assert [error.connection for error in vault.unopenable_records()] == ["postgres-broken"]


def test_refuse_unopenable_connections_stops_on_a_record_that_wont_open(tmp_path):
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("postgres", "sales", {"password": CANARY})
    refuse_unopenable_connections(vault=vault)
    LocalDataVault(tmp_path).save("mysql", "orders", {"password": "plain"})

    with pytest.raises(SealedRecordError, match="mysql/orders"):
        refuse_unopenable_connections(vault=vault)


def test_a_turn_error_names_the_connection():
    error = SealedRecordError(connection="postgres/sales", cause="its record file is a link")

    assert friendly_turn_error(error) == (
        GENERIC_TURN_ERROR_CODE,
        "The saved connection postgres/sales can't be opened: its record file is a link. "
        "Ask your operator: Cowork's log names it too.",
    )


# ── the one builder ─────────────────────────────────────────────────────────


def test_without_a_key_file_the_server_builds_anton_vaults_and_writes_plain_json(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_KEY_FILE", raising=False)
    sealing_keys.cache_clear()

    vault = data_vault_at(tmp_path)
    vault.save("postgres", "sales", {"password": CANARY})

    assert type(vault) is LocalDataVault
    assert json.loads(_record_file(vault, "postgres", "sales").read_text())["fields"] == {"password": CANARY}


def test_with_a_key_file_the_server_builds_sealed_vaults(key_file, tmp_path):
    vault = data_vault_at(tmp_path / "vault")
    vault.save("postgres", "sales", {"password": CANARY})

    assert isinstance(vault, SealedDataVault)
    assert CANARY not in _record_file(vault, "postgres", "sales").read_text()
    key = Fernet(key_file.read_bytes().strip())
    token = json.loads(_record_file(vault, "postgres", "sales").read_text())["fields"]["password"]
    assert json.loads(key.decrypt(token.encode())) == CANARY


# Calls that may build anton's vault directly: an empty folder the probe hands
# its check, and a path helper that never opens a record.
_ALLOWED_VAULT_CALLS = {
    ("cowork/services/connectors/persist.py", "data_vault_at"),
    ("cowork/services/connectors/probe.py", "run"),
    ("cowork/ops/connection_inventory.py", "_vault_filename"),
}


def _vault_calls(root: Path) -> set[tuple[str, str]]:
    found = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LocalDataVault":
                    found.add((path.relative_to(root.parent).as_posix(), function.name))
    return found


def test_every_vault_the_server_builds_goes_through_the_builder():
    root = Path(__file__).resolve().parents[1] / "cowork"

    assert _vault_calls(root) <= _ALLOWED_VAULT_CALLS


# ── a turn and an app backend ───────────────────────────────────────────────


@pytest.fixture
def local_vault(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    get_app_settings.cache_clear()
    yield tmp_path / "vault"
    get_app_settings.cache_clear()


async def test_a_turn_stops_on_a_sealed_connection_that_wont_open(
    monkeypatch, caplog, key_file, local_vault, tmp_path
):
    from tests._turns import built_turn

    vault = data_vault_at(local_vault)
    vault.save("postgres", "sales", {"password": CANARY})
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
        assert turn.config.data_vault.env_for("postgres", "sales")["DS_POSTGRES_SALES__PASSWORD"] == CANARY
    path = _record_file(vault, "postgres", "sales")
    path.write_bytes(path.read_bytes()[:100])

    # A file too short to parse can't say its engine and name, so the error
    # names the record's file.
    with caplog.at_level("ERROR"), pytest.raises(SealedRecordError, match="postgres-sales"):
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None):
            pass

    logged = [
        record
        for record in caplog.records
        if record.name == "cowork.harnesses.anton_harness.harness" and record.levelname == "ERROR"
    ]
    assert [(record.event, record.connection) for record in logged] == [
        ("sealed_connection_unopenable", "postgres-sales")
    ]
    assert CANARY not in caplog.text


async def test_an_app_backend_doesnt_start_on_a_sealed_connection_that_wont_open(
    monkeypatch, key_file, local_vault, tmp_path
):
    from cowork.services import artifacts

    vault = data_vault_at(local_vault)
    vault.save("postgres", "sales", {"password": CANARY})
    LocalDataVault(local_vault).save("postgres", "sales", {"password": CANARY})
    folder = tmp_path / "project" / ".anton" / "artifacts" / "app"
    folder.mkdir(parents=True)
    (folder / "backend.py").write_text("print('hi')\n")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": "app", "datasources": [{"engine": "postgres", "name": "sales"}]})
    )

    async def must_not_launch(**_kwargs):
        pytest.fail("a backend whose connection won't open must not start")

    monkeypatch.setattr("anton.core.artifacts.backend_launcher.launch_artifact_backend", must_not_launch)

    status = await artifacts._launch_backend_locked(folder, project_root=tmp_path / "project", scope=None)

    assert status.running is False
    assert "postgres/sales" in status.detail and "isn't sealed" in status.detail
    assert CANARY not in status.detail
