"""Saved connections sealed under COWORK_KEY_FILE.

Every vault the server builds goes through persist.data_vault_at: anton's
plain LocalDataVault with no key file, a SealedDataVault with one, whichever
path saves. A sealed record keeps anton's layout and its bookkeeping fields,
and each other value opens with the type and text it was saved with. A record
that exists but won't open names the connection, and never lets a same-named
connection from another layer answer in its place, in a turn, an app backend
or a request, which answers 409. A save whose sealed record would be too
large to read back, or that holds a field named like the sealed mark, is
refused before anything is written, and answers 422.
"""
from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from cryptography.fernet import Fernet, MultiFernet
from fastapi.testclient import TestClient

import cowork.server as cowork_server
from cowork.common.encryption import sealing_keys
from cowork.common.record_files import MAX_RECORD_BYTES
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, friendly_turn_error
from cowork.services.connectors import sealed_vault
from cowork.services.connectors.layered_vault import LayeredDataVault
from cowork.services.connectors.persist import data_vault_at, own_vault_for_scope, own_vault_root
from cowork.services.connectors.sealed_vault import (
    SEALED_FIELD,
    RecordSealing,
    RecordTooLargeError,
    ReservedFieldError,
    SealedDataVault,
    SealedRecordError,
    is_fernet_token,
    record_file_bytes,
    record_sealing,
    refuse_unopenable_connections,
)
from tests._logging_assertions import assert_log_fields

CANARY = "zx-canary-7781"
# The longest stamp anton's save writes: microseconds and the UTC offset.
LONGEST_STAMP = "2026-01-01T00:00:00.000000+00:00"


def _keys(*keys: bytes) -> MultiFernet:
    return MultiFernet([Fernet(key) for key in keys])


def _record_file(vault: LocalDataVault, engine: str, name: str) -> Path:
    return vault._path_for(engine, name)


def _change_one_sealed_value(path: Path, field: str) -> None:
    stored = json.loads(path.read_text())
    token = stored["fields"][field]
    stored["fields"][field] = token[:20] + ("A" if token[20] != "A" else "B") + token[21:]
    path.write_text(json.dumps(stored))


def _loopback_client() -> TestClient:
    """create_app's app, with its exception handlers, as the desktop reaches it."""
    return TestClient(cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


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
    _change_one_sealed_value(_record_file(vault, "postgres", "sales"), "password")

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


@pytest.mark.parametrize("mark", ["1", True, 1.0], ids=["string", "bool", "float"])
def test_only_the_integer_mark_reads_as_sealed(tmp_path, mark):
    """True and 1.0 equal 1 in Python, and a plain record can hold any of
    these under a field of the mark's name."""
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("postgres", "sales", {"password": CANARY})
    path = _record_file(vault, "postgres", "sales")
    stored = json.loads(path.read_text())
    stored["fields"][SEALED_FIELD] = mark
    path.write_text(json.dumps(stored))

    with pytest.raises(SealedRecordError, match="isn't sealed"):
        vault.load("postgres", "sales")


def test_a_fernet_token_is_told_from_other_text_by_its_shape():
    token = Fernet(Fernet.generate_key()).encrypt(b'"value"').decode()

    assert is_fernet_token(value=token)
    for value in (
        "1",
        CANARY,
        token[:-4],
        token.replace("=", "") + "A",
        base64.urlsafe_b64encode(bytes(73)).decode(),
        token + "\n",
        5,
        None,
    ):
        assert not is_fernet_token(value=value), value


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


def test_a_record_that_wont_open_answers_409_naming_it_on_the_list_and_the_delete(
    key_file, sharepoint_stores, owned_logger
):
    vault = own_vault_for_scope(LOCAL_SCOPE)
    vault.save("postgres", "sales", {"password": CANARY})
    path = _record_file(vault, "postgres", "sales")
    _change_one_sealed_value(path, "password")
    client = _loopback_client()
    logged = owned_logger("cowork.common.logger", level=logging.ERROR)

    listed = client.get("/api/v1/connectors/connections/")
    deleted = client.delete("/api/v1/connectors/connections/postgres/sales")

    expected = (
        "The saved connection postgres/sales can't be opened: "
        "its password field doesn't open with the server's key."
    )
    assert (listed.status_code, listed.json()) == (409, {"detail": expected})
    assert (deleted.status_code, deleted.json()) == (409, {"detail": expected})
    assert path.exists()
    assert_log_fields(
        logged.output(),
        log_level="ERROR",
        event="sealed_connection_unopenable",
        connection="postgres/sales",
        reason="its password field doesn't open with the server's key",
    )
    assert CANARY not in logged.output()


def test_the_start_counts_managed_connections_as_unknown_when_one_wont_open(
    key_file, managed_dir, caplog, owned_logger
):
    """create_app leaves the record to the lifespan's sealed-data check,
    which names every one that won't open."""
    (managed_dir.vault_dir / "postgres-broken").write_text("{not json")

    owned_logger("cowork.common.logger", level=logging.INFO)
    cowork_server.create_app()

    lines = [record.getMessage() for record in caplog.records if record.name == "cowork.common.logger"]
    assert (
        f"managed folder: {managed_dir.root} (0 skills, an unknown number of connections, 0 shared connections)"
        in lines
    )


def test_a_turn_error_names_the_connection():
    error = SealedRecordError(connection="postgres/sales", cause="its record file is a link")

    assert friendly_turn_error(error) == (
        GENERIC_TURN_ERROR_CODE,
        "The saved connection postgres/sales can't be opened: its record file is a link. "
        "Ask your operator: Cowork's log names it too.",
    )


# ── reading a record file ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "cause"),
    [
        ("link", "its record file is a link"),
        ("second-name", "its record file isn't a regular file with one name"),
        ("fifo", "its record file isn't a regular file with one name"),
        ("too-large", "its record file is too large"),
        pytest.param(
            "unreadable",
            "its record file can't be read (EACCES)",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file"),
        ),
    ],
)
def test_the_record_reader_names_why_it_refuses_a_file(tmp_path, kind, cause):
    path = tmp_path / "postgres-sales"
    target = tmp_path / "elsewhere"
    target.write_text(json.dumps({"engine": "postgres", "name": "sales", "fields": {"password": CANARY}}))
    if kind == "link":
        path.symlink_to(target)
    elif kind == "second-name":
        os.link(target, path)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "too-large":
        path.write_bytes(b" " * (MAX_RECORD_BYTES + 1))
    else:
        path.write_text(target.read_text())
        path.chmod(0)

    with pytest.raises(SealedRecordError) as raised:
        sealed_vault.read_record_file(path=path, connection="postgres/sales")

    assert raised.value.cause == cause and CANARY not in str(raised.value)


def test_the_record_reader_answers_none_where_no_file_is(tmp_path):
    assert sealed_vault.read_record_file(path=tmp_path / "postgres-sales", connection="postgres/sales") is None
    assert sealed_vault.read_record_file(path=tmp_path / "gone" / "postgres-sales", connection="postgres/sales") is None


# ── a record too large to read back ─────────────────────────────────────────


def test_the_size_helper_measures_the_file_anton_writes(tmp_path):
    vault = LocalDataVault(tmp_path)
    vault.save("postgres", "sales", {"password": CANARY, "_label": "Sales"}, secure_keys=["password"])
    path = _record_file(vault, "postgres", "sales")
    record = json.loads(path.read_text())

    measured = record_file_bytes(
        engine="postgres", name="sales", fields=record["fields"], secure_keys=record["secure_keys"], prior=record
    )

    # The helper stamps updated_at in its longest form; anton's drops a zero microsecond.
    assert measured - path.stat().st_size == len(LONGEST_STAMP) - len(record["updated_at"])


def test_a_save_too_large_to_read_back_is_refused_and_writes_nothing(tmp_path):
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("postgres", "big", {"host": "db.example"})
    path = _record_file(vault, "postgres", "big")
    before = path.read_bytes()

    # About 790 KB in plain text seals to over 1 MiB.
    with pytest.raises(RecordTooLargeError) as raised:
        vault.save("postgres", "big", {"host": "db.example", "sslrootcert": "A" * 790_000})

    assert str(raised.value).startswith("The saved connection postgres/big is too large to save")
    assert raised.value.connection == "postgres/big" and "AAAA" not in str(raised.value)
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir()] == ["postgres-big"]


def test_every_record_the_save_writes_reads_back(monkeypatch, tmp_path):
    """Up to the limit, one byte of plain text at a time: each save that
    succeeds opens again, and the first that wouldn't is refused."""
    monkeypatch.setattr(sealed_vault, "MAX_RECORD_BYTES", 4096)
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    saved = 0
    for length in range(2_500, 4_096):
        value = "A" * length
        try:
            vault.save("postgres", "edge", {"sslrootcert": value})
        except RecordTooLargeError:
            break
        assert vault.load("postgres", "edge") == {"sslrootcert": value}
        saved += 1
    else:
        pytest.fail("no save reached the limit")

    assert saved > 0
    assert _record_file(vault, "postgres", "edge").stat().st_size <= 4096


def test_the_save_route_answers_422_for_a_record_too_large_to_read_back(key_file, sharepoint_stores):
    response = _loopback_client().post(
        "/api/v1/connectors/connections/save",
        json={"connector_id": "postgres", "name": "big", "values": {"host": "db.example", "password": "A" * 790_000}},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"].startswith("The saved connection postgres/big is too large to save")
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []


# ── a field named like the sealed mark ──────────────────────────────────────


_TOKEN_KEYS = _keys(Fernet.generate_key())
_TOKEN = _TOKEN_KEYS.encrypt(b'"value"').decode()
_OTHER_TOKEN = Fernet(Fernet.generate_key()).encrypt(b'"value"').decode()
_NOT_JSON_TOKEN = _TOKEN_KEYS.encrypt(b"not json").decode()


@pytest.mark.parametrize(
    ("fields", "sealing"),
    [
        ({SEALED_FIELD: 1, "password": _TOKEN, "_label": "Sales"}, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "_label": "Sales"}, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "password": _TOKEN, "host": "db.example"}, RecordSealing.PARTLY_SEALED),
        ({SEALED_FIELD: 1, "password": _TOKEN, "port": 5432}, RecordSealing.PARTLY_SEALED),
        ({SEALED_FIELD: "1", "password": _TOKEN}, RecordSealing.PARTLY_SEALED),
        ({SEALED_FIELD: True, "password": _TOKEN}, RecordSealing.PARTLY_SEALED),
        ({SEALED_FIELD: 1, "api_token": CANARY}, RecordSealing.RESERVED_FIELD),
        ({SEALED_FIELD: True, "api_token": CANARY}, RecordSealing.RESERVED_FIELD),
        # A bookkeeping field is never sealed, so a token there is no sealed value.
        ({SEALED_FIELD: 1, "_label": _TOKEN, "host": "db.example"}, RecordSealing.RESERVED_FIELD),
        ({SEALED_FIELD: None}, RecordSealing.RESERVED_FIELD),
        ({"password": _TOKEN, "_label": "Sales"}, RecordSealing.PLAIN),
        ({}, RecordSealing.PLAIN),
    ],
)
def test_a_record_reads_as_sealed_partly_sealed_reserved_or_plain(fields, sealing):
    assert record_sealing(fields=fields) is sealing


@pytest.mark.parametrize(
    ("fields", "without_keys", "with_keys"),
    [
        ({SEALED_FIELD: 1, "password": _TOKEN + "\n"}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "password": " " + _TOKEN}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "password": _TOKEN + "\n", "host": _TOKEN}, RecordSealing.PARTLY_SEALED, RecordSealing.SEALED),
        ({SEALED_FIELD: "1", "password": _TOKEN + "\n"}, RecordSealing.RESERVED_FIELD, RecordSealing.PARTLY_SEALED),
        # Fernet's decoder skips a character that isn't base64, so these open, and their text doesn't
        # start as a token's does: only opening them reads them as sealed. The second opens to text
        # that isn't the JSON a save seals, so open_fields refuses it, and the re-seal names it as a
        # value no key opens.
        ({SEALED_FIELD: 1, "password": _TOKEN[:2] + "." + _TOKEN[2:]}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        (
            {SEALED_FIELD: 1, "password": _NOT_JSON_TOKEN[:2] + "." + _NOT_JSON_TOKEN[2:]},
            RecordSealing.RESERVED_FIELD,
            RecordSealing.SEALED,
        ),
        # Text that can't be encoded to bytes, as a lone surrogate a JSON file can hold, opens with no key.
        ({SEALED_FIELD: 1, "api_token": "\ud800"}, RecordSealing.RESERVED_FIELD, RecordSealing.RESERVED_FIELD),
        # The rows below open with no key and have no token's shape; their text starts as a token's
        # does, whitespace aside, so given keys they read as sealed, and the re-seal refuses them as
        # values no key opens.
        ({SEALED_FIELD: 1, "password": _OTHER_TOKEN + "\n"}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        (
            {SEALED_FIELD: 1, "password": _OTHER_TOKEN[:3] + "\n " + _OTHER_TOKEN[3:]},
            RecordSealing.RESERVED_FIELD,
            RecordSealing.SEALED,
        ),
        ({SEALED_FIELD: 1, "password": _TOKEN[:-10]}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        # Fernet opens it, but not to the JSON text seal_fields seals, so open_fields refuses it.
        (
            {SEALED_FIELD: 1, "password": " " + _NOT_JSON_TOKEN},
            RecordSealing.RESERVED_FIELD,
            RecordSealing.SEALED,
        ),
        ({SEALED_FIELD: 1, "password": '"' + _OTHER_TOKEN + '"'}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "api_token": "gAAAAA" + CANARY}, RecordSealing.RESERVED_FIELD, RecordSealing.SEALED),
        ({SEALED_FIELD: 1, "api_token": "gAAAA-" + CANARY}, RecordSealing.RESERVED_FIELD, RecordSealing.RESERVED_FIELD),
        ({SEALED_FIELD: 1, "api_token": CANARY + " gAAAAA"}, RecordSealing.RESERVED_FIELD, RecordSealing.RESERVED_FIELD),
        ({SEALED_FIELD: 1, "api_token": CANARY}, RecordSealing.RESERVED_FIELD, RecordSealing.RESERVED_FIELD),
    ],
    ids=[
        "trailing-newline", "leading-space", "padded-beside-a-token", "string-mark-beside-padded",
        "skipped-character", "skipped-character-opens-to-text-that-isnt-json", "text-that-cant-be-encoded",
        "padded-under-another-key", "whitespace-inside-the-prefix-under-another-key", "truncated",
        "opens-to-text-that-isnt-json", "quoted-under-another-key", "plain-text-with-the-prefix",
        "plain-text-with-five-of-the-prefix", "plain-text-with-the-prefix-later", "plain",
    ],
)
def test_given_keys_a_value_they_open_reads_as_sealed_whatever_its_shape(fields, without_keys, with_keys):
    """Fernet opens a token with stray whitespace around it, as a hand edit can leave, and the shape
    test refuses that. Given keys, as the re-seal passes its opening keys, a value Fernet opens with
    them counts as sealed, whether or not it opens to JSON, and so does one whose text, with each
    character outside a token's alphabet removed, starts as a token's does. Without them, as at a start with no key file, the
    shape alone decides."""
    assert record_sealing(fields=fields) is without_keys
    assert record_sealing(fields=fields, keys=_TOKEN_KEYS) is with_keys


@pytest.mark.parametrize("mark", ["1", 1, True, None], ids=["string", "int", "bool", "null"])
def test_a_keyed_save_with_a_field_named_like_the_mark_is_refused_and_writes_nothing(tmp_path, mark):
    """The mark would replace the field, so the person's value would be gone."""
    vault = SealedDataVault(tmp_path, keys=_keys(Fernet.generate_key()))
    vault.save("acme_api", "acme", {"api_token": "first"})
    path = _record_file(vault, "acme_api", "acme")
    before = path.read_bytes()

    with pytest.raises(ReservedFieldError) as raised:
        vault.save("acme_api", "acme", {"api_token": CANARY, SEALED_FIELD: mark})

    assert str(raised.value) == (
        "The saved connection acme_api/acme can't be saved: the field name _sealed is reserved for the sealed mark. "
        "Rename the field and save again."
    )
    assert raised.value.connection == "acme_api/acme"
    assert isinstance(raised.value, ValueError)
    assert path.read_bytes() == before
    assert [entry.name for entry in tmp_path.iterdir()] == ["acme_api-acme"]


def test_without_a_key_file_a_save_keeps_a_field_named_like_the_mark(monkeypatch, sharepoint_stores):
    """No key file, no sealed mark: the form's save writes the field through anton's vault as it
    always has."""
    from cowork.services.connectors.persist import persist_connection

    monkeypatch.delenv("COWORK_KEY_FILE", raising=False)
    sealing_keys.cache_clear()

    persist_connection("acme_api", None, "acme", {"api_token": CANARY, SEALED_FIELD: "1"}, scope=LOCAL_SCOPE)

    vault = own_vault_for_scope(LOCAL_SCOPE)
    assert type(vault) is LocalDataVault
    fields = json.loads(_record_file(vault, "acme_api", "acme").read_text())["fields"]
    assert fields[SEALED_FIELD] == "1" and fields["api_token"] == CANARY


def test_the_save_route_answers_422_for_a_field_named_like_the_mark(key_file, sharepoint_stores):
    response = _loopback_client().post(
        "/api/v1/connectors/connections/save",
        json={"connector_id": "postgres", "name": "acme", "values": {"host": "db.example", SEALED_FIELD: "1"}},
    )

    assert response.status_code == 422, response.text
    assert response.json() == {
        "detail": "The saved connection postgres/acme can't be saved: the field name _sealed is reserved for the "
        "sealed mark. Rename the field and save again."
    }
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []


def test_the_validate_and_save_route_answers_422_for_a_field_named_like_the_mark(
    monkeypatch, key_file, sharepoint_stores
):
    """Code's personal-token setup saves through the same direct save as /save once the provider
    accepts the token, so a field named like the mark answers 422 there too."""
    from cowork.api.v1.endpoints.connectors import connections as connections_endpoint
    from cowork.services.connectors.developer_validation import ValidatedDeveloperIdentity

    monkeypatch.setattr(
        connections_endpoint,
        "validate_developer_connection",
        lambda connector_id, method, values: ValidatedDeveloperIdentity(account_email="dev@example.com"),
    )

    response = _loopback_client().post(
        "/api/v1/connectors/connections/validate-and-save",
        json={
            "connector_id": "github",
            "method": "fine-grained-pat",
            "name": "acme",
            "values": {"access_token": CANARY, SEALED_FIELD: "1"},
        },
    )

    assert response.status_code == 422, response.text
    assert response.json() == {
        "detail": "The saved connection github/acme can't be saved: the field name _sealed is reserved for the "
        "sealed mark. Rename the field and save again."
    }
    assert CANARY not in response.text
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []


@pytest.mark.parametrize("value", ["1", "", None], ids=["one", "empty", "null"])
@pytest.mark.parametrize(
    ("engine", "fields"),
    [
        ("acme_api", {"api_token": CANARY}),
        ("postgres", {"host": "db.example", "port": "5432", "database": "sales", "user": "reader", "password": CANARY}),
    ],
    ids=["unknown-engine", "postgres"],
)
async def test_the_chat_connect_tool_refuses_a_field_named_like_the_mark_before_anton_writes_anything(
    monkeypatch, key_file, sharepoint_stores, tmp_path, engine, fields, value
):
    """anton's connect tool writes an engine it doesn't know to ~/.anton/datasources.md before it
    saves. Were the save refused there, the field would stay among that engine's fields, and the
    model's retry with the field renamed would save without the renamed field, and say nothing. So
    the tool refuses first: no engine is written and nothing is saved. It refuses by name, as the
    vault does, where anton would drop the field and save without a word too: an empty value, no
    value, and a field postgres's engine doesn't list."""
    from anton.core.datasources.datasource_registry import DatasourceRegistry

    from tests._turns import built_turn, connect_in_chat

    home = tmp_path / "anton-home"
    engines = home / ".anton" / "datasources.md"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(DatasourceRegistry, "_USER_PATH", engines)
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
        connected = await connect_in_chat(
            turn, monkeypatch, engine=engine, known_variables={**fields, SEALED_FIELD: value}
        )

    assert connected.answer == (
        f"The saved connection {engine} can't be saved: the field name _sealed is reserved for the sealed mark. "
        "Rename the field and save again."
    )
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []
    assert not engines.exists()


async def test_the_chat_connect_tool_answers_with_the_vaults_reason_for_a_record_too_large_to_read_back(
    monkeypatch, caplog, key_file, sharepoint_stores, tmp_path
):
    """The tool answers as it does for a field named like the mark: with the vault's reason, which
    names the connection and never a value. Nothing is saved, and nothing is logged as an error.
    anton has already written the engine it doesn't know to ~/.anton/datasources.md with these
    field names, the entry any connect with them writes. A retry with a smaller value keeps the
    names, so anton drops none of its fields."""
    from anton.core.datasources.datasource_registry import DatasourceRegistry

    from tests._turns import built_turn, connect_in_chat

    home = tmp_path / "anton-home"
    engines = home / ".anton" / "datasources.md"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(DatasourceRegistry, "_USER_PATH", engines)
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
        with caplog.at_level(logging.ERROR):
            # About 790 KB in plain text seals to over 1 MiB.
            connected = await connect_in_chat(
                turn, monkeypatch, engine="acme_api", known_variables={"api_token": CANARY + "A" * 790_000}
            )

    assert re.fullmatch(
        r"The saved connection acme_api/[0-9a-f]{8} is too large to save: sealed, its record file would be "
        rf"\d+ bytes, and the limit is {MAX_RECORD_BYTES}\.",
        connected.answer,
    ), connected.answer
    assert own_vault_for_scope(LOCAL_SCOPE).list_connections() == []
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == []
    engine = DatasourceRegistry().find_by_name("acme_api")
    assert engine is not None and [field.name for field in engine.fields] == ["api_token"]


async def test_without_a_key_file_the_chat_connect_tool_saves_a_field_named_like_the_mark(
    monkeypatch, sharepoint_stores, tmp_path
):
    """No key file, no sealed mark: the tool answers as anton's save does, the record keeps the
    field, and anton writes the engine it doesn't know to ~/.anton/datasources.md, this field among
    its fields, as it always has."""
    from anton.core.datasources.datasource_registry import DatasourceRegistry

    from tests._turns import built_turn, connect_in_chat

    monkeypatch.delenv("COWORK_KEY_FILE", raising=False)
    sealing_keys.cache_clear()
    home = tmp_path / "anton-home"
    engines = home / ".anton" / "datasources.md"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(DatasourceRegistry, "_USER_PATH", engines)
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
        connected = await connect_in_chat(
            turn, monkeypatch, engine="acme_api", known_variables={"api_token": CANARY, SEALED_FIELD: "1"}
        )

    vault = own_vault_for_scope(LOCAL_SCOPE)
    [saved] = vault.list_connections()
    assert connected.answer.startswith(
        f"Saved connection `{saved['engine']}-{saved['name']}` (label \"acme_api\") to vault with fields: "
        "_sealed, _user_label, api_token."
    ), connected.answer
    assert type(vault) is LocalDataVault
    fields = vault.load("acme_api", saved["name"])
    assert fields[SEALED_FIELD] == "1" and fields["api_token"] == CANARY
    engine = DatasourceRegistry().find_by_name("acme_api")
    assert engine is not None and {field.name for field in engine.fields} == {"api_token", SEALED_FIELD}


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
    ("cowork/ops/connection_inventory.py", "vault_filename"),
}


def _builds_local_data_vault(*, call: ast.Call, aliases: frozenset[str]) -> bool:
    """Whether ``call`` calls anton's LocalDataVault: by a name it was
    imported under, or as an attribute such as ``data_vault.LocalDataVault``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in aliases
    return isinstance(func, ast.Attribute) and func.attr == "LocalDataVault"


def _calls_in(*, node: ast.AST, enclosing: str, aliases: frozenset[str]) -> Iterator[str]:
    """The enclosing function's name, or ``<module>``, of each such call under
    ``node``. A class body and a lambda belong to what encloses them."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _calls_in(node=child, enclosing=child.name, aliases=aliases)
            continue
        if isinstance(child, ast.Call) and _builds_local_data_vault(call=child, aliases=aliases):
            yield enclosing
        yield from _calls_in(node=child, enclosing=enclosing, aliases=aliases)


def _vault_calls(*, root: Path) -> set[tuple[str, str]]:
    """Each place under ``root`` that builds anton's LocalDataVault directly,
    as (file, enclosing function or ``<module>``)."""
    found = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = frozenset({"LocalDataVault"}) | {
            alias.asname
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "LocalDataVault" and alias.asname
        }
        file = path.relative_to(root.parent).as_posix()
        found.update((file, enclosing) for enclosing in _calls_in(node=tree, enclosing="<module>", aliases=aliases))
    return found


def test_every_vault_the_server_builds_goes_through_the_builder():
    root = Path(__file__).resolve().parents[1] / "cowork"

    assert _vault_calls(root=root) <= _ALLOWED_VAULT_CALLS


def test_the_builder_guard_finds_every_way_to_build_a_plain_vault(tmp_path):
    root = tmp_path / "cowork"
    root.mkdir()
    imports = (
        "from pathlib import Path\n"
        "from anton.core.datasources import data_vault\n"
        "from anton.core.datasources.data_vault import LocalDataVault\n"
        "from anton.core.datasources.data_vault import LocalDataVault as Plain\n"
    )
    planted = {
        "at_import.py": "VAULT = LocalDataVault(Path('/vault'))\n",
        "class_body.py": "class Holder:\n    VAULT = Plain(Path('/vault'))\n",
        "lambda_body.py": "build = lambda root: data_vault.LocalDataVault(root)\n",
        "functions.py": (
            "def by_attribute(root):\n    return data_vault.LocalDataVault(root)\n"
            "def by_alias(root):\n    return Plain(root)\n"
            "async def direct(root):\n    return LocalDataVault(root)\n"
            "def outer(root):\n    def inner():\n        return Plain(root)\n    return inner\n"
            "def only_named(vault: LocalDataVault) -> bool:\n    return isinstance(vault, LocalDataVault)\n"
        ),
    }
    for name, body in planted.items():
        (root / name).write_text(imports + body, encoding="utf-8")

    assert _vault_calls(root=root) == {
        ("cowork/at_import.py", "<module>"),
        ("cowork/class_body.py", "<module>"),
        ("cowork/lambda_body.py", "<module>"),
        ("cowork/functions.py", "by_attribute"),
        ("cowork/functions.py", "by_alias"),
        ("cowork/functions.py", "direct"),
        ("cowork/functions.py", "inner"),
    }


@pytest.mark.parametrize("managed", [False, True], ids=["no managed folder", "managed folder"])
@pytest.mark.parametrize("save_path", ["form", "sharepoint", "chat"])
async def test_every_save_path_seals_under_a_key_file(
    request, monkeypatch, key_file, sharepoint_stores, tmp_path, save_path, managed
):
    """The form's save, the SharePoint route and the agent's connect tool,
    each through the vault the server builds for it. The SharePoint key check
    and the chat's connection test pass without a network."""
    from cowork.api.v1.endpoints.connectors import sharepoint as sharepoint_ep
    from cowork.services.connectors.persist import persist_connection
    from tests._turns import built_turn, connect_in_chat

    if managed:
        request.getfixturevalue("managed_dir")
    credentials = {"host": "db.example.com", "port": "5432", "user": "u", "database": "d", "password": CANARY}
    if save_path == "form":
        persist_connection("postgres", None, "formsave", credentials, scope=LOCAL_SCOPE)
    elif save_path == "sharepoint":
        monkeypatch.setattr(sharepoint_ep, "_check", lambda key, sites: None)
        sharepoint_ep.create_connection(
            {
                "name": "spsave",
                "tenant_id": "contoso.onmicrosoft.com",
                "client_id": "11111111-2222-3333-4444-555555555555",
                "client_secret": CANARY,
                "site_urls": "https://contoso.sharepoint.com/sites/team",
            },
            LOCAL_SCOPE,
        )
    else:
        async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
            connected = await connect_in_chat(turn, monkeypatch, known_variables=credentials)
        assert "Connection test failed" not in connected.answer

    [record] = list(own_vault_root(LOCAL_SCOPE).iterdir())
    text = record.read_text()
    assert CANARY not in text
    assert f'"{SEALED_FIELD}": 1' in text


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


@pytest.mark.parametrize("sealed", [False, True], ids=["no key file", "key file"])
async def test_a_turn_leaves_out_a_custom_mcp_connection_while_no_app_secret_is_held(
    request, monkeypatch, local_vault, tmp_path, sealed
):
    """The turn's vault hides custom MCP connections without the app secret
    (app_held.turn_vault), and the sealed-connection check reads that vault."""
    from cowork import app_secret
    from cowork.services.connectors.custom_mcp import CUSTOM_MCP_ENGINE
    from tests._turns import built_turn

    if sealed:
        request.getfixturevalue("key_file")
    monkeypatch.setattr(app_secret, "_secret", None)
    vault = data_vault_at(local_vault)
    vault.save(CUSTOM_MCP_ENGINE, "cm-" + "0123456789abcdef" * 2, {"_method": "mcp", "_label": "Tracker"})
    vault.save("postgres", "sales", {"password": CANARY})

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=None) as turn:
        listed = [(c["engine"], c["name"]) for c in turn.config.data_vault.list_connections()]

    assert listed == [("postgres", "sales")]


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
