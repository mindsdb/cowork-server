"""At-rest encryption for the connector data vault.

All tests run against a temp vault dir (pytest ``tmp_path``); the Fernet master
key is the throwaway one the test bootstrap (conftest) points ``MASTER_KEY_PATH``
at. The real ``~/.cowork/data-vault`` is never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

from cowork.services.connectors.encrypted_vault import (
    _ENC_PREFIX,
    EncryptedDataVault,
    build_vault,
)


def _record_file(vault_dir: Path) -> Path:
    """The single file written under the temp vault dir."""
    files = [p for p in vault_dir.iterdir() if p.is_file()]
    assert len(files) == 1, f"expected one vault file, found {files}"
    return files[0]


def test_write_is_not_plaintext_on_disk_and_round_trips(tmp_path: Path):
    vault = build_vault(tmp_path)
    secret = "super-secret-password-123"
    vault.save("postgres", "prod", {"host": "db.example.com", "password": secret})

    # On-disk bytes must not contain the plaintext secret.
    raw_text = _record_file(tmp_path).read_text(encoding="utf-8")
    assert secret not in raw_text
    assert "db.example.com" not in raw_text

    # Field values on disk are tagged ciphertext.
    on_disk = json.loads(raw_text)
    assert on_disk["fields"]["password"].startswith(_ENC_PREFIX)
    assert on_disk["fields"]["host"].startswith(_ENC_PREFIX)
    # Metadata stays plaintext so list_connections keeps working.
    assert on_disk["engine"] == "postgres"
    assert on_disk["name"] == "prod"

    # Read round-trips back to plaintext.
    assert vault.load("postgres", "prod") == {
        "host": "db.example.com",
        "password": secret,
    }


def test_read_record_round_trips_and_lists(tmp_path: Path):
    vault = build_vault(tmp_path)
    vault.save("hubspot", "main", {"api_key": "tok_abc"}, secure_keys=["api_key"])

    record = vault.read_record("hubspot", "main")
    assert record is not None
    assert record["fields"] == {"api_key": "tok_abc"}
    assert record["secure_keys"] == ["api_key"]

    listed = vault.list_connections()
    assert listed == [{"engine": "hubspot", "name": "main", "created_at": record["created_at"]}]


def test_legacy_plaintext_is_transparently_migrated_on_first_read(tmp_path: Path):
    """A pre-seeded plaintext record (as the old vault wrote it) must read
    correctly AND be re-written encrypted on first read."""
    # Hand-write a legacy plaintext record exactly like the old LocalDataVault.
    legacy = {
        "engine": "mysql",
        "name": "legacy_db",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "fields": {"host": "10.0.0.5", "password": "plaintext-legacy-pw"},
        "secure_keys": ["password"],
    }
    path = tmp_path / "mysql-legacy_db"
    path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    # Sanity: starts as plaintext on disk.
    assert "plaintext-legacy-pw" in path.read_text(encoding="utf-8")

    vault = build_vault(tmp_path)

    # First read returns the correct plaintext value.
    assert vault.load("mysql", "legacy_db") == {
        "host": "10.0.0.5",
        "password": "plaintext-legacy-pw",
    }

    # ...and the on-disk file is now encrypted (migration happened in place).
    migrated_text = path.read_text(encoding="utf-8")
    assert "plaintext-legacy-pw" not in migrated_text
    assert "10.0.0.5" not in migrated_text
    migrated = json.loads(migrated_text)
    assert migrated["fields"]["password"].startswith(_ENC_PREFIX)
    assert migrated["fields"]["host"].startswith(_ENC_PREFIX)
    # created_at preserved, secure_keys preserved across migration.
    assert migrated["created_at"] == "2025-01-01T00:00:00+00:00"
    assert migrated["secure_keys"] == ["password"]

    # A second read still returns the right value (idempotent).
    assert vault.load("mysql", "legacy_db") == {
        "host": "10.0.0.5",
        "password": "plaintext-legacy-pw",
    }


def test_migrated_value_is_decryptable_by_a_fresh_vault_instance(tmp_path: Path):
    """Encryption is keyed by the shared master key, not per-instance state —
    a brand-new vault object reads what an earlier one wrote."""
    build_vault(tmp_path).save("redis", "cache", {"password": "r3dis-pw"})
    # Fresh instance, same dir + same master key.
    assert build_vault(tmp_path).load("redis", "cache") == {"password": "r3dis-pw"}


def test_update_preserves_created_at_and_re_encrypts(tmp_path: Path):
    vault = build_vault(tmp_path)
    vault.save("postgres", "prod", {"password": "v1"})
    created = vault.read_record("postgres", "prod")["created_at"]

    vault.save("postgres", "prod", {"password": "v2"})
    record = vault.read_record("postgres", "prod")
    assert record["fields"] == {"password": "v2"}
    assert record["created_at"] == created  # preserved across update

    raw = _record_file(tmp_path).read_text(encoding="utf-8")
    assert "v2" not in raw  # still encrypted on disk
    assert json.loads(raw)["fields"]["password"].startswith(_ENC_PREFIX)


def test_load_missing_connection_returns_none(tmp_path: Path):
    assert build_vault(tmp_path).load("nope", "missing") is None
    assert build_vault(tmp_path).read_record("nope", "missing") is None


def test_build_vault_returns_encrypted_vault(tmp_path: Path):
    assert isinstance(build_vault(tmp_path), EncryptedDataVault)
