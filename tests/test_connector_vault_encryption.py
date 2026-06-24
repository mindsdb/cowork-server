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


def test_legacy_migration_preserves_non_string_fields(tmp_path: Path):
    """Regression: migrating a legacy plaintext record must not drop non-string
    fields (int ``port``, bool flags, nested dicts).

    The old migration re-saved only ``str`` fields, so any non-string in a
    legacy record was permanently dropped from disk on the first read —
    silently corrupting real connector credentials. ``save`` already passes
    non-strings through unencrypted, so the migration must re-save the full
    decrypted dict.
    """
    # Legacy plaintext record with mixed value types, as the old vault wrote it.
    legacy = {
        "engine": "mysql",
        "name": "mixed_db",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "fields": {
            "password": "pw",          # str -> encrypted
            "port": 3306,              # int -> preserved as-is
            "ssl": True,               # bool -> preserved as-is
            "opts": {"x": 1},          # nested dict -> preserved as-is
        },
        "secure_keys": ["password"],
    }
    path = tmp_path / "mysql-mixed_db"
    path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    vault = build_vault(tmp_path)
    expected = {"password": "pw", "port": 3306, "ssl": True, "opts": {"x": 1}}

    # (a) First read returns ALL fields intact, including the non-strings.
    assert vault.load("mysql", "mixed_db") == expected

    # (b) The on-disk file after migration still contains every field —
    #     port/ssl/opts are not dropped.
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert set(migrated["fields"]) == {"password", "port", "ssl", "opts"}
    # (d) Strings are encrypted on disk; non-strings preserved verbatim.
    assert migrated["fields"]["password"].startswith(_ENC_PREFIX)
    assert "pw" not in path.read_text(encoding="utf-8")
    assert migrated["fields"]["port"] == 3306
    assert migrated["fields"]["ssl"] is True
    assert migrated["fields"]["opts"] == {"x": 1}

    # (c) A second read from a fresh vault instance still returns all fields.
    assert build_vault(tmp_path).load("mysql", "mixed_db") == expected


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


# ── test-result metadata (slice 2) ──────────────────────────────────────────

def test_record_test_result_stamps_plaintext_metadata(tmp_path: Path):
    vault = build_vault(tmp_path)
    vault.save("postgres", "prod", {"password": "pw"})

    stamped = vault.record_test_result("postgres", "prod", result="pass")
    assert stamped is True

    record = vault.read_record("postgres", "prod")
    assert record["last_test_result"] == "pass"
    assert record["last_tested_at"]  # ISO timestamp set
    # A pass clears any error string.
    assert record["last_test_error"] == ""
    # Metadata is plaintext on disk; the password stays encrypted.
    raw = json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["last_test_result"] == "pass"
    assert raw["fields"]["password"].startswith(_ENC_PREFIX)


def test_record_test_result_retains_error_on_failure(tmp_path: Path):
    vault = build_vault(tmp_path)
    vault.save("postgres", "prod", {"password": "pw"})
    vault.record_test_result("postgres", "prod", result="fail", error="auth denied")

    record = vault.read_record("postgres", "prod")
    assert record["last_test_result"] == "fail"
    assert record["last_test_error"] == "auth denied"


def test_record_test_result_missing_connection_returns_false(tmp_path: Path):
    assert build_vault(tmp_path).record_test_result("nope", "x", result="pass") is False


def test_test_metadata_survives_credential_resave(tmp_path: Path):
    """A credential re-save (e.g. an OAuth token refresh) must not wipe the
    stamped test metadata — the base LocalDataVault.save only writes a fixed
    top-level key set, so the subclass has to carry it forward."""
    vault = build_vault(tmp_path)
    vault.save("google_drive", "me@example.com", {"access_token": "v1"})
    vault.record_test_result("google_drive", "me@example.com", result="pass")

    # Re-save credentials (simulates a token refresh).
    vault.save("google_drive", "me@example.com", {"access_token": "v2"})

    record = vault.read_record("google_drive", "me@example.com")
    assert record["fields"]["access_token"] == "v2"      # creds updated
    assert record["last_test_result"] == "pass"          # metadata preserved
    assert record["last_tested_at"]


def test_legacy_migration_does_not_recurse(tmp_path: Path):
    """Regression: migrating a legacy plaintext record must re-save EXACTLY
    once. The migration re-saves via ``self.save`` → base ``save`` → which
    re-reads through ``self._read_raw`` (for created_at) on the not-yet-written
    file; without the re-entrancy guard that nested read re-triggers migration
    and recurses ~250 deep (RecursionError under a real request stack)."""
    legacy = {
        "engine": "pg", "name": "p",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "fields": {"host": "h", "password": "pw"},
        "secure_keys": ["password"],
    }
    (tmp_path / "pg-p").write_text(json.dumps(legacy), encoding="utf-8")

    vault = build_vault(tmp_path)
    calls = {"n": 0}
    orig_save = type(vault).save

    def counting_save(self, *a, **k):
        calls["n"] += 1
        return orig_save(self, *a, **k)

    # Patch the instance's class method just for this read.
    import types
    vault.save = types.MethodType(counting_save, vault)

    assert vault.load("pg", "p") == {"host": "h", "password": "pw"}
    assert calls["n"] == 1, f"migration recursed: {calls['n']} save calls"
    # Migration still completed: encrypted on disk, secure_keys preserved.
    on_disk = json.loads((tmp_path / "pg-p").read_text(encoding="utf-8"))
    assert on_disk["fields"]["password"].startswith(_ENC_PREFIX)
    assert on_disk["secure_keys"] == ["password"]


def test_test_metadata_does_not_leak_into_load_fields(tmp_path: Path):
    """The stamped metadata lives at the top level, never inside ``fields`` —
    so it never pollutes the credential dict the agent/probe consumes."""
    vault = build_vault(tmp_path)
    vault.save("postgres", "prod", {"password": "pw"})
    vault.record_test_result("postgres", "prod", result="fail", error="boom")

    fields = vault.load("postgres", "prod")
    assert fields == {"password": "pw"}
    assert "last_test_result" not in fields
