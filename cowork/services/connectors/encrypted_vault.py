"""At-rest encryption for the local connector credential vault.

The connector vault (``LocalDataVault``, from the ``anton-agent`` package)
persists each connection as a JSON file under ``~/.cowork/data-vault``. By
default the credential ``fields`` — passwords, API keys, OAuth tokens — are
written in **plaintext**. Channel credentials are already encrypted at rest in
the DB via :mod:`cowork.common.encryption` (Fernet, keyed by the app's
``master_key_path``); this module brings the connector vault to parity.

Design
------
``EncryptedDataVault`` subclasses ``LocalDataVault`` and overrides only the
on-disk (de)serialization seam. Each value in the record's ``fields`` dict is
Fernet-encrypted on write and decrypted on read; the surrounding record
metadata (``engine``, ``name``, ``created_at``, ``updated_at``,
``secure_keys``) stays plaintext so ``list_connections`` /
``next_connection_number`` — which only ever read that metadata — keep working
untouched and need no override.

The external ``DataVault`` API and behavior are unchanged: callers still pass
and receive plaintext ``credentials`` dicts. The encryption is purely at rest.

Transparent migration
----------------------
Encrypted values are tagged with a short prefix (:data:`_ENC_PREFIX`). On read,
any field value lacking the prefix is treated as a legacy plaintext entry,
returned as-is, and the whole record is **re-written encrypted** so the next
read finds only ciphertext. Migration is therefore lazy (first read per record)
and idempotent. A value that carries the prefix but fails to decrypt (e.g. a
plaintext string that happens to start with the prefix) also falls back to
being returned verbatim rather than raising — over-eager, but never lossy.

Key source
----------
Fernet key comes from :func:`cowork.common.encryption.get_fernet`, i.e. the
same ``master_key_path`` (default ``~/.cowork/.master_key``) used for channel
credentials. No new key material is introduced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.encryption import get_fernet

# Marks a Fernet token written by this class. Fernet tokens are urlsafe-base64
# and never contain ':', so a stored value beginning with this exact string is
# unambiguously one we wrote (a legacy plaintext credential would have to start
# with the literal "enc:v1:" to collide, and even then the decrypt-failure
# fallback keeps the read lossless).
_ENC_PREFIX = "enc:v1:"


def _encrypt_value(plaintext: str) -> str:
    return _ENC_PREFIX + get_fernet().encrypt(plaintext.encode()).decode()


def _looks_encrypted(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_ENC_PREFIX)


def _decrypt_value(value: str) -> str:
    """Decrypt a tagged value. On any failure, return it verbatim.

    Defensive: a corrupt token or a plaintext string that merely starts with
    the prefix must not blow up a read — better to surface the raw value than
    to lose the credential.
    """
    from cryptography.fernet import InvalidToken

    token = value[len(_ENC_PREFIX):]
    try:
        return get_fernet().decrypt(token.encode()).decode()
    except (InvalidToken, ValueError):
        return value


class EncryptedDataVault(LocalDataVault):
    """``LocalDataVault`` with Fernet-encrypted credential fields at rest.

    Only the ``fields`` values are encrypted; record metadata stays plaintext.
    Reads transparently decrypt and migrate legacy plaintext records.
    """

    # ── write seam ──────────────────────────────────────────────────────
    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> Path:
        encrypted = {
            key: (_encrypt_value(value) if isinstance(value, str) else value)
            for key, value in credentials.items()
        }
        return super().save(engine, name, encrypted, secure_keys=secure_keys)

    # ── read seam ───────────────────────────────────────────────────────
    def _read_raw(self, path: Path) -> dict[str, Any] | None:
        """Decrypt fields in the raw record, migrating legacy plaintext in place.

        ``_read_raw`` is the chokepoint for ``read_record`` and for ``save``'s
        prior-record lookup. We decrypt the ``fields`` values here so both see
        plaintext, and if any value was legacy plaintext we re-persist the
        record encrypted (one-time, idempotent migration).
        """
        record = super()._read_raw(path)
        if record is None:
            return None

        fields = record.get("fields")
        if not isinstance(fields, dict):
            return record

        needs_migration = False
        decrypted: dict[str, Any] = {}
        for key, value in fields.items():
            if _looks_encrypted(value):
                decrypted[key] = _decrypt_value(value)
            else:
                # Legacy plaintext (or a non-str value) — keep as-is and flag
                # the record for re-encryption.
                decrypted[key] = value
                if isinstance(value, str):
                    needs_migration = True

        record["fields"] = decrypted

        if needs_migration:
            # Re-save through our own encrypting ``save`` so the on-disk record
            # becomes ciphertext. ``save`` preserves ``created_at`` and the
            # ``secure_keys`` list, so the migration is transparent.
            try:
                self.save(
                    record.get("engine", ""),
                    record.get("name", ""),
                    {k: v for k, v in decrypted.items() if isinstance(v, str)},
                    secure_keys=record.get("secure_keys"),
                )
            except Exception:
                # A failed migration must not break the read — the caller
                # still gets correct plaintext; we just retry next time.
                pass

        return record

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        """Return decrypted credential fields, or None if not found.

        The base ``load`` decodes the file independently of ``_read_raw``, so
        route it through ``_read_raw`` here to reuse the decrypt+migrate path.
        """
        record = self._read_raw(self._path_for(engine, name))
        if record is None:
            return None
        return record.get("fields", {})


def build_vault(vault_dir: Path) -> EncryptedDataVault:
    """Construct the connector vault with at-rest encryption enabled.

    Single factory used by every server call site in place of constructing
    ``LocalDataVault`` directly, so the encryption seam stays in one place.
    The return type is the ``DataVault`` protocol the rest of the code expects.
    """
    return EncryptedDataVault(vault_dir)
