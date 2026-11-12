"""Start checks for secrets sealed under ``COWORK_KEY_FILE``.

Sealing is opt-in: with no key file set, the server stores connections and
settings as it always has. Two checks make the switch fail closed:

* With a key file, every sealed setting, channel credential and saved
  connection must open with it, and every connection must carry the sealed
  mark, before the server serves anything (refuse_unopenable_sealed_data).
* Without one, the server refuses to start where sealing was turned on, so it
  never reads sealed values as missing or makes a new legacy key beside them
  (refuse_sealed_state_without_key_file). The re-seal command writes the
  marker row that records it was turned on.

Org tenancy reads no vault in either check.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from cryptography.fernet import InvalidToken
from sqlmodel import Session as SQLSession
from sqlmodel import select

from cowork.common.encryption import KeyFileError, sealing_keys
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import UserSettings
from cowork.db.session import get_engine
from cowork.models.setting import Setting
from cowork.services.connectors.sealed_vault import (
    BOOKKEEPING_FIELDS,
    SEALED_FIELD,
    SealedDataVault,
    SealedRecordError,
    errno_name,
    is_fernet_token,
    is_sealed_record,
    read_record_file,
)

logger = logging.getLogger(__name__)

# The global settings row the re-seal command writes once it has sealed this
# deployment's connections and settings under the key file.
SEALING_MARKER_KEY = "sealing.turned_on"


class SealedDataError(RuntimeError):
    """Sealed data that the key file doesn't open. The message names each
    record and settings row, never a value."""


def vault_roots() -> list[Path]:
    """Every folder that holds saved connections on a local deployment: the
    base vault, the operator's managed folder, and each person's partition
    under the shared root."""
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.managed import managed_folder
    from cowork.services.connectors.persist import own_vault_root

    roots = [own_vault_root(LOCAL_SCOPE)]
    managed = managed_folder()
    if managed is not None:
        roots.append(managed.vault_dir)
    shared_root = Path(get_app_settings().storage.shared_root)
    try:
        entries = sorted(os.scandir(shared_root), key=lambda entry: entry.name)
    except FileNotFoundError:
        entries = []
    for entry in entries:
        try:
            UUID(entry.name)
        except ValueError:
            continue  # The shared root also holds stores that aren't partitions.
        roots.append(Path(entry.path) / "data-vault")
    return roots


def sealing_marker_present() -> bool:
    """Whether the re-seal command has sealed this deployment.

    It runs before migrations, so it reads any schema since the first: a
    SQLite file that doesn't exist yet, or a database with no settings table,
    has never been sealed, and the query names the ``key`` column alone. Only
    the re-seal command writes the marker, as a global row, so no scope is
    asked for."""
    from cowork.dev_setup import sqlite_database_file

    db_uri = get_app_settings().database.uri
    database_file = sqlite_database_file(db_uri=db_uri)
    if database_file is not None and not database_file.exists():
        return False
    with get_engine(db_uri).connect() as connection:
        if not sa.inspect(connection).has_table(Setting.__tablename__):
            return False
        row = connection.execute(
            sa.select(Setting.key).where(Setting.key == SEALING_MARKER_KEY).limit(1)
        ).first()
    return row is not None


def _holds_sealed_values(*, fields: dict[str, Any]) -> bool:
    """Whether a record reads as one SealedDataVault wrote, checked without a
    key: the sealed mark, and each value outside the bookkeeping fields a
    Fernet token. A plain record can carry a field named like the mark."""
    return is_sealed_record(fields=fields) and all(
        key in BOOKKEEPING_FIELDS or key == SEALED_FIELD or is_fernet_token(value=value)
        for key, value in fields.items()
    )


def _warn_unreadable_vault(*, folder: Path | str | None, error: OSError) -> None:
    logger.warning(
        "A vault folder can't be read, so the no-key start check skips it",
        extra={"event": "vault_unreadable", "vault_folder": str(folder), "reason": errno_name(error=error)},
    )


def refuse_sealed_state_without_key_file() -> None:
    """Refuse when no key file is set on a local deployment whose settings
    record that sealing was turned on, or whose vaults hold a sealed record.

    The vault scan reads each record file the way the vault does, and skips
    one it can't read, as anton's LocalDataVault does without a key, and a
    folder it can't search, with a warning: the marker row is the record of
    the switch, and the scan backs it up."""
    if sealing_keys() is not None or get_app_settings().tenancy_mode == "org":
        return
    if sealing_marker_present():
        raise KeyFileError(
            "Sealing was turned on for this deployment, and COWORK_KEY_FILE isn't set. "
            "Start the server with COWORK_KEY_FILE naming the key file its data was sealed under."
        )
    try:
        roots = vault_roots()
    except OSError as error:
        _warn_unreadable_vault(folder=error.filename, error=error)
        return
    for root in roots:
        try:
            entries = sorted(root.glob("*")) if root.is_dir() else []
        except OSError as error:
            _warn_unreadable_vault(folder=root, error=error)
            continue
        for path in entries:
            if path.name.endswith(".tmp"):
                continue
            try:
                record = read_record_file(path=path, connection=path.name)
            except SealedRecordError:
                continue
            if record is not None and _holds_sealed_values(fields=record["fields"]):
                raise KeyFileError(
                    f"The saved connection file {path} is sealed, and COWORK_KEY_FILE isn't set. "
                    "Start the server with COWORK_KEY_FILE naming the key file it was sealed under."
                )


def sealed_setting_keys() -> frozenset[str]:
    """The settings keys whose rows the server stores sealed: each sensitive setting and each
    channel's secret credential."""
    from cowork.services.channels import secret_credential_keys

    sensitive = frozenset(key for key in UserSettings.model_fields if UserSettings.field_is_sensitive(key))
    return sensitive | secret_credential_keys()


def _row_name(row: Setting) -> str:
    if row.scope is None:
        return f"setting {row.key} (global)"
    if row.scope == "org":
        return f"setting {row.key} (org {row.org_id})"
    return f"setting {row.key} (user {row.user_id} in {row.org_id})"


def refuse_unopenable_sealed_data() -> None:
    """With a key file set, refuse to start unless every sealed setting and
    channel credential opens with it, and, on a local deployment, every saved
    connection is sealed and opens with it. Names every one that fails."""
    keys = sealing_keys()
    if keys is None:
        return
    failures: list[str] = []
    sealed_keys = sealed_setting_keys()
    engine = get_engine(get_app_settings().database.uri)
    with SQLSession(engine) as session:
        rows = session.exec(select(Setting).where(Setting.key.in_(sealed_keys))).all()
    for row in rows:
        try:
            keys.decrypt(row.value.encode())
        except InvalidToken:
            failures.append(_row_name(row))
    if get_app_settings().tenancy_mode != "org":
        try:
            roots = vault_roots()
        except OSError as error:
            failures.append(f"vault {error.filename}: can't be read ({errno_name(error=error)})")
            roots = []
        for root in roots:
            try:
                unopenable = SealedDataVault(root, keys=keys).unopenable_records()
            except OSError as error:
                failures.append(f"vault {root}: can't be read ({errno_name(error=error)})")
                continue
            failures.extend(f"connection {error.connection} in {root}: {error.cause}" for error in unopenable)
    if failures:
        logger.error(
            "Sealed data doesn't open with the key file, so the server won't start",
            extra={"event": "sealed_data_unopenable", "count": len(failures)},
        )
        raise SealedDataError(
            f"COWORK_KEY_FILE's keys don't open {len(failures)} sealed item(s): "
            + "; ".join(failures)
            + ". Check that the pinned key version is the one this data was sealed under, "
            "or run the re-seal command with Cowork stopped."
        )
