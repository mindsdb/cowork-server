"""Saved connections sealed under the server's key file.

With ``COWORK_KEY_FILE`` set (cowork.common.encryption.sealing_keys), every
vault the server builds is a SealedDataVault (persist.data_vault_at). It keeps
anton's record layout and file names, so listing, sharing and the operator's
inventory read the same files, and seals each field value outside a fixed list
of bookkeeping fields. A value is sealed as its JSON text, so it opens with
the type and text it was saved with.

A record that exists but won't open is an error that names the connection,
never a missing record: anton's LocalDataVault reads a file it can't parse as
missing, and a layered vault then lets a same-named connection from a lower
layer answer in its place.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path
from typing import Any

from anton.core.datasources.data_vault import DataVault, LocalDataVault
from cryptography.fernet import InvalidToken, MultiFernet

from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, open_fd, read_bounded
from cowork.services.connectors.layered_vault import CREATED_BY_FIELD

# Fields that describe a connection rather than hold its credentials. They stay
# readable, so listing, sharing and the operator's inventory need no key.
BOOKKEEPING_FIELDS = frozenset(
    {
        CREATED_BY_FIELD,
        "_label",
        "_user_label",
        "_method",
        "_connector_id",
        "_picked_files",
        "_access_mode",
    }
)
SEALED_FIELD = "_sealed"
SEALED_VERSION = 1
MAX_RECORD_BYTES = 1 << 20


class SealedRecordError(Exception):
    """A saved connection whose record file exists but can't be opened.

    ``connection`` is ``engine/name``, or the file name when the file doesn't
    say. The message names it and the cause, never a value.
    """

    def __init__(self, *, connection: str, cause: str) -> None:
        self.connection = connection
        self.cause = cause
        super().__init__(f"The saved connection {connection} can't be opened: {cause}.")


class SealedDataVault(LocalDataVault):
    """anton's LocalDataVault with every non-bookkeeping value sealed."""

    def __init__(self, vault_dir: Path, *, keys: MultiFernet) -> None:
        super().__init__(vault_dir)
        self._keys = keys

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, Any],
        *,
        secure_keys: list[str] | None = None,
    ) -> Path:
        sealed: dict[str, Any] = {
            key: value if key in BOOKKEEPING_FIELDS else self._seal(value=value)
            for key, value in credentials.items()
            if key != SEALED_FIELD
        }
        sealed[SEALED_FIELD] = SEALED_VERSION
        return super().save(engine, name, sealed, secure_keys=secure_keys)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        connection = f"{engine}/{name}"
        record = read_record_file(path=self._path_for(engine, name), connection=connection)
        if record is None:
            return None
        return {**record, "fields": open_fields(fields=record["fields"], keys=self._keys, connection=connection)}

    def load(self, engine: str, name: str) -> dict[str, Any] | None:
        record = self.read_record(engine, name)
        return None if record is None else record["fields"]

    def list_connections(self) -> list[dict[str, str]]:
        """anton's listing, except that a record file that can't be read or
        parsed raises instead of dropping out of the list."""
        results = []
        for path in self._record_paths():
            record = read_record_file(path=path, connection=path.name)
            if record is None:
                continue  # removed since the folder was listed
            results.append(
                {
                    "engine": record.get("engine", ""),
                    "name": record.get("name", ""),
                    "created_at": record.get("created_at", ""),
                }
            )
        return results

    def unopenable_records(self) -> list[SealedRecordError]:
        """Each record in the folder that can't be read, parsed or opened, or
        that lacks the sealed mark, for a check that names them all at once."""
        failures = []
        for path in self._record_paths():
            try:
                record = read_record_file(path=path, connection=path.name)
                if record is not None:
                    connection = f"{record.get('engine', '')}/{record.get('name', '')}"
                    open_fields(fields=record["fields"], keys=self._keys, connection=connection)
            except SealedRecordError as error:
                failures.append(error)
        return failures

    def _record_paths(self) -> list[Path]:
        """The folder's record files: every entry except subfolders and the
        temporary file anton's save renames into place."""
        try:
            entries = sorted(os.scandir(self._dir), key=lambda entry: entry.name)
        except FileNotFoundError:
            return []
        return [
            Path(entry.path)
            for entry in entries
            if not entry.name.endswith(".tmp") and not entry.is_dir(follow_symlinks=False)
        ]

    def _seal(self, *, value: Any) -> str:
        return self._keys.encrypt(json.dumps(value).encode()).decode()


def open_fields(*, fields: dict[str, Any], keys: MultiFernet, connection: str) -> dict[str, Any]:
    """A sealed record's fields, each value opened with ``keys``. A record without the sealed
    mark, or a value that isn't a token ``keys`` opens, raises."""
    if fields.get(SEALED_FIELD) != SEALED_VERSION:
        raise SealedRecordError(
            connection=connection,
            cause="its record isn't sealed; stop Cowork and run the re-seal command",
        )
    opened: dict[str, Any] = {}
    for key, value in fields.items():
        if key == SEALED_FIELD:
            continue
        if key in BOOKKEEPING_FIELDS:
            opened[key] = value
            continue
        if not isinstance(value, str):
            raise SealedRecordError(connection=connection, cause=f"its {key} field isn't sealed")
        try:
            opened[key] = json.loads(keys.decrypt(value.encode()))
        except (InvalidToken, ValueError):
            raise SealedRecordError(
                connection=connection, cause=f"its {key} field doesn't open with the server's key"
            ) from None
    return opened


def read_record_file(*, path: Path, connection: str) -> dict[str, Any] | None:
    """The record at ``path``, or None when no file is there. A link, a
    file with a second name, or anything that isn't a readable JSON record
    raises."""
    try:
        fd = open_fd(path, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SealedRecordError(connection=connection, cause="its record file is a link") from None
        raise SealedRecordError(
            connection=connection, cause=f"its record file can't be read ({errno.errorcode.get(error.errno, 'error')})"
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SealedRecordError(
                connection=connection, cause="its record file isn't a regular file with one name"
            )
        if info.st_size > MAX_RECORD_BYTES:
            raise SealedRecordError(connection=connection, cause="its record file is too large")
        raw = b"".join(read_bounded(fd, info.st_size))
    finally:
        os.close(fd)
    try:
        record = json.loads(raw)
    except ValueError:
        raise SealedRecordError(connection=connection, cause="its record file isn't valid JSON") from None
    if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
        raise SealedRecordError(connection=connection, cause="its record file isn't a saved connection")
    return record


def refuse_unopenable_connections(*, vault: DataVault) -> None:
    """Open every connection ``vault`` lists, so a turn or an app backend stops
    on the first one that won't open rather than starting without it. anton
    builds a cell's environment one connection at a time and leaves out one
    that fails, logged at DEBUG."""
    for connection in vault.list_connections():
        vault.read_record(connection["engine"], connection["name"])
