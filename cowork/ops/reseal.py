"""Seal every saved connection and every encrypted setting under the key file, with the server stopped.

    python -m cowork.ops.reseal [--managed-vault PATH] [--old-key-file PATH]

Turning sealing on and rotating its key both run this command. In order, it:

1. refuses unless ``COWORK_KEY_FILE`` names a key file the server would start with
   (cowork.common.encryption.sealing_keys), on a local deployment;
2. checks every entry in every vault before it writes anything: the base vault, each person's
   partition under the shared root, and the managed folder's (``--managed-vault``, a writable path
   to it). It reads each through descriptors that follow no link, whatever ``COWORK_CODE_SANDBOX``
   says, and stops, writing nothing, at a link, a file with a second name, a record whose file name
   isn't its engine and name's, or one that doesn't parse, naming each path. Agent code that ran
   before the code sandbox could have planted any of them, to send these writes into another
   person's space;
3. opens every record, and every sealed setting, channel credential and live-app preview cookie,
   with the key file's keys and the legacy key: ``--old-key-file``, else ``COWORK_MASTER_KEY``,
   else the file at ``MASTER_KEY_PATH``. It stops, changing nothing, when a record, setting or
   credential opens with none, naming every one. A preview cookie no key opens is cleared: the
   preview already reads one as no grant (cowork.services.live_app_preview);
4. records that sealing was turned on (cowork.services.sealing.SEALING_MARKER_KEY), then writes
   every record through the sealed vault with its own ``secure_keys``, absent where it was absent,
   and every row under the key file's first key in one transaction, and removes leftover
   temporary files;
5. deletes the legacy key file it opened with, last.

It prints counts and paths, never a value. Run again, it changes nothing but each record's
``updated_at``.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import stat
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from sqlmodel import Session, select

from cowork.common.encryption import KeyFileError, key_file_path, read_key_file
from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_lstat,
    dir_open,
    dir_scandir,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_engine
from cowork.managed import managed_folder
from cowork.models.live_app_preview import LiveAppPreview
from cowork.models.setting import Setting
from cowork.ops import OperatorCommandError, run_operator_command
from cowork.ops.connection_inventory import _vault_filename
from cowork.services.connectors.persist import own_vault_root
from cowork.services.connectors.sealed_vault import (
    MAX_RECORD_BYTES,
    SEALED_FIELD,
    SealedDataVault,
    SealedRecordError,
    open_fields,
)
from cowork.services.sealing import SEALING_MARKER_KEY, sealed_setting_keys


class Problem(NamedTuple):
    """An entry that stops the command before it writes anything."""

    path: Path
    reason: str


class StoredRecord(NamedTuple):
    """A saved connection, read and opened, to write back sealed."""

    vault: Path
    engine: str
    name: str
    fields: dict[str, Any]
    secure_keys: list[str] | None


class VaultContents(NamedTuple):
    """What the check found in one vault folder."""

    records: list[tuple[str, dict[str, Any]]]
    leftovers: list[str]
    problems: list[Problem]


class LegacyKey(NamedTuple):
    """The key the server sealed settings with before the key file, and the file it came from."""

    key: Fernet
    path: Path | None


def _read_entry(*, directory: PinnedDir, name: str) -> bytes:
    descriptor = dir_open(directory, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError(errno.EMLINK, "not a regular file with one name")
        if info.st_size > MAX_RECORD_BYTES:
            raise OSError(errno.EFBIG, "too large")
        return b"".join(read_bounded(descriptor, info.st_size))
    finally:
        os.close(descriptor)


def _check_folder(*, directory: PinnedDir) -> VaultContents:
    """Every entry of one vault folder: the records to seal, the leftover temporary files to
    remove, and each entry that stops the command."""
    records: list[tuple[str, dict[str, Any]]] = []
    leftovers: list[str] = []
    problems: list[Problem] = []
    with dir_scandir(directory) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        path = directory.path / name
        info = dir_lstat(directory, name)
        if stat.S_ISLNK(info.st_mode):
            problems.append(Problem(path, "is a link"))
            continue
        if stat.S_ISDIR(info.st_mode):
            continue  # anton's vault keeps no folder; it lists files only
        if not stat.S_ISREG(info.st_mode):
            problems.append(Problem(path, "isn't a regular file"))
            continue
        if info.st_nlink != 1:
            problems.append(Problem(path, "has a second name"))
            continue
        if name.endswith(".tmp"):
            leftovers.append(name)
            continue
        try:
            record = json.loads(_read_entry(directory=directory, name=name))
        except OSError as error:
            problems.append(Problem(path, f"can't be read ({errno.errorcode.get(error.errno, 'error')})"))
            continue
        except ValueError:
            problems.append(Problem(path, "isn't valid JSON"))
            continue
        if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
            problems.append(Problem(path, "isn't a saved connection"))
            continue
        engine, record_name = record.get("engine"), record.get("name")
        if not isinstance(engine, str) or not isinstance(record_name, str):
            problems.append(Problem(path, "isn't a saved connection"))
            continue
        if _vault_filename(engine, record_name) != name:
            problems.append(Problem(path, f"holds {engine}/{record_name}, whose file name differs"))
            continue
        records.append((name, record))
    return VaultContents(records=records, leftovers=leftovers, problems=problems)


def _vault_folders(*, managed_vault: Path | None) -> list[Path]:
    """Every vault folder on this deployment, the managed one at ``managed_vault``."""
    if get_app_settings().tenancy_mode == "org":
        raise OperatorCommandError("The re-seal command runs on a local deployment only.")
    if managed_folder() is not None and managed_vault is None:
        raise OperatorCommandError(
            "COWORK_MANAGED_DIR is set: pass --managed-vault, a writable path to its data-vault folder."
        )
    folders = [own_vault_root(LOCAL_SCOPE)]
    if managed_vault is not None:
        folders.append(managed_vault)
    shared_root = Path(get_app_settings().storage.shared_root)
    try:
        with pinned_dir(shared_root, nofollow_base=True) as root, dir_scandir(root) as entries:
            partitions = sorted(entry.name for entry in entries)
    except FileNotFoundError:
        partitions = []
    for partition in partitions:
        try:
            UUID(partition)
        except ValueError:
            continue  # The shared root also holds stores that aren't partitions.
        folders.append(shared_root / partition / "data-vault")
    return folders


def _folder_contents(*, folder: Path, shared_root: Path) -> VaultContents:
    """``folder``'s check, opened without following a link at it, or, for a partition's vault, at
    the partition folder on the way to it."""
    if folder.parent.parent == shared_root and folder.name == "data-vault":
        with pinned_dir(shared_root, nofollow_base=True) as root, ExitStack() as opened:
            partition = open_pinned_child(root, folder.parent.name)
            opened.callback(partition.close)
            vault = open_pinned_child(partition, "data-vault")
            opened.callback(vault.close)
            return _check_folder(directory=vault)
    with pinned_dir(folder, nofollow_base=True) as directory:
        return _check_folder(directory=directory)


def _check_vaults(*, folders: list[Path]) -> tuple[dict[Path, VaultContents], list[Problem]]:
    shared_root = Path(get_app_settings().storage.shared_root)
    found: dict[Path, VaultContents] = {}
    problems: list[Problem] = []
    for folder in folders:
        try:
            contents = _folder_contents(folder=folder, shared_root=shared_root)
        except FileNotFoundError:
            continue
        except OSError as error:
            reason = "is a link, or on the way to one" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "can't be read"
            problems.append(Problem(folder, reason))
            continue
        found[folder] = contents
        problems.extend(contents.problems)
    return found, problems


def _legacy_key(*, old_key_file: Path | None) -> LegacyKey | None:
    """The key settings were sealed with before the key file, found as the server found it."""
    settings = get_app_settings()
    path = old_key_file
    if path is None and settings.master_key:
        return LegacyKey(key=_fernet(material=settings.master_key.strip().encode(), source="COWORK_MASTER_KEY"), path=None)
    if path is None:
        path = Path(settings.master_key_path)
        if not path.is_file():
            return None
    try:
        material = path.read_bytes().strip()
    except OSError as error:
        raise OperatorCommandError(f"The old key file {path} can't be read: {error.strerror}.") from None
    return LegacyKey(key=_fernet(material=material, source=str(path)), path=path)


def _fernet(*, material: bytes, source: str) -> Fernet:
    try:
        return Fernet(material)
    except ValueError:
        raise OperatorCommandError(f"The old key in {source} isn't a Fernet key.") from None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reseal(*, managed_vault: Path | None, old_key_file: Path | None) -> str:
    """Seals every record and row under the key file's first key; see the module's docstring."""
    path = key_file_path()
    if path is None:
        raise OperatorCommandError("COWORK_KEY_FILE isn't set, so there is no key to seal under: nothing changed.")
    try:
        file_keys = read_key_file(path=path)
    except KeyFileError as error:
        raise OperatorCommandError(f"{error} Nothing changed.") from None
    sealing = MultiFernet(file_keys)
    legacy = _legacy_key(old_key_file=old_key_file)
    opening = MultiFernet([*file_keys, *([legacy.key] if legacy else [])])

    found, problems = _check_vaults(folders=_vault_folders(managed_vault=managed_vault))
    if problems:
        listed = "".join(f"\n  {problem.path}: {problem.reason}" for problem in problems)
        raise OperatorCommandError(
            f"These vault entries stop the re-seal, which wrote nothing:{listed}\n"
            "Agent code from before the code sandbox may have left them: run make residue-check, "
            "remove what it names, and run this again."
        )

    unopened: list[str] = []
    records: list[StoredRecord] = []
    for folder, contents in found.items():
        for _file, record in contents.records:
            connection = f"{record['engine']}/{record['name']}"
            fields = record["fields"]
            try:
                opened = (
                    open_fields(fields=fields, keys=opening, connection=connection)
                    if SEALED_FIELD in fields
                    else dict(fields)
                )
            except SealedRecordError as error:
                unopened.append(f"connection {connection} in {folder}: {error.cause}")
                continue
            secure_keys = record.get("secure_keys")
            records.append(StoredRecord(folder, record["engine"], record["name"], opened, secure_keys))

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        rows = session.exec(select(Setting).where(Setting.key.in_(sealed_setting_keys()))).all()
        settings: list[tuple[Setting, bytes]] = []
        for row in rows:
            try:
                settings.append((row, opening.decrypt(row.value.encode())))
            except InvalidToken:
                unopened.append(f"setting {row.key} (scope {row.scope or 'global'})")
        if unopened:
            listed = "".join(f"\n  {item}" for item in unopened)
            raise OperatorCommandError(
                f"No key opens these, so the re-seal changed nothing:{listed}\n"
                "Check that the key file lists the key they were sealed under, or name the old key with --old-key-file."
            )
        cookies = session.exec(select(LiveAppPreview).where(LiveAppPreview.viewer_cookie.is_not(None))).all()

        marker = session.exec(
            select(Setting).where(Setting.key == SEALING_MARKER_KEY, Setting.scope.is_(None))
        ).first()
        if marker is None:
            session.add(Setting(key=SEALING_MARKER_KEY, value=_now()))
            session.commit()

        for record in records:
            SealedDataVault(record.vault, keys=sealing).save(
                record.engine, record.name, record.fields, secure_keys=record.secure_keys
            )
        for row, plaintext in settings:
            row.value = sealing.encrypt(plaintext).decode()
        cleared = 0
        for cookie in cookies:
            try:
                cookie.viewer_cookie = sealing.encrypt(opening.decrypt(cookie.viewer_cookie.encode())).decode()
            except InvalidToken:
                cookie.viewer_cookie = None
                cleared += 1
        session.commit()

    removed = 0
    for folder, contents in found.items():
        for name in contents.leftovers:
            (folder / name).unlink(missing_ok=True)
            removed += 1

    lines = [
        f"Sealed {len(records)} saved connection(s) in {len(found)} vault(s), {len(settings)} setting(s) and "
        f"channel credential(s), and {len(cookies) - cleared} live-app preview cookie(s) under the key file's first key.",
    ]
    if cleared:
        lines.append(f"Cleared {cleared} preview cookie(s) no key opened; those previews ask to sign in again.")
    if removed:
        lines.append(f"Removed {removed} leftover temporary file(s).")
    if legacy is not None and legacy.path is not None:
        legacy.path.unlink(missing_ok=True)
        lines.append(f"Removed the old key file {legacy.path}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.reseal",
        description="Seal every saved connection and encrypted setting under COWORK_KEY_FILE, with Cowork stopped.",
    )
    parser.add_argument(
        "--managed-vault", type=Path, help="the managed folder's data-vault, writable; required with COWORK_MANAGED_DIR"
    )
    parser.add_argument(
        "--old-key-file", type=Path, help="the legacy key file, in place of COWORK_MASTER_KEY or MASTER_KEY_PATH's"
    )
    arguments = parser.parse_args(argv)
    return run_operator_command(
        lambda: reseal(managed_vault=arguments.managed_vault, old_key_file=arguments.old_key_file)
    )


if __name__ == "__main__":
    sys.exit(main())
