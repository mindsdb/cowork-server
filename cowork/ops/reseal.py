"""Seal every saved connection and every encrypted setting under the key file, with the server stopped.

    python -m cowork.ops.reseal [--managed-vault PATH] [--old-key-file PATH]

Turning sealing on and rotating its key both run this command, as the server's own user. In order,
it:

1. refuses unless ``COWORK_KEY_FILE`` names a key file the server would start with
   (cowork.common.encryption.sealing_keys), on a local deployment, and refuses when the legacy key
   file is the key file itself, which step 5 would delete;
2. checks every entry in every vault before it writes anything: the base vault, each person's
   partition under the shared root, and the managed folder's (``--managed-vault``, a writable path
   to it). It reads each through descriptors that follow no link
   (cowork.common.record_files.read_record_bytes), whatever ``COWORK_CODE_SANDBOX`` says, and
   stops, writing nothing, at a link, a file with a second name, a temporary name that isn't a
   regular file, a record whose file name isn't its engine and name's, or one that doesn't parse
   as a saved connection (SavedRecord), naming each path. Agent code that ran before the code
   sandbox could have planted any of them, to send these writes into another person's space. It
   also stops at a vault folder it would write in that this user doesn't own or can't write;
3. opens every record, and every sealed setting, channel credential and live-app preview cookie,
   with the key file's keys and the legacy key: ``--old-key-file``, else ``COWORK_MASTER_KEY``,
   else the file at ``MASTER_KEY_PATH``. A record is sealed when it carries the mark
   SealedDataVault.save writes (cowork.services.connectors.sealed_vault.is_sealed_record), and
   plain otherwise. It stops, changing nothing, when a record, setting or credential opens with
   none, naming every one; when a record, sealed, would make a file larger than the server reads
   (cowork.common.record_files.MAX_RECORD_BYTES); and when a setting outside those opens with the
   legacy key or an older key in the key file but not with its first key. A channel plugin that
   failed to import leaves its credentials out of cowork.services.sealing.sealed_setting_keys, and
   such a row would stop opening once step 5 deletes the legacy key, or a rotation drops the older
   key from the file. A preview cookie no key opens is cleared: the preview already reads one as no
   grant (cowork.services.live_app_preview);
4. records that sealing was turned on (cowork.services.sealing.SEALING_MARKER_KEY), then writes
   every record through the sealed vault with its own ``secure_keys``, absent where it was absent,
   and every row under the key file's first key in one transaction, and removes leftover
   temporary files;
5. deletes the legacy key file it opened with, last. A legacy key it read from
   ``COWORK_MASTER_KEY`` it leaves, and it names where to remove it.

It prints counts and paths, never a value. It exits 0 when it sealed everything, 1 when it refused
and changed nothing, and 2 when an error stopped it part-way through steps 4 and 5: some values may
then be sealed under the key file's first key, the legacy key is kept, and the same command with
the same key file finishes the job. Run again, it seals the same values again: each record's
``updated_at`` and every sealed value's ciphertext change, and everything opens as before.
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
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlmodel import Session, select

from cowork.common.encryption import KeyFileError, key_file_path, read_key_file
from cowork.common.paths import PinnedDir, dir_lstat, dir_scandir, open_pinned_child, pinned_dir
from cowork.common.record_files import MAX_RECORD_BYTES, read_record_bytes
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_engine
from cowork.managed import managed_folder
from cowork.models.live_app_preview import LiveAppPreview
from cowork.models.setting import Setting
from cowork.ops import OperatorCommandError, OperatorCommandStopped, run_operator_command
from cowork.ops.connection_inventory import vault_filename
from cowork.services.connectors.persist import own_vault_root
from cowork.services.connectors.sealed_vault import (
    SealedDataVault,
    SealedRecordError,
    is_sealed_record,
    open_fields,
    record_file_bytes,
    seal_fields,
)
from cowork.services.sealing import SEALING_MARKER_KEY, row_name, sealed_setting_keys


class Problem(NamedTuple):
    """An entry that stops the command before it writes anything."""

    path: Path
    reason: str


class SavedRecord(BaseModel):
    """A saved connection's record file, as anton's LocalDataVault writes it: the parts the command
    reads and writes back. anton's save keeps ``created_at`` and stamps ``updated_at`` again."""

    model_config = ConfigDict(strict=True, frozen=True)

    engine: str
    name: str
    created_at: str | None = None
    fields: dict[str, Any]
    secure_keys: list[str] | None = None


class StoredRecord(NamedTuple):
    """A saved connection, read and opened, to write back sealed."""

    vault: Path
    engine: str
    name: str
    fields: dict[str, Any]
    secure_keys: list[str] | None


class VaultContents(NamedTuple):
    """What the check found in one vault folder."""

    records: list[SavedRecord]
    leftovers: list[str]
    problems: list[Problem]


class LegacyKey(NamedTuple):
    """The key the server sealed settings with before the key file, and the file it came from."""

    key: Fernet
    path: Path | None


class RecordsCheck(NamedTuple):
    """What opening every record found."""

    records: list[StoredRecord]  # each record, opened, to write back sealed
    unopened: list[str]  # each record no key opens, by name
    too_large: list[Problem]  # each record whose sealed file would be larger than the server reads


class SettingsCheck(NamedTuple):
    """What opening every settings row found."""

    sealed: list[tuple[Setting, bytes]]  # each sealed row, with its value opened
    unopened: list[str]  # each sealed row no key opens, by name
    stranded: list[str]  # each row outside the sealed set that only a key the run lets go opens, by name


def _check_folder(*, directory: PinnedDir) -> VaultContents:
    """Every entry of one vault folder: the records to seal, the leftover temporary files to
    remove, and each entry that stops the command."""
    records: list[SavedRecord] = []
    leftovers: list[str] = []
    problems: list[Problem] = []
    with dir_scandir(directory) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        path = directory.path / name
        info = dir_lstat(directory, name)
        # anton's save writes each record to its name plus .tmp, then renames that over the record,
        # so anything but a lone regular file at a .tmp name would stop the writes part-way.
        temporary = name.endswith(".tmp")
        if stat.S_ISLNK(info.st_mode):
            problems.append(Problem(path, "is a link"))
            continue
        if stat.S_ISDIR(info.st_mode) and not temporary:
            continue  # anton's vault keeps no folder; it lists files only
        if not stat.S_ISREG(info.st_mode):
            problems.append(Problem(path, "isn't a regular file"))
            continue
        if info.st_nlink != 1:
            problems.append(Problem(path, "has a second name"))
            continue
        if temporary:
            leftovers.append(name)
            continue
        try:
            raw = read_record_bytes(directory=directory, name=name)
        except OSError as error:
            problems.append(Problem(path, f"can't be read ({errno.errorcode.get(error.errno, 'error')})"))
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            problems.append(Problem(path, "isn't valid JSON"))
            continue
        try:
            record = SavedRecord.model_validate(parsed)
        except ValidationError:
            problems.append(Problem(path, "isn't a saved connection"))
            continue
        if vault_filename(record.engine, record.name) != name:
            problems.append(Problem(path, f"holds {record.engine}/{record.name}, whose file name differs"))
            continue
        records.append(record)
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


def _unwritable_folders(*, found: dict[Path, VaultContents]) -> list[Problem]:
    """Each vault folder the command writes in that this user doesn't own or can't write. anton's
    save sets the folder's mode on every write, which only its owner can, so a run as another user
    (root, say) would stop part-way, or leave records the server can't read."""
    problems: list[Problem] = []
    for folder, contents in found.items():
        if not contents.records and not contents.leftovers:
            continue
        owner = os.stat(folder, follow_symlinks=False).st_uid
        if hasattr(os, "geteuid") and owner != os.geteuid():
            problems.append(Problem(folder, f"belongs to user {owner}, not to this command's user {os.geteuid()}"))
        elif not os.access(folder, os.W_OK | os.X_OK):
            problems.append(Problem(folder, "isn't writable by this command's user"))
    return problems


def _open_records(*, found: dict[Path, VaultContents], opening: MultiFernet, sealing: MultiFernet) -> RecordsCheck:
    """Every record, opened; each one no key opens; and each one that, sealed in memory and
    measured as anton's save writes it over the record on disk, is larger than the server reads.
    SealedDataVault.save refuses such a record, which would stop the writes part-way."""
    check = RecordsCheck(records=[], unopened=[], too_large=[])
    for folder, contents in found.items():
        for record in contents.records:
            connection = f"{record.engine}/{record.name}"
            try:
                opened = (
                    open_fields(fields=record.fields, keys=opening, connection=connection)
                    if is_sealed_record(fields=record.fields)
                    else dict(record.fields)
                )
            except SealedRecordError as error:
                check.unopened.append(f"connection {connection} in {folder}: {error.cause}")
                continue
            size = record_file_bytes(
                engine=record.engine,
                name=record.name,
                fields=seal_fields(fields=opened, keys=sealing),
                secure_keys=record.secure_keys,
                prior={"created_at": record.created_at},
            )
            if size > MAX_RECORD_BYTES:
                check.too_large.append(
                    Problem(
                        folder / vault_filename(record.engine, record.name),
                        f"sealed, its file would be {size} bytes, and the server reads at most {MAX_RECORD_BYTES}",
                    )
                )
                continue
            check.records.append(StoredRecord(folder, record.engine, record.name, opened, record.secure_keys))
    return check


def _opens(*, keys: Fernet | MultiFernet, value: str) -> bool:
    try:
        keys.decrypt(value.encode())
    except InvalidToken:
        return False
    return True


def _check_settings(*, session: Session, opening: MultiFernet, first_key: Fernet) -> SettingsCheck:
    """Every sealed row, opened; each one no key opens; and each row outside the sealed set that a
    key the run lets go opens and the key file's first key doesn't: the legacy key, which it
    deletes, or an older key in the key file, which a rotation drops next."""
    sealed_keys = sealed_setting_keys()
    check = SettingsCheck(sealed=[], unopened=[], stranded=[])
    for row in session.exec(select(Setting)).all():
        if row.key in sealed_keys:
            try:
                check.sealed.append((row, opening.decrypt(row.value.encode())))
            except InvalidToken:
                check.unopened.append(row_name(row))
        elif _opens(keys=opening, value=row.value) and not _opens(keys=first_key, value=row.value):
            check.stranded.append(row_name(row))
    return check


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


def _legacy_key_left(*, key_file: Path) -> list[str]:
    """Where the legacy key a run read from COWORK_MASTER_KEY is still kept, for the operator to
    remove: the variable, and a key file at MASTER_KEY_PATH that isn't the key file itself."""
    settings = get_app_settings()
    env_files = [str(Path(name).absolute()) for name in type(settings).model_config.get("env_file") or () if Path(name).is_file()]
    if os.environ.get("COWORK_MASTER_KEY"):
        where = "the environment that starts Cowork"
    elif env_files:
        where = f"the .env file that sets it ({', '.join(env_files)})"
    else:
        where = "wherever it is set"
    lines = [
        f"The old key is still set as COWORK_MASTER_KEY, which the server ignores while COWORK_KEY_FILE is set: remove it from {where}."
    ]
    old_file = Path(settings.master_key_path)
    if old_file.is_file() and not os.path.samefile(old_file, key_file):
        lines.append(f"The old key file {old_file} is still there too: delete it as well.")
    return lines


def _cause(*, error: BaseException) -> str:
    """What stopped the writes, without a value: an OSError's reason and path, else the error's
    type. A database error's text can carry the statement's parameters, so it isn't printed."""
    if isinstance(error, OSError) and error.strerror:
        return f"{error.strerror}: {error.filename}" if error.filename else error.strerror
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    return type(error).__name__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _listed(items: list[str]) -> str:
    return "".join(f"\n  {item}" for item in items)


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
    if legacy is not None and legacy.path is not None and os.path.samefile(legacy.path, path):
        raise OperatorCommandError(
            f"COWORK_KEY_FILE names the legacy key file {path}: copy the key to a file outside cowork-home, "
            "point COWORK_KEY_FILE at the copy, and run this again. Nothing changed."
        )
    opening = MultiFernet([*file_keys, *([legacy.key] if legacy else [])])

    found, problems = _check_vaults(folders=_vault_folders(managed_vault=managed_vault))
    if problems:
        raise OperatorCommandError(
            "These vault entries stop the re-seal, which wrote nothing:"
            + _listed([f"{problem.path}: {problem.reason}" for problem in problems])
            + "\nRemove or move each entry named above, then run this again."
        )
    unwritable = _unwritable_folders(found=found)
    if unwritable:
        raise OperatorCommandError(
            "The re-seal writes in these vault folders, and this user can't, so it wrote nothing:"
            + _listed([f"{problem.path}: {problem.reason}" for problem in unwritable])
            + "\nRun it as the server's own user, with the server stopped."
        )
    record_check = _open_records(found=found, opening=opening, sealing=sealing)
    if record_check.too_large:
        raise OperatorCommandError(
            "These saved connections would be too large for the server to read once sealed, so the re-seal "
            "wrote nothing:"
            + _listed([f"{problem.path}: {problem.reason}" for problem in record_check.too_large])
            + "\nRemove each one, or save it again smaller, with Cowork started as before; then run this again."
        )
    records = record_check.records

    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        settings = _check_settings(session=session, opening=opening, first_key=file_keys[0])
        unopened = record_check.unopened + settings.unopened
        if unopened:
            raise OperatorCommandError(
                "No key opens these, so the re-seal changed nothing:"
                + _listed(unopened)
                + "\nCheck that the key file lists the key they were sealed under, or name the old key with --old-key-file. "
                "A row no key you hold opens can be entered again, or deleted, with Cowork started as before; "
                "then run this again."
            )
        if settings.stranded:
            raise OperatorCommandError(
                "These open with the legacy key or an older key in the key file, not with its first key, and "
                "aren't among the settings the re-seal seals, so they would stop opening once that key goes. "
                "It changed nothing:"
                + _listed(settings.stranded)
                + "\nA channel plugin that failed to import leaves its credentials out: fix the error logged above, "
                "then run this again."
            )
        cookies = session.exec(select(LiveAppPreview).where(LiveAppPreview.viewer_cookie.is_not(None))).all()
        marker = session.exec(
            select(Setting).where(Setting.key == SEALING_MARKER_KEY, Setting.scope.is_(None))
        ).first()

        # From the marker on, an error leaves some values sealed and others not. The legacy key goes
        # last, so the same command with the same key file opens both and finishes the job.
        cleared = removed = 0
        try:
            if marker is None:
                session.add(Setting(key=SEALING_MARKER_KEY, value=_now()))
                session.commit()
            for record in records:
                SealedDataVault(record.vault, keys=sealing).save(
                    record.engine, record.name, record.fields, secure_keys=record.secure_keys
                )
            for row, plaintext in settings.sealed:
                row.value = sealing.encrypt(plaintext).decode()
            for cookie in cookies:
                try:
                    cookie.viewer_cookie = sealing.encrypt(opening.decrypt(cookie.viewer_cookie.encode())).decode()
                except InvalidToken:
                    cookie.viewer_cookie = None
                    cleared += 1
            session.commit()
            for folder, contents in found.items():
                for name in contents.leftovers:
                    (folder / name).unlink(missing_ok=True)
                    removed += 1
            if legacy is not None and legacy.path is not None:
                legacy.path.unlink(missing_ok=True)
        except (Exception, KeyboardInterrupt) as error:
            kept = ", and the legacy key was kept" if legacy is not None else ""
            raise OperatorCommandStopped(
                f"The re-seal stopped part-way through its writes ({_cause(error=error)}). Some values may already be "
                f"sealed under the key file's first key{kept}. Fix the cause, then run the same command again with "
                "the same key file."
            ) from error

    lines = [
        f"Sealed {len(records)} saved connection(s) in {len(found)} vault(s), {len(settings.sealed)} setting(s) and "
        f"channel credential(s), and {len(cookies) - cleared} live-app preview cookie(s) under the key file's first key.",
    ]
    if cleared:
        lines.append(f"Cleared {cleared} preview cookie(s) no key opened; those previews ask to sign in again.")
    if removed:
        lines.append(f"Removed {removed} leftover temporary file(s).")
    if legacy is not None and legacy.path is not None:
        lines.append(f"Removed the old key file {legacy.path}.")
    if legacy is not None and legacy.path is None:
        lines.extend(_legacy_key_left(key_file=path))
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
