"""Read-only operator inventory of stored connections, including inactive stores.

This enumerates storage, not a tester's connection picker. Visibility describes
the configured sharing policy; notes identify managed/name-prefix overrides.
Only metadata crosses the output boundary. Reads are bounded and descriptor
relative, and errors never include record contents or validation inputs. One
person's listing opens no record in another person's partition.
"""

from __future__ import annotations

import os
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from anton.core.datasources.data_vault import LocalDataVault, _slug_env_prefix
from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError

from cowork.common.paths import (
    O_NOFOLLOW,
    O_NONBLOCK,
    PinnedDir,
    dir_open,
    dir_scandir,
    open_pinned_child,
    pinned_dir,
    read_bounded,
)
from cowork.code_sandbox import PlantedLinkError, refuse_planted_links
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import LOCAL_SCOPE
from cowork.managed import MAX_REGISTRY_BYTES, SharedConnection, SharedConnections, managed_folder
from cowork.ops import OperatorCommandError, Person, require_trusted_identity
from cowork.services.connectors.persist import own_vault_root

MAX_RECORD_BYTES = 1 << 20


class RecordMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str = Field(min_length=1)
    name: str = Field(min_length=1)
    creator: UUID | None = Field(default=None, validation_alias=AliasPath("fields", "_created_by"))
    fields: dict[str, object] = Field(exclude=True, repr=False)


class ConnectionName(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str
    name: str

    @property
    def prefix(self) -> str:
        return _slug_env_prefix(self.engine, self.name)


class ConnectionInventoryRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str
    name: str
    origin: Literal["managed", "base", "personal", "registry"]
    user_id: str | None = None
    partition_id: str | None = None
    visibility: Literal["private", "shared", "operator", "inactive", "stale"]
    shared_as: tuple[ConnectionName, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        return _slug_env_prefix(self.engine, self.name)


class ConnectionInventory(BaseModel):
    model_config = ConfigDict(frozen=True)

    personal_connections: bool
    connections: tuple[ConnectionInventoryRow, ...]


def _vault_filename(engine: str, name: str) -> str:
    # Runtime reads resolve registry names through this same filename helper.
    # Different names (sales.db and sales_db) can reach one stored record.
    return LocalDataVault(Path("."))._path_for(engine, name).name


def _read(d: PinnedDir, name: str, *, limit: int) -> bytes:
    fd = dir_open(d, name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OperatorCommandError("Connection inventory refuses linked or non-regular records.")
        if info.st_size > limit:
            raise OperatorCommandError("Connection inventory record exceeds the read limit.")
        return b"".join(read_bounded(fd, info.st_size))
    finally:
        os.close(fd)


def _records(d: PinnedDir) -> list[RecordMetadata]:
    records = []
    with dir_scandir(d) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        # LocalDataVault.save atomically replaces this temporary file; it is
        # never a saved connection's canonical filename.
        if name.endswith(".tmp"):
            continue
        try:
            raw = _read(d, name, limit=MAX_RECORD_BYTES)
        except FileNotFoundError:
            continue  # A concurrent delete; this is an inventory, not a snapshot.
        try:
            record = RecordMetadata.model_validate_json(raw)
        except ValidationError:
            raise OperatorCommandError("Connection inventory found a malformed vault record.") from None
        if _vault_filename(record.engine, record.name) != name:
            raise OperatorCommandError("Connection inventory found a record with a noncanonical filename.")
        records.append(record)
    return records


def _root_records(path: Path, *, required: bool = False) -> list[RecordMetadata]:
    try:
        with pinned_dir(path, nofollow_base=True) as d:
            return _records(d)
    except FileNotFoundError:
        if required:
            raise OperatorCommandError("Connection inventory cannot find the configured managed vault.") from None
        return []


@contextmanager
def _partition_vault(d: PinnedDir, partition: str) -> Iterator[PinnedDir]:
    with ExitStack() as opened:
        child = open_pinned_child(d, partition)
        opened.callback(child.close)
        vault = open_pinned_child(child, "data-vault")
        opened.callback(vault.close)
        yield vault


def _personal_records(root: Path, *, only: str | None = None) -> dict[str, list[RecordMetadata]]:
    """Every partition's records, or only the ``only`` partition's."""
    result = {}
    try:
        with pinned_dir(root, nofollow_base=True) as d:
            with dir_scandir(d) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                if only is not None and name != only:
                    continue
                try:
                    canonical = str(UUID(name))
                except ValueError:
                    continue  # The shared root also contains non-partition stores.
                if canonical != name:
                    raise OperatorCommandError("Connection inventory found a noncanonical partition id.")
                try:
                    with _partition_vault(d, name) as vault:
                        result[name] = _records(vault)
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        pass
    return result


def _shared_filenames(root: Path, partitions: Iterable[str]) -> set[tuple[str, str]]:
    """The record filenames in ``partitions``' vaults, read from their listings alone.

    One person's listing still needs to know which other people's shares
    reach a stored record, for the precedence notes on their own rows, but
    it opens none of those records: a malformed one can't fail it, and no
    other person's fields load. A vault refuse_planted_links refuses (a link,
    or a file with two names, in it or on the way to it) shares nothing, as
    _shared_sources leaves it out at runtime; a missing one shares nothing
    too. Any other read error refuses the command. Registry partition ids are
    canonical UUIDs (SharedEntry), so each is one child name.
    """
    found = set()
    try:
        with pinned_dir(root, nofollow_base=True) as d:
            for partition in sorted(set(partitions)):
                try:
                    refuse_planted_links(root / partition / "data-vault")
                    with _partition_vault(d, partition) as vault, dir_scandir(vault) as entries:
                        found.update(
                            (partition, entry.name)
                            for entry in entries
                            if entry.is_file(follow_symlinks=False) and not entry.name.endswith(".tmp")
                        )
                except (PlantedLinkError, FileNotFoundError):
                    continue
    except FileNotFoundError:
        pass
    return found


def _registry(root: Path) -> SharedConnections:
    try:
        with pinned_dir(root, nofollow_base=True) as d, ExitStack() as opened:
            shared = open_pinned_child(d, "shared")
            opened.callback(shared.close)
            raw = _read(shared, "connections.json", limit=MAX_REGISTRY_BYTES)
    except FileNotFoundError:
        return SharedConnections(())
    try:
        return SharedConnections.model_validate_json(raw)
    except ValidationError:
        raise OperatorCommandError("Connection inventory found a malformed sharing registry.") from None


def connection_inventory(*, person: Person | None = None) -> ConnectionInventory:
    """All stored connections, or one partition plus the managed/base stores.

    No Keycloak lookup is needed for the complete inventory, so spaces whose
    accounts have been deleted remain visible by partition id. A legacy
    record without a creator stamp keeps user_id null instead of guessing.
    With ``person``, only that partition's records are read; other people's
    shares are matched by filename (_shared_filenames).
    """
    require_trusted_identity()
    settings = get_app_settings()
    if settings.tenancy_mode != "local" or os.name != "posix":
        raise OperatorCommandError("Connection inventory requires a POSIX local trusted-identity deployment.")
    personal = PersonalResource.CONNECTIONS in settings.personal_resources
    # With connections personal, no person reads the base vault. Under audit,
    # a request with no identity pair is still the operator view on
    # LOCAL_SCOPE, which reads and changes it, so those records aren't inactive.
    base_visibility = ("operator" if settings.identity_enforce == "audit" else "inactive") if personal else "shared"
    managed = managed_folder()
    shared_root = Path(settings.storage.shared_root)
    try:
        registry = SharedConnections(()) if managed is None else _registry(managed.root)
        managed_records = [] if managed is None else _root_records(managed.vault_dir, required=True)
        base_records = _root_records(own_vault_root(LOCAL_SCOPE))
        if person is None:
            partitions = _personal_records(shared_root)
            stored = set()
        else:
            partitions = _personal_records(shared_root, only=person.partition_id)
            others = {entry.partition_id for entry in registry.root} - {person.partition_id}
            stored = _shared_filenames(shared_root, others)
    except OSError:
        raise OperatorCommandError(
            "Connection inventory cannot read a store: check permissions and remove linked or non-directory paths."
        ) from None

    rows: list[ConnectionInventoryRow] = []
    for origin, records in (("managed", managed_records), ("base", base_records)):
        for record in records:
            rows.append(ConnectionInventoryRow(
                engine=record.engine, name=record.name, origin=origin,
                user_id=None if record.creator is None else str(record.creator),
                visibility=base_visibility if origin == "base" else "shared",
            ))
    shares: dict[tuple[str, str], list[SharedConnection]] = defaultdict(list)
    for entry in registry.root:
        shares[entry.partition_id, _vault_filename(entry.engine, entry.name)].append(entry)
    for partition, records in partitions.items():
        for record in records:
            key = (partition, _vault_filename(record.engine, record.name))
            stored.add(key)
            registered = shares.get(key, [])
            creator = str(record.creator) if record.creator is not None else None
            notes = ()
            if registered:
                if creator is not None and any(creator != entry.user_id for entry in registered):
                    notes = ("registry-owner-differs-from-creator",)
                creator = creator or registered[0].user_id
                if any((entry.engine, entry.name) != (record.engine, record.name) for entry in registered):
                    notes += ("registry-name-alias",)
                if not personal:
                    notes += ("registered-share-inactive",)
            rows.append(ConnectionInventoryRow(
                engine=record.engine, name=record.name, origin="personal",
                user_id=creator, partition_id=partition,
                visibility=("shared" if registered else "private") if personal else "inactive",
                shared_as=tuple(ConnectionName(engine=entry.engine, name=entry.name) for entry in registered),
                notes=notes,
            ))
    for entry in registry.root:
        if (entry.partition_id, _vault_filename(entry.engine, entry.name)) not in stored:
            rows.append(ConnectionInventoryRow(
                engine=entry.engine, name=entry.name, origin="registry",
                user_id=entry.user_id, partition_id=entry.partition_id,
                visibility="stale" if personal else "inactive",
            ))

    managed_prefixes = {_slug_env_prefix(record.engine, record.name) for record in managed_records}
    counts = Counter((row.origin, row.partition_id, row.prefix) for row in rows if row.origin != "registry")
    first_share = {}
    for entry in registry.root:
        if (entry.partition_id, _vault_filename(entry.engine, entry.name)) in stored:
            first_share.setdefault((entry.engine, entry.name), entry.partition_id)
    shared_prefixes = Counter(_slug_env_prefix(engine, name) for engine, name in first_share)
    result = []
    for row in rows:
        notes = list(row.notes)
        if row.visibility in {"private", "shared", "operator"}:
            if row.origin != "managed" and row.prefix in managed_prefixes:
                notes.append("hidden-by-managed-prefix")
            if counts[row.origin, row.partition_id, row.prefix] > 1:
                notes.append("same-vault-prefix-collision")
            if row.origin == "personal" and row.visibility == "shared":
                notes.append("recipient-own-prefix-takes-priority")
                if any(alias.prefix in managed_prefixes and alias.prefix != row.prefix for alias in row.shared_as):
                    notes.append("shared-alias-hidden-by-managed-prefix")
                if any(shared_prefixes[alias.prefix] > 1 for alias in row.shared_as):
                    notes.append("shared-prefix-collision-for-other-recipients")
                if any(first_share[alias.engine, alias.name] != row.partition_id for alias in row.shared_as):
                    notes.append("earlier-share-takes-priority-for-other-recipients")
        if person is None or row.partition_id in {None, person.partition_id}:
            result.append(row.model_copy(update={"notes": tuple(notes)}))
    return ConnectionInventory(personal_connections=personal, connections=tuple(result))
