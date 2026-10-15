"""Sharing one person's saved connection: ``python -m cowork.ops.connections share|unshare``.

``share --name <name> --managed-dir <folder>`` adds the person's saved
connection named ``<name>`` to the shared-connection registry,
``<folder>/shared/connections.json`` (cowork.managed.SharedConnections).
``<folder>`` is the managed folder, writable by the caller; cowork-server
reads the same registry read-only. Everyone else then reads the connection
live from the person's own vault and can't change it, and the person still
changes, reconnects and deletes it as their own (LayeredDataVault).

``unshare --name <name> --managed-dir <folder>`` removes the person's entries
by that name from the registry. The person's connection stays in their vault.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from anton.core.datasources.data_vault import LocalDataVault
from pydantic import BaseModel, ConfigDict

from cowork.db.personal_resources import PersonalResource
from cowork.managed import (
    ManagedFolder,
    SharedConnection,
    SharedConnections,
    SharedConnectionsError,
    managed_folder,
)
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    run_operator_command,
)
from cowork.services.connectors.persist import own_vault_for_scope


class RegistryChange(BaseModel):
    """What share or unshare did to the registry file."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[SharedConnection, ...]
    registry: Path


class _Registry(NamedTuple):
    path: Path
    connections: SharedConnections


def _registry_at(managed_dir: Path) -> _Registry:
    if not managed_dir.is_dir():
        raise OperatorCommandError(f"{managed_dir} is not a folder.")
    path = ManagedFolder.at(managed_dir).shared_connections_file
    try:
        return _Registry(path=path, connections=SharedConnections.read(path))
    except SharedConnectionsError as error:
        # Never rewrite a registry this command can't read: it would drop
        # every entry in it.
        raise OperatorCommandError(f"{error}. Fix or remove it first.") from error


def _write_registry(path: Path, registry: SharedConnections) -> None:
    """Replace the registry file at once, so the server never reads half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".connections-", suffix=".json", delete=False
    ) as handle:
        handle.write(registry.model_dump_json(indent=2) + "\n")
    temporary = Path(handle.name)
    try:
        # Readable by the server's user, whatever the caller's umask.
        temporary.chmod(0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def share_connection(*, person: Person, name: str, managed_dir: Path) -> RegistryChange:
    scope = personal_scope(person)
    if not scope.partitioned_for(PersonalResource.CONNECTIONS):
        raise OperatorCommandError(
            "COWORK_PERSONAL_RESOURCES leaves connections shared, so everyone already "
            "uses every connection."
        )
    registry = _registry_at(managed_dir)
    own = own_vault_for_scope(scope)
    engines = sorted(
        entry["engine"] for entry in own.list_connections() if entry["name"] == name
    )
    if not engines:
        raise OperatorCommandError(f"User {person.user_id} has no saved connection named {name}.")
    if len(engines) > 1:
        raise OperatorCommandError(
            f"User {person.user_id} has {len(engines)} saved connections named {name}, "
            f"for {', '.join(engines)}."
        )
    (engine,) = engines
    managed = managed_folder()
    if managed is not None and LocalDataVault(managed.vault_dir).read_record(engine, name):
        # Everyone reads the operator's own entry first, so the share would
        # reach nobody.
        raise OperatorCommandError(
            f"The managed folder already provides a {engine} connection named {name}."
        )
    for entry in registry.connections.root:
        if (entry.engine, entry.name) != (engine, name):
            continue
        if entry.partition_id == person.partition_id:
            raise OperatorCommandError(
                f"Connection {engine}/{name} of user {person.user_id} is already shared."
            )
        # Everyone else reads the first entry for an engine and name, so a
        # second one would reach nobody.
        raise OperatorCommandError(
            f"User {entry.user_id} already shares a {engine} connection named {name}."
        )
    added = SharedConnection(
        user_id=person.user_id, partition_id=person.partition_id, engine=engine, name=name
    )
    _write_registry(registry.path, SharedConnections((*registry.connections.root, added)))
    return RegistryChange(entries=(added,), registry=registry.path)


def unshare_connection(*, person: Person, name: str, managed_dir: Path) -> RegistryChange:
    registry = _registry_at(managed_dir)
    entries = registry.connections.root
    removed = tuple(
        entry
        for entry in entries
        if entry.partition_id == person.partition_id and entry.name == name
    )
    if not removed:
        raise OperatorCommandError(f"User {person.user_id} shares no connection named {name}.")
    kept = tuple(entry for entry in entries if entry not in removed)
    _write_registry(registry.path, SharedConnections(kept))
    return RegistryChange(entries=removed, registry=registry.path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.connections",
        description="Share one person's saved connection with everyone, or stop sharing it.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    share = actions.add_parser(
        "share", help="share the person's saved connection with everyone, read-only"
    )
    unshare = actions.add_parser(
        "unshare", help="stop sharing it; the person's connection stays theirs"
    )
    for action in (share, unshare):
        add_person_arguments(action)
        action.add_argument("--name", required=True, help="the connection's name")
        action.add_argument(
            "--managed-dir",
            required=True,
            type=Path,
            help="the managed folder, writable: the registry is its shared/connections.json",
        )
    arguments = parser.parse_args(argv)
    person = person_from_arguments(arguments)

    def command() -> str:
        if arguments.action == "share":
            change = share_connection(
                person=person, name=arguments.name, managed_dir=arguments.managed_dir
            )
            return "\n".join(
                f"Shared connection {entry.engine}/{entry.name} of user {person.user_id} "
                f"with everyone, read-only: added it to {change.registry}."
                for entry in change.entries
            )
        change = unshare_connection(
            person=person, name=arguments.name, managed_dir=arguments.managed_dir
        )
        return "\n".join(
            f"Stopped sharing connection {entry.engine}/{entry.name} of user {person.user_id}: "
            f"removed it from {change.registry}. Their own connection stays in their vault."
            for entry in change.entries
        )

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
