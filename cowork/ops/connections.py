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
import sys
from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.db.personal_resources import PersonalResource
from cowork.managed import SharedConnection, SharedConnections, managed_folder
from cowork.ops import (
    OperatorCommandError,
    Person,
    RegistryChange,
    add_person_arguments,
    managed_folder_at,
    person_from_arguments,
    personal_scope,
    read_registry,
    run_operator_command,
    write_registry,
)
from cowork.services.connectors.persist import own_vault_for_scope


def share_connection(
    *, person: Person, name: str, managed_dir: Path
) -> RegistryChange[SharedConnection]:
    scope = personal_scope(person)
    if not scope.partitioned_for(PersonalResource.CONNECTIONS):
        raise OperatorCommandError(
            "COWORK_PERSONAL_RESOURCES leaves connections shared, so everyone already "
            "uses every connection."
        )
    path = managed_folder_at(managed_dir).shared_connections_file
    registry = read_registry(registry_type=SharedConnections, path=path)
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
    for entry in registry.root:
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
    write_registry(path=path, registry=SharedConnections((*registry.root, added)))
    return RegistryChange(entries=(added,), registry=path)


def unshare_connection(
    *, person: Person, name: str, managed_dir: Path
) -> RegistryChange[SharedConnection]:
    path = managed_folder_at(managed_dir).shared_connections_file
    entries = read_registry(registry_type=SharedConnections, path=path).root
    removed = tuple(
        entry
        for entry in entries
        if entry.partition_id == person.partition_id and entry.name == name
    )
    if not removed:
        raise OperatorCommandError(f"User {person.user_id} shares no connection named {name}.")
    kept = tuple(entry for entry in entries if entry not in removed)
    write_registry(path=path, registry=SharedConnections(kept))
    return RegistryChange(entries=removed, registry=path)


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
