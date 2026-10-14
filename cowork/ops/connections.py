"""One person's saved connections: ``python -m cowork.ops.connections share``.

``share --name <name> --managed-dir <folder>`` copies the person's saved
connection named ``<name>`` into ``<folder>``, in the vault's own file format.
``<folder>`` is a copy of the managed ``data-vault/`` folder the caller can
write, which the caller then places in the managed folder itself, read-only
for everyone. The person's own entry stays in their vault, hidden behind the
managed one, which every vault read takes first (LayeredDataVault), and the
person can no longer change or delete it under that name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault
from pydantic import BaseModel, ConfigDict

from cowork.managed import managed_folder
from cowork.ops import (
    OperatorCommandError,
    Person,
    add_person_arguments,
    person_from_arguments,
    personal_scope,
    run_operator_command,
)
from cowork.services.connectors.persist import own_vault_for_scope


class SharedConnection(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str
    name: str
    destination: Path


def share_connection(*, person: Person, name: str, managed_dir: Path) -> SharedConnection:
    if not managed_dir.is_dir():
        raise OperatorCommandError(f"{managed_dir} is not a folder.")
    # The person's own entries only: a managed entry is never copied again.
    own = own_vault_for_scope(personal_scope(person))
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
        raise OperatorCommandError(
            f"The managed folder already provides a {engine} connection named {name}."
        )
    destination = LocalDataVault(managed_dir)
    if destination.read_record(engine, name) is not None:
        raise OperatorCommandError(f"{managed_dir} already holds a {engine} connection named {name}.")
    record = own.read_record(engine, name) or {}
    path = destination.save(
        engine, name, record.get("fields") or {}, secure_keys=record.get("secure_keys")
    )
    return SharedConnection(engine=engine, name=name, destination=Path(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.connections",
        description="Share one person's saved connection with everyone.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    share = actions.add_parser(
        "share", help="copy the person's saved connection into the managed connections"
    )
    add_person_arguments(share)
    share.add_argument("--name", required=True, help="the connection's name")
    share.add_argument(
        "--managed-dir",
        required=True,
        type=Path,
        help="the writable copy of the managed data-vault folder the connection is copied into",
    )
    arguments = parser.parse_args(argv)
    person = person_from_arguments(arguments)

    def command() -> str:
        shared = share_connection(
            person=person, name=arguments.name, managed_dir=arguments.managed_dir
        )
        return (
            f"Copied connection {shared.engine}/{shared.name} of user {person.user_id} "
            f"into {shared.destination}."
        )

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
