"""Operator commands, run from a shell on the deployment's host.

Each module is one command, ``python -m cowork.ops.<module> <action>``, and
acts for one person on a local deployment that trusts identity headers
(COWORK_TRUST_IDENTITY_HEADERS). Every action takes the pair that deployment's
proxy sends for the person: ``--organization-id``, the deployment's own id
(X-Organization-Id), and ``--user-id``, the person's ``sub`` (X-User-Id).
The one exception is memory left shared, where the memory actions act on the
deployment's one store and take neither (cowork.ops.memory). cowork-server
works out the person's space from the pair, as it does for their requests, so
a caller never repeats the storage layout.

Results go to stdout as plain text, and errors and log lines to stderr. A
command that can't do what it was asked prints why and exits with status 1.
A connection share or unshare given ``--registry-stdout`` writes nothing: it
prints the new registry on stdout, exactly the bytes it would have written,
and its message on stderr, so a caller can check the registry and write it
itself. A skill share or unshare has no such flag: Cowork writes the
shared-skill list itself (cowork.shared_skills), so the command runs as
Cowork's own user and changes the list as the person's Skills page does.

The helpers below are shared by every module: the two person arguments, the
Person they make, the scope that person's own requests get, and the reading
and writing of the managed folder's shared-connection registry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

# A command runs in a fresh interpreter that imports only the models it queries,
# but a model's relationships name others by string, which SQLAlchemy resolves
# on the first query. So every model module loads here first, as
# cowork/db/alembic/env.py loads them for a migration.
import cowork.models.artifact_identity  # noqa: F401
import cowork.models.code_control  # noqa: F401
import cowork.models.conversation  # noqa: F401
import cowork.models.file  # noqa: F401
import cowork.models.message  # noqa: F401
import cowork.models.message_event  # noqa: F401
import cowork.models.pin  # noqa: F401
import cowork.models.project  # noqa: F401
import cowork.models.schedule  # noqa: F401
import cowork.models.setting  # noqa: F401
import cowork.models.shared_resource  # noqa: F401
import cowork.models.skill  # noqa: F401
import cowork.models.task_object  # noqa: F401
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope, scope_from_principal
from cowork.managed import ManagedFolder, SharedEntry, SharedRegistry, SharedRegistryError
from cowork.principal import Principal, personal_partition_id


class OperatorCommandError(Exception):
    """A command that can't do what it was asked. run_operator_command prints
    the message on stderr and exits with status 1."""


class Person(BaseModel):
    """One person on a trusted-identity deployment: the pair its proxy sends,
    each id canonical (lowercase, hyphenated)."""

    model_config = ConfigDict(frozen=True)

    organization_id: str
    user_id: str

    @field_validator("organization_id", "user_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))

    @property
    def partition_id(self) -> str:
        """The person's space: cowork.principal.personal_partition_id."""
        return personal_partition_id(self.organization_id, self.user_id)


def _uuid_argument(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a UUID") from None


def add_person_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the two person arguments to one action's parser. A command whose
    need for a person depends on the deployment passes ``required=False`` and
    checks them itself."""
    parser.add_argument(
        "--organization-id",
        required=required,
        type=_uuid_argument,
        help="the deployment's id, the X-Organization-Id its proxy sends",
    )
    parser.add_argument(
        "--user-id",
        required=required,
        type=_uuid_argument,
        help="the person's id, their sub, the X-User-Id the proxy sends",
    )


def person_from_arguments(arguments: argparse.Namespace) -> Person:
    return Person(organization_id=arguments.organization_id, user_id=arguments.user_id)


def require_trusted_identity() -> None:
    """Refuse a deployment that does not trust identity headers, which keeps
    no space per person for a command to act on."""
    if not get_app_settings().trust_identity_headers:
        raise OperatorCommandError(
            "COWORK_TRUST_IDENTITY_HEADERS is off, so this deployment keeps no "
            "space per person."
        )


def personal_scope(person: Person) -> TenantScope:
    """The scope the person's own requests get.

    Built as a request's is: TrustedHeaderMiddleware's principal, whose org id
    is the partition id, through scope_from_principal. So it keeps the
    resources COWORK_PERSONAL_RESOURCES lists in the person's space and the
    others shared, exactly as their requests do.
    """
    require_trusted_identity()
    return scope_from_principal(
        Principal(user_id=person.user_id, org_id=person.partition_id)
    )


def _logs_to_stderr() -> None:
    """Point cowork's console log handler (cowork.common.logger), which writes
    to stdout, at stderr, so stdout holds only the command's own output."""
    for handler in logging.getLogger().handlers:
        if (
            type(handler) is logging.StreamHandler
            and handler.stream in (sys.stdout, sys.__stdout__)
        ):
            handler.setStream(sys.__stderr__)


def run_operator_command(command: Callable[[], str]) -> int:
    """Run one command: print what it returns on stdout and return 0, or
    print an OperatorCommandError on stderr and return 1."""
    _logs_to_stderr()
    try:
        output = command()
    except OperatorCommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output if output.endswith("\n") else output + "\n")
    return 0


# ── the managed folder's shared-connection registry ─────────────────────────

EntryT = TypeVar("EntryT", bound=SharedEntry)
RegistryT = TypeVar("RegistryT", bound=SharedRegistry)


class RegistryChange(BaseModel, Generic[EntryT]):
    """What share or unshare does to a registry file: the entries it adds or
    removes, and the registry it leaves. finish_registry_change writes it."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[EntryT, ...]
    registry: Path
    updated: SharedRegistry


def managed_folder_at(managed_dir: Path) -> ManagedFolder:
    """The managed folder at ``managed_dir``, the writable path the caller
    passes; its registries sit in its ``shared/``."""
    if not managed_dir.is_dir():
        raise OperatorCommandError(f"{managed_dir} is not a folder.")
    return ManagedFolder.at(managed_dir)


def read_registry(*, registry_type: type[RegistryT], path: Path) -> RegistryT:
    """The registry at ``path``; none when the file is missing."""
    try:
        return registry_type.read(path)
    except SharedRegistryError as error:
        # Never rewrite a registry this command can't read: it would drop
        # every entry in it.
        raise OperatorCommandError(f"{error}. Fix or remove it first.") from error


# Where the new registry goes with --registry-stdout, for the messages.
REGISTRY_ON_STDOUT = "the registry on stdout"


def add_registry_stdout_argument(parser: argparse.ArgumentParser) -> None:
    """Add --registry-stdout to one connection share or unshare parser."""
    parser.add_argument(
        "--registry-stdout",
        action="store_true",
        help=(
            "write nothing: print the new registry on stdout, exactly the bytes the "
            "command would write, and the message on stderr, for a caller that "
            "checks and writes the registry itself"
        ),
    )


def finish_registry_change(
    change: RegistryChange, *, message: str, registry_stdout: bool
) -> str:
    """Write the change's registry and return ``message`` for stdout. With
    ``registry_stdout``, write nothing: ``message`` goes to stderr, and the
    new registry's text (SharedRegistry.dump_json) is returned for stdout
    instead."""
    if registry_stdout:
        print(message, file=sys.stderr)
        return change.updated.dump_json()
    change.updated.write(change.registry)
    return message
