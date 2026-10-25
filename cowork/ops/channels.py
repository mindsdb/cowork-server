"""A local deployment's channels: ``python -m cowork.ops.channels set|delete|agent``.

A channel's config is the deployment's one global row, and every message to
the channel's bot runs through it. On a deployment that trusts identity
headers, every person's request to change it answers 403
(cowork.api.v1.permissions.AuthenticatedOrgAdmin), and under
COWORK_IDENTITY_ENFORCE=enforce no request is the operator's either. So the
operator sets a channel here, from a shell where cowork-server runs. Every
action acts on the deployment's one installation and takes no person.

``set <channel>`` reads the channel's credential fields as ``name=value``
lines on stdin, so no secret reaches the command line or the shell's history,
and saves them as the channel's config route does
(ChannelConfigService.set_config). ``delete <channel>`` removes the channel's
credentials and installation. ``agent <harness>`` sets the harness that serves
channel conversations and detaches existing chats, so their next message
starts fresh (cowork.services.channels.set_channels_harness).

cowork-server builds each channel's live adapter and starts its ingress when it
starts (cowork.server._start_channels), so a change applies at its next
restart. Binding changes apply to the next message without a restart.

``bindings list`` shows which groups need setup. On a trusted-identity local
deployment, every binding from before operator configuration was recorded
needs ``bindings set`` once, including rows with no recorded owner. Until
then the group receives a fixed setup-required reply and no agent turn runs.
``bindings set`` reads the full configuration as JSON on stdin; omitted
settings reset to defaults. It replaces stale routing, preserving the former
project and conversation. ``bindings delete <id>`` removes only the routing.
Results print on stdout and never hold a secret.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import TextIO
from uuid import UUID

from pydantic import ValidationError
from sqlmodel import Session

from cowork.channels.registry import load_first_party_plugins
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.channel import ChannelBinding
from cowork.ops import OperatorCommandError, run_operator_command
from cowork.schemas.channels import BindingCreateRequest
from cowork.services.channel_bindings import BindingConflictError, ChannelBindingService
from cowork.services.channels import ChannelConfigService, UnknownChannelError, set_channels_harness

APPLIES_AT_RESTART = "It applies when cowork-server next starts."


def _deployment_wide(action: Callable[[Session, ScopedSession], str]) -> str:
    """Run ``action`` on the deployment's one channel installation."""
    if get_app_settings().tenancy_mode == "org":
        raise OperatorCommandError(
            "This command sets a local deployment's channels; in org tenancy each "
            "org's admin sets its own."
        )
    load_first_party_plugins()
    session = get_open_session()
    try:
        return action(session, ScopedSession(session, LOCAL_SCOPE))
    finally:
        session.close()


def _credential_lines(stdin: TextIO) -> dict[str, str]:
    if stdin.isatty():
        raise OperatorCommandError("Pipe the channel's fields as name=value lines on stdin.")
    values: dict[str, str] = {}
    for number, line in enumerate(stdin.read().splitlines(), start=1):
        if not line.strip():
            continue
        name, separator, value = line.partition("=")
        if not separator or not name.strip():
            raise OperatorCommandError(f"Line {number} of stdin is not name=value.")
        if not value.strip():
            raise OperatorCommandError(f"Line {number} of stdin gives {name.strip()} no value.")
        values[name.strip()] = value.strip()
    if not values:
        raise OperatorCommandError("Give the channel's fields as name=value lines on stdin.")
    return values


def set_channel(channel: str, values: dict[str, str]) -> str:
    def action(_session: Session, scoped: ScopedSession) -> str:
        try:
            saved = ChannelConfigService(scoped).set_config(channel, values)
        except UnknownChannelError:
            raise OperatorCommandError(f"No channel is named {channel}.") from None
        except ValueError as error:
            raise OperatorCommandError(str(error)) from None
        state = "configured" if saved.configured else "not configured yet: a required field is unset"
        return f"Saved {', '.join(sorted(values))} for {channel}, which is {state}. {APPLIES_AT_RESTART}"

    return _deployment_wide(action)


def delete_channel(channel: str) -> str:
    def action(_session: Session, scoped: ScopedSession) -> str:
        try:
            deleted = ChannelConfigService(scoped).delete_config(channel)
        except UnknownChannelError:
            raise OperatorCommandError(f"No channel is named {channel}.") from None
        if not deleted:
            raise OperatorCommandError(f"{channel} has no stored config.")
        return f"Deleted {channel}'s credentials. {APPLIES_AT_RESTART}"

    return _deployment_wide(action)


def set_agent(harness: str) -> str:
    def action(session: Session, scoped: ScopedSession) -> str:
        try:
            changed = set_channels_harness(session, scoped, harness)
        except ValueError as error:
            raise OperatorCommandError(str(error)) from None
        return (
            f"Channel conversations use {changed.harness}; "
            f"{changed.reset_conversations} chat(s) start fresh on their next message."
        )

    return _deployment_wide(action)


def set_binding(stdin: TextIO) -> str:
    if stdin.isatty():
        raise OperatorCommandError("Pipe the binding configuration as JSON on stdin.")
    try:
        request = BindingCreateRequest.model_validate_json(stdin.read())
    except ValidationError:
        raise OperatorCommandError("Invalid binding JSON; see bindings set --help for the fields.") from None

    def action(_session: Session, scoped: ScopedSession) -> str:
        try:
            binding = ChannelBindingService(scoped).configure(request)
        except (ValueError, BindingConflictError) as error:
            raise OperatorCommandError(str(error)) from None
        return f"Configured binding {binding.id}. It applies to the group's next message."

    return _deployment_wide(action)


def list_bindings() -> str:
    def action(_session: Session, scoped: ScopedSession) -> str:
        service = ChannelBindingService(scoped)
        rows = scoped.exec(scoped.select(ChannelBinding)).all()
        lines = []
        for row in rows:
            state = "configured" if service.operator_ready(row) else "needs operator setup"
            lines.append(f"{row.id} {row.channel_type} {row.external_group_id} {row.external_thread_key}: {state}")
        return "\n".join(lines) or "No channel bindings."

    return _deployment_wide(action)


def delete_binding(binding_id: UUID) -> str:
    def action(_session: Session, scoped: ScopedSession) -> str:
        if not ChannelBindingService(scoped).delete(binding_id):
            raise OperatorCommandError(f"No binding is named {binding_id}.")
        return f"Deleted binding {binding_id}. Its project and conversation remain."

    return _deployment_wide(action)


def main(argv: list[str] | None = None, *, stdin: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cowork.ops.channels",
        description="Manage local channels (restart to apply) and bindings (apply to the next message).",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "set", help="save a channel's fields, read as name=value lines on stdin"
    ).add_argument("channel")
    actions.add_parser("delete", help="delete a channel's credentials").add_argument("channel")
    actions.add_parser("agent", help="set the harness that serves channel conversations").add_argument(
        "harness"
    )
    bindings = actions.add_parser("bindings", help="list, configure or remove group routing")
    binding_actions = bindings.add_subparsers(dest="binding_action", required=True)
    binding_actions.add_parser("list", help="list bindings and whether operator setup is needed")
    binding_actions.add_parser(
        "set", help="replace a binding from JSON on stdin; applies to the next message",
        description=("JSON fields: channel_type, external_group_id, external_thread_id (optional), "
                     "anton_project_id and anton_conversation_id (optional deployment-owned UUIDs), "
                     "display_name, trigger_rule (always, mention_only or regex), trigger_pattern, instructions. "
                     "Omitted settings reset to defaults. Legacy bindings require this one-time setup; "
                     "their old projects and conversations are preserved."),
    )
    binding_actions.add_parser("delete", help="remove routing; preserve project and conversation").add_argument("binding_id", type=UUID)
    arguments = parser.parse_args(argv)

    def command() -> str:
        if arguments.action == "bindings":
            if arguments.binding_action == "set":
                return set_binding(stdin or sys.stdin)
            if arguments.binding_action == "list":
                return list_bindings()
            return delete_binding(arguments.binding_id)
        if arguments.action == "set":
            return set_channel(arguments.channel, _credential_lines(stdin or sys.stdin))
        if arguments.action == "delete":
            return delete_channel(arguments.channel)
        return set_agent(arguments.harness)

    return run_operator_command(command)


if __name__ == "__main__":
    sys.exit(main())
