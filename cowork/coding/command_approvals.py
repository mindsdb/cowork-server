"""Conservative, task-local matching for remembered command approvals.

This is deliberately not a shell interpreter. Only literal, single POSIX
commands qualify; shell expressions and other shells still require approval.
Never send an execpolicy amendment to Codex: it writes to the shared home.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass


_LITERAL_COMMAND = re.compile(r"[A-Za-z0-9_./:=@%+,\- '\"]+")
_SHELLS = {"bash", "zsh", "/bin/bash", "/bin/zsh", "/usr/bin/bash", "/usr/bin/zsh"}


@dataclass(frozen=True)
class CommandRule:
    context: tuple[str, ...]
    command: tuple[str, ...]
    prefix: tuple[str, ...]

    def _key(self, prefix: tuple[str, ...]) -> str:
        # Rules may contain credentials. Persist only their fingerprints, not
        # raw command arguments, and bind them to their execution context.
        payload = json.dumps(["command-approval-v1", self.context, prefix])
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def grant(self) -> str:
        return self._key(self.prefix)

    def matches(self, grants: list[str]) -> bool:
        known = set(grants)
        return any(self._key(self.command[:size]) in known for size in range(1, len(self.command) + 1))


def command_rule(method: str, params: dict) -> CommandRule | None:
    """Accept an engine-proposed prefix only for an unambiguous literal command.

    A proposal alone is not proof that the whole command matches: the engine
    can suggest a rule for just one segment of a compound shell expression.
    Refuse expressions, redirects, expansions, env assignments and extra
    permission requests instead of trying to reproduce the engine's parser.
    """
    if method != "item/commandExecution/requestApproval":
        return None
    if any(params.get(key) for key in (
        "additionalPermissions", "networkApprovalContext", "proposedNetworkPolicyAmendments", "approvalId",
    )):
        return None
    command, cwd = params.get("command"), params.get("cwd")
    environment = params.get("environmentId", "local")
    prefix = params.get("proposedExecpolicyAmendment", params.get("proposedExecPolicyAmendment"))
    if not isinstance(command, str) or len(command) > 8192 or not isinstance(cwd, str) or not cwd:
        return None
    if environment != "local" or not isinstance(prefix, list) or not 0 < len(prefix) <= 32:
        return None
    if any(not isinstance(word, str) or not word for word in prefix):
        return None
    try:
        outer = shlex.split(command)
        # Codex renders commandExecution requests as the actual shell argv.
        # Do not guess PowerShell/cmd quoting or accept arbitrary shell flags.
        if len(outer) != 3 or outer[0] not in _SHELLS or outer[1] not in {"-c", "-lc"}:
            return None
        if not _LITERAL_COMMAND.fullmatch(outer[2]):
            return None
        words = tuple(shlex.split(outer[2]))
    except ValueError:
        return None
    if not words or len(words) > 128 or "=" in words[0] or words[:len(prefix)] != tuple(prefix):
        return None
    return CommandRule((cwd, environment, *outer[:2]), words, tuple(prefix))
