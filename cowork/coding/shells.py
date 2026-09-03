from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from cowork.coding.contracts import (
    TerminalShellInventory,
    TerminalShellOption,
    TerminalShellPreference,
)


def shell_inventory(
    *,
    platform_name: str | None = None,
    commands: dict[TerminalShellPreference, str] | None = None,
    system_shell: str | None = None,
) -> TerminalShellInventory:
    """Describe the interactive shells available on this device."""
    windows = _is_windows(platform_name)
    available = commands if commands is not None else _available_commands(windows=windows)
    automatic = resolve_shell(
        TerminalShellPreference.auto,
        commands=available,
        windows=windows,
        system_shell=system_shell,
    )
    automatic_name = _display_name(automatic[0])
    items = [
        TerminalShellOption(
            id=TerminalShellPreference.auto,
            label=f"Automatic — {automatic_name}",
        ),
    ]

    if windows:
        candidates = (
            (TerminalShellPreference.bash, "Git Bash"),
            (TerminalShellPreference.pwsh, "PowerShell 7"),
            (TerminalShellPreference.powershell, "Windows PowerShell"),
            (TerminalShellPreference.cmd, "Command Prompt"),
        )
    else:
        candidates = (
            (TerminalShellPreference.bash, "Bash"),
            (TerminalShellPreference.zsh, "zsh"),
            (TerminalShellPreference.fish, "fish"),
        )

    for preference, label in candidates:
        if preference in available:
            items.append(TerminalShellOption(id=preference, label=label))

    system = _system_shell(available, windows=windows, configured=system_shell)
    if system:
        items.append(TerminalShellOption(
            id=TerminalShellPreference.system,
            label=f"System default — {_display_name(system)}",
        ))

    return TerminalShellInventory(
        platform=platform_name or sys.platform,
        resolved=_preference_for_command(automatic[0], available, windows=windows),
        items=items,
    )


def resolve_shell(
    preference: TerminalShellPreference | str = TerminalShellPreference.auto,
    *,
    commands: dict[TerminalShellPreference, str] | None = None,
    windows: bool | None = None,
    system_shell: str | None = None,
) -> list[str]:
    """Resolve a stored preference, safely falling back if software changed."""
    try:
        selected = TerminalShellPreference(preference)
    except ValueError:
        selected = TerminalShellPreference.auto
    on_windows = os.name == "nt" if windows is None else windows
    available = commands if commands is not None else _available_commands(windows=on_windows)

    if selected == TerminalShellPreference.system:
        executable = _system_shell(available, windows=on_windows, configured=system_shell)
    elif selected == TerminalShellPreference.auto:
        order = (
            TerminalShellPreference.bash,
            TerminalShellPreference.pwsh,
            TerminalShellPreference.powershell,
            TerminalShellPreference.cmd,
        ) if on_windows else (
            TerminalShellPreference.bash,
            TerminalShellPreference.zsh,
            TerminalShellPreference.fish,
        )
        executable = next((available[item] for item in order if item in available), None)
        executable = executable or _system_shell(available, windows=on_windows, configured=system_shell)
    else:
        executable = available.get(selected)
        if executable is None:
            return resolve_shell(
                TerminalShellPreference.auto,
                commands=available,
                windows=on_windows,
                system_shell=system_shell,
            )

    if not executable:
        executable = os.environ.get("COMSPEC", "cmd.exe") if on_windows else "/bin/sh"
    executable_name = Path(executable.replace("\\", "/")).name.casefold()
    if on_windows and executable_name in {
        "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    }:
        return [executable]
    if executable_name in {"bash", "bash.exe", "zsh", "zsh.exe", "fish", "fish.exe"}:
        return [executable, "--login"]
    # POSIX only standardises login-shell behaviour through argv[0]. Shells
    # such as dash (commonly /bin/sh) reject the GNU-style --login flag, so a
    # portable fallback must start without shell-specific arguments.
    return [executable]


def shell_environment(command: list[str], working_directory: Path | None = None) -> dict[str, str]:
    # Development builds are commonly launched by npm. Its process-scoped
    # prefix belongs to the app's bundled Node runtime (Hermes in Cowork), not
    # to the user's interactive shell. Leaving it set makes NVM reject zsh
    # startup and exposes an internal implementation detail in the terminal.
    environment: dict[str, str] = {
        "npm_config_prefix": "",
        "PREFIX": "",
    }
    executable = Path(command[0]).name.casefold()
    if executable in {"bash", "bash.exe"}:
        environment["BASH_SILENCE_DEPRECATION_WARNING"] = "1"
    if working_directory is not None:
        # Bash and zsh preserve a logical PWD when it points to the same inode
        # as the process cwd. That lets the normal user prompt show our human
        # workspace alias without replacing the user's PS1/PROMPT.
        environment["PWD"] = str(working_directory)
    # The sidecar is launched by the desktop app, whose environment carries no
    # terminal type, so the PTY would otherwise start without one.
    if not os.environ.get("TERM"):
        environment["TERM"] = "xterm-256color"
    if not os.environ.get("COLORTERM"):
        environment["COLORTERM"] = "truecolor"
    return environment


def _available_commands(*, windows: bool | None = None) -> dict[TerminalShellPreference, str]:
    on_windows = os.name == "nt" if windows is None else windows
    available: dict[TerminalShellPreference, str] = {}
    candidates = {
        TerminalShellPreference.bash: ("bash", "/bin/bash"),
        TerminalShellPreference.zsh: ("zsh", "/bin/zsh"),
        TerminalShellPreference.fish: ("fish",),
        TerminalShellPreference.pwsh: ("pwsh",),
        TerminalShellPreference.powershell: ("powershell",),
        TerminalShellPreference.cmd: (os.environ.get("COMSPEC", "cmd.exe"),),
    }
    for preference, names in candidates.items():
        executable = next(
            (resolved for name in names if name and (resolved := _resolve_executable(name))),
            None,
        )
        if executable and (
            preference != TerminalShellPreference.bash
            or not on_windows
            or _windows_bash_is_compatible(executable)
        ):
            available[preference] = executable
    return available


def _system_shell(
    commands: dict[TerminalShellPreference, str],
    *,
    windows: bool | None = None,
    configured: str | None = None,
) -> str | None:
    on_windows = os.name == "nt" if windows is None else windows
    if on_windows:
        return (
            commands.get(TerminalShellPreference.pwsh)
            or commands.get(TerminalShellPreference.powershell)
            or commands.get(TerminalShellPreference.cmd)
        )
    selected = configured or os.environ.get("SHELL")
    if selected:
        resolved = _resolve_executable(selected)
        if resolved:
            return resolved
        normalized = _normalized_command(selected, windows=False)
        matching = next(
            (
                command
                for command in commands.values()
                if _normalized_command(command, windows=False) == normalized
            ),
            None,
        )
        if matching:
            return matching
    return _resolve_executable("/bin/sh")


def _preference_for_command(
    executable: str,
    commands: dict[TerminalShellPreference, str],
    *,
    windows: bool = False,
) -> TerminalShellPreference:
    resolved = _normalized_command(executable, windows=windows)
    return next(
        (
            preference
            for preference, command in commands.items()
            if _normalized_command(command, windows=windows) == resolved
        ),
        TerminalShellPreference.system,
    )


def _display_name(executable: str) -> str:
    name = Path(executable.replace("\\", "/")).stem.casefold()
    return {
        "bash": "Bash",
        "zsh": "zsh",
        "fish": "fish",
        "pwsh": "PowerShell 7",
        "powershell": "Windows PowerShell",
        "cmd": "Command Prompt",
    }.get(name, Path(executable).stem)


def _resolve_executable(candidate: str) -> str | None:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return str(path) if path.is_file() else None
    return shutil.which(candidate)


def _windows_bash_is_compatible(executable: str) -> bool:
    """Reject the legacy WSL launcher, whose paths do not match task workspaces."""
    normalized = executable.replace("\\", "/").casefold()
    return not (
        normalized.endswith("/windows/system32/bash.exe")
        or "/windowsapps/bash.exe" in normalized
    )


def _is_windows(platform_name: str | None) -> bool:
    return os.name == "nt" if platform_name is None else platform_name.casefold().startswith("win")


def _normalized_command(executable: str, *, windows: bool) -> str:
    if windows:
        return executable.replace("\\", "/").casefold()
    return str(Path(executable).resolve())
