from __future__ import annotations

from cowork.coding import shells
from cowork.coding.contracts import TerminalShellPreference


def test_explicit_unavailable_shell_falls_back_to_automatic() -> None:
    commands = {TerminalShellPreference.bash: "/bin/bash"}

    assert shells.resolve_shell(TerminalShellPreference.fish, commands=commands) == [
        "/bin/bash",
        "--login",
    ]


def test_posix_fallback_does_not_assume_a_gnu_login_flag() -> None:
    assert shells.resolve_shell(
        TerminalShellPreference.system,
        commands={},
        windows=False,
        system_shell="/bin/sh",
    ) == ["/bin/sh"]


def test_shell_inventory_starts_with_an_explained_automatic_choice() -> None:
    inventory = shells.shell_inventory()

    assert inventory.items
    assert inventory.items[0].id == TerminalShellPreference.auto
    assert inventory.items[0].label.startswith("Automatic — ")
    assert len({item.id for item in inventory.items}) == len(inventory.items)


def test_windows_inventory_prefers_compatible_git_bash_and_lists_native_choices() -> None:
    commands = {
        TerminalShellPreference.bash: r"C:\Program Files\Git\bin\bash.exe",
        TerminalShellPreference.pwsh: r"C:\Program Files\PowerShell\7\pwsh.exe",
        TerminalShellPreference.powershell: r"C:\Windows\System32\WindowsPowerShell\powershell.exe",
        TerminalShellPreference.cmd: r"C:\Windows\System32\cmd.exe",
    }

    inventory = shells.shell_inventory(platform_name="win32", commands=commands)

    assert inventory.resolved == TerminalShellPreference.bash
    assert [(item.id, item.label) for item in inventory.items] == [
        (TerminalShellPreference.auto, "Automatic — Bash"),
        (TerminalShellPreference.bash, "Git Bash"),
        (TerminalShellPreference.pwsh, "PowerShell 7"),
        (TerminalShellPreference.powershell, "Windows PowerShell"),
        (TerminalShellPreference.cmd, "Command Prompt"),
        (TerminalShellPreference.system, "System default — PowerShell 7"),
    ]
    assert shells.resolve_shell(
        TerminalShellPreference.pwsh,
        commands=commands,
        windows=True,
    ) == [r"C:\Program Files\PowerShell\7\pwsh.exe"]


def test_linux_inventory_only_lists_installed_shells() -> None:
    commands = {
        TerminalShellPreference.bash: "/bin/bash",
        TerminalShellPreference.fish: "/usr/bin/fish",
    }

    inventory = shells.shell_inventory(
        platform_name="linux",
        commands=commands,
        system_shell="/usr/bin/fish",
    )

    assert [(item.id, item.label) for item in inventory.items] == [
        (TerminalShellPreference.auto, "Automatic — Bash"),
        (TerminalShellPreference.bash, "Bash"),
        (TerminalShellPreference.fish, "fish"),
        (TerminalShellPreference.system, "System default — fish"),
    ]
