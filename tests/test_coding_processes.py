from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import psutil
import pytest

from cowork.coding import processes
from cowork.coding.processes import executable_name, runs_command, terminate_command_trees


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("cmd.exe", "cmd"),
        (r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.EXE", "powershell"),
        ("/bin/zsh", "zsh"),
        ("/usr/local/bin/codex-code-mode-host", "codex-code-mode-host"),
        ("node", "node"),
    ],
)
def test_executable_names_drop_directories_and_windows_suffixes(path: str, expected: str) -> None:
    assert executable_name(path) == expected


def test_runs_command_matches_a_configured_program_and_its_arguments() -> None:
    assert runs_command(["/usr/bin/python3", "-m", "server", "--port", "9"], "python3", ["-m", "server"])
    assert runs_command([r"C:\nodejs\node.exe", "mcp.js"], "node", ["mcp.js"])
    assert not runs_command(["node", "server.js"], "node", ["mcp.js"])
    assert not runs_command([], "node", [])


class _FakeProcess:
    """Enough of psutil.Process for the tree walk: name, cmdline, children, terminate, kill."""

    def __init__(self, pid: int, name: str, children: list[_FakeProcess] | None = None, cmdline: list[str] | None = None) -> None:
        self.pid = pid
        self._name = name
        self._cmdline = cmdline or [name]
        self._children = children or []
        self.ended: list[str] = []

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return list(self._cmdline)

    def children(self, recursive: bool = False) -> list[_FakeProcess]:
        if not recursive:
            return list(self._children)
        found: list[_FakeProcess] = []
        for child in self._children:
            found.append(child)
            found.extend(child.children(recursive=True))
        return found

    def terminate(self) -> None:
        self.ended.append("terminate")

    def kill(self) -> None:
        self.ended.append("kill")


def _flatten(process: _FakeProcess) -> list[_FakeProcess]:
    return [process, *process.children(recursive=True)]


def test_every_unprotected_tree_under_the_app_server_ends_and_helpers_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    # The tree WIN-QA-007 captured on Windows: npm test under cmd.exe and a game
    # server, next to the helpers Codex keeps for the next turn.
    workers = [_FakeProcess(7, "node.exe"), _FakeProcess(8, "node.exe")]
    test_runner = _FakeProcess(6, "cmd.exe", [_FakeProcess(9, "node.exe", workers)])
    npm = _FakeProcess(5, "node.exe", [test_runner])
    npm_cmd = _FakeProcess(4, "cmd.exe", [npm])
    game_server = _FakeProcess(11, "node", [_FakeProcess(12, "node")])
    mcp_server = _FakeProcess(2, "python.exe", cmdline=["python.exe", "-m", "cowork.coding.integration_mcp"])
    code_mode_host = _FakeProcess(3, "codex-code-mode-host.exe")
    app_server = _FakeProcess(1, "codex.exe", [mcp_server, code_mode_host, npm_cmd, game_server])

    order: list[int] = []

    def wait_procs(procs, timeout):
        order.extend(p.pid for p in procs)
        return list(procs), []

    monkeypatch.setattr(processes, "psutil", SimpleNamespace(Process=lambda pid: {1: app_server}[pid], Error=psutil.Error, wait_procs=wait_procs))
    protected = lambda p: p.name().startswith("codex-code-mode-host") or "integration_mcp" in " ".join(p.cmdline())

    ended = terminate_command_trees(1, protected=protected)

    assert ended == 8
    assert {p.pid for p in _flatten(app_server) if p.ended} == {4, 5, 6, 7, 8, 9, 11, 12}
    assert app_server.ended == [] and mcp_server.ended == [] and code_mode_host.ended == []
    # Leaves go before their parents so a shell cannot restart a dying child.
    assert order.index(7) < order.index(9) < order.index(6) < order.index(5) < order.index(4)
    assert order.index(12) < order.index(11)


def test_a_missing_app_server_or_uninspectable_child_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class Opaque(_FakeProcess):
        def name(self) -> str:
            raise psutil.AccessDenied(self.pid)

    opaque = Opaque(5, "unknown")
    app_server = _FakeProcess(1, "codex", [opaque])

    def process(pid: int):
        if pid == 1:
            return app_server
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(processes, "psutil", SimpleNamespace(Process=process, Error=psutil.Error, wait_procs=lambda procs, timeout: (list(procs), [])))
    protect_by_name = lambda p: p.name() == "keep"

    assert terminate_command_trees(None, protected=protect_by_name) == 0
    assert terminate_command_trees(424242, protected=protect_by_name) == 0
    assert terminate_command_trees(1, protected=protect_by_name) == 0
    assert opaque.ended == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX process tree; the Windows shape is covered by the fake tree")
def test_a_real_command_tree_ends_while_a_protected_sibling_and_the_parent_survive() -> None:
    # A stand-in app-server: python with a shell-rooted command tree and a
    # helper as direct children. The compound command keeps sh alive as the
    # tree's root instead of exec-ing straight into sleep.
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        shell = subprocess.Popen(["sh", "-c", "sleep 60; sleep 60"])
        helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)  # keep-me"])
        print(shell.pid, helper.pid, flush=True)
        time.sleep(60)
        """
    )
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        shell_pid, helper_pid = (int(x) for x in parent.stdout.readline().split())
        shell, helper = psutil.Process(shell_pid), psutil.Process(helper_pid)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not shell.children():
            time.sleep(0.05)
        sleeper = shell.children()[0]

        ended = terminate_command_trees(parent.pid, protected=lambda p: "keep-me" in " ".join(p.cmdline()), timeout=2.0)

        assert ended == 2
        _, alive = psutil.wait_procs([shell, sleeper], timeout=2.0)
        assert all(p.status() == psutil.STATUS_ZOMBIE for p in alive)
        assert helper.is_running() and helper.status() != psutil.STATUS_ZOMBIE
        assert parent.poll() is None
    finally:
        processes.terminate_descendants(parent.pid, timeout=2.0)
        parent.kill()
        parent.wait(timeout=5)
