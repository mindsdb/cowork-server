"""Desktop sidecar spawn contract.

The Electron desktop app runs this server as a subprocess and talks to it
over an env-var contract (see cowork/src/main/server-process.ts in the
desktop repo): it picks a per-user port, passes it as COWORK_SERVER_PORT,
stamps the process with COWORK_SERVER_OWNER, points COWORK_HOME at the
build's data home (non-prod builds only; prod leaves it unset so the
server defaults to ~/.cowork), and then polls /api/v1/health/ on the port
it chose.

These tests spawn the REAL server process with that exact env and assert
the contract end to end. Unit tests on AppSettings alone cannot catch a
broken spawn path: removing the COWORK_SERVER_PORT alias passed unit tests
while making every packaged desktop install health-check a port the server
never bound.
"""

import os
import socket
import subprocess
import sys
import time
import uuid

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The desktop spawns the sidecar from a clean login environment, not from
# whatever this pytest session has accumulated (conftest.py and other tests
# mutate os.environ). Passing only these OS plumbing vars through keeps the
# spawned server hermetic and the test immune to suite ordering.
_PASSTHROUGH_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "TMPDIR",
    # Windows process plumbing.
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
)


def _spawn_like_desktop(tmp_path, port: int, owner: str, port_var: str) -> tuple[subprocess.Popen, "os.PathLike[str]"]:
    # Mirror the env block the desktop app builds before spawning the
    # sidecar. DATABASE_URI is pinned to a fresh SQLite file so the spawned
    # process cannot touch the shared test DB conftest.py points at.
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update(
        {
            port_var: str(port),
            "COWORK_SERVER_HOST": "127.0.0.1",
            "COWORK_SERVER_OWNER": owner,
            "COWORK_HOME": str(tmp_path / "home"),
            "DATABASE_URI": f"sqlite:///{tmp_path / 'contract.db'}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Server output goes to a file, not a PIPE: nobody drains a pipe while
    # we poll /health, so chatty startup logging could fill the pipe buffer
    # and block the child. The file doubles as failure diagnostics.
    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-c", "from cowork.cli import main; main()"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return proc, log_path


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _read_log(log_path) -> str:
    try:
        return log_path.read_text(errors="replace")[-4000:]
    except OSError:
        return "<no server output captured>"


def _wait_for_health(proc: subprocess.Popen, port: int, log_path, deadline_s: float = 90.0) -> dict:
    url = f"http://127.0.0.1:{port}/api/v1/health/"
    deadline = time.monotonic() + deadline_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"server exited before becoming healthy (code {proc.returncode}):\n{_read_log(log_path)}"
            )
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError as err:
            last_error = err
        time.sleep(0.25)
    _stop(proc)
    pytest.fail(
        f"server never served /health on the desktop-assigned port {port} "
        f"within {deadline_s}s (last error: {last_error!r}). The desktop app "
        f"would report 'Server did not respond on /health'.\n{_read_log(log_path)}"
    )


# COWORK_SERVER_PORT is what every shipped desktop build passes; the
# COWORK_LISTEN_PORT case locks the newer name so neither alias can be
# dropped again without this suite failing.
@pytest.mark.parametrize("port_var", ["COWORK_SERVER_PORT", "COWORK_LISTEN_PORT"])
def test_desktop_spawn_contract(tmp_path, port_var):
    port = _free_port()
    owner = uuid.uuid4().hex
    proc, log_path = _spawn_like_desktop(tmp_path, port, owner, port_var)
    try:
        payload = _wait_for_health(proc, port, log_path)

        # The desktop adopts a server only when /health echoes the owner
        # token it stamped at spawn (server-process.ts adoption check).
        assert payload["status"] == "ok"
        assert payload["owner"] == owner

        # The About panel folds these into the unified version display.
        assert payload["server_version"]
        assert payload["anton_version"]
    finally:
        _stop(proc)
