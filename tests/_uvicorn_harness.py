"""Serve a small app with a real Uvicorn process and keep everything it printed.

Uvicorn runs with its own default log config, as the images start it, so a
test sees exactly what a container's stdout and stderr would hold.
"""
from __future__ import annotations

import http.client
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

# How each launcher reaches the app. "module" is `python -m uvicorn
# module:app`, as the images run it: Uvicorn configures logging, then imports
# the app. "run" calls setup_logging before Uvicorn's dictConfig, as the
# cowork-server CLI does through its cowork.dev_setup import.
Launch = Literal["module", "run"]

_STARTED = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")
_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class UvicornRun:
    statuses: tuple[int, ...]
    stdout: str
    stderr: str


def _collect(*, stream: IO[str], lines: list[str]) -> None:
    for line in stream:
        lines.append(line)


def _port(*, stderr: list[str], process: subprocess.Popen) -> int:
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for line in list(stderr):
            if match := _STARTED.search(line):
                return int(match.group(1))
        if process.poll() is not None:
            break
        time.sleep(0.05)
    process.kill()
    raise AssertionError("Uvicorn never reported its port:\n" + "".join(stderr))


def run_uvicorn_app(
    *, tmp_path: Path, app_source: str, paths: tuple[str, ...], launch: Launch,
) -> UvicornRun:
    """Serve ``app`` from ``app_source``, GET each path, then stop the server."""
    app_dir = tmp_path / "served"
    app_dir.mkdir()
    (app_dir / "served_app.py").write_text(textwrap.dedent(app_source))
    if launch == "module":
        command = [sys.executable, "-m", "uvicorn", "served_app:app", "--app-dir", str(app_dir),
                   "--host", "127.0.0.1", "--port", "0"]
    else:
        command = [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import served_app, uvicorn; "
                   "uvicorn.run(served_app.app, host='127.0.0.1', port=0)", str(app_dir)]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"},
    )
    stdout: list[str] = []
    stderr: list[str] = []
    readers = [threading.Thread(target=_collect, kwargs={"stream": process.stdout, "lines": stdout}, daemon=True),
               threading.Thread(target=_collect, kwargs={"stream": process.stderr, "lines": stderr}, daemon=True)]
    for reader in readers:
        reader.start()
    try:
        port = _port(stderr=stderr, process=process)
        statuses = []
        for path in paths:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=_TIMEOUT_SECONDS)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                response.read()
                statuses.append(response.status)
            finally:
                connection.close()
        process.terminate()
        process.wait(timeout=_TIMEOUT_SECONDS)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for reader in readers:
            reader.join(timeout=_TIMEOUT_SECONDS)
    return UvicornRun(statuses=tuple(statuses), stdout="".join(stdout), stderr="".join(stderr))
