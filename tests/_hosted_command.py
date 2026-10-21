"""The command a hosted cowork-server pod runs, read from the chart and the image.

The chart's ``deployment.image.command`` and ``args`` become the container's
``command`` and ``args``, and Kubernetes lets them replace the image's own
start command: ``command`` replaces the ENTRYPOINT and drops the CMD, and
``args`` alone replace the CMD. With neither set, the image's CMD runs. So a
test that read only the Dockerfile would pass while the chart ran a command
without the image's shutdown bound.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"
CHART_VALUES = REPO / "deployment" / "cowork-server" / "values.yaml"
GRACEFUL_SHUTDOWN_FLAG = "--timeout-graceful-shutdown"


def image_cmd() -> list[str]:
    """The Dockerfile's exec-form CMD."""
    lines = DOCKERFILE.read_text().splitlines()
    # args alone would replace the CMD after an ENTRYPOINT; with none they
    # are the whole command, which hosted_command assumes.
    assert not [line for line in lines if line.startswith("ENTRYPOINT")], "the image gained an ENTRYPOINT"
    return next(json.loads(line.removeprefix("CMD ")) for line in lines if line.startswith("CMD ["))


def hosted_command(values: Path = CHART_VALUES) -> list[str]:
    """The command the chart's values make a hosted pod run."""
    image = ((yaml.safe_load(values.read_text()) or {}).get("deployment") or {}).get("image") or {}
    command, args = image.get("command") or [], image.get("args") or []
    if command:
        return [*command, *args]
    if args:
        return list(args)
    return image_cmd()


def graceful_shutdown_seconds(command: list[str]) -> float:
    """Uvicorn's request-drain bound in ``command``; fails when it has none,
    since uvicorn's default then waits for every open response forever."""
    assert GRACEFUL_SHUTDOWN_FLAG in command, f"no {GRACEFUL_SHUTDOWN_FLAG} in {command}"
    return float(command[command.index(GRACEFUL_SHUTDOWN_FLAG) + 1])
