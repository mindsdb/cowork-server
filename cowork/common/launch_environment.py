"""The server's own ``COWORK_*`` and ``MINDSHUB_*`` variables, held out of its environment.

Every process the server starts inherits ``os.environ``: git, project
commands, terminals, the Codex app-server and anton's cells. The Codex SDK
and anton copy the whole environment and can only add to it, so a variable
that configures this server would reach code the agent runs. A local launch
(``cowork.cli.main``) therefore moves each ``COWORK_*`` and ``MINDSHUB_*``
variable out of ``os.environ`` into this module once the cached settings have
loaded (``scrub_launch_environment``), and the server reads them from here
afterwards.

``launch_value`` reads the live environment first, so a test that sets a
variable still wins, and then the value captured at launch. Every settings
class reads the captured values too, ranked just under the live environment
(``cowork.common.settings.app_settings.LaunchEnvironmentSource``), so a
settings object built after the scrub sees what it saw before.

A server started by ``uvicorn cowork.server:app`` (the hosted image) never
scrubs, and this module then holds nothing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType

#: The variable prefixes the scrub moves out of the server's environment.
SCRUBBED_PREFIXES = ("COWORK_", "MINDSHUB_")

_captured: Mapping[str, str] = MappingProxyType({})


def scrub_launch_environment() -> None:
    """Move every ``COWORK_*`` and ``MINDSHUB_*`` variable out of ``os.environ``
    into this module. A second call keeps what the first captured and adds
    what has appeared since."""
    global _captured
    moved = {name: value for name, value in os.environ.items() if name.startswith(SCRUBBED_PREFIXES)}
    _captured = MappingProxyType({**_captured, **moved})
    for name in moved:
        os.environ.pop(name, None)


def launch_value(name: str) -> str | None:
    """The variable's value in the live environment, else the one captured at
    launch, else None."""
    value = os.environ.get(name)
    return value if value is not None else _captured.get(name)


def captured_environment() -> Mapping[str, str]:
    """The variables the scrub moved out of the environment, read-only."""
    return _captured
