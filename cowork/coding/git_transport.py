from __future__ import annotations

import re
from urllib.parse import urlparse

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_SCP_STYLE = re.compile(r"^[A-Za-z0-9._-]+@([A-Za-z0-9.-]+):([^\s]+)$")
ALLOWED_GIT_PROTOCOLS = "file:https:ssh"


def validate_git_source(value: str) -> str:
    """Accept only inert, explicitly supported Git transports.

    Git's ``ext::`` remote helper executes an arbitrary command.  Repository
    metadata crosses the control/execution-plane boundary, so validation must
    happen both while modelling a project and immediately before cloning it.
    """

    source = value.strip()
    if not source or source.startswith("-") or "\x00" in source or "\n" in source or "\r" in source:
        raise ValueError("repository URL is invalid")
    if "::" in source:
        raise ValueError("repository URL uses an unsupported Git transport")

    if source.startswith("/") or _WINDOWS_ABSOLUTE.match(source):
        return source
    if source.startswith("\\\\"):
        raise ValueError("network filesystem repository URLs are not supported")
    if _SCP_STYLE.fullmatch(source):
        return source

    parsed = urlparse(source)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        raise ValueError("repository URL must use HTTPS or SSH")
    if parsed.scheme == "https" and (parsed.username or parsed.password):
        raise ValueError("repository credentials must come from a connector, not the URL")
    return source


def is_local_git_source(value: str) -> bool:
    source = value.strip()
    return source.startswith("/") or bool(_WINDOWS_ABSOLUTE.match(source))
