from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CliConfig:
    """Pure data describing one CLI's invocation shape — no callables.
    Behavior (arg construction, output parsing) lives in the harness
    subclass, not here. See BaseCliHarness.build_arguments/parse_line."""

    executable: str
    print_flag: str = "-p"
    model_flag: str | None = "--model"
    resume_flag: str | None = "--resume"
    """Flag to resume an existing session by id, e.g. claude's `--resume <id>`."""
    session_flag: str | None = None
    """Flag to start a FRESH session with an explicit id, e.g. claude's
    `--session-id <id>` (used when there's no resume_flag, or on first turn)."""
    skip_permissions_flag: str | None = None
    default_args: tuple[str, ...] = ()
    supports_resume: bool = True
    supports_images: bool = False
    supports_mcp: bool = False
