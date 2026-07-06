"""Antigravity coworker — plain-text CLI output, no structured event
protocol at all (unlike Claude Code's stream-json). Every stdout line is
just a text_chunk; BaseCliHarness's generic accumulation fallback
assembles final_text from those since this CLI never emits an explicit
completion marker.

supports_resume=False: `agy --conversation <id>` and `agy --continue`
both hang and time out in headless (`-p`) mode on this machine — a real
CLI-level limitation confirmed by direct testing (30s and 90s timeouts,
both mechanisms), not a harness bug. Every turn starts fresh rather than
risk a multi-minute hang. Revisit if a future agy release fixes headless
resume.
"""
from __future__ import annotations

import shutil
import subprocess

from cowork.harnesses.base import register
from cowork.harnesses.cli_agents.base import BaseCliHarness
from cowork.harnesses.cli_agents.config import CliConfig
from cowork.harnesses.cli_agents.events import NormalizedEvent

AGY_CONFIG = CliConfig(
    executable="agy",
    print_flag="-p",
    model_flag="--model",
    resume_flag=None,
    session_flag=None,
    skip_permissions_flag="--dangerously-skip-permissions",
    supports_resume=False,
    supports_images=False,
    supports_mcp=False,
)


@register
class AntigravityHarness(BaseCliHarness):
    id = "antigravity"
    label = "Antigravity"
    config = AGY_CONFIG

    category = "CLI"
    priority = 6
    tags = ("subscription", "fast", "coding")

    # Populated on first successful `agy models` run; the CLI accepts the
    # display names it prints verbatim as --model values (verified:
    # `agy -p … --model "Gemini 3.5 Flash (Low)"` works).
    _models_cache: tuple[str, ...] | None = None

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        if cls._models_cache is not None:
            return cls._models_cache
        cli = shutil.which(cls.config.executable)
        if cli is None:
            return ()  # not installed — don't cache, it may appear later
        try:
            result = subprocess.run(
                [*cls.spawn_argv(cli), "models"],
                capture_output=True, text=True, timeout=20,
            )
        except Exception:
            return ()
        if result.returncode != 0:
            return ()
        models = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if models:
            cls._models_cache = models
        return models

    def parse_line(self, line: str) -> NormalizedEvent | None:
        return NormalizedEvent(type="text_chunk", text=line)

    def check_status(self) -> dict:
        # agy has no login/whoami/auth subcommand (confirmed: `agy --help`
        # lists only changelog/help/install/models/plugin/update). `agy
        # models` succeeding with a non-empty list is the closest available
        # proxy — it needs a working session to reach the backend at all.
        base = super().check_status()
        if not base["installed"]:
            return base
        try:
            result = subprocess.run(
                [base["path"], "models"], capture_output=True, text=True, timeout=20,
            )
        except Exception as exc:
            base["detail"] = f"Could not check status: {exc}"
            return base
        base["loggedIn"] = result.returncode == 0 and bool(result.stdout.strip())
        base["detail"] = (
            "Responding normally (no dedicated login-status command exists for this CLI)."
            if base["loggedIn"] else "Not responding as expected — run `agy` directly to check login."
        )
        return base
