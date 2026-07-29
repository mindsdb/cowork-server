"""Derive the CLI's ``.env`` from the DB settings (ENG-1127, Phase A).

DB is source of truth; the ``anton`` CLI still reads ``.env`` so a desktop install
re-exports the overlapping settings on write. Only ``SETTING_ENV_ALIASES`` keys are
managed — auth token, ``ANTON_*_MODEL`` pins (CLI-only per ENG-739), comments preserved.
"""
from __future__ import annotations

import os
import tempfile
from enum import Enum
from pathlib import Path

from pydantic import SecretStr

from cowork.common.settings.user_settings import (
    SETTING_ENV_ALIASES,
    Provider,
    UserSettings,
)

# the ANTON_* vars this export owns; every other line is left untouched.
MANAGED_ENV_VARS: tuple[str, ...] = tuple(SETTING_ENV_ALIASES.values())


def _env_str(value: object) -> str:
    """Format a loaded ``UserSettings`` value as its ``.env`` string.

    Dash-form providers (``minds-cloud``), decrypted-plaintext secrets, lowercased bools.
    """
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, Provider):
        return value.ui_value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def build_env_export(settings: UserSettings, present_keys: set[str]) -> dict[str, str]:
    """Map the STORED aliased settings to ``{ANTON_VAR: value}``.

    Only ``present_keys`` are exported — ``settings`` resolves a default for every
    unset field, so exporting on non-``None`` alone would push server defaults to the CLI.
    """
    out: dict[str, str] = {}
    for db_key, env_var in SETTING_ENV_ALIASES.items():
        if db_key not in present_keys:
            continue
        value = getattr(settings, db_key, None)
        if value is None:
            continue
        text = _env_str(value)
        if text == "":
            continue
        out[env_var] = text
    return out


def merge_env_lines(existing: str, managed: dict[str, str]) -> str:
    """Rewrite the managed lines in ``existing``, preserving everything else.

    Managed ANTON_* lines are dropped then re-appended in alias order (byte-stable);
    a key absent from ``managed`` loses its line — how logout wipes the CLI's creds.
    """
    drop = tuple(f"{var}=" for var in MANAGED_ENV_VARS)
    kept = [ln for ln in existing.split("\n") if ln and not ln.startswith(drop)]
    kept.extend(f"{var}={value}" for var, value in managed.items())
    return "\n".join(kept) + "\n"


def atomic_write_env(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically, owner-only (0o600).

    Temp file + ``os.replace`` so a crash/concurrent read never sees a truncated ``.env``; 0o600 for plaintext keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(tmp, 0o600)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
