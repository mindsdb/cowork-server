"""Derive the CLI's ``.env`` from the DB settings (ENG-1127, Phase A).

The DB is the source of truth for cowork-server. The standalone ``anton`` CLI
still reads its config from ``.env``, so on a single-user desktop install the
server exports the overlapping settings back out to ``.env`` after every write,
treating the file as an export rather than a second source of truth.

This module holds the pure (no-DB) derivation so it is unit-testable; the gate,
path resolution, and the write itself live in
``SettingService._export_env_for_cli``.

Only the keys in ``SETTING_ENV_ALIASES`` are managed. Everything else in the
file is preserved verbatim: ``COWORK_AUTH_TOKEN`` (written by the auth
middleware), the ``ANTON_*_MODEL`` lines (CLI-only, deliberately excluded from
the sync per ENG-739 so a login/refresh can't re-pin a picker choice),
``ANTON_FIRST_RUN_DONE`` (written by the CLI itself), and any comments.
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

# The ANTON_* variables this export owns. A line for any of these is replaced on
# export; every other line in the file is left untouched.
MANAGED_ENV_VARS: tuple[str, ...] = tuple(SETTING_ENV_ALIASES.values())


def _env_str(value: object) -> str:
    """Format a loaded ``UserSettings`` value as its ``.env`` string.

    Providers use the dash form the CLI expects (``minds-cloud``, not the DB's
    ``minds_cloud``); secrets are the decrypted plaintext (``.env`` is plaintext
    while the DB is Fernet-encrypted); booleans are lowercased.
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
    """Map the aliased settings that are actually STORED to ``{ANTON_VAR: value}``.

    Only keys in ``present_keys`` (i.e. that have a DB row) are exported. This is
    deliberate: ``settings`` carries a resolved default for every unset field, so
    exporting on non-``None`` alone would push the server's defaults (and their
    drift) onto the CLI and would leave a line behind after a key is cleared.
    Mirroring only stored rows keeps ``.env`` a faithful export of the DB and
    lets the CLI apply its own defaults for anything unset. Unset/empty stored
    values are still skipped. Model keys are absent from ``SETTING_ENV_ALIASES``
    (ENG-739), so they are never exported here.
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

    Every line for a managed ANTON_* var is dropped, then the currently-set
    managed vars are appended (in ``SETTING_ENV_ALIASES`` order, so repeated
    exports of the same state produce identical output). Unmanaged lines keep
    their place. A managed key absent from ``managed`` (unset or cleared) loses
    its line — that is how a logout also wipes credentials from the CLI's file.
    """
    drop = tuple(f"{var}=" for var in MANAGED_ENV_VARS)
    kept = [ln for ln in existing.split("\n") if ln and not ln.startswith(drop)]
    kept.extend(f"{var}={value}" for var, value in managed.items())
    return "\n".join(kept) + "\n"


def atomic_write_env(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically, owner-only (0o600).

    Temp file in the same directory + ``os.replace`` so a crash or a concurrent
    CLI read never observes a truncated ``.env``. 0o600 because the file holds
    plaintext API keys.
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
