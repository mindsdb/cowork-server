"""Finish a desktop sign-out's credential clear before this store serves.

The desktop app drops ``PENDING_CLEAR_MARKER`` beside a store's database when
it signs out, because the sidecar for that store may be down, may not answer,
or may belong to an organization that is not open. Consuming the marker during
startup, before Uvicorn binds, means no request is ever served from a store
whose provider keys the user was told were removed.

The marker's name is shared with the desktop app (``pending-credential-clear.ts``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from sqlalchemy.engine import make_url
from sqlmodel import Session

from cowork.db.session import get_engine
from cowork.services.settings import SettingService

logger = logging.getLogger(__name__)

PENDING_CLEAR_MARKER = ".pending-credential-clear"

# Windows antivirus and indexers hold a just-written file open for a moment.
_UNLINK_ATTEMPTS = 5
_UNLINK_RETRY_S = 0.2


class PendingCredentialClearFailed(RuntimeError):
    """The store holds a pending clear that could not be completed."""


def _store_dir(database_uri: str) -> Path | None:
    """Directory of a file-backed SQLite database, or None for anything else.

    Args:
        database_uri: The store's SQLAlchemy URI.

    Returns:
        The directory the desktop writes the marker into, or None when the
        store is not a local SQLite file.
    """
    url = make_url(database_uri)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).parent


def _unlink_marker(marker: Path) -> None:
    """Remove the marker, retrying a transient lock.

    Args:
        marker: Path of the marker file.

    Raises:
        OSError: The last failure, when every attempt failed.
    """
    for attempt in range(_UNLINK_ATTEMPTS):
        try:
            marker.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == _UNLINK_ATTEMPTS - 1:
                raise
            time.sleep(_UNLINK_RETRY_S)


def consume_pending_credential_clear(database_uri: str, tenancy_mode: str) -> bool:
    """Clear stored credentials when a sign-out left a marker beside the store.

    Run after migrations and before the server binds. Clears the same rows as
    ``POST /settings/logout`` and then removes the marker. A failure leaves the
    marker for the next start, which is safe to retry because a store whose
    start failed has served nothing in between.

    Args:
        database_uri: The store's SQLAlchemy URI.
        tenancy_mode: ``"local"`` or ``"org"``. Org mode ignores the marker,
            since credentials there are org owned, not one desktop user's.

    Returns:
        True when a pending clear was found and completed.

    Raises:
        PendingCredentialClearFailed: The clear or the marker removal failed.
            Raised outside any handler so no traceback carries a settings row.
    """
    if tenancy_mode == "org":
        return False
    store = _store_dir(database_uri)
    if store is None:
        return False
    marker = store / PENDING_CLEAR_MARKER
    if not marker.exists():
        return False

    failure: str | None = None
    try:
        with Session(get_engine(database_uri)) as session:
            SettingService(session, None).clear_credentials()
        _unlink_marker(marker)
    except Exception as exc:
        # The type only: a database error's text can carry the settings keys.
        failure = type(exc).__name__
    if failure is not None:
        logger.error("pending credential clear failed for store %s (%s)", store, failure)
        raise PendingCredentialClearFailed(f"pending credential clear failed for store {store}")

    logger.info("cleared stored credentials left by a desktop sign-out in %s", store)
    return True
