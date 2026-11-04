"""What follows a write to a doc, for the editor's and the agent's writes alike.

The editors and the chat watching hear of it, a shared page republishes a few
seconds later, and the live copy behind its link follows. (An edit from the
shared page itself takes the same steps in ``cloud.apply_edit``, mirroring at
once rather than later.) At startup, shared docs catch up on what was still
waiting when the app quit.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from cowork.db.scoped import LOCAL_SCOPE, TenantScope, tenant_scope_required

from . import cloud, republish
from .model import Doc
from .service import announce
from .sharing import shared_doc_folders

logger = logging.getLogger(__name__)


def after_write(
    folder: Path,
    doc: Doc,
    result: dict[str, Any] | None,
    *,
    by: str,
    scope: TenantScope | None,
    client_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    announce(doc["id"], doc, result, by=by, client_id=client_id, conversation_id=conversation_id)
    republish.schedule(folder, scope=scope)
    cloud.schedule_mirror(folder)


def catch_up_shared_docs() -> None:
    """Bring every shared doc's page and live copy up to date, then take the owner's edits."""
    try:
        folders = shared_doc_folders()
    except Exception:  # noqa: BLE001
        logger.warning("could not list shared docs", exc_info=True)
        return
    for folder in folders:
        try:
            republish.catch_up(folder, scope=LOCAL_SCOPE)
            cloud.mirror_now(folder)
        except Exception:  # noqa: BLE001 - one doc must not keep the others behind
            logger.warning("could not bring shared doc %s up to date", folder, exc_info=True)
    if folders:
        cloud.ensure_listening()


def start() -> None:
    """At startup, on a background thread so boot never waits.

    Only on a one-person desktop: a server several people share has no one set
    of projects to walk, and in org mode autopublish owns what a shared page shows.
    """
    cloud.install()
    if tenant_scope_required():
        return
    threading.Thread(target=catch_up_shared_docs, name="shared-doc-start", daemon=True).start()


def stop() -> None:
    """Shutdown: drop the follow-ups still waiting; the next start catches them up."""
    cloud.stop()
    republish.stop()
