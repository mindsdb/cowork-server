"""The one write path for docs, shared by the agent's tools and the editor API.

Both callers create docs and apply ops through here, so an agent edit and a
person's keystrokes are serialised by the same lock and kept in the same
history; ``announce`` tells the editors and chats watching (``sync.after_write``
does that and the rest of what follows a write).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .hub import conv_topic, doc_topic, hub
from .model import Doc, DocError
from .ops import apply_ops, build_new_doc
from .store import NewDoc, create_doc_folder, mutate_doc, with_doc, write_html
from .undo import undo_ops

logger = logging.getLogger(__name__)


def render_page(folder: Path, doc: Doc) -> None:
    """Rewrite the doc's page (doc.html) from ``doc``."""
    write_html(folder, doc)


def refresh_page(folder: Path) -> None:
    """Re-render the page from the doc as it is now (for a change that wasn't a write)."""
    with_doc(folder, lambda doc: render_page(folder, doc))


def create_doc(
    artifacts_dir: Path,
    *,
    title: str,
    markdown: str | None = None,
    tabs: list[dict[str, Any]] | None = None,
    description: str = "",
    by: str = "agent",
    turn: str | None = None,
) -> NewDoc:
    """Create the artifact folder, its doc and its page. ``turn``: the agent reply writing it."""
    if not (title or "").strip():
        raise DocError("bad_title", "give the doc a title as its reader would say it.")
    doc = build_new_doc(title, markdown=markdown, tabs=tabs, by=by)
    made = create_doc_folder(Path(artifacts_dir), title=doc["title"], description=description or "", doc=doc,
                             by=by, turn=turn)
    render_page(made.folder, made.doc)
    return made


def apply(
    folder: Path,
    ops: list[dict[str, Any]],
    *,
    tab: str | None = None,
    by: str,
    if_rev: int | None = None,
    turn: str | None = None,
) -> tuple[Doc, dict[str, Any]]:
    """Apply ``ops`` atomically and re-render the page; returns (new doc, apply_ops result).

    ``turn``: the agent reply the edit is part of, so one reply is one version.
    The person's writes (``by="user"``) also answer ``undo``: the ops that put
    back what this one changed.
    """

    def change(current: Doc) -> tuple[Doc, dict[str, Any]]:
        new, result = apply_ops(current, ops, tab=tab, by=by, if_rev=if_rev)
        if by == "user":
            # Worked out here, against the doc this write replaced (undo.py).
            result["undo"] = undo_ops(current, new, result)
        return new, result

    return mutate_doc(folder, change, by=by, turn=turn, after=lambda doc: render_page(folder, doc))


# ---------------------------------------------------------------- events


def announce(
    artifact_id: str,
    doc: Doc,
    result: dict[str, Any] | None,
    *,
    by: str,
    client_id: str | None = None,
    conversation_id: str | None = None,
    created: bool = False,
    card: dict[str, Any] | None = None,
) -> None:
    """Tell watching editors (and the conversation the change came from)."""
    try:
        hub.publish(doc_topic(artifact_id), {
            "type": "doc.updated",
            "artifactId": artifact_id,
            "rev": doc["rev"],
            "by": by,
            "clientId": client_id,
            "touched": (result or {}).get("touched", {}),
            "removed": (result or {}).get("removed", {}),
            "doc": doc,
        })
        if conversation_id:
            hub.publish(conv_topic(conversation_id), keep=True, event={
                "type": "doc.created" if created else "doc.updated",
                "artifactId": artifact_id,
                "conversationId": conversation_id,
                "title": doc["title"],
                "rev": doc["rev"],
                "by": by,
                "card": card,
            })
    except Exception:  # announcing is best-effort; the write already landed
        logger.warning("could not announce doc change for %s", artifact_id, exc_info=True)


__all__ = ["announce", "apply", "create_doc", "refresh_page", "render_page"]
