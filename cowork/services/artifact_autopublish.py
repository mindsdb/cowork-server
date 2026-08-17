"""Autopublish reconciliation: deciding what to publish, and publishing it.

No new state is introduced. The source of truth for "already published" is the
artifact's own `.published.json`, which is why retry is free: a publish that
failed, timed out, or died with the replica left no record, so the next turn sees
the artifact as new again and republishes it.

cowork-server has no metrics backend (no prometheus, no statsd, no counter
facade), so the "metrics" this feature needs are structured log lines with the
stable `artifact_autopublish` prefix, ready to be wired to a collector.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Reasons a candidate is skipped. Kept distinct because a single None would make
# the skip log unusable: "nothing to do" and "cannot be published at all" demand
# different follow-up.
REASON_NO_CONTENT = "no_content"
REASON_NOT_PUBLISHABLE = "not_publishable"
REASON_UNPUBLISHED = "unpublished"
REASON_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class PublishDecision:
    action: Literal["new", "changed"] | None
    reason: str | None = None
    is_fullstack: bool = False
    content_mtime: int = 0


def needs_publish(folder: Path, artifacts_base: Path) -> PublishDecision:
    """Whether this artifact should be (re)published, and why not when it shouldn't.

    Order of the checks matters: the cheap ones come first, and the md5 bundle
    rebuild happens only after the mtime gate has already said something moved.
    """
    from cowork.services.artifacts import _user_files, content_mtime, load_published_map
    from cowork.services.publish import (
        PUBLISHABLE_STATIC_SUFFIXES,
        _resolve_publish_target,
        compute_publish_md5,
    )

    if not _user_files(folder):
        return PublishDecision(None, REASON_NO_CONTENT)

    try:
        publish_target, published_dir, published_key, is_fullstack = _resolve_publish_target(
            folder, container_dirs=[artifacts_base]
        )
    except Exception:
        logger.warning("Could not resolve publish target for %s", folder, exc_info=True)
        return PublishDecision(None, REASON_NOT_PUBLISHABLE)

    if is_fullstack:
        # /upload is strict here: literally backend.py plus static/index.html.
        if not (folder / "backend.py").is_file() or not (folder / "static" / "index.html").is_file():
            return PublishDecision(None, REASON_NOT_PUBLISHABLE, True)
    elif publish_target.suffix.lower() not in PUBLISHABLE_STATIC_SUFFIXES:
        # Suffix, not the on-disk name: _zip_html renames a single file to
        # index.html inside the archive, so an artifact whose primary is
        # `report.html` publishes normally.
        return PublishDecision(None, REASON_NOT_PUBLISHABLE)

    current_mtime = content_mtime(folder)
    entry = load_published_map(published_dir).get(published_key)

    if not isinstance(entry, dict) or not entry.get("report_id"):
        return PublishDecision("new", None, is_fullstack, current_mtime)
    if entry.get("published") is False:
        # Explicitly unpublished — never resurrect it on our own.
        return PublishDecision(None, REASON_UNPUBLISHED, is_fullstack, current_mtime)

    published_mtime = entry.get("published_mtime")
    if isinstance(published_mtime, (int, float)) and current_mtime <= published_mtime:
        return PublishDecision(None, REASON_UNCHANGED, is_fullstack, current_mtime)

    digest = compute_publish_md5(folder, artifacts_base=artifacts_base)
    if digest is None or digest == entry.get("last_md5"):
        # None means "can't tell" — never republish on a guess.
        return PublishDecision(None, REASON_UNCHANGED, is_fullstack, current_mtime)
    return PublishDecision("changed", None, is_fullstack, current_mtime)
