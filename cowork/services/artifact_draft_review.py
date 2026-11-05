"""The owner's record that one draft is open for same-org review.

A project is shared by the organization but a conversation workspace under it is
private to whoever created it, and artifact bytes live inside that workspace
(`artifact_roots._project_artifact_bases`, ENG-1910). So org membership alone
cannot be what lets a co-member read a draft — otherwise every artifact of every
chat would be readable by the whole organization, which is the case that fix
closed.

This marker is the grant that reopens exactly one artifact, and only because its
owner asked. It sits next to the revision journal, so it is already excluded from
publishing, from `files[]` and from the draft preview itself.

The auth rule minted alongside it (`artifact_access.provision_draft_review_access`)
is what mindshub_inference checks for comments. This file is the local half:
cowork-server gates its own preview and review routes on it instead of calling
auth on every request, and the two are written by the same owner action.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from cowork.common.logger import log_context
from cowork.services.artifact_journal import (
    MAX_JOURNAL_RECORD_BYTES,
    read_journal_json,
    remove_journal_file,
    write_journal_file,
)

logger = logging.getLogger(__name__)

#: Inside the journal, `.revisions/`, which every artifact listing, publish
#: bundle and draft preview already refuses to serve. It's read and written
#: through artifact_journal, which follows no link on the way.
_MARKER_NAME = "draft-review.json"

SCOPE_ORGANIZATION = "organization"


def enable_draft_review(folder: Path, *, org_id: str, enabled_by: str) -> dict:
    """Record that `folder`'s draft is open to `org_id` for review.

    Idempotent: re-enabling keeps the original `enabledAt` so a client that
    calls this on every mount does not rewrite the record.
    """
    existing = draft_review_grant(folder)
    if existing is not None:
        return existing
    grant = {
        "version": 1,
        "scope": SCOPE_ORGANIZATION,
        "orgId": str(org_id),
        "enabledBy": str(enabled_by),
        "enabledAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Replaced in one step: a half-written grant would read as absent, so the
    # risk is a draft that silently stays private, but a reader must never see
    # a truncated file either way.
    write_journal_file(folder, _MARKER_NAME, content=json.dumps(grant, indent=2).encode("utf-8"))
    return grant


def draft_review_grant(folder: Path) -> dict | None:
    """The grant on `folder`, or None when its draft is still private.

    Unreadable or malformed content reads as absent: the grant only ever widens
    access, so a damaged record must fail closed.
    """
    try:
        payload = read_journal_json(folder, _MARKER_NAME, max_bytes=MAX_JOURNAL_RECORD_BYTES)
    except (OSError, ValueError):
        logger.warning(
            "Ignoring unreadable draft review grant",
            extra={
                "event": "artifact_draft_review_grant_unreadable",
                **log_context(artifact_slug=folder.name),
            },
            exc_info=True,
        )
        return None
    if not isinstance(payload, dict) or payload.get("scope") != SCOPE_ORGANIZATION:
        return None
    return payload


def draft_review_allows(folder: Path, org_id: str | None) -> bool:
    """Whether a member of `org_id` may review `folder`'s draft.

    The organization is compared, not just the presence of a grant: artifact
    roots are resolved from a project the caller can already read, but a project
    row can move between organizations, and a stale grant must not survive that.
    """
    if not org_id:
        return False
    grant = draft_review_grant(folder)
    return grant is not None and grant.get("orgId") == str(org_id)


def disable_draft_review(folder: Path) -> bool:
    """Drop the grant. True when one was there."""
    try:
        return remove_journal_file(folder, _MARKER_NAME)
    except OSError:
        logger.warning(
            "Could not remove draft review grant",
            extra={
                "event": "artifact_draft_review_grant_not_removed",
                **log_context(artifact_slug=folder.name),
            },
            exc_info=True,
        )
        return False
