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
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Inside `.revisions/`, which every artifact listing, publish bundle and draft
#: preview already refuses to serve.
_MARKER_RELPATH = (".revisions", "draft-review.json")

SCOPE_ORGANIZATION = "organization"


def _marker_path(folder: Path) -> Path:
    return folder.joinpath(*_MARKER_RELPATH)


def _write_atomically(path: Path, payload: dict) -> None:
    """Replace `path` in one step, like the sibling comments journal does.

    A half-written grant would read as absent (`draft_review_grant` fails
    closed), so the risk is a draft that silently stays private rather than one
    that leaks — but a reader must never see a truncated file either way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=".draft-review-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
    except BaseException:
        os.unlink(name)
        raise


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
    _write_atomically(_marker_path(folder), grant)
    return grant


def draft_review_grant(folder: Path) -> dict | None:
    """The grant on `folder`, or None when its draft is still private.

    Unreadable or malformed content reads as absent: the grant only ever widens
    access, so a damaged record must fail closed.
    """
    path = _marker_path(folder)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable draft review grant: %s", path, exc_info=True)
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
        _marker_path(folder).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("Could not remove draft review grant in %s", folder, exc_info=True)
        return False
