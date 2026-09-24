"""Who owns an artifact in organization mode (ENG-2961).

Project-level artifacts roots (ENG-2056) are shared by every member of a
project, so a position in the tree no longer names a creator. The owner of an
artifact under a project root is a server-written row in
``shared_resource_attributions`` (kind ``artifact``), recorded when the turn
that created it ends, or by the one-time startup backfill.

Legacy per-conversation roots (``<project>/conversations/<uuid>/.anton/
artifacts``) keep path-derived ownership. That is a deliberate exception to
"never derive an owner from a directory name": nothing new is written there,
the directory is server-controlled, and on prod those roots hold almost every
artifact created before 2026-09-23.

``metadata.json`` provenance sits on the project-wide writable mount, so any
agent in the project can rewrite it. At runtime it may only DROP a slug from a
turn's claim (``turn_created_slugs``), never grant ownership.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from cowork.models.conversation import Conversation
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.services.artifact_access import ArtifactAccessUnavailable
from cowork.services.artifact_roots import _ARTIFACTS_SUBPATH
from cowork.services.artifacts import ProjectArtifacts

logger = logging.getLogger(__name__)

ARTIFACT = "artifact"

_KEY_MAX_LENGTH = 255  # SharedResourceAttribution.resource_key
_CONVERSATIONS_DIRNAME = "conversations"

OwnerState = Literal["recorded", "legacy_path", "unknown", "local"]

# Keys already reported as `unknown`, so a polled listing logs each one once
# per process instead of on every refresh (deployments run at WARNING).
# Bounded: cleared when it grows past the cap, which at worst repeats a line.
_REPORTED_UNKNOWN: set[tuple[str, str]] = set()
_REPORTED_UNKNOWN_CAP = 10_000


class ArtifactOwnerUnknown(ArtifactAccessUnavailable):
    """No owner is recorded for this artifact, so no one may act as owner."""


@dataclass(frozen=True)
class OwnerResolution:
    owner_user_id: str | None
    state: OwnerState

    @property
    def unknown(self) -> bool:
        return self.state == "unknown"


def artifact_resource_key(project_id, slug: str) -> str:
    """The attribution key of one project-root artifact.

    Slugs may be 255 characters long (`TaskObjectService._unique_slug`), which
    with the project id would overflow the column; a long slug is hashed so the
    claim never fails on length and silently leaves the artifact ownerless.
    """
    key = f"{project_id}/{slug}"
    if len(key) <= _KEY_MAX_LENGTH:
        return key
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()
    return f"{project_id}/sha256:{digest}"


def _org_scope(session):
    scope = getattr(session, "scope", None)
    return scope if scope is not None and scope.org_mode else None


def _root_parts(source) -> tuple[str, ...]:
    return tuple(getattr(source, "root_parts", ()) or ())


def _anchored(source, parts: tuple[str, ...]) -> bool:
    anchor = getattr(source, "trusted_anchor", None)
    return anchor is not None and Path(source.base) == Path(anchor).joinpath(*parts)


def project_root_source(project) -> ProjectArtifacts:
    """The project's shared ``<project>/.anton/artifacts`` root, from its row.

    Built directly rather than picked out of `artifacts_sources_for_project`,
    which also lists every legacy conversation root on the shared mount: a
    caller that already knows its base is the project root has nothing to
    discover. It is the same shape `artifact_roots._sources_for` gives that
    root, so `is_project_root` holds for it.
    """
    project_path = Path(project.path)
    return ProjectArtifacts(
        base=project_path.joinpath(*_ARTIFACTS_SUBPATH),
        project_id=str(project.id),
        project_name=project.name,
        trusted_anchor=project_path,
        root_parts=_ARTIFACTS_SUBPATH,
    )


def is_project_root(source) -> bool:
    """True for the shared ``<project>/.anton/artifacts`` root."""
    parts = _root_parts(source)
    return parts == _ARTIFACTS_SUBPATH and _anchored(source, parts)


def _legacy_conversation_id(source) -> UUID | None:
    """The conversation a legacy root belongs to, or None for any other shape."""
    parts = _root_parts(source)
    if (
        len(parts) != 4
        or parts[0] != _CONVERSATIONS_DIRNAME
        or parts[2:] != _ARTIFACTS_SUBPATH
        or not _anchored(source, parts)
    ):
        return None
    try:
        return UUID(parts[1])
    except ValueError:
        return None


def _legacy_owner(session, source, conversation_id: UUID) -> str | None:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or str(conversation.project_id) != str(source.project_id):
        return None
    return conversation.created_by or None


def _report_unknown(project_id, slug: str) -> None:
    key = (str(project_id), slug)
    if key in _REPORTED_UNKNOWN:
        return
    if len(_REPORTED_UNKNOWN) >= _REPORTED_UNKNOWN_CAP:
        _REPORTED_UNKNOWN.clear()
    _REPORTED_UNKNOWN.add(key)
    logger.warning("artifact_owner unknown project=%s slug=%s", project_id, slug)


def resolve_artifact_owners(session, source, slugs: Iterable[str]) -> dict[str, OwnerResolution]:
    """Owner of each slug under one root. Read-only; one query per root."""
    wanted = list(dict.fromkeys(slugs))
    scope = _org_scope(session)
    if scope is None:
        user_id = getattr(getattr(session, "scope", None), "user_id", None)
        return {slug: OwnerResolution(user_id, "local") for slug in wanted}

    conversation_id = _legacy_conversation_id(source)
    if conversation_id is not None:
        owner = _legacy_owner(session, source, conversation_id)
        state: OwnerState = "legacy_path" if owner else "unknown"
        return {slug: OwnerResolution(owner, state) for slug in wanted}

    if source.project_id and getattr(source, "trusted_anchor", None) is None:
        # Every org root built by `artifact_roots._sources_for` carries an anchor;
        # one without it would silently resolve `unknown` for every artifact.
        logger.error(
            "artifact_owner source without trusted_anchor project=%s base=%s",
            source.project_id, source.base,
        )
    if not is_project_root(source) or not source.project_id or not wanted:
        return {slug: OwnerResolution(None, "unknown") for slug in wanted}

    keys = {artifact_resource_key(source.project_id, slug): slug for slug in wanted}
    rows = session.exec(
        session.select(SharedResourceAttribution).where(
            SharedResourceAttribution.resource_kind == ARTIFACT,
            SharedResourceAttribution.resource_key.in_(list(keys)),
        )
    ).all()
    owners = {
        keys[row.resource_key]: row.created_by_id
        for row in rows
        if row.created_by_id and not row.pending_claim_token
    }
    resolutions: dict[str, OwnerResolution] = {}
    for slug in wanted:
        owner = owners.get(slug)
        if owner:
            resolutions[slug] = OwnerResolution(owner, "recorded")
        else:
            _report_unknown(source.project_id, slug)
            resolutions[slug] = OwnerResolution(None, "unknown")
    return resolutions


def resolve_artifact_owner(session, source, slug: str) -> OwnerResolution:
    return resolve_artifact_owners(session, source, [slug])[slug]


def record_artifact_owner(
    session, project_id, slug: str, owner_user_id: str | None, *, action: str
) -> str | None:
    """Claim ``slug`` for ``owner_user_id``; the first writer wins.

    Returns the owner actually recorded, which differs from the argument when
    another writer got there first. None outside org mode or without an owner.
    """
    if _org_scope(session) is None or not owner_user_id or not project_id:
        return None
    from cowork.services.shared_resources import SharedResourceAccess

    row, _created = SharedResourceAccess(session).claim_as(
        ARTIFACT,
        artifact_resource_key(project_id, slug),
        creator_id=str(owner_user_id),
        action=action,
    )
    winner = row.created_by_id if row is not None else None
    if winner and winner != str(owner_user_id):
        logger.warning(
            "artifact_owner conflict project=%s slug=%s kept=%s rejected=%s",
            project_id, slug, winner, owner_user_id,
        )
    return winner


def rekey_artifact_owner(
    session, source, old_slug: str, new_project_id, new_slug: str, *, actor_id: str
) -> bool:
    """Carry ownership along when a task moves an artifact to another project."""
    if _org_scope(session) is None or not is_project_root(source) or not source.project_id:
        return False
    from cowork.services.shared_resources import SharedResourceAccess

    row = SharedResourceAccess(session).rekey_as(
        ARTIFACT,
        artifact_resource_key(source.project_id, old_slug),
        artifact_resource_key(new_project_id, new_slug),
        actor_id=actor_id,
    )
    return row is not None


def forget_artifact_owner(session, source, slug: str, *, actor_id: str) -> bool:
    """Drop ownership of a deleted project-root artifact. Legacy roots have none."""
    if _org_scope(session) is None or not is_project_root(source) or not source.project_id:
        return False
    from cowork.services.shared_resources import SharedResourceAccess

    return SharedResourceAccess(session).delete_as(
        ARTIFACT, artifact_resource_key(source.project_id, slug), actor_id=actor_id
    )


def provenance_origin(folder: Path) -> UUID | None:
    """``provenance[0].conversation`` of one artifact folder, or None.

    Opened with O_NOFOLLOW and checked with lstat so a link planted on the
    shared mount cannot point the server at a file outside the project.
    """
    try:
        if not stat.S_ISDIR(Path(folder).lstat().st_mode):
            return None
        fd = os.open(Path(folder) / "metadata.json", os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    from cowork.services.artifacts import origin_conversation_id

    raw = origin_conversation_id(data if isinstance(data, dict) else None)
    try:
        return UUID(raw) if raw else None
    except ValueError:
        return None


def turn_created_slugs(
    base,
    before_slugs: set[str],
    conversation_id,
    *,
    after: set[str] | None = None,
    accept_unattributed: bool = False,
) -> set[str]:
    """Slugs that appeared during this turn AND name it as their creator.

    A drop-only filter: every conversation in a project shares one artifacts
    root, so a folder that merely appeared may be a concurrent sibling turn's
    (ENG-1933). Anton's artifact tools record the cowork conversation id first
    in ``provenance``; rewriting it can only remove a slug from this turn, never
    add one, because the slug must also have appeared in this turn's window.

    ``accept_unattributed`` is for a turn that did not finish cleanly (Stop,
    cancel, failure). Anton writes ``metadata.json`` with an empty provenance
    first and appends this turn's entry right after, so a turn cut short in
    between leaves a folder with no provenance at all; dropping it would leave
    it ownerless forever. Such a slug is then kept, while one naming another
    conversation is still dropped. The only way this claims a sibling's folder
    is a Stop that coincides with a parallel sibling turn caught in that same
    window, which is why a cleanly completed turn never accepts one.

    ``after`` is the caller's post-turn listing of ``base``; given, the root is
    not listed again.

    Never raises: it runs inside a turn's ``finally``.
    """
    try:
        expected = UUID(str(conversation_id))
        if after is None:
            from cowork.services.task_objects import snapshot_artifact_slugs

            after = snapshot_artifact_slugs(base)
        appeared = set(after) - set(before_slugs or ())
        kept: set[str] = set()
        for slug in sorted(appeared):
            origin = provenance_origin(Path(base) / slug)
            if origin == expected:
                kept.add(slug)
                continue
            if origin is None and accept_unattributed:
                logger.info(
                    "artifact_attribution accepted slug=%s reason=unfinished_turn", slug
                )
                kept.add(slug)
                continue
            reason = "no_provenance" if origin is None else "foreign_provenance"
            logger.info("artifact_attribution skipped slug=%s reason=%s", slug, reason)
        return kept
    except Exception:
        # ERROR, not WARNING: WARNING is kept for write-time conflicts and the
        # backfill summary, and reaching this means a bug or an unreadable root.
        logger.error("artifact attribution failed; claiming nothing this turn", exc_info=True)
        return set()
