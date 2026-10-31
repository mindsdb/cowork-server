"""The sharing an artifact's owner last chose (cowork.models.artifact_owner_choice).

Written only by the owner's own save of the artifact's sharing
(publish_artifact's ``owner_save``), never from the artifact's .published.json
entry, which an agent can rewrite. Read before
every other re-publish in MindsHub mode (live sync, autopublish), which is
refused when it would open the artifact wider than its owner chose
(republish_widening): a wider mode than theirs (public over password over
restricted), or published while the record says unpublished. A save always
publishes, and MindsHub mode has no unpublish of its own besides deleting the
artifact, so a record says unpublished only where a later unpublish route
records it.

An artifact the owner hasn't saved since the record existed has none, and its
re-publishes are judged as before: by the restricted rule MindsHub holds
(cowork.services.publish._access_mode_refusal).

Each call opens its own session for the scope's organization
(cowork.services.artifact_authorization_identity.authorization_session), as
the publish that calls it runs in a worker thread. A database failure is
ArtifactAccessUnavailable, so a publish that can't read or write the record
fails, retryably, before anything is uploaded or reported as saved.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import NamedTuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.artifact_owner_choice import (
    ARTIFACT_KEY_MAX_CHARS,
    AccessMode,
    ArtifactOwnerChoice,
)
from cowork.services.artifact_access import ArtifactAccessUnavailable
from cowork.services.artifact_authorization_identity import authorization_session
from cowork.services.mindshub_client import canonical_uuid

# How far each mode opens the link: a mode is wider than another when its
# number is higher.
_BREADTH = {AccessMode.RESTRICTED: 0, AccessMode.PASSWORD: 1, AccessMode.PUBLIC: 2}


class OwnerChoice(NamedTuple):
    """What the owner chose: the mode, and whether the artifact is published."""

    access_mode: AccessMode
    published: bool

    def narrowest(self, other: OwnerChoice) -> OwnerChoice:
        """The narrower of the two in each respect: the narrower mode, and
        unpublished when either is."""
        mode = min(self.access_mode, other.access_mode, key=_BREADTH.__getitem__)
        return OwnerChoice(access_mode=mode, published=self.published and other.published)


class Widening(StrEnum):
    """How a publish would open an artifact wider than its owner chose."""

    # The owner unpublished it, and this publishes it.
    UNPUBLISHED_BY_OWNER = "unpublished_by_owner"
    # This publishes it in a wider mode than the owner's.
    WIDER_THAN_OWNER_CHOICE = "wider_than_owner_choice"


def widening(*, mode: AccessMode, choice: OwnerChoice) -> Widening | None:
    """How a publish in ``mode`` opens the artifact wider than ``choice``, or
    None when it doesn't."""
    if not choice.published:
        return Widening.UNPUBLISHED_BY_OWNER
    if _BREADTH[mode] > _BREADTH[choice.access_mode]:
        return Widening.WIDER_THAN_OWNER_CHOICE
    return None


def _keyable(key: str) -> bool:
    return 0 < len(key) <= ARTIFACT_KEY_MAX_CHARS


def read_owner_choice(*, keys: Iterable[str], scope: TenantScope) -> OwnerChoice | None:
    """The narrowest choice recorded under any of ``keys``, or None when none
    is recorded. A failed read raises ArtifactAccessUnavailable, so a
    re-publish never guesses."""
    wanted = [key for key in dict.fromkeys(keys) if _keyable(key)]
    if not wanted:
        return None
    try:
        with authorization_session(scope) as session:
            rows = session.exec(
                session.select(ArtifactOwnerChoice).where(ArtifactOwnerChoice.artifact_key.in_(wanted))
            ).all()
    except SQLAlchemyError as exc:
        raise ArtifactAccessUnavailable("Could not read the sharing the artifact's owner chose") from exc
    choices = [
        OwnerChoice(access_mode=AccessMode(row.access_mode), published=bool(row.published))
        for row in rows
    ]
    if not choices:
        return None
    narrowest = choices[0]
    for choice in choices[1:]:
        narrowest = narrowest.narrowest(choice)
    return narrowest


def republish_widening(
    *, keys: Iterable[str], mode: AccessMode, scope: TenantScope
) -> Widening | None:
    """How a re-publish that isn't the owner's save, in ``mode``, would open
    the artifact wider than its owner chose, judged against the narrowest
    choice recorded under any key its rule could sit under (``keys``: the
    artifact's identity first, then each key its entry recorded, which the
    pod can write but which only ever adds a narrower record). None when it
    wouldn't, or when no choice is recorded."""
    choice = read_owner_choice(keys=keys, scope=scope)
    return None if choice is None else widening(mode=mode, choice=choice)


def narrow_before_save(*, key: str, chosen: OwnerChoice, scope: TenantScope) -> None:
    """Before the owner's save uploads anything, narrow the record to
    ``chosen`` where it is wider; record_owner_choice records ``chosen``
    once the save is live. So the record is never wider than what the owner
    chose or what is live, whichever way the upload ends. Nothing while no
    choice is recorded."""
    stored = read_owner_choice(keys=[key], scope=scope)
    if stored is None:
        return
    narrowed = stored.narrowest(chosen)
    if narrowed != stored:
        record_owner_choice(key=key, choice=narrowed, scope=scope)


def record_owner_choice(*, key: str, choice: OwnerChoice, scope: TenantScope) -> None:
    """Record ``choice`` as the owner's for the artifact under ``key``,
    replacing what was there. Nothing for a key the column can't hold, as a
    legacy loose file's service-generated key may be: that artifact keeps no
    record. A failed write raises ArtifactAccessUnavailable."""
    if not _keyable(key):
        return
    try:
        with authorization_session(scope) as session:
            _stage_choice(session, key=key, choice=choice)
            try:
                session.commit()
            except IntegrityError:
                # Another save of the same artifact inserted its row first;
                # this one replaces it.
                session.rollback()
                _stage_choice(session, key=key, choice=choice)
                session.commit()
    except SQLAlchemyError as exc:
        raise ArtifactAccessUnavailable("Could not record the sharing the artifact's owner chose") from exc


def _owner_user_id(scope: TenantScope) -> UUID:
    """The owner making the choice: the scope's person. Only the owner's own
    save records a choice, and it always names them."""
    owner = canonical_uuid(scope.user_id) if scope.user_id else None
    if owner is None:
        raise ArtifactAccessUnavailable("Artifact ownership could not be established")
    return UUID(owner)


def _stage_choice(session: ScopedSession, *, key: str, choice: OwnerChoice) -> None:
    owner_user_id = _owner_user_id(session.scope)
    row = session.exec(
        session.select(ArtifactOwnerChoice).where(ArtifactOwnerChoice.artifact_key == key)
    ).first()
    if row is None:
        row = ArtifactOwnerChoice(
            artifact_key=key, access_mode=choice.access_mode.value, published=choice.published,
            owner_user_id=owner_user_id,
        )
    row.access_mode = choice.access_mode.value
    row.published = choice.published
    row.owner_user_id = owner_user_id
    session.add(row)
