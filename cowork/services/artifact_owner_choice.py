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

The owner's save narrows the record before its upload (narrow_before_save)
and records its choice once the upload is live (record_owner_choice). When
MindsHub's upload service refuses the save before it writes anything, the
save puts back what it narrowed (restore_after_refused_save), but only while
the row still holds that narrowing's stamp: every owner save stamps the row
before its upload, so a restore never undoes another save's write, nor a
save that started since and may have changed what is live.

Each call opens its own session for the scope's organization
(cowork.services.artifact_authorization_identity.authorization_session), as
the publish that calls it runs in a worker thread. A database failure is
ArtifactAccessUnavailable, so a publish that can't read or write the record
fails, retryably, before anything is uploaded or reported as saved.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict
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
    """Whether ``key`` can be a record's key: no longer than the column, and
    an artifact id as MindsHub reads it (mindshub_artifact_id), so a key the
    pod writes into an entry with a NUL or a lone surrogate is never asked
    about."""
    from cowork.services.sharing_contract import mindshub_artifact_id

    return 0 < len(key) <= ARTIFACT_KEY_MAX_CHARS and mindshub_artifact_id(key) == key


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


class Narrowing(BaseModel):
    """What an owner's save wrote under ``key`` before its upload: the choice
    it ``replaced``, the narrower one it ``wrote``, and the ``stamp`` it
    wrote them with."""

    model_config = ConfigDict(frozen=True)

    key: str
    replaced: OwnerChoice
    wrote: OwnerChoice
    stamp: datetime


def narrow_before_save(*, key: str, chosen: OwnerChoice, scope: TenantScope) -> Narrowing | None:
    """Before the owner's save uploads anything, narrow the record to
    ``chosen`` where it is wider; record_owner_choice records ``chosen``
    once the save is live, and restore_after_refused_save puts back what
    this replaced when the upload service refuses the save before writing.
    So the record is never wider than the owner's last confirmed choice:
    narrowed while the upload is out, ``chosen`` once it is live, and what
    it held before after such a refusal. Nothing while no choice is recorded.

    The row is read and written under one lock, and stamped even when its
    mode stays, so a save that started earlier and narrowed it can tell that
    this one has started (restore_after_refused_save). The Narrowing when
    this save narrowed the record, else None. A failed read or write raises
    ArtifactAccessUnavailable."""
    if not _keyable(key):
        return None
    try:
        with authorization_session(scope) as session:
            row = session.exec(
                session.select(ArtifactOwnerChoice)
                .where(ArtifactOwnerChoice.artifact_key == key)
                .with_for_update()
            ).first()
            if row is None:
                return None
            stored = OwnerChoice(access_mode=AccessMode(row.access_mode), published=bool(row.published))
            narrowed = stored.narrowest(chosen)
            stamp = _stage_choice(session, key=key, choice=narrowed)
            session.commit()
    except SQLAlchemyError as exc:
        raise ArtifactAccessUnavailable("Could not record the sharing the artifact's owner chose") from exc
    if narrowed == stored:
        return None
    return Narrowing(key=key, replaced=stored, wrote=narrowed, stamp=stamp)


def restore_after_refused_save(*, narrowing: Narrowing, scope: TenantScope) -> None:
    """Put back the choice ``narrowing`` replaced, once the upload service
    refused the owner's save before writing anything. Only while the row
    still holds that narrowing, stamp included: another save that wrote the
    row since, even the same mode, keeps its write, and so does one that
    started since, which may have changed what is live without narrowing
    anything. A failed read or write raises ArtifactAccessUnavailable and
    leaves the record narrowed."""
    try:
        with authorization_session(scope) as session:
            row = session.exec(
                session.select(ArtifactOwnerChoice)
                .where(
                    ArtifactOwnerChoice.artifact_key == narrowing.key,
                    ArtifactOwnerChoice.access_mode == narrowing.wrote.access_mode.value,
                    ArtifactOwnerChoice.published == narrowing.wrote.published,
                    # Compared in SQL: SQLite returns the stamp without its
                    # time zone, which never equals the aware one in Python.
                    ArtifactOwnerChoice.modified_at == narrowing.stamp,
                )
                .with_for_update()
            ).first()
            if row is None:
                return
            _stage_choice(session, key=narrowing.key, choice=narrowing.replaced)
            session.commit()
    except SQLAlchemyError as exc:
        raise ArtifactAccessUnavailable("Could not restore the sharing the artifact's owner chose") from exc


def record_owner_choice(*, key: str, choice: OwnerChoice, scope: TenantScope) -> None:
    """Record ``choice`` as the owner's for the artifact under ``key``,
    replacing what was there, under the row's lock (_stage_choice), so a
    restore in progress can't put a wider mode back over it. Nothing for a
    key the column can't hold, as a legacy loose file's service-generated
    key may be: that artifact keeps no record. A failed write raises
    ArtifactAccessUnavailable."""
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


def _stage_choice(session: ScopedSession, *, key: str, choice: OwnerChoice) -> datetime:
    """Stage ``choice`` as the row under ``key``, and return the stamp it
    sets. Set here on every write, so a write of the same mode still changes
    it, which ``onupdate`` doesn't: SQLAlchemy sends no UPDATE for a row
    whose values didn't change.

    The row is read under its lock. An UPDATE carries only the columns that
    changed from what was read, so a write that read the row before a
    restore committed would keep the restore's wider mode and change only
    the stamp. Under the lock it waits for the restore and writes its mode
    over it."""
    owner_user_id = _owner_user_id(session.scope)
    row = session.exec(
        session.select(ArtifactOwnerChoice)
        .where(ArtifactOwnerChoice.artifact_key == key)
        .with_for_update()
    ).first()
    if row is None:
        row = ArtifactOwnerChoice(
            artifact_key=key, access_mode=choice.access_mode.value, published=choice.published,
            owner_user_id=owner_user_id,
        )
    stamp = datetime.now(UTC)
    row.access_mode = choice.access_mode.value
    row.published = choice.published
    row.owner_user_id = owner_user_id
    row.modified_at = stamp
    session.add(row)
    return stamp
