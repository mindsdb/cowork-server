"""The ``skill_access`` table's reads and writes.

MindsHub mode only: who each skill of an org's store is shared with
(cowork.models.skill_access.SkillAccess; the rules are
cowork.services.skill_sharing's). Every call takes the request's or the unit's
ScopedSession, which keys the rows by the scope's org. Writes are staged and
flushed, never committed: the caller commits them with the attribution and
audit rows of the same change (cowork.services.shared_resources), or rolls
them back with it. The one exception is backfill_legacy_access, which commits
in a session of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from cowork.db.scoped import ScopedSession
from cowork.models.setting import Setting
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.skill_access import SkillAccess
from cowork.services.mindshub_client import canonical_uuid
from cowork.services.shared_resources import SKILL
from cowork.services.skill_sharing import SkillAccessRule

# The settings row that marks an org's skills from before sharing as given
# their rows (backfill_legacy_access). It lives in the settings table, not
# beside the rows it vouches for, so the revision that creates skill_access
# (c5a8e2f1d7b3) deletes it on downgrade along with the rows.
BACKFILL_SETTING_KEY = "_skill_access_backfill_v1"


def access_rows(session: ScopedSession) -> dict[str, SkillAccess]:
    """Every row of the scope's org, by skill slug: one query, for a listing
    or a turn that reads many skills. Read from the database even when the
    session already holds the rows, so a check made under a skill's lock sees
    a change committed while it waited."""
    return {
        row.skill_slug: row
        for row in session.exec(
            session.select(SkillAccess).execution_options(populate_existing=True)
        ).all()
    }


def access_row(session: ScopedSession, slug: str) -> SkillAccess | None:
    """``slug``'s row, read from the database as access_rows reads them."""
    return session.exec(
        session.select(SkillAccess)
        .where(SkillAccess.skill_slug == slug)
        .execution_options(populate_existing=True)
    ).first()


def _owner_column(owner_user_id: str | None) -> UUID | None:
    """``owner_user_id`` as the owner column holds it: a UUID, or None for a
    skill nobody owns. A ValueError for an id that isn't a UUID: every owner
    is a scope's person, which the identity middleware parsed as one."""
    return UUID(owner_user_id) if owner_user_id else None


def stage_access(session: ScopedSession, slug: str, rule: SkillAccessRule) -> SkillAccess:
    """Write ``rule`` as ``slug``'s row, creating it when there is none.
    ``always_on`` is dropped from a rule shared with neither everyone nor a
    team."""
    rule = rule.normalized()
    row = access_row(session, slug) or SkillAccess(skill_slug=slug)
    row.owner_user_id = _owner_column(rule.owner_user_id)
    row.org_allowed = rule.org_allowed
    row.allowed_team_ids = list(rule.allowed_team_ids)
    row.allowed_user_ids = list(rule.allowed_user_ids)
    row.always_on = rule.always_on
    session.add(row)
    session.flush()
    return row


def stage_only_owner(session: ScopedSession, *, slug: str, owner_user_id: str | None) -> SkillAccess:
    """A new skill's row: shared with nobody but ``owner_user_id``. Replaces
    whatever a skill of the same name left behind."""
    return stage_access(session, slug, SkillAccessRule(owner_user_id=owner_user_id))


def stage_rename(session: ScopedSession, *, old_slug: str, new_slug: str) -> None:
    """Move ``old_slug``'s row to ``new_slug``, dropping any stale row the
    new name still had. Nothing to move for a skill with no row."""
    if old_slug == new_slug:
        return
    row = access_row(session, old_slug)
    if row is None:
        return
    stale = access_row(session, new_slug)
    if stale is not None:
        session.delete(stale)
        session.flush()
    row.skill_slug = new_slug
    session.add(row)
    session.flush()


def stage_delete(session: ScopedSession, slug: str) -> None:
    row = access_row(session, slug)
    if row is not None:
        session.delete(row)
        session.flush()


def _backfilled(session: ScopedSession) -> bool:
    return (
        session.exec(
            session.select(Setting).where(
                Setting.key == BACKFILL_SETTING_KEY,
                Setting.scope == "org",
                Setting.org_id == session.scope.org_id,
            )
        ).first()
        is not None
    )


def _insert_once(session: Session, row: Setting | SkillAccess) -> None:
    """Insert ``row`` in its own transaction; a unique key that already holds
    one (another replica's backfill, or a create that got there first) wins."""
    try:
        session.add(row)
        session.commit()
    except IntegrityError:
        session.rollback()


def backfill_legacy_access(session: ScopedSession, list_slugs: Callable[[], Iterable[str]]) -> None:
    """Give every skill of the org's store that predates sharing its row,
    once per org, before its rows are first read.

    Each slug of ``list_slugs()`` with no row gets one shared with everyone
    (what every member saw before sharing), owned by the creator its
    SharedResourceAttribution names, or by nobody when it names none or one
    that isn't a UUID. A slug whose create is still in flight (a pending claim)
    is left to that create, which commits its own row with the claim
    (cowork.api.v1.endpoints.skills._record_new_skill), or to the recovery
    that finalizes it after a crash (skills._recover_skill_claim), which
    commits the same row. Then a settings row marks the org done. A folder
    with no row after that, such as a skill a server from before sharing
    created during the rollout, reads as its creator's alone, or as nobody's
    with no creator (cowork.services.skill_sharing.access_rule_for).

    Its writes commit in a session of its own, so the caller's unfinished
    work is never committed with them, and each insert is its own
    transaction, so two replicas backfilling one org at once write each row
    once.
    """
    if _backfilled(session):
        return
    with Session(session.get_bind()) as raw:
        own = ScopedSession(raw, session.scope)
        if _backfilled(own):
            return
        stored = access_rows(own)
        missing = [slug for slug in dict.fromkeys(list_slugs()) if slug not in stored]
        claims: dict[str, SharedResourceAttribution] = {}
        if missing:
            claims = {
                claim.resource_key: claim
                for claim in own.exec(
                    own.select(SharedResourceAttribution).where(
                        SharedResourceAttribution.resource_kind == SKILL,
                        SharedResourceAttribution.resource_key.in_(missing),
                    )
                ).all()
            }
        for slug in missing:
            claim = claims.get(slug)
            if claim is not None and claim.pending_claim_token:
                continue
            owner = canonical_uuid(claim.created_by_id) if claim is not None and claim.created_by_id else None
            _insert_once(
                raw,
                SkillAccess(
                    org_id=UUID(session.scope.org_id),
                    skill_slug=slug,
                    owner_user_id=_owner_column(owner),
                    org_allowed=True,
                ),
            )
        _insert_once(
            raw,
            Setting(key=BACKFILL_SETTING_KEY, value="1", scope="org", org_id=session.scope.org_id),
        )
