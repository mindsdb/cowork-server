"""The ``skill_access`` table's reads and writes (ENG-3787).

Hosted org mode only: who each skill of an org's store is shared with
(cowork.models.skill_access.SkillAccess; the rules are
cowork.services.skill_sharing's). Every call takes the request's or the unit's
ScopedSession, which keys the rows by the scope's org. Writes are staged and
flushed, never committed: the caller commits them with the attribution and
audit rows of the same change (cowork.services.shared_resources), or rolls
them back with it.
"""

from __future__ import annotations

from cowork.db.scoped import ScopedSession
from cowork.models.skill_access import SkillAccess
from cowork.services.skill_sharing import SkillAccessRule


def access_rows(session: ScopedSession) -> dict[str, SkillAccess]:
    """Every row of the scope's org, by skill slug: one query, for a listing
    or a turn that reads many skills."""
    return {row.skill_slug: row for row in session.exec(session.select(SkillAccess)).all()}


def access_row(session: ScopedSession, slug: str) -> SkillAccess | None:
    return session.exec(
        session.select(SkillAccess).where(SkillAccess.skill_slug == slug)
    ).first()


def stage_access(session: ScopedSession, slug: str, rule: SkillAccessRule) -> SkillAccess:
    """Write ``rule`` as ``slug``'s row, creating it when there is none.
    ``always_on`` is dropped from a rule that is neither org-wide nor shared
    with a team."""
    rule = rule.normalized()
    row = access_row(session, slug) or SkillAccess(skill_slug=slug)
    row.owner_user_id = rule.owner_user_id
    row.org_allowed = rule.org_allowed
    row.allowed_team_ids = list(rule.allowed_team_ids)
    row.allowed_user_ids = list(rule.allowed_user_ids)
    row.always_on = rule.always_on
    session.add(row)
    session.flush()
    return row


def stage_only_owner(session: ScopedSession, slug: str, owner_user_id: str | None) -> SkillAccess:
    """A new skill's row: shared with nobody but ``owner_user_id``. Replaces
    whatever a skill of the same name left behind."""
    return stage_access(session, slug, SkillAccessRule(owner_user_id=owner_user_id))


def stage_rename(session: ScopedSession, old_slug: str, new_slug: str) -> None:
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
