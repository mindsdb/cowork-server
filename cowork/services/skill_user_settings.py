"""The ``skill_user_setting`` table's reads and writes (ENG-3787).

A person's own on/off per skill (cowork.models.skill_user_setting): a skill
they turned off is left out of their turns only. Every call takes the
request's or the unit's ScopedSession, which keys the rows by the scope's org
where skills are partitioned; the person is the scope's (setting_user_id).
Reads are one query for every skill. Writes are staged and flushed, and the
caller commits, except set_enabled_for_me, which is a change of its own.
"""

from __future__ import annotations

from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.skill_user_setting import SkillUserSetting

# The desktop's one person, and the operator view of a deployment, which
# name nobody in their scope.
LOCAL_USER = "local"


def setting_user_id(scope: TenantScope | None) -> str | None:
    """Whose settings ``scope`` reads and writes: its person, else the
    desktop's one person where skills are not partitioned, else None (a
    partitioned scope with nobody in it has no settings of its own)."""
    if scope is None:
        return LOCAL_USER
    if scope.user_id:
        return scope.user_id
    if scope.partitioned_for(PersonalResource.SKILLS):
        return None
    return LOCAL_USER


def _mine(session: ScopedSession, user_id: str):
    return session.select(SkillUserSetting).where(SkillUserSetting.user_id == user_id)


def off_for_me(session: ScopedSession | None) -> frozenset[str]:
    """The slugs the session's person turned off for themselves, in one
    query. Empty without a session or a person."""
    if session is None:
        return frozenset()
    user_id = setting_user_id(session.scope)
    if user_id is None:
        return frozenset()
    rows = session.exec(_mine(session, user_id).where(SkillUserSetting.enabled.is_(False))).all()
    return frozenset(row.skill_slug for row in rows)


def stage_enabled_for_me(session: ScopedSession, slug: str, enabled: bool) -> bool:
    """Stage the session's person's setting for ``slug``: a row for off, none
    for on (the default). False, staging nothing, when it already reads so.
    PermissionError for a scope with no person."""
    user_id = setting_user_id(session.scope)
    if user_id is None:
        raise PermissionError("A personal skill setting needs a signed-in person.")
    row = session.exec(_mine(session, user_id).where(SkillUserSetting.skill_slug == slug)).first()
    if enabled:
        if row is None:
            return False
        session.delete(row)
        session.flush()
        return True
    if row is not None and not row.enabled:
        return False
    row = row or SkillUserSetting(user_id=user_id, skill_slug=slug)
    row.enabled = False
    session.add(row)
    session.flush()
    return True


def set_enabled_for_me(session: ScopedSession, slug: str, enabled: bool) -> None:
    """stage_enabled_for_me, committed."""
    try:
        if stage_enabled_for_me(session, slug, enabled):
            session.commit()
    except Exception:
        session.rollback()
        raise


def _every_persons(session: ScopedSession, slug: str) -> list[SkillUserSetting]:
    return list(
        session.exec(
            session.select(SkillUserSetting).where(SkillUserSetting.skill_slug == slug)
        ).all()
    )


def stage_rename(session: ScopedSession, old_slug: str, new_slug: str) -> None:
    """Move every person's setting for ``old_slug`` to ``new_slug``, dropping
    any stale row the new name still had, as skill_access's stage_rename."""
    if old_slug == new_slug:
        return
    moving = _every_persons(session, old_slug)
    stale = _every_persons(session, new_slug)
    for row in stale:
        session.delete(row)
    if stale:
        session.flush()
    for row in moving:
        row.skill_slug = new_slug
        session.add(row)
    session.flush()


def stage_delete(session: ScopedSession, slug: str) -> None:
    """Remove every person's setting for ``slug``."""
    rows = _every_persons(session, slug)
    for row in rows:
        session.delete(row)
    if rows:
        session.flush()
