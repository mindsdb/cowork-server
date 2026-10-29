"""Who sees, uses and changes a skill in hosted org mode (ENG-3787).

In an org store every member reads one folder, so sharing is recorded on each
skill, in its SKILL.md ``metadata`` (cowork.models.skill META_SHARE_*):
org-wide, a list of team ids, and a list of emails. Nothing set is "only me".

- Only me: the owner sees and changes it.
- People: each email's owner sees and uses it; only the owner changes it.
- Team: every member of a team it's shared with sees it and changes it.
- Org-wide: everyone sees and uses it; changing it takes ``skills.write_org``.

An org-wide or team skill is recalled on demand like any other unless it is
marked always on (META_ALWAYS_ON), which puts it in every turn's prompt of the
people it reaches. Only such a skill keeps the mark.

An org admin (cowork.principal.can_manage_org) sees, changes and shares
every skill on the skills page, but their turns use only what they'd see
without the role (SkillViewer.for_turns). A skill with no
``share_org`` key predates sharing and reads as org-wide, which is what every
member saw before. Built-in and managed skills stay visible to everyone and
read-only. Outside hosted org mode none of this applies: no viewer is built
(resolve_skill_viewer) and the store behaves as it always has.

Teams come from auth per caller (cowork.services.hub_teams). When auth can't
be asked, the caller is in no team: they see their own, org-wide and
email-shared skills, never one shared only with a team. That fails closed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

from cowork.db.scoped import TenantScope
from cowork.models.skill import (
    META_ALWAYS_ON,
    META_SHARE_EMAILS,
    META_SHARE_ORG,
    META_SHARE_TEAMS,
    Skill,
)
from cowork.principal import Principal, can_manage_org
from cowork.services.hub_teams import (
    SKILLS_SHARE_ORG,
    SKILLS_WRITE_ORG,
    fetch_teams_and_permissions,
)

SkillSection = Literal["mine", "shared_with_me", "team", "org"]

# Deliberately loose: one @, no spaces, a dot in the domain. Auth is the
# source of truth for who an address names.
_EMAIL_RE = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")
_MAX_EMAILS = 200
_MAX_TEAMS = 100


def sharing_applies(scope: TenantScope | None) -> bool:
    """Whether skills carry per-skill sharing: hosted org tenancy only."""
    return scope is not None and scope.org_mode


def _split(raw: str | None) -> tuple[str, ...]:
    return tuple(part for part in (s.strip() for s in (raw or "").split(",")) if part)


@dataclass(frozen=True)
class SkillSharing:
    """Who one skill is shared with. The default is only its owner.
    ``always_on`` puts an org-wide or team skill in every turn's prompt."""

    org: bool = False
    team_ids: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    always_on: bool = False

    @property
    def broad(self) -> bool:
        """Shared org-wide or with a team: the skills that can be always on."""
        return self.org or bool(self.team_ids)

    @classmethod
    def of(cls, skill: Skill) -> SkillSharing:
        """The sharing ``skill`` records; org-wide when it records none."""
        metadata = skill.metadata or {}
        raw_org = metadata.get(META_SHARE_ORG)
        return cls(
            org=True if raw_org is None else raw_org.strip().lower() == "true",
            team_ids=_split(metadata.get(META_SHARE_TEAMS)),
            emails=_split(metadata.get(META_SHARE_EMAILS)),
            always_on=(metadata.get(META_ALWAYS_ON) or "").strip().lower() == "true",
        )

    def apply(self, metadata: dict[str, str]) -> None:
        """Write this sharing into ``metadata``. ``share_org`` is always
        written, so the skill never reads as legacy org-wide again.
        ``always_on`` is kept only on a skill shared org-wide or with a team;
        anything else drops it."""
        metadata[META_SHARE_ORG] = "true" if self.org else "false"
        for key, values in ((META_SHARE_TEAMS, self.team_ids), (META_SHARE_EMAILS, self.emails)):
            if values:
                metadata[key] = ",".join(values)
            else:
                metadata.pop(key, None)
        if self.always_on and self.broad:
            metadata[META_ALWAYS_ON] = "true"
        else:
            metadata.pop(META_ALWAYS_ON, None)


ONLY_ME = SkillSharing()


def normalize_emails(values: Iterable[str]) -> tuple[str, ...]:
    """Lowercased, stripped and deduplicated in first-seen order. Raises
    ValueError naming the first address that doesn't look like one."""
    seen: dict[str, None] = {}
    for raw in values:
        email = str(raw).strip().lower()
        if not email:
            continue
        if not _EMAIL_RE.match(email):
            raise ValueError(f"{raw!r} is not a valid email address.")
        seen.setdefault(email, None)
    if len(seen) > _MAX_EMAILS:
        raise ValueError(f"A skill can be shared with at most {_MAX_EMAILS} people.")
    return tuple(seen)


def normalize_team_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Stripped and deduplicated in first-seen order. A comma would corrupt
    the stored list, so it is refused."""
    seen: dict[str, None] = {}
    for raw in values:
        team_id = str(raw).strip()
        if not team_id:
            continue
        if "," in team_id:
            raise ValueError(f"{raw!r} is not a valid team id.")
        seen.setdefault(team_id, None)
    if len(seen) > _MAX_TEAMS:
        raise ValueError(f"A skill can be shared with at most {_MAX_TEAMS} teams.")
    return tuple(seen)


@dataclass(frozen=True)
class SkillViewer:
    """One caller in hosted org mode, as skill sharing needs them."""

    user_id: str | None
    email: str = ""
    team_ids: frozenset[str] = frozenset()
    # False when auth couldn't list the caller's teams: then team_ids is empty
    # and no team-only skill is visible (fail closed).
    teams_known: bool = False
    is_admin: bool = False
    # Both already true for an admin.
    can_share_org: bool = False
    can_write_org: bool = False

    def owns(self, owner_id: str | None) -> bool:
        return bool(owner_id and self.user_id and owner_id == self.user_id)

    def in_team_of(self, sharing: SkillSharing) -> bool:
        return not self.team_ids.isdisjoint(sharing.team_ids)

    def sees(self, skill: Skill) -> bool:
        """Owner, org-wide, one of the caller's teams, or the caller's email;
        an admin sees every skill."""
        sharing = SkillSharing.of(skill)
        return (
            self.is_admin
            or self.owns(skill.created_by)
            or sharing.org
            or self.in_team_of(sharing)
            or bool(self.email and self.email.lower() in sharing.emails)
        )

    def for_turns(self) -> SkillViewer:
        """This viewer as their turns select skills: an admin's role grants
        them every skill on the skills page, never another member's private
        skills in their agent."""
        return replace(self, is_admin=False) if self.is_admin else self

    def can_edit(self, skill: Skill, *, owner_id: str | None) -> bool:
        """Change the content, enable or disable, rename or delete. Org-wide
        takes skills.write_org, even for its owner; a team member changes a
        skill shared with their team; people-shared recipients read only."""
        if self.is_admin:
            return True
        sharing = SkillSharing.of(skill)
        if sharing.org:
            return self.can_write_org
        return self.owns(owner_id) or self.in_team_of(sharing)

    def can_manage_sharing(self, *, owner_id: str | None) -> bool:
        """Change who a skill is shared with: its owner or an admin. Whether
        they may also set or clear org-wide is can_share_org's question."""
        return self.is_admin or self.owns(owner_id)

    def section(self, skill: Skill, *, owner_id: str | None) -> SkillSection:
        """The bucket the skills page lists ``skill`` under for this caller."""
        if self.owns(owner_id):
            return "mine"
        sharing = SkillSharing.of(skill)
        if sharing.org:
            return "org"
        if self.in_team_of(sharing):
            return "team"
        return "shared_with_me"


def fail_closed_viewer(scope: TenantScope | None) -> SkillViewer:
    """A viewer for a caller whose teams and permissions are unknown: their
    own skills, org-wide ones, and nothing team-only."""
    return SkillViewer(user_id=scope.user_id if scope is not None else None)


async def resolve_skill_viewer(
    *, scope: TenantScope | None, principal: Principal | None, bearer_token: str
) -> SkillViewer | None:
    """The caller as skill sharing sees them, or None outside hosted org mode.

    Teams and permissions are read from auth with the caller's own bearer
    (cowork.principal.hub_credential), cached per caller. Admin standing is
    the gateway's forwarded role, the same check every shared resource uses
    (cowork.services.shared_resources.SharedResourceAccess.is_admin).
    """
    if not sharing_applies(scope):
        return None
    matches = (
        principal is not None
        and principal.user_id == scope.user_id
        and principal.org_id == scope.org_id
    )
    is_admin = matches and can_manage_org(principal)
    teams, permissions = await fetch_teams_and_permissions(
        bearer_token=bearer_token, org_id=scope.org_id or "", user_id=scope.user_id or ""
    )
    return SkillViewer(
        user_id=scope.user_id,
        email=(principal.email.strip().lower() if matches and principal.email else ""),
        team_ids=teams.team_ids,
        teams_known=teams.reachable,
        is_admin=is_admin,
        can_share_org=is_admin or SKILLS_SHARE_ORG in permissions.permissions,
        can_write_org=is_admin or SKILLS_WRITE_ORG in permissions.permissions,
    )
