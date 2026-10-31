"""Who sees, uses and changes a skill in hosted org mode (ENG-3787).

In an org store every member reads one folder, so who each skill is shared
with is recorded beside it, in the ``skill_access`` table
(cowork.models.skill_access.SkillAccess, read and written through
cowork.services.skill_access). Its names mirror auth's ArtifactAccess:
``org_allowed``, ``allowed_team_ids`` and ``allowed_user_ids`` (Keycloak ids
of people in the org). Nothing set is "only me".

- Only me: the owner sees and changes it.
- People: each one sees and uses it; only the owner changes it.
- Team: every member of a team it's shared with sees it and changes it.
- Org-wide: everyone sees and uses it; changing it takes ``skills.write_org``.

An org-wide or team skill is recalled on demand like any other unless it is
marked ``always_on``, which puts it in every turn's prompt of the people it
reaches. Only such a skill keeps the mark (SkillAccessRule.normalized).

An org admin (cowork.principal.can_manage_org) sees, changes and shares
every skill on the skills page, but their turns use only what they'd see
without the role (SkillViewer.for_turns). A skill with no row and no creator
in its SKILL.md predates sharing and reads as org-wide, which is what every
member saw before; one with no row but a recorded creator is that creator's
alone (access_rule_for). Built-in and managed skills stay visible to everyone
and read-only. Outside hosted org mode none of this applies: no viewer is
built (resolve_skill_viewer) and the store behaves as it always has.

The org's default team (auth's ``is_default``) has the whole organization
as its members, so it IS org-wide: a share naming it is stored as
``org_allowed`` (and takes ``skills.share_org``), and a row that already
lists it among its teams reads as org-wide (SkillViewer.effective).

Teams come from auth per caller (cowork.services.hub_teams). When auth can't
be asked, the caller is in no team: they see their own, org-wide and
people-shared skills, never one shared only with a team. That fails closed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from cowork.db.scoped import TenantScope
from cowork.principal import Principal, can_manage_org
from cowork.services.hub_teams import (
    SKILLS_SHARE_ORG,
    SKILLS_WRITE_ORG,
    fetch_teams_and_permissions,
)

if TYPE_CHECKING:
    from cowork.models.skill import Skill
    from cowork.models.skill_access import SkillAccess

SkillSection = Literal["mine", "shared_with_me", "team", "org"]

_MAX_PEOPLE = 200
_MAX_TEAMS = 100
# A Keycloak id or a team id is a UUID; this only bounds what is stored.
_MAX_ID_CHARS = 64


def sharing_applies(scope: TenantScope | None) -> bool:
    """Whether skills carry per-skill sharing: hosted org tenancy only."""
    return scope is not None and scope.org_mode


@dataclass(frozen=True)
class SkillAccessRule:
    """Who one skill is shared with, as its ``skill_access`` row records it.
    The default is only its owner."""

    org_allowed: bool = False
    allowed_team_ids: tuple[str, ...] = ()
    allowed_user_ids: tuple[str, ...] = ()
    always_on: bool = False
    owner_user_id: str | None = None

    @property
    def broad(self) -> bool:
        """Shared org-wide or with a team: the skills that can be always on."""
        return self.org_allowed or bool(self.allowed_team_ids)

    @classmethod
    def from_row(cls, row: SkillAccess) -> SkillAccessRule:
        return cls(
            org_allowed=bool(row.org_allowed),
            allowed_team_ids=tuple(row.allowed_team_ids or ()),
            allowed_user_ids=tuple(row.allowed_user_ids or ()),
            always_on=bool(row.always_on),
            owner_user_id=row.owner_user_id,
        ).normalized()

    def normalized(self) -> SkillAccessRule:
        """``always_on`` kept only on a skill shared org-wide or with a team."""
        return self if not self.always_on or self.broad else replace(self, always_on=False)

    def with_default_team_as_org(self, default_team_id: str) -> SkillAccessRule:
        """The org's default team named among the teams means org-wide: drop
        it from the teams and set ``org_allowed``."""
        if not default_team_id or default_team_id not in self.allowed_team_ids:
            return self
        return replace(
            self,
            org_allowed=True,
            allowed_team_ids=tuple(t for t in self.allowed_team_ids if t != default_team_id),
        )


# Everyone, on demand: a skill that predates sharing, and a packaged built-in.
LEGACY_ORG_WIDE = SkillAccessRule(org_allowed=True)


def access_rule_for(row: SkillAccess | None, skill: Skill) -> SkillAccessRule:
    """``skill``'s access: its row's, else org-wide when its SKILL.md records
    no creator (a skill from before sharing: what every member saw then),
    else its creator's alone."""
    if row is not None:
        return SkillAccessRule.from_row(row)
    if skill.created_by:
        return SkillAccessRule(owner_user_id=skill.created_by)
    return LEGACY_ORG_WIDE


def _normalize_ids(values: Iterable[str], *, what: str, most: int) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if len(value) > _MAX_ID_CHARS or "," in value or any(c.isspace() for c in value):
            raise ValueError(f"{raw!r} is not a valid {what} id.")
        seen.setdefault(value, None)
    if len(seen) > most:
        raise ValueError(f"A skill can be shared with at most {most} {what}s.")
    return tuple(seen)


def normalize_team_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Stripped and deduplicated in first-seen order; a malformed id is a
    ValueError naming it."""
    return _normalize_ids(values, what="team", most=_MAX_TEAMS)


def normalize_user_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Keycloak user ids, stripped, lowercased (a UUID's canonical form, as
    the gateway forwards ``sub``) and deduplicated in first-seen order."""
    return tuple(
        dict.fromkeys(
            v.lower() for v in _normalize_ids(values, what="person", most=_MAX_PEOPLE)
        )
    )


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
    # The org's default team (whole-org membership), when auth named one.
    default_team_id: str = ""

    def effective(self, rule: SkillAccessRule) -> SkillAccessRule:
        """``rule`` with the org's default team read as org-wide."""
        return rule.with_default_team_as_org(self.default_team_id)

    def owns(self, owner_id: str | None) -> bool:
        return bool(owner_id and self.user_id and owner_id == self.user_id)

    def in_team_of(self, rule: SkillAccessRule) -> bool:
        return not self.team_ids.isdisjoint(rule.allowed_team_ids)

    def named_in(self, rule: SkillAccessRule) -> bool:
        return bool(self.user_id) and self.user_id.lower() in rule.allowed_user_ids

    def sees(self, rule: SkillAccessRule) -> bool:
        """Owner, org-wide, one of the caller's teams, or the caller's own
        id among the people; an admin sees every skill."""
        rule = self.effective(rule)
        return (
            self.is_admin
            or self.owns(rule.owner_user_id)
            or rule.org_allowed
            or self.in_team_of(rule)
            or self.named_in(rule)
        )

    def for_turns(self) -> SkillViewer:
        """This viewer as their turns select skills: an admin's role grants
        them every skill on the skills page, never another member's private
        skills in their agent."""
        return replace(self, is_admin=False) if self.is_admin else self

    def can_edit(self, rule: SkillAccessRule) -> bool:
        """Change the content, enable or disable, rename or delete. Org-wide
        takes skills.write_org, even for its owner; a team member changes a
        skill shared with their team; people-shared recipients read only."""
        if self.is_admin:
            return True
        rule = self.effective(rule)
        if rule.org_allowed:
            return self.can_write_org
        return self.owns(rule.owner_user_id) or self.in_team_of(rule)

    def can_manage_sharing(self, rule: SkillAccessRule) -> bool:
        """Change who a skill is shared with: its owner or an admin. Whether
        they may also set or clear org-wide is can_share_org's question."""
        return self.is_admin or self.owns(rule.owner_user_id)

    def section(self, rule: SkillAccessRule) -> SkillSection:
        """The bucket the skills page lists a skill under for this caller,
        by its audience, the owner's included: org-wide (or the default
        team) is "org"; else shared with a team is "team" (for its owner, or
        a member of one of the teams); else the caller's own is "mine" (its
        owner's, even shared with people); else "shared_with_me"."""
        owner = self.owns(rule.owner_user_id)
        rule = self.effective(rule)
        if rule.org_allowed:
            return "org"
        if rule.allowed_team_ids and (owner or self.in_team_of(rule)):
            return "team"
        if owner:
            return "mine"
        return "shared_with_me"


def fail_closed_viewer(scope: TenantScope | None) -> SkillViewer:
    """A viewer for a caller whose teams and permissions are unknown: their
    own skills, org-wide ones, ones naming them, and nothing team-only."""
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
        default_team_id=teams.default_team_id,
    )
