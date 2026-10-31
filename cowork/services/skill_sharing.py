"""Who sees, uses and changes a skill in MindsHub mode.

In an org store every member reads one folder, so who each skill is shared
with is recorded beside it, in the ``skill_access`` table
(cowork.models.skill_access.SkillAccess, read and written through
cowork.services.skill_access). Its names mirror MindsHub's artifact rule
(cowork.services.sharing_contract.AudienceField): ``org_allowed``,
``allowed_team_ids`` and ``allowed_user_ids`` (user ids of people in the
org). Nothing set is "Only me".

- Only me: the owner sees and changes it.
- Specific people: each one sees and uses it; only the owner changes it.
- A team: every member of a team it's shared with sees it and changes it.
- Everyone at the org (``org_allowed``): everyone sees and uses it; changing
  it takes being its owner or ``skills.write_org``, and sharing a skill with
  everyone takes ``skills.share_org``.

A skill shared with everyone or with a team is recalled on demand like any
other unless it is marked ``always_on``, which puts it in every turn's prompt
of the org or the team it reaches. It stays on demand for people named
personally. Only such a skill keeps the mark (SkillAccessRule.normalized).

An org admin (cowork.principal.can_manage_org) sees, changes and shares
every skill on the skills page, but their turns use only what they'd see
without the role (SkillViewer.for_turns). Every skill of the store has a row:
a new skill gets one with its folder, and a skill from before sharing gets
one shared with everyone, owned by its recorded creator, the first time its
org's rows are read (cowork.services.skill_access.backfill_legacy_access). A
folder with no row after that is its recorded creator's alone, or nobody's
when it has none, so only an admin sees it (access_rule_for).
Built-in and managed skills stay visible to everyone and read-only. Outside
MindsHub mode none of this applies: no viewer is built (resolve_skill_viewer)
and the store behaves as it always has.

The org's default team (MindsHub's ``is_default``) has everyone in the
organization as its members, so sharing with it IS sharing with everyone: a
share naming it is stored as ``org_allowed`` (and takes
``skills.share_org``), and a row that already lists it among its teams reads
as ``org_allowed`` (SkillViewer.effective). Artifacts read it the same way
(default_team_as_org).

Teams come from MindsHub per caller (cowork.services.mindshub_teams). When it
can't be asked, the caller is in no team: they see their own skills, the ones
shared with everyone and the ones naming them, never one shared only with a
team. That fails closed. A change whose answer rests on what it couldn't be
asked is a 503 ``mindshub_unreachable``
(cowork.services.mindshub_client.MindsHubUnreachable), never a 403 that
blames the caller.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from fastapi import HTTPException, Request, status

from cowork.db.scoped import TenantScope
from cowork.principal import Principal, can_manage_org, caller_bearer
from cowork.schemas.skills import SkillSection
from cowork.services.mindshub_client import (
    PEOPLE_UNREACHABLE_REFUSAL,
    MindsHubUnreachable,
    canonical_uuid,
)
from cowork.services.mindshub_people import org_members_by_ids
from cowork.services.mindshub_teams import fetch_teams_and_permissions
from cowork.services.sharing_contract import (
    ARTIFACTS_SHARE_ORG,
    MAX_PEOPLE_PER_RULE,
    MAX_TEAMS_PER_RULE,
    SKILLS_SHARE_ORG,
    SKILLS_WRITE_ORG,
)

if TYPE_CHECKING:
    from cowork.models.skill_access import SkillAccess

# Which of some user ids (canonical UUIDs) are members of the caller's active
# org, or None when MindsHub couldn't be asked
# (cowork.services.mindshub_people.org_members_by_ids).
OrgMemberLookup = Callable[[Iterable[str]], Awaitable[frozenset[str] | None]]


def get_org_member_lookup(request: Request) -> OrgMemberLookup:
    """The org membership check, asked with the Authorization the gateway
    validated (cowork.principal.caller_bearer)."""
    bearer_token = caller_bearer(request)

    async def lookup(ids: Iterable[str]) -> frozenset[str] | None:
        members = await org_members_by_ids(bearer_token=bearer_token, ids=ids)
        if members is None:
            return None
        return frozenset(found for m in members if (found := canonical_uuid(m.id)) is not None)

    return lookup


async def members_checked(
    members: OrgMemberLookup, *, added: Iterable[str], kept: Iterable[str] = (), refusal: str
) -> frozenset[str]:
    """The org members among ``added`` and ``kept``, asked in one batched
    lookup (MindsHub's ``?ids=``), for a skill or an artifact a save shares
    with people. 400 naming every added id outside the org, after
    ``refusal``; 503 ``mindshub_unreachable`` while the lookup can't be
    asked. A kept id the answer leaves out is the caller's to drop."""
    added = list(added)
    found = await members([*added, *kept])
    if found is None:
        raise MindsHubUnreachable(PEOPLE_UNREACHABLE_REFUSAL)
    outside = [u for u in added if u not in found]
    if outside:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{refusal}: {', '.join(outside)}"
        )
    return found


def sharing_applies(scope: TenantScope | None) -> bool:
    """Whether a scope's skills and artifacts carry per-item sharing: MindsHub
    mode only. The scope's half of the rule; SkillService.org_sharing is a
    store's (the general skills store, in such a scope)."""
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
        """Shared with everyone or with a team: the skills that can be always
        on."""
        return self.org_allowed or bool(self.allowed_team_ids)

    @classmethod
    def from_row(cls, row: SkillAccess) -> SkillAccessRule:
        return cls(
            org_allowed=bool(row.org_allowed),
            allowed_team_ids=tuple(row.allowed_team_ids or ()),
            allowed_user_ids=tuple(row.allowed_user_ids or ()),
            always_on=bool(row.always_on),
            owner_user_id=str(row.owner_user_id) if row.owner_user_id is not None else None,
        ).normalized()

    def normalized(self) -> SkillAccessRule:
        """``always_on`` kept only on a skill shared with everyone or with a
        team."""
        return self if not self.always_on or self.broad else replace(self, always_on=False)

    def with_default_team_as_org(self, default_team_id: str) -> SkillAccessRule:
        """The org's default team named among the teams means everyone: drop
        it from the teams and set ``org_allowed`` (default_team_as_org)."""
        teams = default_team_as_org(self.allowed_team_ids, default_team_id)
        if not teams.named_default:
            return self
        return replace(self, org_allowed=True, allowed_team_ids=teams.team_ids)


class TeamsWithoutDefault(NamedTuple):
    """Teams a share names, the org's default team taken out
    (default_team_as_org)."""

    team_ids: tuple[str, ...]
    # Whether the share named the default team: sharing with everyone.
    named_default: bool


def default_team_as_org(team_ids: Iterable[str], default_team_id: str) -> TeamsWithoutDefault:
    """``team_ids`` without the org's default team, and whether they named
    it. Its members are everyone in the organization, so naming it is sharing with
    everyone (``org_allowed``), for a skill and an artifact alike. Nothing
    is dropped while the default team is unknown (``default_team_id``
    empty)."""
    teams = tuple(team_ids)
    if not default_team_id or default_team_id not in teams:
        return TeamsWithoutDefault(team_ids=teams, named_default=False)
    return TeamsWithoutDefault(team_ids=tuple(t for t in teams if t != default_team_id), named_default=True)


# Everyone, on demand: a packaged built-in, which has no row.
BUILTIN_RULE = SkillAccessRule(org_allowed=True)


def access_rule_for(row: SkillAccess | None, creator_id: str | None = None) -> SkillAccessRule:
    """A skill's access: its row's. Every skill from before sharing got its
    row from the backfill (cowork.services.skill_access.backfill_legacy_access)
    and a new skill gets one when its create commits, so a folder with no row
    is one a server from before sharing created during the rollout, a create
    still in flight or cut short by a crash, or one written around the API.
    With no row it is shared with nobody but ``creator_id``, the creator its
    SharedResourceAttribution records: the row its create commits, or would
    have. With neither it is nobody's, so only an admin sees it."""
    if row is not None:
        return SkillAccessRule.from_row(row)
    return SkillAccessRule(owner_user_id=creator_id)


def _normalize_ids(
    values: Iterable[str], *, what: str, plural: str, most: int, subject: str
) -> tuple[str, ...]:
    """Each value parsed as a UUID, in its canonical lowercase form, and
    deduplicated in first-seen order; a blank one is skipped. A value that
    isn't a UUID is a ValueError naming it, so a save answers 400 before any
    lookup or publish."""
    seen: dict[str, None] = {}
    for raw in values:
        if not str(raw).strip():
            continue
        value = canonical_uuid(raw)
        if value is None:
            raise ValueError(f"{raw!r} is not a valid {what} id.")
        seen.setdefault(value, None)
    if len(seen) > most:
        raise ValueError(f"{subject[:1].upper()}{subject[1:]} can be shared with at most {most} {plural}.")
    return tuple(seen)


def normalize_team_ids(values: Iterable[str], *, subject: str = "a skill") -> tuple[str, ...]:
    """Team UUIDs, canonical and deduplicated (_normalize_ids); a malformed
    id, or more than MAX_TEAMS_PER_RULE, is a ValueError saying so about
    ``subject`` ("a skill", "an artifact")."""
    return _normalize_ids(
        values, what="team", plural="teams", most=MAX_TEAMS_PER_RULE, subject=subject
    )


def normalize_user_ids(values: Iterable[str], *, subject: str = "a skill") -> tuple[str, ...]:
    """User ids, canonical and deduplicated (_normalize_ids), the form the
    gateway forwards ``sub`` in; a malformed id, or more than
    MAX_PEOPLE_PER_RULE, is a ValueError saying so about ``subject``."""
    return _normalize_ids(
        values, what="person", plural="people", most=MAX_PEOPLE_PER_RULE, subject=subject
    )


@dataclass(frozen=True)
class SkillViewer:
    """One caller in MindsHub mode, as sharing a skill or an artifact needs
    them."""

    user_id: str | None
    email: str = ""
    team_ids: frozenset[str] = frozenset()
    # False when MindsHub couldn't list the caller's teams: then team_ids is
    # empty and no team-only skill is visible (fail closed).
    teams_known: bool = False
    is_admin: bool = False
    # MindsHub's skills.share_org, skills.write_org and artifacts.share_org;
    # all three already true for an admin.
    can_share_skills_org: bool = False
    can_write_skills_org: bool = False
    can_share_artifacts_org: bool = False
    # False when MindsHub couldn't list the caller's permissions: then the
    # capabilities read false, and a refusal that rests on them is a 503
    # (MindsHubUnreachable), not a 403.
    permissions_known: bool = False
    # The org's default team (everyone in the organization), when MindsHub named one.
    default_team_id: str = ""

    def effective(self, rule: SkillAccessRule) -> SkillAccessRule:
        """``rule`` with the org's default team read as ``org_allowed``."""
        return rule.with_default_team_as_org(self.default_team_id)

    def owns(self, owner_id: str | None) -> bool:
        return bool(owner_id and self.user_id and owner_id == self.user_id)

    def in_team_of(self, rule: SkillAccessRule) -> bool:
        return not self.team_ids.isdisjoint(rule.allowed_team_ids)

    def named_in(self, rule: SkillAccessRule) -> bool:
        return bool(self.user_id) and self.user_id.lower() in rule.allowed_user_ids

    def sees(self, rule: SkillAccessRule) -> bool:
        """Owner, everyone, one of the caller's teams, or the caller's own id
        among the people; an admin sees every skill."""
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
        """Change the content, enable or disable, rename or delete. Its owner
        always may, whoever it is shared with; anyone else needs
        skills.write_org for a skill shared with everyone, or to be a member
        of a team a team skill is shared with; people it is shared with
        personally read only."""
        if self.is_admin or self.owns(rule.owner_user_id):
            return True
        rule = self.effective(rule)
        if rule.org_allowed:
            return self.can_write_skills_org
        return self.in_team_of(rule)

    def unsure_of(self, rule: SkillAccessRule) -> bool:
        """Whether a can_edit refusal of ``rule`` rests on what MindsHub
        couldn't be asked: the caller's permissions for a skill shared with
        everyone, or their teams for a team one."""
        rule = self.effective(rule)
        if rule.org_allowed:
            return not self.permissions_known
        return bool(rule.allowed_team_ids) and not self.teams_known

    def can_manage_sharing(self, rule: SkillAccessRule) -> bool:
        """Change who a skill is shared with, taking it back from everyone
        included: its owner or an admin. Whether they may also share it with
        everyone is can_share_skills_org's question."""
        return self.is_admin or self.owns(rule.owner_user_id)

    def section(self, rule: SkillAccessRule) -> SkillSection:
        """The bucket the skills page lists a skill under for this caller,
        by its audience, the owner's included: everyone (or the default
        team) is "org"; else shared with a team is "team" (for its owner, or
        a member of one of the teams); else the caller's own is "mine" (its
        owner's, even shared with people); else "shared_with_me"."""
        owner = self.owns(rule.owner_user_id)
        rule = self.effective(rule)
        if rule.org_allowed:
            return SkillSection.ORG
        if rule.allowed_team_ids and (owner or self.in_team_of(rule)):
            return SkillSection.TEAM
        if owner:
            return SkillSection.MINE
        return SkillSection.SHARED_WITH_ME


def fail_closed_viewer(scope: TenantScope | None) -> SkillViewer:
    """A viewer for a caller whose teams and permissions are unknown: their
    own skills, the ones shared with everyone, the ones naming them, and
    nothing team-only."""
    return SkillViewer(user_id=scope.user_id if scope is not None else None)


async def resolve_skill_viewer(
    *, scope: TenantScope | None, principal: Principal | None, bearer_token: str
) -> SkillViewer | None:
    """The caller as sharing sees them, or None outside MindsHub mode.

    Teams and permissions are read from MindsHub with ``bearer_token``,
    cached per caller. Callers pass the Authorization the gateway validated
    (cowork.principal.caller_bearer), never the client-set
    X-MindsHub-Authorization: MindsHub answers for whichever org that
    credential belongs to, and its answer carries no org to check it against.
    Admin standing is the gateway's forwarded role, the same check every
    shared resource uses (cowork.services.shared_resources.SharedResourceAccess.is_admin).
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
        can_share_skills_org=is_admin or SKILLS_SHARE_ORG in permissions.permissions,
        can_write_skills_org=is_admin or SKILLS_WRITE_ORG in permissions.permissions,
        can_share_artifacts_org=is_admin or ARTIFACTS_SHARE_ORG in permissions.permissions,
        permissions_known=is_admin or permissions.reachable,
        default_team_id=teams.default_team_id,
    )
