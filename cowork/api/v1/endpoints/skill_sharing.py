"""The sharing routes of the skills page: who a skill is shared with, what
the caller may share, and the caller's own off switch for a skill.

- ``GET /skills/sharing-capabilities``: what the caller may do with sharing
  in MindsHub mode (skill_sharing_capabilities).
- ``PUT /skills/{skill_id}/me``: the caller's own on/off for one skill they
  see, in every mode (set_skill_enabled_for_me).
- ``PUT /skills/{skill_id}/sharing``: a skill's audience in MindsHub mode
  (_set_org_skill_sharing), or one skill shared with everyone, read-only, in
  single-tenant mode (cowork.api.v1.endpoints.skills._set_personal_skill_sharing).

Mounted under ``/skills`` ahead of the skills router, so
``/sharing-capabilities`` is never read as a skill's id. Who sees, uses and
changes a skill is cowork.services.skill_sharing's.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from cowork.api.v1.endpoints.skills import (
    SkillViewerDep,
    _set_personal_skill_sharing,
    _skill_response,
)
from cowork.api.v1.permissions import Authenticated, AuthenticatedInOrgMode, require
from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.principal import Principal, get_principal
from cowork.schemas.skills import (
    SkillEnabledForMeRequest,
    SkillResponse,
    SkillSharingCapabilities,
    SkillSharingRequest,
)
from cowork.services.mindshub_client import (
    PERMISSIONS_UNREACHABLE,
    TEAMS_UNREACHABLE,
    MindsHubUnreachable,
)
from cowork.services.shared_resources import SKILL, SharedResourceAccess
from cowork.services.sharing_contract import SKILLS_SHARE_ORG
from cowork.services.skill_sharing import (
    OrgMemberLookup,
    SkillAccessRule,
    SkillViewer,
    get_org_member_lookup,
    members_checked,
    normalize_team_ids,
    normalize_user_ids,
)
from cowork.services.skill_user_settings import stage_enabled_for_me
from cowork.services.skills import (
    SkillService,
    SkillSharingError,
    builtin_skill_refusal,
    is_builtin_skill,
)

# AuthenticatedInOrgMode, as the skills router declares it: ScopedSessionDep
# already fails closed whenever MindsHub mode has no org in scope, and
# declaring it makes the requirement visible to a route walker.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

OrgMemberLookupDep = Annotated[OrgMemberLookup, Depends(get_org_member_lookup)]


@router.get("/sharing-capabilities", response_model=SkillSharingCapabilities)
def skill_sharing_capabilities(viewer: SkillViewerDep = None):
    """What the caller may do with sharing in MindsHub mode.

    ``enabled`` is MindsHub mode. The three capabilities are MindsHub's
    ``skills.share_org``, ``skills.write_org`` and ``artifacts.share_org``,
    read from the caller's entitlements, and all true for an org admin.
    ``permissions_known`` and ``teams_known`` say whether MindsHub could be
    asked for the caller's permissions and teams: while it can't, a false
    capability means "couldn't ask", and the app says so instead of showing
    a permission lock. It names no organization: the app names it from its
    own list of the caller's organizations.
    """
    if viewer is None:
        return SkillSharingCapabilities(permissions_known=True, teams_known=True).model_dump(
            by_alias=True
        )
    return SkillSharingCapabilities(
        enabled=True,
        can_share_skills_org=viewer.can_share_skills_org,
        can_write_skills_org=viewer.can_write_skills_org,
        can_share_artifacts_org=viewer.can_share_artifacts_org,
        permissions_known=viewer.permissions_known,
        teams_known=viewer.teams_known,
    ).model_dump(by_alias=True)


@router.put("/{skill_id}/me", response_model=SkillResponse)
def set_skill_enabled_for_me(
    skill_id: str,
    body: SkillEnabledForMeRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    """The caller's own on/off for one skill they see: off leaves
    it out of their turns only, whoever else reads it. Any skill the caller
    sees, built-ins and managed skills included; it needs no edit rights.
    404 for one they don't see. On (the default) keeps no row.

    Turning a skill off for everyone is the global ``enabled`` (PUT
    /skills/{slug}), which takes edit rights."""
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    service.ensure_builtin_skills()
    try:
        visible = service.get_visible_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    try:
        if stage_enabled_for_me(scoped, visible.skill.name, body.enabled):
            scoped.commit()
    except PermissionError as e:
        scoped.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except IntegrityError:
        # A concurrent request turned it off first: the row it wrote reads
        # as this one would have.
        scoped.rollback()
    except Exception:
        scoped.rollback()
        raise
    service.forget_off_for_me()
    return _skill_response(
        visible, access=SharedResourceAccess(scoped, principal), service=service
    ).model_dump(by_alias=True)


PEOPLE_OUTSIDE_ORG_REFUSAL = "Skills can only be shared with people in your organization"
SHARING_RACED_REFUSAL = "This skill's sharing changed while saving. Reload it and try again."


async def stored_sharing_rule(
    skill_id: str, scoped: ScopedSessionDep, viewer: SkillViewerDep
) -> SkillAccessRule | None:
    """The skill's stored access as ``viewer`` reads it (the default team as
    everyone), for the check PUT /{skill_id}/sharing asks MindsHub
    before it takes the skill's lock (verify_people_in_org): it only decides
    what that asks, and the route reads the rule again under the lock. None
    outside MindsHub mode, and for a skill this caller can't see or whose
    name is refused (SkillNotFoundError, a ValueError), which the route
    refuses itself. Any other failure, a database error included, raises:
    read as "no stored rule" it would ask MindsHub about every person kept
    and answer 503 for an edit that adds nobody."""
    if viewer is None:
        return None

    def read() -> SkillAccessRule | None:
        service = SkillService(scoped.scope, viewer=viewer, session=scoped)
        try:
            skill = service.get_seen_skill(skill_id)
        except ValueError:
            return None
        return viewer.effective(service.access_rule(skill))

    return await run_in_threadpool(read)


StoredRuleDep = Annotated[SkillAccessRule | None, Depends(stored_sharing_rule)]


async def verify_people_in_org(
    body: SkillSharingRequest,
    viewer: SkillViewerDep,
    stored: StoredRuleDep,
    members: OrgMemberLookupDep,
) -> frozenset[str]:
    """The user ids ``body`` adds to the skill's people, confirmed members of
    the caller's org in one batched lookup (MindsHub's ``?ids=``).

    Only new ids are asked about: one already stored is kept as is, so an
    outage doesn't block an edit that adds nobody. The caller's own id needs
    no check. 400 naming every id outside the org; 503 ``mindshub_unreachable``
    while MindsHub can't be asked. A malformed id or a skill that
    can't be read is left to _set_org_skill_sharing to refuse; it re-checks
    the added ids against this answer under the skill's lock.

    A dependency of PUT /{skill_id}/sharing rather than a step in it, so the
    route stays sync. Nothing to check outside MindsHub mode.
    """
    if viewer is None or body.allowed_user_ids is None:
        return frozenset()
    try:
        user_ids = normalize_user_ids(body.allowed_user_ids)
    except ValueError:
        return frozenset()
    kept = stored.allowed_user_ids if stored is not None else ()
    me = (viewer.user_id or "").lower()
    added = [u for u in user_ids if u not in kept and u != me]
    if not added:
        return frozenset()
    await members_checked(members, added=added, refusal=PEOPLE_OUTSIDE_ORG_REFUSAL)
    return frozenset(added)


VerifiedPeopleDep = Annotated[frozenset[str], Depends(verify_people_in_org)]


@router.put("/{skill_id}/sharing", response_model=SkillResponse)
def set_skill_sharing(
    skill_id: str,
    body: SkillSharingRequest,
    scoped: ScopedSessionDep,
    # Authenticated, beside the router's AuthenticatedInOrgMode: a person
    # shares their own skill, so a caller with no person (the operator view,
    # the desktop) has nothing to share and gets a 401 in either mode.
    principal: Principal = Depends(require(Authenticated)),
    viewer: SkillViewerDep = None,
    verified_people: VerifiedPeopleDep = frozenset(),
):
    """Change who one skill is shared with: in MindsHub mode
    _set_org_skill_sharing, elsewhere _set_personal_skill_sharing."""
    if viewer is not None:
        return _set_org_skill_sharing(skill_id, body, scoped, principal, viewer, verified_people)
    if body.shared is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="shared is required."
        )
    return _set_personal_skill_sharing(skill_id, body.shared, scoped, principal)


SHARING_OWNER_REFUSAL = (
    "Only the skill's owner or an organization admin can change who it is shared with."
)
SHARING_ORG_REFUSAL = (
    f"Sharing a skill with everyone in the organization requires the {SKILLS_SHARE_ORG} "
    "permission and being the skill's owner, or an organization admin."
)
ALWAYS_ON_UNSHARED_REFUSAL = "Only a skill shared with everyone or with a team can be always on."
ALWAYS_ON_ORG_REFUSAL = (
    "Making a skill shared with everyone always on, or taking that back, requires the "
    f"{SKILLS_SHARE_ORG} permission or an organization admin."
)
ALWAYS_ON_TEAM_REFUSAL = (
    "Only the skill's owner, a member of a team it's shared with, or an organization "
    "admin can make it always on."
)


def _set_org_skill_sharing(
    skill_id: str,
    body: SkillSharingRequest,
    scoped: ScopedSession,
    principal: Principal,
    viewer: SkillViewer,
    verified_people: frozenset[str] = frozenset(),
) -> dict:
    """MindsHub mode: set a skill's access (its skill_access row):
    ``org_allowed``, ``allowed_team_ids``, ``allowed_user_ids`` and
    ``always_on``.

    A field the body leaves out keeps what the skill has. Teams, people and
    taking a skill back from the org are the owner's or an admin's to
    change; sharing it with everyone takes ``skills.share_org`` as well. A team
    added must be one the caller belongs to (admins exempt), checked against
    their listing from MindsHub, 400 for a team they aren't in. A
    kept team is kept unchecked: a deleted team's id grants nobody anything.
    The org's default team is never stored: named among the teams it means
    everyone (default_team_as_org), so while the caller's teams can't be
    read, a change that adds a team is a 503, an admin's included. User ids
    are stripped, lowercased and deduplicated; a malformed one is a 400. An
    id added must be a member of the org, which verify_people_in_org
    confirmed before this runs (``verified_people``); one added since that
    check read the skill is a 409. A refusal that rests on teams or
    permissions MindsHub couldn't list is a 503 ``mindshub_unreachable``
    (MindsHubUnreachable).

    ``always_on`` can be set only on a skill that, after this change, is
    shared with everyone or with a team (else 400). Changing it on a skill
    shared with everyone takes ``skills.share_org``; on a team skill, edit
    rights (owner, a member of one of its teams, or admin); else 403. A
    change that leaves the skill shared with neither everyone nor a team
    clears it.

    404 for a skill the caller doesn't see or that doesn't exist, 403 for a
    built-in, a managed skill, or a change the caller may not make. Nothing
    is written when the access already reads this way. The rule is read
    again from the database under the skill's lock, so this change applies
    on top of one committed while it waited: what the body leaves out keeps
    that change's values.
    """
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    access = SharedResourceAccess(scoped, principal)
    try:
        service.refuse_unchangeable(skill_id)
        skill = service.get_seen_skill(skill_id)
        if is_builtin_skill(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        try:
            team_ids = (
                normalize_team_ids(body.allowed_team_ids)
                if body.allowed_team_ids is not None
                else None
            )
            user_ids = (
                normalize_user_ids(body.allowed_user_ids)
                if body.allowed_user_ids is not None
                else None
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        with access.coordination_lock(SKILL, skill.name):
            service.forget_access()
            skill = service.get_seen_skill(skill.name)
            stored = service.access_rule(skill)
            owner_id = stored.owner_user_id
            # The org's default team is "everyone in the org": named among
            # the teams it means everyone, with everyone's rules
            # (skills.share_org), and a stored one is rewritten as such.
            current = viewer.effective(stored)
            wanted = replace(
                current,
                org_allowed=current.org_allowed if body.org_allowed is None else body.org_allowed,
                allowed_team_ids=current.allowed_team_ids if team_ids is None else team_ids,
                allowed_user_ids=current.allowed_user_ids if user_ids is None else user_ids,
                always_on=current.always_on if body.always_on is None else body.always_on,
            ).with_default_team_as_org(viewer.default_team_id)
            if wanted.always_on and not wanted.broad:
                if body.always_on:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=ALWAYS_ON_UNSHARED_REFUSAL,
                    )
                # Unshared from the org and every team: no longer always on.
                wanted = replace(wanted, always_on=False)
            if wanted.always_on != (current.always_on and current.broad) and wanted.broad:
                if wanted.org_allowed and not viewer.can_share_skills_org:
                    if not viewer.permissions_known:
                        raise MindsHubUnreachable(PERMISSIONS_UNREACHABLE)
                    raise PermissionError(ALWAYS_ON_ORG_REFUSAL)
                if not wanted.org_allowed and not (
                    viewer.is_admin or viewer.owns(owner_id) or viewer.in_team_of(wanted)
                ):
                    if not viewer.teams_known:
                        raise MindsHubUnreachable(TEAMS_UNREACHABLE)
                    raise PermissionError(ALWAYS_ON_TEAM_REFUSAL)
            if (
                wanted.allowed_team_ids != current.allowed_team_ids
                or wanted.allowed_user_ids != current.allowed_user_ids
                or wanted.org_allowed != current.org_allowed
            ) and not viewer.can_manage_sharing(stored):
                raise PermissionError(SHARING_OWNER_REFUSAL)
            if wanted.org_allowed and not current.org_allowed and not viewer.can_share_skills_org:
                if not viewer.permissions_known:
                    raise MindsHubUnreachable(PERMISSIONS_UNREACHABLE)
                raise PermissionError(SHARING_ORG_REFUSAL)
            added = [t for t in wanted.allowed_team_ids if t not in current.allowed_team_ids]
            if added and not viewer.teams_known:
                # Without the caller's teams the org's default team can't be
                # told apart, and it is never stored as a team: an admin's
                # save waits too.
                raise MindsHubUnreachable(TEAMS_UNREACHABLE)
            if added and not viewer.is_admin:
                foreign = [t for t in added if t not in viewer.team_ids]
                if foreign:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"You can only share with teams you belong to: {', '.join(foreign)}.",
                    )
            me = (viewer.user_id or "").lower()
            unverified = [
                u
                for u in wanted.allowed_user_ids
                if u not in current.allowed_user_ids and u != me and u not in verified_people
            ]
            if unverified:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail=SHARING_RACED_REFUSAL
                )
            if service.stage_org_access(skill.name, wanted):
                try:
                    access.record_update(
                        SKILL, skill.name, action="share", fallback_creator_id=owner_id
                    )
                    # record_update commits only in MindsHub mode, which this is;
                    # committed here too so the row never waits on it.
                    access.session.commit()
                except Exception:
                    access.session.rollback()
                    raise
    except SkillSharingError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    service.forget_access()
    return _skill_response(
        service.as_own(service.get_skill(skill.name)), access=access, service=service
    ).model_dump(by_alias=True)
