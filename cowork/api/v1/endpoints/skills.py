import logging
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from cowork.api.v1.permissions import Authenticated, AuthenticatedInOrgMode, require
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import (
    ScopedSession,
    ScopedSessionDep,
    TenantScope,
    get_tenant_scope,
)
from cowork.db.session import get_open_session
from cowork.models.skill import Skill
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.principal import Principal, caller_bearer, get_principal
from cowork.schemas.skills import (
    SkillAccessView,
    SkillCreateRequest,
    SkillEnabledForMeRequest,
    SkillListResponse,
    SkillOwner,
    SkillResponse,
    SkillSharingCapabilities,
    SkillSharingRequest,
    SkillUpdateRequest,
)
from cowork.schemas.shared_resources import SkillCapabilities
from cowork.services.projects import ProjectService
from cowork.services.shared_resources import (
    SKILL,
    SKILL_PROJECT_REFERENCES,
    SharedResourceAccess,
)
from cowork.services.skills import (
    SharedSkillsBusyError,
    SkillAlreadyExistsError,
    SkillOrigin,
    SkillService,
    SkillSharingError,
    VisibleSkill,
    builtin_skill_refusal,
    is_builtin_skill,
)
from cowork.services.skill_access import (
    stage_delete as stage_access_delete,
    stage_only_owner,
    stage_rename as stage_access_rename,
)
from cowork.services.skill_user_settings import (
    set_enabled_for_me,
    stage_delete as stage_settings_delete,
    stage_enabled_for_me,
    stage_rename as stage_settings_rename,
)
from cowork.services.skill_sharing import (
    PERMISSIONS_UNREACHABLE,
    TEAMS_UNREACHABLE,
    HubUnavailable,
    OrgMemberLookup,
    SkillAccessRule,
    SkillViewer,
    get_org_member_lookup,
    members_checked,
    normalize_team_ids,
    normalize_user_ids,
    resolve_skill_viewer,
)

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
logger = logging.getLogger(__name__)


async def get_skill_viewer(
    request: Request,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
) -> SkillViewer | None:
    """The caller as hosted org sharing sees them, with their teams and
    permissions from the auth service (cached per caller), asked with the
    Authorization the gateway validated; None outside hosted org mode, where
    nothing changes."""
    return await resolve_skill_viewer(
        scope=scope, principal=principal, bearer_token=caller_bearer(request)
    )


SkillViewerDep = Annotated[SkillViewer | None, Depends(get_skill_viewer)]

OrgMemberLookupDep = Annotated[OrgMemberLookup, Depends(get_org_member_lookup)]

# Every change to a skill the caller may not change in hosted org mode.
ORG_WIDE_EDIT_REFUSAL = (
    "Changing an organization-wide skill requires being its owner, the skills.write_org "
    "permission, or an organization admin."
)
SKILL_EDIT_REFUSAL = (
    "Only the skill's owner, a member of a team it is shared with, or an organization "
    "admin can change this skill."
)


def _require_skill_change(
    access: SharedResourceAccess,
    service: SkillService,
    skill: Skill,
    creator_id: str | None,
    *,
    detail: str,
) -> None:
    """The change rule: hosted org sharing's (SkillViewer.can_edit) where it
    applies, else creator or org admin (SharedResourceAccess.can_change).

    In hosted org mode the skill's row is read again first, so a re-check
    under the skill's lock judges the rule a change committed while it
    waited, not the one read before. A refusal that rests on what the auth
    service couldn't be asked is a 503 (HubUnavailable)."""
    viewer = service.viewer
    if not service.org_sharing or viewer is None:
        access.require_change(creator_id, detail=detail)
        return
    service.forget_access()
    rule = service.access_rule(skill)
    if access.has_trusted_actor and viewer.can_edit(rule):
        return
    if access.has_trusted_actor and viewer.unsure_of(rule):
        raise HubUnavailable(
            PERMISSIONS_UNREACHABLE if viewer.effective(rule).org_allowed else TEAMS_UNREACHABLE
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=ORG_WIDE_EDIT_REFUSAL if viewer.effective(rule).org_allowed else SKILL_EDIT_REFUSAL,
    )


def _project_reference_lock(access: SharedResourceAccess):
    if access.org_mode:
        return access.coordination_lock(SKILL_PROJECT_REFERENCES, "all")
    return nullcontext()


def _require_known_projects(scoped: ScopedSessionDep, project_names: list[str]) -> None:
    project_service = ProjectService(scoped)
    missing = sorted(
        name
        for name in set(project_names)
        if project_service.get_or_provision_by_name_or_none(name) is None
    )
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project selection is stale: {', '.join(missing)}",
        )


def _builtins_immutable(scope: TenantScope) -> bool:
    """Packaged builtins are read-only wherever the skill store is partitioned:
    each org, or each person, gets its own seeded copy (SkillService)."""
    return scope.partitioned_for(PersonalResource.SKILLS)


def _skill_response(
    visible: VisibleSkill, *, access: SharedResourceAccess, service: SkillService
) -> dict:
    skill = visible.skill
    shared = visible.shared
    viewer = service.viewer if service.org_sharing else None
    rule: SkillAccessRule | None = None
    if visible.origin is SkillOrigin.MANAGED:
        # The operator's managed skill: built-in and read-only for every
        # caller, and authored by nobody here.
        creator_id = None
        builtin = True
        can_change = False
    elif visible.origin is SkillOrigin.SHARED:
        # Someone else's skill, shared with the caller: read-only here, and
        # its creator is named to nobody (ResourceAttribution outside org mode).
        creator_id = None
        builtin = False
        can_change = False
    else:
        if access.org_mode and access.has_trusted_actor:
            with access.coordination_lock(SKILL_PROJECT_REFERENCES, "all"):
                with access.coordination_lock(SKILL, skill.name):
                    _recover_skill_claim(access, service, skill.name)
        pending = access.claim_is_pending(SKILL, skill.name)
        creator_id = access.creator_id(SKILL, skill.name)
        if viewer is not None:
            rule = service.access_rule(skill)
        builtin = is_builtin_skill(skill.name)
        theirs = not service.may_change(skill)
        may_change = (
            access.has_trusted_actor and viewer.can_edit(rule)
            if viewer is not None
            else access.can_change(creator_id)
        )
        can_change = (
            not pending
            and not (builtin and _builtins_immutable(access.session.scope))
            and may_change
            and not theirs
        )
        # Where skills are left shared, the base store holds everyone's
        # skills, and one the caller didn't create is shared with them,
        # read-only, as a skill someone shares is where skills are personal.
        shared = shared or theirs
    return SkillResponse(
        id=skill.id,
        label=skill.label,
        name=skill.display_name,
        description=skill.description,
        instructions=skill.instructions,
        created_at=skill.created_at,
        updated_at=skill.updated_at,
        enabled=skill.enabled,
        enabled_for_me=skill.name not in service.off_for_me(),
        can_disable_for_everyone=can_change and not builtin,
        visible_to_others=_visible_to_others(visible, service=service, viewer=viewer, rule=rule),
        projects=skill.projects,
        attribution=access.attribution(
            SKILL,
            skill.name,
            fallback_creator_id=creator_id,
            fallback_modified_at=skill.updated_at,
        ),
        is_builtin=builtin,
        shared=shared,
        capabilities=SkillCapabilities(
            can_edit=can_change,
            can_delete=can_change,
            can_disable=can_change,
            # Only the caller's own skill, where sharing is available, and
            # never a packaged built-in's copy. A share check may still
            # refuse it, saying why, when the caller submits.
            can_share=(
                visible.origin is SkillOrigin.OWN
                and not is_builtin_skill(skill.name)
                and (
                    viewer.can_manage_sharing(rule)
                    if viewer is not None
                    else service.sharing_available
                )
            ),
        ),
        can_edit=can_change,
        **_org_access_fields(visible, viewer=viewer, rule=rule),
    ).model_dump(by_alias=True)


def _visible_to_others(
    visible: VisibleSkill,
    *,
    service: SkillService,
    viewer: SkillViewer | None,
    rule: SkillAccessRule | None,
) -> bool:
    """Whether anyone besides the caller reads ``visible``: a managed or
    packaged built-in skill, someone else's, one shared with people, a team
    or the org, or one of a store every person reads (skills left shared on
    a deployment that trusts identity headers). False for the caller's own
    private skill, whose switch is its global ``enabled``."""
    skill = visible.skill
    if visible.origin is not SkillOrigin.OWN or is_builtin_skill(skill.name) or visible.shared:
        return True
    if viewer is not None and rule is not None:
        if not viewer.owns(rule.owner_user_id):
            return True
        effective = viewer.effective(rule)
        return effective.broad or bool(effective.allowed_user_ids)
    changer = service._changer
    return changer is not None and changer.deployment_wide


def _org_access_fields(
    visible: VisibleSkill, *, viewer: SkillViewer | None, rule: SkillAccessRule | None
) -> dict:
    """``access``, ``section`` and ``owner`` for one skill's response.

    Managed skills, and packaged built-ins in an org store, read as org-wide
    and owned by nobody. Outside hosted org mode every other skill is the
    caller's own bucket with no access set."""
    skill = visible.skill
    if visible.origin is SkillOrigin.MANAGED or (
        viewer is not None and is_builtin_skill(skill.name)
    ):
        return {"access": SkillAccessView(org_allowed=True), "section": "org", "owner": SkillOwner()}
    if viewer is None or rule is None:
        return {"access": SkillAccessView(), "section": "mine", "owner": SkillOwner()}
    effective = viewer.effective(rule)
    owner_id = rule.owner_user_id
    return {
        "access": SkillAccessView(
            org_allowed=effective.org_allowed,
            allowed_team_ids=list(effective.allowed_team_ids),
            # The people only to whoever may change them.
            allowed_user_ids=(
                list(effective.allowed_user_ids) if viewer.can_manage_sharing(rule) else []
            ),
            always_on=effective.always_on and effective.broad,
        ),
        "section": viewer.section(rule),
        # Only the caller's own address is known here; the app names anyone
        # else's id through GET /hub/people/?ids=.
        "owner": SkillOwner(
            id=owner_id, email=viewer.email if viewer.owns(owner_id) and viewer.email else None
        ),
    }


def _record_new_skill(
    skill: Skill,
    service: SkillService,
    access: SharedResourceAccess,
    *,
    action: str,
    claim: SharedResourceAttribution | None = None,
    claim_token: str | None = None,
) -> None:
    try:
        if service.org_sharing:
            # Hosted org mode: the new skill is shared with nobody but its
            # creator, whatever an uploaded SKILL.md says. Staged before the
            # claim is finalized so the two commit together: a skill whose
            # claim is no longer pending always has its row, which the
            # one-time backfill relies on to leave it alone
            # (cowork.services.skill_access.backfill_legacy_access).
            stage_only_owner(access.session, skill.name, access.session.scope.user_id)
        if claim is not None and claim_token is not None:
            attribution = access.finalize_claim(
                claim,
                claim_token,
                action=action,
            )
            created = attribution is not None
        else:
            _attribution, created = access.claim(SKILL, skill.name, action=action)
    except Exception:
        access.session.rollback()
        try:
            service.delete_skill(skill.name)
        except Exception:
            logger.exception("Could not remove a skill after an attribution failure")
        if claim is not None and claim_token is not None:
            access.session.rollback()
            access.release_claim(claim, claim_token=claim_token)
        raise
    if access.org_mode and not created:
        # The filesystem create was exclusive, so this directory belongs to
        # this request. A pre-existing DB claim must not gain somebody else's
        # bytes or let this request report a second successful create. The
        # row staged above goes with the rollback.
        access.session.rollback()
        service.delete_skill(skill.name)
        _release_new_skill_claim(access, claim, claim_token)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A skill named '{skill.name}' already exists.",
        )
    if service.org_sharing:
        service.forget_access()


def _release_new_skill_claim(
    access: SharedResourceAccess,
    claim: SharedResourceAttribution | None,
    claim_token: str | None,
) -> None:
    if claim is None or claim_token is None:
        return
    access.session.rollback()
    access.release_claim(claim, claim_token=claim_token)


def _recover_skill_claim(
    access: SharedResourceAccess,
    service: SkillService,
    slug: str,
) -> bool:
    """recover_stale_claim for a skill. In hosted org mode a complete skill
    it finalizes gets, in the same commit, the row its create would have
    committed with the claim (_record_new_skill): shared with nobody but its
    creator. The one-time backfill skips a skill whose claim was pending
    (cowork.services.skill_access.backfill_legacy_access), so nothing else
    would give it one."""
    recovered = access.recover_stale_claim(
        SKILL,
        slug,
        resource_exists=lambda: service.has_complete_skill(slug),
        on_finalize=(
            (lambda creator_id: stage_only_owner(access.session, slug, creator_id))
            if service.org_sharing
            else None
        ),
    )
    if recovered:
        service.forget_access()
    return recovered


def _recover_stale_skill_claim(
    access: SharedResourceAccess,
    service: SkillService,
    slug: str,
) -> None:
    """Recover only a complete canonical skill, removing a crashed partial."""
    with access.coordination_lock(SKILL, slug):
        recovered = _recover_skill_claim(access, service, slug)
        if recovered and not service.has_complete_skill(slug):
            service.discard_incomplete_skill(slug)


def _reserve_free_slug(
    access: SharedResourceAccess,
    service: SkillService,
    slug: str,
) -> tuple[SharedResourceAttribution, str]:
    """Reserve ``slug`` for a create, refusing one that is already occupied.

    The caller holds ``slug``'s coordination lock. The directory is checked
    before the claim because a claim reserved over existing bytes survives a
    crash, and recovery would then hand a skill this request never wrote to
    whoever reserved it.
    """
    _recover_stale_skill_claim(access, service, slug)
    if service._skill_dir(slug).exists():
        raise FileExistsError(f"A skill named '{slug}' already exists.")
    claim, claim_token = access.reserve_claim(SKILL, slug)
    if claim is None or claim_token is None:
        raise FileExistsError(f"A skill named '{slug}' already exists.")
    return claim, claim_token


@router.get("/", response_model=SkillListResponse)
def list_skills(
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    # Seeded here for an org that opens this menu before it has ever chatted.
    # The turn payload seeds too, so whichever comes first wins; see
    # `build_turn_skills`. In hosted org mode, only what the caller sees.
    skill_service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    skill_service.ensure_builtin_skills()
    access = SharedResourceAccess(scoped, principal)
    # The typed response carries can_share's "canShare" alias from the model.
    return SkillListResponse(
        skills=[
            _skill_response(visible, access=access, service=skill_service)
            for visible in skill_service.list_visible_skills()
        ],
        can_share=skill_service.sharing_available,
    ).model_dump(by_alias=True)


# Declared before GET /{skill_id}, which would otherwise take this path.
@router.get("/sharing-capabilities", response_model=SkillSharingCapabilities)
def skill_sharing_capabilities(viewer: SkillViewerDep = None):
    """What the caller may do with hosted org skill sharing (ENG-3787).

    ``enabled`` is hosted org mode. The two capabilities are the auth
    service's ``skills.share_org`` and ``skills.write_org``, read from the
    caller's entitlements (false while it can't be asked), and both true for
    an org admin. This server keeps no organization display name, so ``org_name`` is
    null for now.
    """
    if viewer is None:
        return SkillSharingCapabilities().model_dump(by_alias=True)
    return SkillSharingCapabilities(
        enabled=True,
        can_share_org=viewer.can_share_org,
        can_write_org=viewer.can_write_org,
        org_name=None,
    ).model_dump(by_alias=True)


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=SkillResponse,
)
def create_skill(
    body: SkillCreateRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    access = SharedResourceAccess(scoped, principal)
    access.require_actor()
    # In hosted org mode a new skill records its creator
    # (SkillService._stamp_creator) and gets a skill_access row shared with
    # nobody else (_record_new_skill), so it is never org-wide by default.
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    claim = None
    claim_token = None
    with _project_reference_lock(access):
        with ExitStack() as resource_locks:
            try:
                if scoped.scope.org_mode and body.projects is not None:
                    _require_known_projects(scoped, body.projects)
                slug = service._slug_from_label(body.label)
                if scoped.scope.org_mode:
                    resource_locks.enter_context(access.coordination_lock(SKILL, slug))
                    claim, claim_token = _reserve_free_slug(access, service, slug)
                skill = service.create_skill(
                    label=body.label,
                    name=body.name,
                    instructions=body.instructions or "",
                    description=body.description,
                    enabled=body.enabled,
                    projects=body.projects,
                    shared=body.shared,
                )
            except SkillSharingError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
            except SharedSkillsBusyError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
                )
            except PermissionError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=str(e)
                )
            except (FileExistsError, ValueError) as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
            except Exception:
                _release_new_skill_claim(access, claim, claim_token)
                raise
            _record_new_skill(
                skill,
                service,
                access,
                action="create",
                claim=claim,
                claim_token=claim_token,
            )
    return _skill_response(service.as_own(skill), access=access, service=service)


def _upload_skill_bytes(
    raw: bytes,
    filename: str | None,
    scoped: ScopedSession,
    principal: Principal | None,
    *,
    shared: bool = False,
    viewer: SkillViewer | None = None,
) -> dict:
    access = SharedResourceAccess(scoped, principal)
    access.require_actor()
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    claim = None
    claim_token = None

    with _project_reference_lock(access):
        with ExitStack() as resource_locks:

            def reserve_import(slug: str) -> None:
                nonlocal claim, claim_token
                resource_locks.enter_context(access.coordination_lock(SKILL, slug))
                claim, claim_token = _reserve_free_slug(access, service, slug)

            try:
                skill = service.import_skill(
                    raw,
                    filename=filename,
                    before_persist=reserve_import if scoped.scope.org_mode else None,
                    shared=shared,
                )
            # Before ValueError's 422: a refused share is a conflict.
            except SkillSharingError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
            except SharedSkillsBusyError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
                )
            except PermissionError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=str(e)
                )
            except (FileExistsError, SkillAlreadyExistsError) as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
            except ValueError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e),
                )
            except Exception:
                _release_new_skill_claim(access, claim, claim_token)
                raise
            _record_new_skill(
                skill,
                service,
                access,
                action="create",
                claim=claim,
                claim_token=claim_token,
            )
    return _skill_response(service.as_own(skill), access=access, service=service)


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=SkillResponse,
)
async def upload_skill(
    file: UploadFile,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
    shared: Annotated[bool, Form()] = False,
    viewer: SkillViewerDep = None,
):
    raw = await file.read()
    return await run_in_threadpool(
        _upload_skill_in_new_session,
        raw,
        file.filename,
        scope,
        principal,
        shared=shared,
        viewer=viewer,
    )


def _upload_skill_in_new_session(
    raw: bytes,
    filename: str | None,
    scope: TenantScope,
    principal: Principal | None,
    *,
    shared: bool = False,
    viewer: SkillViewer | None = None,
) -> dict:
    raw_session = get_open_session()
    try:
        scoped = ScopedSession(raw_session, scope)
        return _upload_skill_bytes(
            raw, filename, scoped, principal, shared=shared, viewer=viewer
        )
    finally:
        raw_session.close()


@router.get("/{skill_id}", response_model=SkillResponse)
def get_skill(
    skill_id: str,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    try:
        visible = service.get_visible_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _skill_response(
        visible, access=SharedResourceAccess(scoped, principal), service=service
    )


@router.put("/{skill_id}", response_model=SkillResponse)
def update_skill(
    skill_id: str,
    body: SkillUpdateRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    access = SharedResourceAccess(scoped, principal)
    with _project_reference_lock(access):
        resource_coordination = ExitStack()
        try:
            # Before any read of the caller's own store, which may not hold a
            # managed or shared slug, or holds one hidden behind it.
            service.refuse_unchangeable(skill_id)
            if scoped.scope.org_mode and body.projects is not None:
                _require_known_projects(scoped, body.projects)
            current = service.get_seen_skill(skill_id)
            # Every scope needs the target: the rename checks below run wherever
            # builtins are immutable, not only where the locks are taken.
            target_slug = (
                service._slug_from_label(body.label)
                if body.label is not None
                else current.name
            )
            service.refuse_unchangeable(current.name)
            if target_slug != current.name:
                # A rename into a name someone else shares is making a new
                # skill of the caller's own, so it answers "choose another
                # name", not the shared-skill refusal that fits an edit.
                service.refuse_unchangeable(target_slug, creating=True)
            if scoped.scope.org_mode:
                for slug in sorted({current.name, target_slug}):
                    resource_coordination.enter_context(
                        access.coordination_lock(SKILL, slug)
                    )
                _recover_stale_skill_claim(access, service, current.name)
            creator_id = access.creator_id(SKILL, current.name)
            if _builtins_immutable(scoped.scope) and is_builtin_skill(current.name):
                raise PermissionError(builtin_skill_refusal(current.name))
            _require_skill_change(
                access,
                service,
                current,
                creator_id,
                detail="Only the skill creator or an organization admin can edit this skill",
            )
            if _builtins_immutable(scoped.scope) and target_slug != current.name:
                if is_builtin_skill(target_slug):
                    raise PermissionError(builtin_skill_refusal(target_slug))
                _recover_stale_skill_claim(access, service, target_slug)
                if access.has_attribution(
                    SKILL,
                    target_slug,
                ) or service._skill_dir(target_slug).exists():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"A skill named '{target_slug}' already exists.",
                    )
            with access.mutation_lock(
                SKILL,
                current.name,
                resource_exists=lambda: service._skill_dir(current.name).is_dir(),
            ) as attribution:
                # Re-read by the locked identity: resolving the request
                # segment again could land on a different directory than the
                # one this mutation lock covers.
                current = service.get_skill(current.name)
                if attribution is not None:
                    # Against the locked copy: its sharing may have changed
                    # while this request waited.
                    _require_skill_change(
                        access,
                        service,
                        current,
                        attribution.created_by_id,
                        detail="Only the skill creator or an organization admin can edit this skill",
                    )
                original_skill_bytes = (
                    service._skill_dir(current.name) / "SKILL.md"
                ).read_bytes()
                skill = service.update_skill(
                    current.name,
                    label=body.label,
                    name=body.name,
                    description=body.description,
                    instructions=body.instructions,
                    enabled=body.enabled,
                    projects=body.projects,
                )
                renamed = skill.name != current.name
                content_changed = any(
                    (
                        body.name is not None
                        and skill.display_name != current.display_name,
                        body.description is not None
                        and skill.description != current.description,
                        body.instructions is not None
                        and skill.instructions != current.instructions,
                        body.projects is not None
                        and skill.projects != current.projects,
                    )
                )
                enabled_changed = (
                    body.enabled is not None and skill.enabled != current.enabled
                )
                actions = ["rename"] if renamed else []
                if content_changed:
                    actions.append("update")
                if enabled_changed:
                    actions.append("enable" if skill.enabled else "disable")
                try:
                    if renamed and access.has_attribution(SKILL, skill.name):
                        # The destination slug gained attribution after this
                        # request cleared it. Name the taken slug rather than
                        # letting its unique key fail the commit.
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"A skill named '{skill.name}' already exists.",
                        )
                    if renamed and service.org_sharing:
                        # Who the skill is shared with follows it to its new
                        # name, in the commit that moves its attribution.
                        stage_access_rename(access.session, current.name, skill.name)
                    if renamed:
                        # So does every person's own on/off for it.
                        stage_settings_rename(access.session, current.name, skill.name)
                    access.record_updates(
                        SKILL,
                        current.name,
                        actions=actions,
                        fallback_creator_id=creator_id,
                        new_key=skill.name if renamed else None,
                    )
                    if renamed and not access.org_mode:
                        # record_updates commits only in org mode.
                        access.session.commit()
                except Exception as audit_failure:
                    access.session.rollback()
                    try:
                        if renamed:
                            service._rename_dir(skill.name, current.name)
                        service._restore_skill_file(
                            current.name,
                            original_skill_bytes,
                        )
                    except Exception:
                        logger.exception(
                            "Could not restore a skill after an audit failure"
                        )
                    if renamed and isinstance(audit_failure, IntegrityError):
                        # Another writer took the destination slug between the
                        # check above and this commit.
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"A skill named '{skill.name}' already exists.",
                        )
                    raise
        except SharedSkillsBusyError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )
        except SkillAlreadyExistsError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        except PermissionError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        finally:
            resource_coordination.close()
    if body.enabled:
        # Turning a skill on for everyone turns it on for its editor too:
        # their own off would otherwise keep it out of their turns.
        _set_enabled_for_me_or_log(scoped, skill.name, True)
    # A rename moved the skill's access row and settings; read them again
    # under the new name.
    service.forget_access()
    service.forget_off_for_me()
    return _skill_response(service.as_own(skill), access=access, service=service)


def _set_enabled_for_me_or_log(scoped: ScopedSession, slug: str, enabled: bool) -> None:
    """set_enabled_for_me after a change that already succeeded: a failure
    here is logged rather than reported as the change's."""
    try:
        set_enabled_for_me(scoped, slug, enabled)
    except Exception:
        logger.exception("Could not update the caller's own setting for skill %r", slug)


@router.put("/{skill_id}/me", response_model=SkillResponse)
def set_skill_enabled_for_me(
    skill_id: str,
    body: SkillEnabledForMeRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    """The caller's own on/off for one skill they see (ENG-3787): off leaves
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
    return _skill_response(visible, access=SharedResourceAccess(scoped, principal), service=service)


PEOPLE_OUTSIDE_ORG_REFUSAL = "Skills can only be shared with people in your organization"
SHARING_RACED_REFUSAL = "This skill's sharing changed while saving. Reload it and try again."


async def stored_sharing_rule(
    skill_id: str, scoped: ScopedSessionDep, viewer: SkillViewerDep
) -> SkillAccessRule | None:
    """The skill's stored access as ``viewer`` reads it (the default team as
    org-wide), for the check PUT /{skill_id}/sharing asks the auth service
    before it takes the skill's lock (verify_people_in_org): it only decides
    what that asks, and the route reads the rule again under the lock. None
    outside hosted org mode, and for a skill that can't be read here, which
    the route refuses itself."""
    if viewer is None:
        return None

    def read() -> SkillAccessRule | None:
        try:
            service = SkillService(scoped.scope, viewer=viewer, session=scoped)
            return viewer.effective(service.access_rule(service.get_seen_skill(skill_id)))
        except Exception:
            return None

    return await run_in_threadpool(read)


StoredRuleDep = Annotated[SkillAccessRule | None, Depends(stored_sharing_rule)]


async def verify_people_in_org(
    body: SkillSharingRequest,
    viewer: SkillViewerDep,
    stored: StoredRuleDep,
    members: OrgMemberLookupDep,
) -> frozenset[str]:
    """The user ids ``body`` adds to the skill's people, confirmed members of
    the caller's org in one batched lookup (the auth service's ``?ids=``).

    Only new ids are asked about: one already stored is kept as is, so an
    outage doesn't block an edit that adds nobody. The caller's own id needs
    no check. 400 naming every id outside the org; 503 ``hub_unavailable``
    while the auth service can't be asked. A malformed id or a skill that
    can't be read is left to _set_org_skill_sharing to refuse; it re-checks
    the added ids against this answer under the skill's lock.

    A dependency of PUT /{skill_id}/sharing rather than a step in it, so the
    route stays sync. Nothing to check outside hosted org mode.
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
    """Change who one skill is shared with: in hosted org mode
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
    "Sharing a skill with everyone in the organization requires the skills.share_org "
    "permission and being the skill's owner, or an organization admin."
)
ALWAYS_ON_UNSHARED_REFUSAL = "Only skills shared with the organization or a team can be always on."
ALWAYS_ON_ORG_REFUSAL = (
    "Making an organization-wide skill always on, or taking that back, requires the "
    "skills.share_org permission or an organization admin."
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
    """Hosted org mode: set a skill's access (its skill_access row):
    ``org_allowed``, ``allowed_team_ids``, ``allowed_user_ids`` and
    ``always_on``.

    A field the body leaves out keeps what the skill has. Teams, people and
    taking a skill back from the org are the owner's or an admin's to
    change; making it org-wide takes ``skills.share_org`` as well. A team
    added must be one the caller belongs to (admins exempt), checked against
    their listing from the auth service, 400 for a team they aren't in. A
    kept team is kept unchecked: a deleted team's id grants nobody anything.
    The org's default team is never stored: named among the teams it means
    org-wide (default_team_as_org), so while the caller's teams can't be
    read, a change that adds a team is a 503, an admin's included. User ids
    are stripped, lowercased and deduplicated; a malformed one is a 400. An
    id added must be a member of the org, which verify_people_in_org
    confirmed before this runs (``verified_people``); one added since that
    check read the skill is a 409. A refusal that rests on teams or
    permissions the auth service couldn't list is a 503 ``hub_unavailable``
    (HubUnavailable).

    ``always_on`` can be set only on a skill that, after this change, is
    shared org-wide or with a team (else 400). Changing it on an org-wide
    skill takes ``skills.share_org``; on a team skill, edit rights (owner, a
    member of one of its teams, or admin); else 403. A change that leaves
    the skill neither org-wide nor team-shared clears it.

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
            # the teams it means org-wide, with org-wide's rules
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
                if wanted.org_allowed and not viewer.can_share_org:
                    if not viewer.permissions_known:
                        raise HubUnavailable(PERMISSIONS_UNREACHABLE)
                    raise PermissionError(ALWAYS_ON_ORG_REFUSAL)
                if not wanted.org_allowed and not (
                    viewer.is_admin or viewer.owns(owner_id) or viewer.in_team_of(wanted)
                ):
                    if not viewer.teams_known:
                        raise HubUnavailable(TEAMS_UNREACHABLE)
                    raise PermissionError(ALWAYS_ON_TEAM_REFUSAL)
            if (
                wanted.allowed_team_ids != current.allowed_team_ids
                or wanted.allowed_user_ids != current.allowed_user_ids
                or wanted.org_allowed != current.org_allowed
            ) and not viewer.can_manage_sharing(stored):
                raise PermissionError(SHARING_OWNER_REFUSAL)
            if wanted.org_allowed and not current.org_allowed and not viewer.can_share_org:
                if not viewer.permissions_known:
                    raise HubUnavailable(PERMISSIONS_UNREACHABLE)
                raise PermissionError(SHARING_ORG_REFUSAL)
            added = [t for t in wanted.allowed_team_ids if t not in current.allowed_team_ids]
            if added and not viewer.teams_known:
                # Without the caller's teams the org's default team can't be
                # told apart, and it is never stored as a team: an admin's
                # save waits too.
                raise HubUnavailable(TEAMS_UNREACHABLE)
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
                    # record_update commits only in org mode, which this is;
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
    )


def _set_personal_skill_sharing(
    skill_id: str, shared: bool, scoped: ScopedSession, principal: Principal
) -> dict:
    """Share one of the caller's own skills with everyone, read-only, or stop
    sharing it. Both directions are idempotent.

    401 for a caller with no person (the operator view, the desktop), from
    the require(Authenticated) dependency before the body runs; 409 where
    sharing is unavailable, before anything else is read; 403 for a managed
    skill's name or one someone else shares with the caller; 404 when the
    caller's own store holds no such skill; 409 for a share a check refuses,
    with the reason; 503 while another change holds the list.
    """
    service = SkillService(scoped.scope)
    try:
        service.refuse_unless_sharing_available()
        service.refuse_unchangeable(skill_id)
        skill = service.get_skill(skill_id)
        if shared:
            service.share(skill.name)
        else:
            service.stop_sharing(skill.name)
        skill = service.get_skill(skill.name)
    except SkillSharingError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except SharedSkillsBusyError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _skill_response(
        service.as_own(skill), access=SharedResourceAccess(scoped, principal), service=service
    )


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(
    skill_id: str,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    service = SkillService(scoped.scope, viewer=viewer, session=scoped)
    access = SharedResourceAccess(scoped, principal)
    reference_coordination = _project_reference_lock(access)
    reference_coordination.__enter__()
    resource_coordination = None
    try:
        service.refuse_unchangeable(skill_id)
        skill = service.get_seen_skill(skill_id)
        if scoped.scope.org_mode:
            resource_coordination = access.coordination_lock(SKILL, skill.name)
            resource_coordination.__enter__()
            _recover_stale_skill_claim(access, service, skill.name)
        if _builtins_immutable(scoped.scope) and is_builtin_skill(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        creator_id = access.creator_id(SKILL, skill.name)
        _require_skill_change(
            access,
            service,
            skill,
            creator_id,
            detail="Only the skill creator or an organization admin can delete this skill",
        )
        if scoped.scope.org_mode:
            with access.mutation_lock(
                SKILL,
                skill.name,
                resource_exists=lambda: service._skill_dir(skill.name).is_dir(),
            ) as attribution:
                if attribution is None:
                    raise RuntimeError("Skill mutation lock was not established")
                _require_skill_change(
                    access,
                    service,
                    service.get_skill(skill.name),
                    attribution.created_by_id,
                    detail="Only the skill creator or an organization admin can delete this skill",
                )
                staged = service.stage_delete(skill.name)
                found = staged is not None
                if found:
                    try:
                        if service.org_sharing:
                            # Gone with the skill, in the same commit.
                            stage_access_delete(access.session, skill.name)
                        stage_settings_delete(access.session, skill.name)
                        access.record_delete(SKILL, skill.name)
                    except Exception:
                        access.session.rollback()
                        service.restore_staged_delete(skill.name, staged)
                        raise
                    service.finalize_staged_delete(staged)
        else:
            found = service.delete_skill(skill.name)
            if found:
                # Every person's own on/off for it goes with it. The folder
                # is gone already, so a failure here leaves rows that name no
                # skill, which nothing reads, and is logged.
                try:
                    stage_settings_delete(access.session, skill.name)
                    access.session.commit()
                except Exception:
                    access.session.rollback()
                    logger.exception("Could not remove the settings of deleted skill %r", skill.name)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    finally:
        if resource_coordination is not None:
            resource_coordination.__exit__(None, None, None)
        reference_coordination.__exit__(None, None, None)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found."
        )
