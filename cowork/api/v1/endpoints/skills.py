import logging
from contextlib import ExitStack, nullcontext
from typing import Annotated, NamedTuple

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
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
    SkillSection,
    SkillCreateRequest,
    SkillListResponse,
    SkillOwner,
    SkillResponse,
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
    stage_rename as stage_settings_rename,
)
from cowork.services.mindshub_client import (
    PERMISSIONS_UNREACHABLE,
    TEAMS_UNREACHABLE,
    MindsHubUnreachable,
)
from cowork.services.sharing_contract import SKILLS_WRITE_ORG
from cowork.services.skill_sharing import (
    SkillAccessRule,
    SkillViewer,
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
    """The caller as sharing sees them in MindsHub mode, with their teams
    and permissions from MindsHub (cached per caller), asked with the
    Authorization the gateway validated; None outside MindsHub mode, where
    nothing changes."""
    return await resolve_skill_viewer(
        scope=scope, principal=principal, bearer_token=caller_bearer(request)
    )


SkillViewerDep = Annotated[SkillViewer | None, Depends(get_skill_viewer)]

# Every change to a skill the caller may not change in MindsHub mode.
ORG_ALLOWED_EDIT_REFUSAL = (
    "Changing a skill shared with everyone in the organization requires being its owner, "
    f"the {SKILLS_WRITE_ORG} permission, or an organization admin."
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
    """The change rule: MindsHub mode's (SkillViewer.can_edit) where it
    applies, else creator or org admin (SharedResourceAccess.can_change).

    In MindsHub mode the skill's row is read again first, so a re-check
    under the skill's lock judges the rule a change committed while it
    waited, not the one read before. A refusal that rests on what MindsHub
    couldn't be asked is a 503 (MindsHubUnreachable)."""
    viewer = service.viewer
    if not service.org_sharing or viewer is None:
        access.require_change(creator_id, detail=detail)
        return
    service.forget_access()
    rule = service.access_rule(skill)
    if access.has_trusted_actor and viewer.can_edit(rule):
        return
    if access.has_trusted_actor and viewer.unsure_of(rule):
        raise MindsHubUnreachable(
            PERMISSIONS_UNREACHABLE if viewer.effective(rule).org_allowed else TEAMS_UNREACHABLE
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=ORG_ALLOWED_EDIT_REFUSAL if viewer.effective(rule).org_allowed else SKILL_EDIT_REFUSAL,
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
) -> SkillResponse:
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
    audience = _org_access_fields(visible, viewer=viewer, rule=rule)
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
        access=audience.access,
        section=audience.section,
        owner=audience.owner,
    )


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


class SkillAudienceFields(NamedTuple):
    """One skill's ``access``, ``section`` and ``owner``, as its response
    names them (_org_access_fields)."""

    access: SkillAccessView
    section: SkillSection
    owner: SkillOwner


def _org_access_fields(
    visible: VisibleSkill, *, viewer: SkillViewer | None, rule: SkillAccessRule | None
) -> SkillAudienceFields:
    """``access``, ``section`` and ``owner`` for one skill's response.

    Managed skills, and packaged built-ins in an org store, read as shared
    with everyone and owned by nobody. Outside MindsHub mode every other
    skill is the caller's own bucket with no audience set."""
    skill = visible.skill
    if visible.origin is SkillOrigin.MANAGED or (
        viewer is not None and is_builtin_skill(skill.name)
    ):
        return SkillAudienceFields(
            access=SkillAccessView(org_allowed=True), section=SkillSection.ORG, owner=SkillOwner()
        )
    if viewer is None or rule is None:
        return SkillAudienceFields(access=SkillAccessView(), section=SkillSection.MINE, owner=SkillOwner())
    effective = viewer.effective(rule)
    owner_id = rule.owner_user_id
    return SkillAudienceFields(
        access=SkillAccessView(
            org_allowed=effective.org_allowed,
            allowed_team_ids=list(effective.allowed_team_ids),
            # The people only to whoever may change them.
            allowed_user_ids=(
                list(effective.allowed_user_ids) if viewer.can_manage_sharing(rule) else []
            ),
            always_on=effective.always_on and effective.broad,
        ),
        section=viewer.section(rule),
        # Only the caller's own address is known here; the app names anyone
        # else's id through GET /hub/people/?ids=.
        owner=SkillOwner(
            id=owner_id, email=viewer.email if viewer.owns(owner_id) and viewer.email else None
        ),
    )


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
            # MindsHub mode: the new skill is shared with nobody but its
            # creator, whatever an uploaded SKILL.md says. Staged before the
            # claim is finalized so the two commit together: a skill whose
            # claim is no longer pending always has its row, which the
            # one-time backfill relies on to leave it alone
            # (cowork.services.skill_access.backfill_legacy_access).
            stage_only_owner(
                access.session, slug=skill.name, owner_user_id=access.session.scope.user_id
            )
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
    """recover_stale_claim for a skill. In MindsHub mode a complete skill
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
            (lambda creator_id: stage_only_owner(access.session, slug=slug, owner_user_id=creator_id))
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


class SlugClaim(NamedTuple):
    """A create's pending claim on a skill's name (_reserve_free_slug), and
    the token that finalizes or releases it."""

    claim: SharedResourceAttribution
    token: str


def _reserve_free_slug(
    access: SharedResourceAccess,
    service: SkillService,
    slug: str,
) -> SlugClaim:
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
    return SlugClaim(claim=claim, token=claim_token)


@router.get("/", response_model=SkillListResponse)
def list_skills(
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
    viewer: SkillViewerDep = None,
):
    # Seeded here for an org that opens this menu before it has ever chatted.
    # The turn payload seeds too, so whichever comes first wins; see
    # `build_turn_skills`. In MindsHub mode, only what the caller sees.
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
    # In MindsHub mode a new skill records its creator
    # (SkillService._stamp_creator) and gets a skill_access row shared with
    # nobody else (_record_new_skill), so it is never shared with everyone
    # by default.
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
                    reserved = _reserve_free_slug(access, service, slug)
                    claim, claim_token = reserved.claim, reserved.token
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
    return _skill_response(
        service.as_own(skill), access=access, service=service
    ).model_dump(by_alias=True)


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
                reserved = _reserve_free_slug(access, service, slug)
                claim, claim_token = reserved.claim, reserved.token

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
    return _skill_response(
        service.as_own(skill), access=access, service=service
    ).model_dump(by_alias=True)


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
    ).model_dump(by_alias=True)


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
                        stage_access_rename(access.session, old_slug=current.name, new_slug=skill.name)
                    if renamed:
                        # So does every person's own on/off for it.
                        stage_settings_rename(access.session, old_slug=current.name, new_slug=skill.name)
                    access.record_updates(
                        SKILL,
                        current.name,
                        actions=actions,
                        fallback_creator_id=creator_id,
                        new_key=skill.name if renamed else None,
                    )
                    if renamed and not access.org_mode:
                        # record_updates commits only in MindsHub mode.
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
    return _skill_response(
        service.as_own(skill), access=access, service=service
    ).model_dump(by_alias=True)


def _set_enabled_for_me_or_log(scoped: ScopedSession, slug: str, enabled: bool) -> None:
    """set_enabled_for_me after a change that already succeeded: a failure
    here is logged rather than reported as the change's."""
    try:
        set_enabled_for_me(scoped, slug, enabled)
    except Exception:
        logger.exception("Could not update the caller's own setting for skill %r", slug)


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
    ).model_dump(by_alias=True)


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
