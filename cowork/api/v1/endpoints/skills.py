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
from cowork.models.skill import META_CREATED_BY_EMAIL, Skill
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.principal import Principal, get_principal, hub_credential
from cowork.schemas.skills import (
    SkillCreateRequest,
    SkillListResponse,
    SkillOwner,
    SkillResponse,
    SkillSharingCapabilities,
    SkillSharingRequest,
    SkillSharingView,
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
from cowork.services.skill_sharing import (
    SkillSharing,
    SkillViewer,
    normalize_emails,
    normalize_team_ids,
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
    permissions from auth (cached per caller); None outside hosted org mode,
    where nothing changes."""
    return await resolve_skill_viewer(
        scope=scope, principal=principal, bearer_token=hub_credential(request)
    )


SkillViewerDep = Annotated[SkillViewer | None, Depends(get_skill_viewer)]

# Every change to a skill the caller may not change in hosted org mode.
ORG_WIDE_EDIT_REFUSAL = (
    "Changing an organization-wide skill requires the skills.write_org permission "
    "or an organization admin."
)
SKILL_EDIT_REFUSAL = (
    "Only the skill's owner, a member of a team it is shared with, or an organization "
    "admin can change this skill."
)


def _skill_owner_id(access: SharedResourceAccess, skill: Skill) -> str | None:
    """Who created ``skill``: the attribution row in org mode, else the
    creator its metadata records."""
    return access.creator_id(SKILL, skill.name, fallback=skill.created_by) or skill.created_by


def _require_skill_change(
    access: SharedResourceAccess,
    service: SkillService,
    skill: Skill,
    creator_id: str | None,
    *,
    detail: str,
) -> None:
    """The change rule: hosted org sharing's (SkillViewer.can_edit) where it
    applies, else creator or org admin (SharedResourceAccess.can_change)."""
    viewer = service.viewer
    if not service.org_sharing or viewer is None:
        access.require_change(creator_id, detail=detail)
        return
    owner_id = creator_id or skill.created_by
    if access.has_trusted_actor and viewer.can_edit(skill, owner_id=owner_id):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=ORG_WIDE_EDIT_REFUSAL if SkillSharing.of(skill).org else SKILL_EDIT_REFUSAL,
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
    owner_id: str | None = None
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
                    access.recover_stale_claim(
                        SKILL,
                        skill.name,
                        resource_exists=lambda: service.has_complete_skill(skill.name),
                    )
        pending = access.claim_is_pending(SKILL, skill.name)
        creator_id = access.creator_id(SKILL, skill.name)
        owner_id = creator_id or skill.created_by
        builtin = is_builtin_skill(skill.name)
        theirs = not service.may_change(skill)
        may_change = (
            access.has_trusted_actor and viewer.can_edit(skill, owner_id=owner_id)
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
                    viewer.can_manage_sharing(owner_id=owner_id)
                    if viewer is not None
                    else service.sharing_available
                )
            ),
        ),
        can_edit=can_change,
        **_org_sharing_fields(visible, viewer=viewer, owner_id=owner_id),
    ).model_dump(by_alias=True)


def _org_sharing_fields(
    visible: VisibleSkill, *, viewer: SkillViewer | None, owner_id: str | None
) -> dict:
    """``sharing``, ``section`` and ``owner`` for one skill's response.

    Managed skills, and packaged built-ins in an org store, read as org-wide
    and owned by nobody. Outside hosted org mode every other skill is the
    caller's own bucket with no sharing."""
    skill = visible.skill
    if visible.origin is SkillOrigin.MANAGED or (
        viewer is not None and is_builtin_skill(skill.name)
    ):
        return {"sharing": SkillSharingView(org=True), "section": "org", "owner": SkillOwner()}
    if viewer is None:
        return {"sharing": SkillSharingView(), "section": "mine", "owner": SkillOwner()}
    sharing = SkillSharing.of(skill)
    owner_email = skill.metadata.get(META_CREATED_BY_EMAIL) or (
        viewer.email if viewer.owns(owner_id) and viewer.email else None
    )
    return {
        "sharing": SkillSharingView(
            org=sharing.org,
            team_ids=list(sharing.team_ids),
            # The recipients' addresses only to whoever may change them.
            emails=list(sharing.emails) if viewer.can_manage_sharing(owner_id=owner_id) else [],
            always_on=sharing.always_on and sharing.broad,
        ),
        "section": viewer.section(skill, owner_id=owner_id),
        "owner": SkillOwner(id=owner_id, email=owner_email),
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
        # bytes or let this request report a second successful create.
        service.delete_skill(skill.name)
        _release_new_skill_claim(access, claim, claim_token)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A skill named '{skill.name}' already exists.",
        )


def _release_new_skill_claim(
    access: SharedResourceAccess,
    claim: SharedResourceAttribution | None,
    claim_token: str | None,
) -> None:
    if claim is None or claim_token is None:
        return
    access.session.rollback()
    access.release_claim(claim, claim_token=claim_token)


def _recover_stale_skill_claim(
    access: SharedResourceAccess,
    service: SkillService,
    slug: str,
) -> None:
    """Recover only a complete canonical skill, removing a crashed partial."""
    with access.coordination_lock(SKILL, slug):
        recovered = access.recover_stale_claim(
            SKILL,
            slug,
            resource_exists=lambda: service.has_complete_skill(slug),
        )
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
    skill_service = SkillService(scoped.scope, viewer=viewer)
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

    ``enabled`` is hosted org mode. The two capabilities are auth's
    ``skills.share_org`` and ``skills.write_org``, read from the caller's
    entitlements (false while auth can't be asked), and both true for an org
    admin. This server keeps no organization display name, so ``org_name`` is
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
    # In hosted org mode a new skill records its creator and only-me sharing
    # (SkillService._stamp_creator), so it is never org-wide by default.
    service = SkillService(scoped.scope, viewer=viewer)
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
    service = SkillService(scoped.scope, viewer=viewer)
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
    service = SkillService(scoped.scope, viewer=viewer)
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
    service = SkillService(scoped.scope, viewer=viewer)
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
                    access.record_updates(
                        SKILL,
                        current.name,
                        actions=actions,
                        fallback_creator_id=creator_id,
                        new_key=skill.name if renamed else None,
                    )
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
    return _skill_response(service.as_own(skill), access=access, service=service)


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
):
    """Change who one skill is shared with: in hosted org mode
    _set_org_skill_sharing, elsewhere _set_personal_skill_sharing."""
    if viewer is not None:
        return _set_org_skill_sharing(skill_id, body, scoped, principal, viewer)
    if body.shared is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="shared is required."
        )
    return _set_personal_skill_sharing(skill_id, body.shared, scoped, principal)


SHARING_OWNER_REFUSAL = (
    "Only the skill's owner or an organization admin can change who it is shared with."
)
SHARING_ORG_REFUSAL = (
    "Sharing a skill with everyone in the organization, or taking it back, requires the "
    "skills.share_org permission or an organization admin."
)
ALWAYS_ON_UNSHARED_REFUSAL = "Only skills shared with the organization or a team can be always on."
ALWAYS_ON_ORG_REFUSAL = "Only admins can make an organization-wide skill always on."
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
) -> dict:
    """Hosted org mode: set a skill's org-wide flag, teams and people.

    A field the body leaves out keeps what the skill has. Teams and people
    are the owner's or an admin's to change; setting or clearing org-wide
    takes ``skills.share_org``. A team added must be one the caller belongs
    to (admins exempt), checked against their listing from auth: 503 while
    auth can't be asked, 400 for a team they aren't in. Emails are
    lowercased and deduplicated; one that isn't an address is a 400.

    ``always_on`` can be set only on a skill that, after this change, is
    shared org-wide or with a team (else 400). Changing it on an org-wide
    skill takes ``skills.share_org``; on a team skill, edit rights (owner, a
    member of one of its teams, or admin); else 403. A change that leaves
    the skill neither org-wide nor team-shared clears it.

    404 for a skill the caller doesn't see or that doesn't exist, 403 for a
    built-in, a managed skill, or a change the caller may not make. Nothing
    is written when the sharing already reads this way.
    """
    service = SkillService(scoped.scope, viewer=viewer)
    access = SharedResourceAccess(scoped, principal)
    try:
        service.refuse_unchangeable(skill_id)
        skill = service.get_seen_skill(skill_id)
        if is_builtin_skill(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        try:
            team_ids = (
                normalize_team_ids(body.team_ids) if body.team_ids is not None else None
            )
            emails = normalize_emails(body.emails) if body.emails is not None else None
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        with access.coordination_lock(SKILL, skill.name):
            skill = service.get_seen_skill(skill.name)
            owner_id = _skill_owner_id(access, skill)
            current = SkillSharing.of(skill)
            wanted = SkillSharing(
                org=current.org if body.org is None else body.org,
                team_ids=current.team_ids if team_ids is None else team_ids,
                emails=current.emails if emails is None else emails,
                always_on=current.always_on if body.always_on is None else body.always_on,
            )
            if wanted.always_on and not wanted.broad:
                if body.always_on:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=ALWAYS_ON_UNSHARED_REFUSAL,
                    )
                # Unshared from the org and every team: no longer always on.
                wanted = replace(wanted, always_on=False)
            if wanted.always_on != (current.always_on and current.broad) and wanted.broad:
                if wanted.org and not viewer.can_share_org:
                    raise PermissionError(ALWAYS_ON_ORG_REFUSAL)
                if not wanted.org and not (
                    viewer.is_admin or viewer.owns(owner_id) or viewer.in_team_of(wanted)
                ):
                    raise PermissionError(ALWAYS_ON_TEAM_REFUSAL)
            if (
                wanted.team_ids != current.team_ids or wanted.emails != current.emails
            ) and not viewer.can_manage_sharing(owner_id=owner_id):
                raise PermissionError(SHARING_OWNER_REFUSAL)
            if wanted.org != current.org and not viewer.can_share_org:
                raise PermissionError(SHARING_ORG_REFUSAL)
            added = [t for t in wanted.team_ids if t not in current.team_ids]
            if added and not viewer.is_admin:
                if not viewer.teams_known:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Couldn't reach MindsHub to confirm your teams. Try again in a moment.",
                    )
                foreign = [t for t in added if t not in viewer.team_ids]
                if foreign:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"You can only share with teams you belong to: {', '.join(foreign)}.",
                    )
            original = (service._skill_dir(skill.name) / "SKILL.md").read_bytes()
            updated = service.set_org_sharing(skill.name, wanted)
            if updated.metadata != skill.metadata:
                try:
                    access.record_update(
                        SKILL, skill.name, action="share", fallback_creator_id=owner_id
                    )
                except Exception:
                    access.session.rollback()
                    try:
                        service._restore_skill_file(skill.name, original)
                    except Exception:
                        logger.exception("Could not restore a skill after an audit failure")
                    raise
    except SkillSharingError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _skill_response(service.as_own(updated), access=access, service=service)


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
    service = SkillService(scoped.scope, viewer=viewer)
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
                        access.record_delete(SKILL, skill.name)
                    except Exception:
                        access.session.rollback()
                        service.restore_staged_delete(skill.name, staged)
                        raise
                    service.finalize_staged_delete(staged)
        else:
            found = service.delete_skill(skill.name)
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
