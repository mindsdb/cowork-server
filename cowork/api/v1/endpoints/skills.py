import logging
from contextlib import ExitStack, nullcontext
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from cowork.db.scoped import (
    ScopedSession,
    ScopedSessionDep,
    TenantScope,
    get_tenant_scope,
)
from cowork.db.session import get_open_session
from cowork.models.skill import Skill
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.principal import Principal, get_principal
from cowork.schemas.skills import (
    SkillCreateRequest,
    SkillListResponse,
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
    SkillService,
    builtin_skill_refusal,
    is_builtin_skill,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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


def _skill_response(skill: Skill, access: SharedResourceAccess) -> dict:
    if access.org_mode and access.has_trusted_actor:
        service = SkillService(access.session.scope)
        with access.coordination_lock(SKILL_PROJECT_REFERENCES, "all"):
            with access.coordination_lock(SKILL, skill.name):
                access.recover_stale_claim(
                    SKILL,
                    skill.name,
                    resource_exists=lambda: service.has_complete_skill(skill.name),
                )
    pending = access.claim_is_pending(SKILL, skill.name)
    creator_id = access.creator_id(SKILL, skill.name)
    builtin = is_builtin_skill(skill.name)
    can_change = (
        not pending
        and not (builtin and access.org_mode)
        and access.can_change(creator_id)
    )
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
        capabilities=SkillCapabilities(
            can_edit=can_change,
            can_delete=can_change,
            can_disable=can_change,
        ),
    ).model_dump(by_alias=True)


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
):
    # Seeded here for an org that opens this menu before it has ever chatted.
    # The turn payload seeds too, so whichever comes first wins; see
    # `build_turn_skills`.
    skill_service = SkillService(scoped.scope)
    skill_service.ensure_builtin_skills()
    skills = skill_service.list_skills()
    access = SharedResourceAccess(scoped, principal)
    return {"skills": [_skill_response(skill, access) for skill in skills]}


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=SkillResponse,
)
def create_skill(
    body: SkillCreateRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    access = SharedResourceAccess(scoped, principal)
    access.require_actor()
    service = SkillService(scoped.scope)
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
    return _skill_response(skill, access)


def _upload_skill_bytes(
    raw: bytes,
    filename: str | None,
    scoped: ScopedSession,
    principal: Principal | None,
) -> dict:
    access = SharedResourceAccess(scoped, principal)
    access.require_actor()
    service = SkillService(scoped.scope)
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
                )
            except PermissionError as e:
                _release_new_skill_claim(access, claim, claim_token)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=str(e)
                )
            except FileExistsError as e:
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
    return _skill_response(skill, access)


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=SkillResponse,
)
async def upload_skill(
    file: UploadFile,
    scope: Annotated[TenantScope, Depends(get_tenant_scope)],
    principal: Principal | None = Depends(get_principal),
):
    raw = await file.read()
    return await run_in_threadpool(
        _upload_skill_in_new_session,
        raw,
        file.filename,
        scope,
        principal,
    )


def _upload_skill_in_new_session(
    raw: bytes,
    filename: str | None,
    scope: TenantScope,
    principal: Principal | None,
) -> dict:
    raw_session = get_open_session()
    try:
        scoped = ScopedSession(raw_session, scope)
        return _upload_skill_bytes(raw, filename, scoped, principal)
    finally:
        raw_session.close()


@router.get("/{skill_id}", response_model=SkillResponse)
def get_skill(
    skill_id: str,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    try:
        skill = SkillService(scoped.scope).get_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _skill_response(skill, SharedResourceAccess(scoped, principal))


@router.put("/{skill_id}", response_model=SkillResponse)
def update_skill(
    skill_id: str,
    body: SkillUpdateRequest,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    service = SkillService(scoped.scope)
    access = SharedResourceAccess(scoped, principal)
    with _project_reference_lock(access):
        resource_coordination = ExitStack()
        try:
            if scoped.scope.org_mode and body.projects is not None:
                _require_known_projects(scoped, body.projects)
            current = service.get_skill(skill_id)
            if scoped.scope.org_mode:
                target_slug = (
                    service._slug_from_label(body.label)
                    if body.label is not None
                    else current.name
                )
                for slug in sorted({current.name, target_slug}):
                    resource_coordination.enter_context(
                        access.coordination_lock(SKILL, slug)
                    )
                _recover_stale_skill_claim(access, service, current.name)
            creator_id = access.creator_id(SKILL, current.name)
            if scoped.scope.org_mode and is_builtin_skill(current.name):
                raise PermissionError(builtin_skill_refusal(current.name))
            access.require_change(
                creator_id,
                detail="Only the skill creator or an organization admin can edit this skill",
            )
            if scoped.scope.org_mode and target_slug != current.name:
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
                if attribution is not None:
                    access.require_change(
                        attribution.created_by_id,
                        detail="Only the skill creator or an organization admin can edit this skill",
                    )
                # Re-read by the locked identity: resolving the request
                # segment again could land on a different directory than the
                # one this mutation lock covers.
                current = service.get_skill(current.name)
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
        except PermissionError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        finally:
            resource_coordination.close()
    return _skill_response(skill, access)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(
    skill_id: str,
    scoped: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    service = SkillService(scoped.scope)
    access = SharedResourceAccess(scoped, principal)
    reference_coordination = _project_reference_lock(access)
    reference_coordination.__enter__()
    resource_coordination = None
    try:
        skill = service.get_skill(skill_id)
        if scoped.scope.org_mode:
            resource_coordination = access.coordination_lock(SKILL, skill.name)
            resource_coordination.__enter__()
            _recover_stale_skill_claim(access, service, skill.name)
        if scoped.scope.org_mode and is_builtin_skill(skill.name):
            raise PermissionError(builtin_skill_refusal(skill.name))
        creator_id = access.creator_id(SKILL, skill.name)
        access.require_change(
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
                access.require_change(
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
