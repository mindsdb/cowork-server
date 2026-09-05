import ipaddress
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from cowork.db.scoped import ScopedSessionDep
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import PROJECT_SLOTS, ProjectMemoryStore
from cowork.models.project import Project
from cowork.principal import Principal, get_principal
from cowork.schemas.projects import (
    ProjectCreateRequest,
    ProjectResponse,
    ProjectUpdateRequest,
)
from cowork.schemas.shared_resources import ProjectCapabilities
from cowork.api.v1.endpoints.guards import require_local
from cowork.services.projects import (
    GENERAL_PROJECT,
    ProjectNotFoundError,
    ProjectPathNotAllowedError,
    ProjectService,
)
from cowork.services.shared_resources import (
    PROJECT,
    PROJECT_INSTRUCTIONS,
    PROJECT_MEMORY,
    SKILL,
    SKILL_PROJECT_REFERENCES,
    ResourceDeletion,
    SharedResourceAccess,
    project_memory_resource_key,
    project_resource_key,
)
from cowork.services.skills import SkillService


router = APIRouter()


def _project_memory_slot_has_content(
    store: ProjectMemoryStore,
    slot: MemorySlot,
) -> bool:
    """Whether a project-memory slot holds content, counting unreadable bytes.

    The agent writes these slots, so a slot can hold non-UTF-8 bytes or be a
    planted symlink. ``read_state`` reports that as present but unreadable,
    which still has ownership to guard, so it answers True. Reading through the
    raising helper instead turned a rename or a delete into a 404 carrying a
    decoder message, or a 500, and because this inventory runs on every attempt
    the project could then never be renamed or deleted at all.
    """
    state = store.read_state(slot)
    if not state.readable:
        return True
    return bool(state.content.strip())


def _guard_project_instructions_for_move(
    project: Project,
    access: SharedResourceAccess,
    locks: ExitStack,
    pending_guards: list[tuple[object, str]],
    legacy_guards: list[object],
) -> None:
    """Block instruction first-writes and edits while a project path moves."""
    key = project_resource_key(project.id)
    path = Path(project.path) / ".anton" / "anton.md"
    locks.enter_context(access.coordination_lock(PROJECT_INSTRUCTIONS, key))
    exists = path.is_file()
    access.recover_stale_claim(
        PROJECT_INSTRUCTIONS,
        key,
        resource_exists=lambda: path.is_file(),
    )
    if access.has_attribution(PROJECT_INSTRUCTIONS, key):
        if access.claim_is_pending(PROJECT_INSTRUCTIONS, key):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project instructions are still being written",
            )
        locks.enter_context(
            access.mutation_lock(
                PROJECT_INSTRUCTIONS,
                key,
                resource_exists=lambda: path.is_file(),
            )
        )
        return
    if exists:
        row, created = access.ensure_mutation_identity(PROJECT_INSTRUCTIONS, key)
        if row.pending_claim_token:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project instructions are still being written",
            )
        if created:
            legacy_guards.append(row)
        locks.enter_context(
            access.mutation_lock(
                PROJECT_INSTRUCTIONS,
                key,
                resource_exists=lambda: path.is_file(),
            )
        )
        return
    row, token = access.reserve_claim(PROJECT_INSTRUCTIONS, key)
    if row is None:
        raise RuntimeError("Could not guard project instructions")
    if token is not None:
        pending_guards.append((row, token))
        return
    if row.pending_claim_token:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project instructions are still being written",
        )
    locks.enter_context(
        access.mutation_lock(
            PROJECT_INSTRUCTIONS,
            key,
            resource_exists=lambda: path.is_file(),
        )
    )


def _guard_project_memory_for_move(
    project: Project,
    access: SharedResourceAccess,
    locks: ExitStack,
    pending_guards: list[tuple[object, str]],
    legacy_guards: list[object],
) -> None:
    """Hold both memory slots across a project directory rename."""
    store = ProjectMemoryStore(Path(project.path))
    for slot in sorted(PROJECT_SLOTS, key=lambda item: item.value):
        key = project_memory_resource_key(project.id, slot.value)
        locks.enter_context(access.coordination_lock(PROJECT_MEMORY, key))

        def exists_check(slot=slot):
            return _project_memory_slot_has_content(store, slot)

        meaningful = _project_memory_slot_has_content(store, slot)
        access.recover_stale_claim(
            PROJECT_MEMORY,
            key,
            resource_exists=exists_check,
        )
        if access.has_attribution(PROJECT_MEMORY, key):
            if access.claim_is_pending(PROJECT_MEMORY, key):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Project memory is still being written",
                )
            locks.enter_context(
                access.mutation_lock(
                    PROJECT_MEMORY,
                    key,
                    resource_exists=exists_check,
                )
            )
            continue
        if meaningful:
            row, created = access.ensure_mutation_identity(PROJECT_MEMORY, key)
            if row.pending_claim_token:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Project memory is still being written",
                )
            if created:
                legacy_guards.append(row)
            locks.enter_context(
                access.mutation_lock(
                    PROJECT_MEMORY,
                    key,
                    resource_exists=exists_check,
                )
            )
            continue
        row, token = access.reserve_claim(PROJECT_MEMORY, key)
        if row is None:
            raise RuntimeError("Could not guard project memory")
        if token is not None:
            pending_guards.append((row, token))
            continue
        if row.pending_claim_token:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project memory is still being written",
            )
        locks.enter_context(
            access.mutation_lock(
                PROJECT_MEMORY,
                key,
                resource_exists=exists_check,
            )
        )


def _project_response(project: Project, access: SharedResourceAccess) -> dict:
    key = project_resource_key(project.id)
    if access.org_mode and access.has_trusted_actor:
        access.recover_stale_claim(
            PROJECT,
            key,
            resource_exists=lambda: access.session.get(Project, project.id) is not None,
        )
    pending = access.claim_is_pending(PROJECT, key)
    creator_id = project.created_by
    is_general = project.name == GENERAL_PROJECT
    can_change = not pending and not is_general and access.can_change(creator_id)
    directory_is_external = ProjectService(access.session).directory_is_external(project)
    return {
        **project.model_dump(),
        "attribution": access.attribution(
            PROJECT,
            key,
            fallback_creator_id=project.created_by,
            fallback_modified_at=project.modified_at,
        ),
        "capabilities": ProjectCapabilities(
            # A rename moves the directory, which is only defined inside the
            # projects root. A folder the user chose is theirs, not ours to move.
            can_rename=can_change and not directory_is_external,
            can_delete=can_change,
            can_edit_instructions=not pending and access.can_change(project.created_by),
            directory_is_external=directory_is_external,
        ),
    }


@router.get("/", response_model=list[ProjectResponse])
def list_projects(
    session: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    service = ProjectService(session)
    # Bootstrap site: adopt the seeded GENERAL project into this org before
    # the first listing (no-op in local mode / once claimed).
    service.ensure_general_for_scope()
    access = SharedResourceAccess(session, principal)
    return [_project_response(project, access) for project in service.list_projects()]


def _is_loopback_address(host: str | None) -> bool:
    """Whether an address literal names the loopback interface."""
    value = (host or "").strip().strip("[]").casefold()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def require_local_for_chosen_folder(
    body: ProjectCreateRequest, request: Request
) -> None:
    """A caller-chosen project folder is only accepted over loopback.

    A chosen path plus the project-file endpoints is read and write anywhere
    the server user can reach, so this has to hold on a deployment that is
    local-mode but not local-only.

    Neither of the obvious signals can carry it. The peer address is forgeable:
    the image runs uvicorn with ``--forwarded-allow-ips "*"``, so
    ``X-Forwarded-For`` rewrites ``request.client``. The configured host is
    blind: that same CMD passes ``--host 0.0.0.0`` on argv and sets no
    ``COWORK_SERVER_HOST``, so ``AppSettings.host`` still reads its loopback
    default inside the container.

    ``scope["server"]`` is the local address of the accepted socket. The proxy
    middleware rewrites only ``client`` and ``scheme``, and a caller cannot
    choose which interface their connection lands on, so a request that
    arrived over loopback really did. The desktop sidecar binds 127.0.0.1, so
    nothing legitimate is refused. ``require_local`` stays as a second layer
    and the service refuses org deployments outright as a third.
    """
    if body.path is None:
        return
    require_local(request)
    server = request.scope.get("server") or ()
    if not _is_loopback_address(server[0] if server else None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="a chosen project folder needs a request over loopback",
        )


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectResponse,
    dependencies=[Depends(require_local_for_chosen_folder)],
)
def create_project(
    body: ProjectCreateRequest,
    session: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    access = SharedResourceAccess(session, principal)
    access.require_actor()
    service = ProjectService(session)
    claim = None
    claim_token = None
    project_id = uuid4() if session.scope.org_mode else None
    try:
        with ExitStack() as locks:
            try:
                if project_id is not None:
                    locks.enter_context(
                        access.coordination_lock(
                            PROJECT,
                            project_resource_key(project_id),
                        )
                    )
                    claim, claim_token = access.reserve_claim(
                        PROJECT,
                        project_resource_key(project_id),
                    )
                    if claim is None or claim_token is None:
                        raise RuntimeError("Project ownership could not be reserved")
                    # This lock is also the project-name namespace: skill project
                    # validation and every org create/rename observe one canonical
                    # name allocation order across replicas.
                    locks.enter_context(
                        access.coordination_lock(SKILL_PROJECT_REFERENCES, "all")
                    )
                project = service.create_project(
                    body.name,
                    project_id=project_id,
                    path=Path(body.path) if body.path is not None else None,
                )
                if claim is not None and claim_token is not None:
                    finalized = access.finalize_claim(
                        claim,
                        claim_token,
                        action="create",
                    )
                    if finalized is None:
                        raise RuntimeError("Project ownership changed during creation")
            except Exception:
                session.rollback()
                if project_id is not None:
                    try:
                        service.delete_project(project_id)
                    except Exception:
                        session.rollback()
                    if claim is not None and claim_token is not None:
                        access.release_claim(claim, claim_token=claim_token)
                raise
    except ProjectPathNotAllowedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _project_response(project, access)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: UUID,
    body: ProjectUpdateRequest,
    session: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    service = ProjectService(session)
    access = SharedResourceAccess(session, principal)
    pending_guards: list[tuple[object, str]] = []
    legacy_guards: list[object] = []
    try:
        current = service.get_project(project_id)
        resource_key = project_resource_key(current.id)
        access.recover_stale_claim(
            PROJECT,
            resource_key,
            resource_exists=lambda: session.get(Project, project_id) is not None,
        )
        if access.claim_is_pending(PROJECT, resource_key):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project creation is still being finalized",
            )
        if body.name is None:
            updated = service.update_project(
                project_id,
                name=None,
                is_active=body.is_active,
            )
        elif not session.scope.org_mode:
            updated = service.update_project(
                project_id,
                name=body.name,
                is_active=body.is_active,
            )
        else:
            with ExitStack() as locks:
                attribution = locks.enter_context(
                    access.mutation_lock(
                        PROJECT,
                        resource_key,
                        fallback_creator_id=current.created_by,
                        resource_exists=lambda: session.get(Project, project_id)
                        is not None,
                    )
                )
                current = service.get_project(project_id)
                # Resolve only after taking the shared project-name/reference
                # namespace. A waiter must see names committed by an earlier
                # create or rename before it chooses a collision suffix.
                locks.enter_context(
                    access.coordination_lock(SKILL_PROJECT_REFERENCES, "all")
                )
                resolved_name = service.resolve_update_name(current, body.name)
                label_target = service.resolve_display_label(current, body.name)
                label_changes = label_target != (
                    current.display_name or current.name
                )
                if resolved_name == current.name and not label_changes:
                    # Active selection remains member-wide and is not a protected
                    # project mutation, even when the client echoes a normalized or
                    # collision-resolved spelling of the current name.
                    updated = service.update_project(
                        project_id,
                        name=None,
                        is_active=body.is_active,
                    )
                elif resolved_name == current.name:
                    # Same slug, different label. Every pair of Cyrillic names
                    # sanitizes to the same slug (ENG-1676), so this is a real
                    # rename that moves nothing: no directory hop, no skill
                    # reference rewrite, so none of the move machinery below
                    # applies. But the label is what every member sees, so it
                    # must clear the same creator/admin gate as a slug rename
                    # rather than riding the unprotected active-selection path
                    # above.
                    if current.name == GENERAL_PROJECT:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="The General project is immutable",
                        )
                    access.require_change(
                        (
                            attribution.created_by_id
                            if attribution is not None
                            else current.created_by
                        )
                        or current.created_by,
                        detail="Only the project creator or an organization admin can rename this project",
                    )
                    updated, label_stage = service.stage_project_update(
                        project_id,
                        resolved_name=None,
                        is_active=body.is_active,
                        display_label=body.name,
                    )
                    try:
                        access.stage_update(
                            PROJECT,
                            resource_key,
                            action="rename",
                            fallback_creator_id=updated.created_by,
                        )
                        updated = service.commit_staged_project_update(
                            updated,
                            label_stage,
                        )
                    except Exception:
                        session.rollback()
                        raise
                else:
                    if current.name == GENERAL_PROJECT:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="The General project is immutable",
                        )
                    access.require_change(
                        (
                            attribution.created_by_id
                            if attribution is not None
                            else current.created_by
                        )
                        or current.created_by,
                        detail="Only the project creator or an organization admin can rename this project",
                    )
                    _guard_project_instructions_for_move(
                        current,
                        access,
                        locks,
                        pending_guards,
                        legacy_guards,
                    )
                    _guard_project_memory_for_move(
                        current,
                        access,
                        locks,
                        pending_guards,
                        legacy_guards,
                    )

                    skill_service = SkillService(session.scope)
                    locked_slugs: set[str] = set()
                    # Slugs the listing keeps returning but that no longer
                    # resolve to a complete skill. They must be remembered, or
                    # the re-scan below offers them forever: `list_skills` only
                    # needs a directory holding SKILL.md, while
                    # `has_complete_skill` also rejects symlinks and frontmatter
                    # whose identity has drifted from the directory, so the two
                    # disagree permanently on a half-written skill.
                    skipped_slugs: set[str] = set()
                    # Re-scan after each lock batch. A skill mutation that
                    # started before this rename is either observed here or
                    # completes before its own resource lock is acquired.
                    while True:
                        unlocked = sorted(
                            set(skill_service.project_reference_slugs(current.name))
                            - locked_slugs
                            - skipped_slugs
                        )
                        if not unlocked:
                            break
                        for slug in unlocked:
                            locks.enter_context(access.coordination_lock(SKILL, slug))
                            access.recover_stale_claim(
                                SKILL,
                                slug,
                                resource_exists=lambda slug=slug: (
                                    skill_service.has_complete_skill(slug)
                                ),
                            )
                            try:
                                locks.enter_context(
                                    access.mutation_lock(
                                        SKILL,
                                        slug,
                                        resource_exists=lambda slug=slug: (
                                            skill_service.has_complete_skill(slug)
                                        ),
                                    )
                                )
                            except HTTPException as exc:
                                if exc.status_code == status.HTTP_404_NOT_FOUND:
                                    skipped_slugs.add(slug)
                                    continue
                                raise
                            locked_slugs.add(slug)

                    rewrites = skill_service.prepare_project_reference_rewrites(
                        current.name,
                        resolved_name,
                        slugs=sorted(locked_slugs),
                    )
                    updated, rename_stage = service.stage_project_update(
                        project_id,
                        resolved_name=resolved_name,
                        is_active=body.is_active,
                        display_label=body.name,
                        skill_rewrites=rewrites,
                    )
                    try:
                        access.stage_update(
                            PROJECT,
                            resource_key,
                            action="rename",
                            fallback_creator_id=updated.created_by,
                        )
                        updated = service.commit_staged_project_update(
                            updated,
                            rename_stage,
                        )
                    except Exception:
                        session.rollback()
                        try:
                            service.rollback_project_rename(rename_stage)
                        except Exception:
                            # The service logs individual restore failures.
                            pass
                        raise
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        for row, token in pending_guards:
            try:
                session.rollback()
                access.release_claim(row, claim_token=token)
            except Exception:
                session.rollback()
        for row in legacy_guards:
            try:
                session.rollback()
                access.release_pristine_identity(row)
            except Exception:
                session.rollback()
    return _project_response(updated, access)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: UUID,
    session: ScopedSessionDep,
    principal: Principal | None = Depends(get_principal),
):
    service = ProjectService(session)
    access = SharedResourceAccess(session, principal)
    pending_guards: list[tuple[object, str]] = []
    legacy_guards: list[object] = []
    project_coordination = None
    child_coordination = None
    try:
        project = service.get_project(project_id)
        if not session.scope.org_mode:
            if project.name == GENERAL_PROJECT:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="The General project is immutable",
                )
            if not service.delete_project(project_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found",
                )
            return
        resource_key = project_resource_key(project.id)
        project_coordination = access.coordination_lock(PROJECT, resource_key)
        project_coordination.__enter__()
        project = service.get_project(project_id)
        session.refresh(project)
        access.recover_stale_claim(
            PROJECT,
            resource_key,
            resource_exists=lambda: session.get(Project, project_id) is not None,
        )
        if access.claim_is_pending(PROJECT, resource_key):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Project creation is still being finalized",
            )
        if project.name == GENERAL_PROJECT:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The General project is immutable",
            )
        access.require_change(
            project.created_by,
            detail="Only the project creator or an organization admin can delete this project",
        )

        # Establish every child identity before taking the lock set. Missing
        # children get a pending reservation, which prevents a first writer
        # from appearing between this inventory and the project rmtree.
        child_locks: list[tuple[str, str, object]] = []
        preexisting_attribution: set[tuple[str, str]] = set()
        child_coordination = ExitStack()
        instructions_key = project_resource_key(project.id)
        child_coordination.enter_context(
            access.coordination_lock(PROJECT_INSTRUCTIONS, instructions_key)
        )
        instructions_path = Path(project.path) / ".anton" / "anton.md"
        instructions_exists = instructions_path.is_file()
        access.recover_stale_claim(
            PROJECT_INSTRUCTIONS,
            instructions_key,
            resource_exists=lambda: instructions_path.is_file(),
        )
        instructions_attributed = access.has_attribution(
            PROJECT_INSTRUCTIONS,
            instructions_key,
        )
        if instructions_attributed:
            preexisting_attribution.add((PROJECT_INSTRUCTIONS, instructions_key))
            child_locks.append(
                (
                    PROJECT_INSTRUCTIONS,
                    instructions_key,
                    lambda: instructions_path.is_file(),
                )
            )
        elif instructions_exists:
            row, created = access.ensure_mutation_identity(
                PROJECT_INSTRUCTIONS,
                instructions_key,
            )
            if row.pending_claim_token:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Project instructions are still being written",
                )
            if created:
                legacy_guards.append(row)
            preexisting_attribution.add((PROJECT_INSTRUCTIONS, instructions_key))
            child_locks.append(
                (
                    PROJECT_INSTRUCTIONS,
                    instructions_key,
                    lambda: instructions_path.is_file(),
                )
            )
        else:
            row, token = access.reserve_claim(
                PROJECT_INSTRUCTIONS,
                instructions_key,
            )
            if row is None:
                raise RuntimeError("Could not guard project instructions")
            if token is not None:
                pending_guards.append((row, token))
            elif row.pending_claim_token:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Project instructions are still being written",
                )
            else:
                preexisting_attribution.add((PROJECT_INSTRUCTIONS, instructions_key))
                child_locks.append(
                    (
                        PROJECT_INSTRUCTIONS,
                        instructions_key,
                        lambda: instructions_path.is_file(),
                    )
                )

        memory_store = ProjectMemoryStore(Path(project.path))
        for slot in sorted(PROJECT_SLOTS, key=lambda item: item.value):
            memory_key = project_memory_resource_key(project.id, slot.value)
            child_coordination.enter_context(
                access.coordination_lock(PROJECT_MEMORY, memory_key)
            )
            has_meaningful_memory = _project_memory_slot_has_content(
                memory_store,
                slot,
            )

            def exists_check(slot=slot):
                return _project_memory_slot_has_content(memory_store, slot)

            access.recover_stale_claim(
                PROJECT_MEMORY,
                memory_key,
                resource_exists=exists_check,
            )
            memory_attributed = access.has_attribution(PROJECT_MEMORY, memory_key)

            if memory_attributed:
                preexisting_attribution.add((PROJECT_MEMORY, memory_key))
                child_locks.append((PROJECT_MEMORY, memory_key, exists_check))
            elif has_meaningful_memory:
                row, created = access.ensure_mutation_identity(
                    PROJECT_MEMORY,
                    memory_key,
                )
                if row.pending_claim_token:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Project memory is still being written",
                    )
                if created:
                    legacy_guards.append(row)
                preexisting_attribution.add((PROJECT_MEMORY, memory_key))
                child_locks.append((PROJECT_MEMORY, memory_key, exists_check))
            else:
                row, token = access.reserve_claim(PROJECT_MEMORY, memory_key)
                if row is None:
                    raise RuntimeError("Could not guard project memory")
                if token is not None:
                    pending_guards.append((row, token))
                elif row.pending_claim_token:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Project memory is still being written",
                    )
                else:
                    preexisting_attribution.add((PROJECT_MEMORY, memory_key))
                    child_locks.append((PROJECT_MEMORY, memory_key, exists_check))

        with ExitStack() as locks:
            locks.enter_context(
                access.mutation_lock(
                    PROJECT,
                    resource_key,
                    fallback_creator_id=project.created_by,
                    resource_exists=lambda: session.get(Project, project_id)
                    is not None,
                )
            )
            for kind, key, exists_check in child_locks:
                locks.enter_context(
                    access.mutation_lock(
                        kind,
                        key,
                        resource_exists=exists_check,
                    )
                )

            # Re-read after waiting for all resource locks. Any writer that won
            # before the guard is now reflected in both bytes and attribution.
            project = service.get_project(project_id)
            access.require_change(
                project.created_by,
                detail="Only the project creator or an organization admin can delete this project",
            )
            instructions_path = Path(project.path) / ".anton" / "anton.md"
            memory_store = ProjectMemoryStore(Path(project.path))
            child_deletions: list[tuple[str, str, str]] = []
            if (
                instructions_path.is_file()
                or (
                    PROJECT_INSTRUCTIONS,
                    instructions_key,
                )
                in preexisting_attribution
            ):
                child_deletions.append(
                    ResourceDeletion(PROJECT_INSTRUCTIONS, instructions_key, "delete")
                )
            for slot in PROJECT_SLOTS:
                memory_key = project_memory_resource_key(project.id, slot.value)
                if (
                    _project_memory_slot_has_content(memory_store, slot)
                    or (
                        PROJECT_MEMORY,
                        memory_key,
                    )
                    in preexisting_attribution
                ):
                    child_deletions.append(
                        ResourceDeletion(PROJECT_MEMORY, memory_key, "clear")
                    )

            deletion_resources = [
                *child_deletions,
                ResourceDeletion(PROJECT, resource_key, "delete"),
            ]
            found = service.delete_project(
                project_id,
                before_commit=lambda: access.stage_deletes(
                    deletion_resources,
                    pending_claim_tokens={token for _row, token in pending_guards},
                ),
            )
            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found",
                )
    # Only a genuinely absent project is a 404. A blanket ValueError handler
    # also swallowed decode and validation faults raised while inventorying the
    # children, which then answered "not found" for a project that exists.
    except ProjectNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    finally:
        for row, token in pending_guards:
            try:
                session.rollback()
                access.release_claim(row, claim_token=token)
            except Exception:
                session.rollback()
        for row in legacy_guards:
            try:
                session.rollback()
                access.release_pristine_identity(row)
            except Exception:
                session.rollback()
        if child_coordination is not None:
            child_coordination.close()
        if project_coordination is not None:
            project_coordination.__exit__(None, None, None)
