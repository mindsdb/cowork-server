from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from cowork.api.v1.endpoints.guards import ensure_loopback_socket
from cowork.api.v1.permissions import AuthenticatedInOrgMode, DesktopOnly, require
from cowork.db.scoped import ScopedSessionDep
from cowork.db.session import get_session
from cowork.models.conversation import Conversation
from cowork.models.conversation_folder import ConversationFolder
from cowork.models.project import Project
from cowork.schemas.conversations import (
    ConversationCreateRequest,
    ConversationFolderAddRequest,
    ConversationListItem,
    ConversationMoveRequest,
    ConversationUpdateRequest,
)
from cowork.services import folder_listing
from cowork.services.conversation_folders import (
    ConversationFolderService,
    FolderAlreadyAttached,
    FolderLimitReached,
    FolderNotFound,
    FolderRefused,
    folder_refusal,
)
from cowork.services.conversations import ConversationService, InvalidPaginationParams
from cowork.services.task_objects import TaskObjectService

# AuthenticatedInOrgMode, declared explicitly: ScopedSessionDep already fails
# closed on its own (MissingTenantScopeError -> 401, cowork/db/scoped.py)
# whenever org mode has no org in scope. Declaring it too makes the
# requirement visible to a route walker instead of something only
# discoverable by reading scoped.py.
router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])
SessionDep = Annotated[Session, Depends(get_session)]


def _serialize_conversation(c, updated_at=None):
    # `updated_at` is the conversation's last-activity time, derived from its
    # messages (ENG-961) — the stored `modified_at` only moves on rename/move,
    # so it can't be trusted as "recent". Callers pass the derived value;
    # falling back to `created_at` keeps a message-less conversation stable.
    return ConversationListItem.serialize({
        "id": c.id,
        "title": c.topic,
        "preview": c.topic,
        "updated_at": updated_at or c.created_at,
        "created_at": c.created_at,
        "project": c.project.name if c.project else None,
        "project_path": c.project.path if c.project else None,
        "project_id": c.project_id,
        "harness": c.harness,
        "model": c.model,
    })


@router.get("/")
def list_conversations(
    scoped: ScopedSessionDep,
    project_id: UUID | None = None,
    project: str | None = None,
    limit: int = 50,
):
    all_projects = project == "all"
    resolved_project_id = project_id
    if not all_projects and resolved_project_id is None and project:
        from cowork.services.projects import ProjectService
        proj = ProjectService(scoped).get_project_by_name_or_none(project)
        if proj is not None:
            resolved_project_id = proj.id
    convs = ConversationService(scoped).list_conversations_with_activity(
        project_id=resolved_project_id, limit=limit, all_projects=all_projects,
    )
    return {"conversations": [_serialize_conversation(c, updated_at=activity) for c, activity in convs]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreateRequest, scoped: ScopedSessionDep):
    svc = ConversationService(scoped)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    try:
        conversation = svc.create_conversation(
            topic=body.topic or body.title or "Untitled task",
            project_id=project_id,
            harness=body.harness,
            model=body.model,
        )
    except ValueError as e:
        # e.g. a project_id that isn't visible in this scope
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize_conversation(conversation)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: UUID, scoped: ScopedSessionDep):
    svc = ConversationService(scoped)
    try:
        conversation = svc.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: UUID, body: ConversationUpdateRequest, scoped: ScopedSessionDep):
    svc = ConversationService(scoped)
    project_id = body.project_id
    if project_id is None and body.project:
        project = svc.project_by_name(body.project)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        project_id = project.id
    try:
        conversation = svc.update_conversation(
            conversation_id, topic=body.topic or body.title, project_id=project_id
        )
        return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{conversation_id}/move")
def move_conversation(conversation_id: UUID, body: ConversationMoveRequest, session: SessionDep, scoped: ScopedSessionDep):
    """Move a task to another project. With `move_objects` (default), the
    artifacts the task created are relocated into the destination project;
    otherwise only the task's project pointer changes. Attachment files
    follow the conversation automatically — purpose tags are keyed by
    conversation id, not project (ENG-338). The destination project must
    already exist (the client creates a new one first, then moves to its
    id)."""
    svc = ConversationService(scoped)
    try:
        conversation = svc.get_conversation(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    dest = None
    if body.project_id is not None:
        dest = scoped.get(Project, body.project_id)
    elif body.project:
        dest = svc.project_by_name(body.project)
    if dest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Destination project not found")

    source = conversation.project
    if body.move_objects and source is not None and dest.id != source.id:
        TaskObjectService(scoped).relocate_to_project(conversation, source, dest)

    conversation = svc.update_conversation(conversation_id, project_id=dest.id)
    return _serialize_conversation(conversation, updated_at=svc.last_message_at(conversation_id))


@router.get("/{conversation_id}/items")
def get_messages(
    conversation_id: UUID,
    scoped: ScopedSessionDep,
    limit: int | None = None,
    before: str | None = None,
):
    """Omitting both `limit` and `before` returns the full, unbounded
    history as a bare list — unchanged from before this endpoint supported
    pagination, so an existing caller that doesn't pass either param (e.g.
    cowork_evals, outside this repo) keeps working exactly as today. Passing
    either opts into the cursor-paginated envelope."""
    svc = ConversationService(scoped)
    try:
        if limit is None and before is None:
            return svc.get_messages(conversation_id)
        page_kwargs = {"before": before}
        if limit is not None:
            page_kwargs["limit"] = limit
        page = svc.get_messages_page(conversation_id, **page_kwargs)
    except InvalidPaginationParams as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return page.model_dump(by_alias=True)


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: UUID, scoped: ScopedSessionDep):
    found = ConversationService(scoped).delete_conversation(conversation_id)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"ok": True}


@router.delete("/{conversation_id}/turns/{message_id}")
def delete_conversation_turn(conversation_id: UUID, message_id: UUID, scoped: ScopedSessionDep):
    """Delete a turn (user+assistant exchange) and everything after it.

    message_id anchors the turn: the visible assistant message it produced,
    or (for a turn stopped/failed before any answer) the opening user
    message itself. A positional index doesn't survive lazy-loaded/
    paginated history, so this took over from an earlier
    `turn_index: int` path param — an old client still sending an int 422s
    here, and a new client sending a UUID would have 422d against the old
    route, so the break is fail-closed both directions.
    """
    svc = ConversationService(scoped)
    try:
        deleted = svc.delete_turn(conversation_id, message_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"ok": True, "deleted": deleted}


def _require_loopback_for_folders(request: Request) -> None:
    """Working folder routes name local paths, so they only answer over loopback."""
    ensure_loopback_socket(request, "working folders need a request over loopback")


_FOLDER_ROUTE_GUARDS = [Depends(require(DesktopOnly)), Depends(_require_loopback_for_folders)]


def _project_path(conversation: Conversation) -> str | None:
    """The chat's project folder, which a working folder may not sit inside."""
    return conversation.project.path if conversation.project else None


def _serialize_folder(folder: ConversationFolder, *, available: bool) -> dict[str, Any]:
    """One folder as the desktop app reads it; `available` is decided by the caller."""
    return {
        "id": folder.id,
        "path": folder.path,
        "name": Path(folder.path).name,
        "available": available,
    }


@router.get("/{conversation_id}/folders", dependencies=_FOLDER_ROUTE_GUARDS)
def list_conversation_folders(conversation_id: UUID, scoped: ScopedSessionDep):
    """The chat's working folders, oldest first, each marked available or not."""
    try:
        conversation, folders = ConversationFolderService(scoped).list_folders(conversation_id)
    except FolderNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    project_path = _project_path(conversation)
    return {
        "folders": [
            _serialize_folder(f, available=folder_refusal(f.path, project_path) is None)
            for f in folders
        ]
    }


@router.post(
    "/{conversation_id}/folders",
    status_code=status.HTTP_201_CREATED,
    dependencies=_FOLDER_ROUTE_GUARDS,
)
def add_conversation_folder(
    conversation_id: UUID, body: ConversationFolderAddRequest, scoped: ScopedSessionDep
):
    """Attach a local folder to the chat. The refusal text is meant for the user."""
    try:
        folder = ConversationFolderService(scoped).add_folder(conversation_id, body.path)
    except FolderNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except FolderRefused as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FolderAlreadyAttached as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except FolderLimitReached as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return _serialize_folder(folder, available=True)


@router.delete(
    "/{conversation_id}/folders/{folder_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=_FOLDER_ROUTE_GUARDS,
)
def remove_conversation_folder(conversation_id: UUID, folder_id: UUID, scoped: ScopedSessionDep):
    """Detach one working folder from the chat. Nothing on disk changes."""
    try:
        ConversationFolderService(scoped).remove_folder(conversation_id, folder_id)
    except FolderNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/{conversation_id}/folders/{folder_id}/files", dependencies=_FOLDER_ROUTE_GUARDS)
def list_conversation_folder_files(
    conversation_id: UUID, folder_id: UUID, scoped: ScopedSessionDep
):
    """Files under one working folder, bounded like the project file listing.

    The path comes from the stored row, never from the request, and is checked
    again here in case the folder changed since it was attached.
    """
    try:
        conversation, folder = ConversationFolderService(scoped).get_folder(
            conversation_id, folder_id
        )
    except FolderNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if folder_refusal(folder.path, _project_path(conversation)) is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Folder not available")
    base = Path(folder.path)
    budget = folder_listing.WalkBudget()
    files: list[dict[str, Any]] = []
    truncated = False
    for p in folder_listing.iter_folder_files(base, budget):
        meta = folder_listing.file_meta(p, base)
        if meta is None:
            continue
        if len(files) >= folder_listing.MAX_LISTED_FILES:
            truncated = True
            break
        files.append(meta)
    files.sort(key=lambda f: f["path"])
    response: dict[str, Any] = {"files": files}
    if truncated or budget.exhausted:
        response["truncated"] = True
    return response
