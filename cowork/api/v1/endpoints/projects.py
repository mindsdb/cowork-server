from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from cowork.db.scoped import ScopedSessionDep
from cowork.schemas.projects import ProjectCreateRequest, ProjectUpdateRequest
from cowork.services.projects import ProjectService


router = APIRouter()


@router.get("/")
def list_projects(session: ScopedSessionDep):
    service = ProjectService(session)
    # Bootstrap site: adopt the seeded GENERAL project into this org before
    # the first listing (no-op in local mode / once claimed).
    service.ensure_general_for_scope()
    return service.list_projects()


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreateRequest, session: ScopedSessionDep):
    return ProjectService(session).create_project(body.name)


@router.patch("/{project_id}")
def update_project(project_id: UUID, body: ProjectUpdateRequest, session: ScopedSessionDep):
    try:
        return ProjectService(session).update_project(
            project_id, name=body.name, is_active=body.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: UUID, session: ScopedSessionDep):
    try:
        found = ProjectService(session).delete_project(project_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
