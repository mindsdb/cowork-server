from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.projects import ProjectCreateRequest, ProjectUpdateRequest
from cowork.services.projects import ProjectService


router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_projects(session: SessionDep):
    return ProjectService(session).list_projects()


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreateRequest, session: SessionDep):
    return ProjectService(session).create_project(body.name)


@router.patch("/{project_id}")
def update_project(project_id: UUID, body: ProjectUpdateRequest, session: SessionDep):
    try:
        return ProjectService(session).update_project(
            project_id, name=body.name, is_active=body.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: UUID, session: SessionDep):
    try:
        found = ProjectService(session).delete_project(project_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
