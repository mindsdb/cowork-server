from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/", response_model=list[SkillResponse])
def list_skills(session: SessionDep):
    return SkillService(session).list_skills()


@router.post("/", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest, session: SessionDep):
    try:
        skill = SkillService(session).create_skill(
            label=body.label,
            name=body.name,
            instructions=body.instructions,
            description=body.description,
            when_to_use=body.when_to_use,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return skill


@router.get("/{skill_id}", response_model=SkillResponse)
def get_skill(skill_id: UUID, session: SessionDep):
    try:
        return SkillService(session).get_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}", response_model=SkillResponse)
def update_skill(skill_id: UUID, body: SkillUpdateRequest, session: SessionDep):
    try:
        skill = SkillService(session).update_skill(
            skill_id,
            label=body.label,
            name=body.name,
            description=body.description,
            when_to_use=body.when_to_use,
            instructions=body.instructions,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return skill


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: UUID, session: SessionDep):
    if not SkillService(session).delete_skill(skill_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
