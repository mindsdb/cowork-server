from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/")
def list_skills(session: SessionDep):
    skills = SkillService(session).list_skills()
    return {"skills": [SkillResponse.serialize(s) for s in skills]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest, session: SessionDep):
    try:
        skill = SkillService(session).create_skill(
            label=body.label,
            name=body.name,
            instructions=body.instructions or "",
            description=body.description,
            enabled=body.enabled,
            projects=body.projects,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return SkillResponse.serialize(skill)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_skill(file: UploadFile, session: SessionDep):
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        )
    try:
        skill = SkillService(session).import_skill(content)
    except FileExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    return SkillResponse.serialize(skill)


@router.get("/{skill_id}")
def get_skill(skill_id: str, session: SessionDep):
    try:
        return SkillResponse.serialize(SkillService(session).get_skill(skill_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}")
def update_skill(skill_id: str, body: SkillUpdateRequest, session: SessionDep):
    try:
        skill = SkillService(session).update_skill(
            skill_id,
            label=body.label,
            name=body.name,
            description=body.description,
            instructions=body.instructions,
            enabled=body.enabled,
            projects=body.projects,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return SkillResponse.serialize(skill)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: str, session: SessionDep):
    if SkillService(session).delete_skill(skill_id):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
