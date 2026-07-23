from fastapi import APIRouter, HTTPException, UploadFile, status

from cowork.schemas.skills import SkillCreateRequest, SkillResponse, SkillUpdateRequest
from cowork.services.skills import SkillService

router = APIRouter()


@router.get("/")
def list_skills():
    skills = SkillService().list_skills()
    return {"skills": [SkillResponse.serialize(s) for s in skills]}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_skill(body: SkillCreateRequest):
    try:
        skill = SkillService().create_skill(
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
async def upload_skill(file: UploadFile):
    raw = await file.read()
    try:
        skill = SkillService().import_skill(raw, filename=file.filename)
    except FileExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    return SkillResponse.serialize(skill)


@router.get("/{skill_id}")
def get_skill(skill_id: str):
    try:
        return SkillResponse.serialize(SkillService().get_skill(skill_id))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/{skill_id}")
def update_skill(skill_id: str, body: SkillUpdateRequest):
    try:
        skill = SkillService().update_skill(
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
def delete_skill(skill_id: str):
    if SkillService().delete_skill(skill_id):
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found.")
