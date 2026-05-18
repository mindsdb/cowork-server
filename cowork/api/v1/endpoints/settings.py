from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.settings import SettingResponse, SettingUpsertRequest
from cowork.services.settings import SettingService

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/", response_model=list[SettingResponse])
def list_settings(session: SessionDep) -> list[SettingResponse]:
    return SettingService(session).list_settings()


@router.put("/{key}", response_model=SettingResponse)
def upsert_setting(
    key: str,
    body: SettingUpsertRequest,
    session: SessionDep,
) -> SettingResponse:
    try:
        return SettingService(session).upsert_setting(key, body.value)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_setting(key: str, session: SessionDep) -> None:
    try:
        deleted = SettingService(session).delete_setting(key)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting '{key}' is not set",
        )
