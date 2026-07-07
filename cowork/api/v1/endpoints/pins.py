from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.pins import PinRequest
from cowork.services.pins import PinService

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

_SUPPORTED_TYPES = {"project", "conversation", "schedule"}


@router.get("/")
def list_pins(session: SessionDep):
    return {"pins": PinService(session).list_pins()}


@router.post("/", status_code=status.HTTP_201_CREATED)
def pin_item(body: PinRequest, session: SessionDep):
    if body.item_type not in _SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported pin type. Must be one of: {sorted(_SUPPORTED_TYPES)}",
        )
    pin = PinService(session).pin_item(body.item_type, body.item_id, body.title)
    return {"pin": pin}


@router.post("/{item_id}/visit")
def record_visit(item_id: str, session: SessionDep, auto_pin: bool = False, title: str | None = None):
    """Record that a conversation was opened. Used for recents ordering."""
    if auto_pin:
        PinService(session).pin_item("conversation", item_id, title)
    return {"ok": True}


@router.delete("/{item_id}")
def unpin_item(item_id: str, session: SessionDep, item_type: str = "conversation"):
    if not PinService(session).unpin_item(item_type, item_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pin not found.")
    return {"ok": True}
