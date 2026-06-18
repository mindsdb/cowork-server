"""Publish API endpoints — publish/unpublish HTML artifacts to 4nton.ai.

Ported from cowork/server/routes/utilities.py (publish section).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from cowork.services.publish import (
    list_publishable,
    publish_artifact as _publish,
    unpublish_artifact as _unpublish,
)

router = APIRouter()


class _AccessBody(BaseModel):
    # Mutually exclusive publish modes (ENG-322):
    #   public     — anyone with the link
    #   password   — visitors must enter `password`
    #   restricted — only `emails` and/or everyone in the owner's org
    mode: Literal["public", "password", "restricted"] = "public"
    password: str | None = None
    emails: list[str] = []
    org_allowed: bool = False


class _PublishBody(BaseModel):
    path: str
    # Back-compat: a bare top-level password still publishes password-protected.
    # New clients send the structured `access` object instead. Only a hash (and,
    # for restricted, the email list) leaves this machine; plaintext stays in
    # .published.json for the in-app reveal.
    password: str | None = None
    access: _AccessBody | None = None


@router.get("/")
async def list_publishable_endpoint():
    return list_publishable()


@router.post("/")
async def publish_artifact(req: _PublishBody):
    try:
        return _publish(req.path, req.password, access=req.access.model_dump() if req.access else None)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)


@router.delete("/")
async def unpublish_artifact(path: str = Query(..., description="Absolute path to the published HTML artifact")):
    try:
        return _unpublish(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)
