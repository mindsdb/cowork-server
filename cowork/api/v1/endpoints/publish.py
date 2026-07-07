"""Publish API endpoints — publish/unpublish HTML artifacts to MindsHub.

Ported from cowork/server/routes/utilities.py (publish section).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from cowork.services.publish import (
    activate_version as _activate_version,
)
from cowork.services.publish import (
    list_publishable,
)
from cowork.services.publish import (
    list_versions as _list_versions,
)
from cowork.services.publish import (
    publish_artifact as _publish,
)
from cowork.services.publish import (
    unpublish_artifact as _unpublish,
)
from cowork.services.publish import (
    update_artifact as _update,
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


class _UpdateBody(BaseModel):
    path: str


class _ActivateBody(BaseModel):
    path: str
    md5: str


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


@router.post("/update")
async def update_artifact(req: _UpdateBody):
    try:
        return _update(req.path)
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


@router.get("/versions")
async def list_versions_endpoint(
    path: str = Query(..., description="Absolute path to the published artifact"),
):
    try:
        return _list_versions(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)


@router.post("/activate")
async def activate_version_endpoint(req: _ActivateBody):
    try:
        return _activate_version(req.path, req.md5)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        detail = str(e)
        if "unavailable" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=502, detail=detail)
