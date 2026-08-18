"""Publish API endpoints — publish/unpublish HTML artifacts to MindsHub.

Ported from cowork/server/routes/utilities.py (publish section).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from cowork.api.v1.endpoints.guards import require_local_tenancy
from cowork.db.scoped import TenantScope, get_tenant_scope

from cowork.services.publish import (
    PublisherUnavailable,
    activate_version as _activate_version,
    desktop_publish_context as _desktop_context,
    list_publishable,
    list_versions as _list_versions,
    publish_artifact as _publish,
    unpublish_artifact as _unpublish,
    update_artifact as _update,
)

# The whole publish surface is desktop-only: it addresses artifacts by absolute
# server path and resolves the credential from stored provider settings, neither of
# which exists in an org deployment. Auto-publish is the org path instead.
router = APIRouter(dependencies=[Depends(require_local_tenancy)])


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
async def publish_artifact(req: _PublishBody, scope: TenantScope = Depends(get_tenant_scope)):
    try:
        artifact, artifacts_base, api_key, publish_url = _desktop_context(req.path)
        # The publisher resolves datasource secrets from the connector vault,
        # which is org-keyed; without the scope it would look in the shared root.
        # This router is local-only (see `require_local_tenancy` above), so the
        # scope is always a local one here and `vault_for_scope` returns the
        # unkeyed vault — threading it anyway keeps one resolution path for
        # every publish call site rather than a desktop-only exception.
        return _publish(
            artifact,
            artifacts_base=artifacts_base,
            api_key=api_key,
            publish_url=publish_url,
            password=req.password,
            access=req.access.model_dump() if req.access else None,
            scope=scope,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PublisherUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/update")
async def update_artifact(req: _UpdateBody):
    try:
        return _update(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PublisherUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/")
async def unpublish_artifact(path: str = Query(..., description="Absolute path to the published HTML artifact")):
    try:
        artifact, artifacts_base, api_key, publish_url = _desktop_context(path)
        return _unpublish(
            artifact, artifacts_base=artifacts_base,
            api_key=api_key, publish_url=publish_url,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PublisherUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


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
    except PublisherUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/activate")
async def activate_version_endpoint(req: _ActivateBody):
    try:
        return _activate_version(req.path, req.md5)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PublisherUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
