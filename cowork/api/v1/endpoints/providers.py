"""Provider registry endpoints — the user's configured LLM connections.

- GET    /                — list all registered providers (redacted)
- POST   /                — add a provider
- PUT    /{slug}          — update a provider (partial)
- DELETE /{slug}          — remove a provider
- POST   /{slug}/ping     — check connectivity with the stored key
- GET    /{slug}/models   — live model-id listing for this connection
"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.base import CamelRequest
from cowork.services.provider_registry import ProviderRegistryService
from cowork.services.providers import (
    fetch_anthropic_models,
    fetch_openai_compatible_models,
    ping_provider,
)

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]

_TYPE_DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
}


class ProviderCreateBody(CamelRequest):
    slug: str
    type: str
    label: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: list[str] = []
    enabled: bool = True
    priority: int = 100


class ProviderUpdateBody(CamelRequest):
    label: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: Optional[list[str]] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None


@router.get("/")
def list_providers(session: SessionDep):
    rows = ProviderRegistryService(session).list(include_disabled=True)
    return [ProviderRegistryService.to_public_dict(r) for r in rows]


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_provider(body: ProviderCreateBody, session: SessionDep):
    try:
        row = ProviderRegistryService(session).create(
            slug=body.slug,
            type=body.type,
            label=body.label,
            api_key=body.api_key,
            base_url=body.base_url,
            models=body.models,
            enabled=body.enabled,
            priority=body.priority,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return ProviderRegistryService.to_public_dict(row)


@router.put("/{slug}")
def update_provider(slug: str, body: ProviderUpdateBody, session: SessionDep):
    try:
        row = ProviderRegistryService(session).update(
            slug,
            label=body.label,
            api_key=body.api_key,
            base_url=body.base_url,
            models=body.models,
            enabled=body.enabled,
            priority=body.priority,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ProviderRegistryService.to_public_dict(row)


@router.delete("/{slug}")
def delete_provider(slug: str, session: SessionDep):
    deleted = ProviderRegistryService(session).delete(slug)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider '{slug}' not found")
    return {"ok": True}


@router.post("/{slug}/ping")
async def ping_provider_endpoint(slug: str, session: SessionDep):
    row = ProviderRegistryService(session).get(slug)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider '{slug}' not found")
    key = ProviderRegistryService.decrypt_key(row) or ""
    base_url = row.base_url or _TYPE_DEFAULT_BASE_URLS.get(row.type)
    ping_type = "openai-compatible" if row.type == "gemini" else row.type
    payload = {"type": ping_type, "apiKey": key, "baseUrl": base_url}
    status_str, detail = await ping_provider(payload)
    return {"status": status_str, "detail": detail}


@router.get("/{slug}/models")
async def list_provider_models(slug: str, session: SessionDep):
    row = ProviderRegistryService(session).get(slug)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider '{slug}' not found")
    key = ProviderRegistryService.decrypt_key(row)
    if not key:
        return {"models": None}

    if row.type == "anthropic":
        models = await fetch_anthropic_models(key)
        return {"models": models}

    base_url = row.base_url or _TYPE_DEFAULT_BASE_URLS.get(row.type)
    if not base_url:
        return {"models": None}
    models = await fetch_openai_compatible_models(base_url, key)
    return {"models": models}
