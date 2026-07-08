"""CRUD for user-configured LLM connections (`provider_configs`).

Each row is one API key + base URL + model list, keyed by a stable
`slug`. Distinct rows let the same provider `type` (e.g. "gemini")
appear more than once with different keys, which is how two free-tier
accounts become two independent failover candidates instead of
fighting over a single-slot field.
"""
from __future__ import annotations

import re
from typing import Any

from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.models.provider_config import ProviderConfig

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]?$")
VALID_TYPES = {"anthropic", "openai", "gemini", "openai-compatible"}


def validate_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise ValueError(
            "slug must be lowercase alphanumeric with hyphens, 1-64 characters "
            "(e.g. 'nvidia', 'gemini-work', 'gemini-personal')"
        )


class ProviderRegistryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list(self, *, include_disabled: bool = True) -> list[ProviderConfig]:
        rows = self.session.exec(
            select(ProviderConfig).order_by(ProviderConfig.priority, ProviderConfig.slug)
        ).all()
        if include_disabled:
            return list(rows)
        return [r for r in rows if r.enabled]

    def get(self, slug: str) -> ProviderConfig | None:
        return self.session.exec(
            select(ProviderConfig).where(ProviderConfig.slug == slug)
        ).first()

    def create(
        self,
        *,
        slug: str,
        type: str,
        label: str,
        api_key: str | None,
        base_url: str | None,
        models: list[str],
        enabled: bool = True,
        priority: int = 100,
    ) -> ProviderConfig:
        validate_slug(slug)
        if type not in VALID_TYPES:
            raise ValueError(f"Unknown provider type '{type}'. Must be one of: {sorted(VALID_TYPES)}")
        if self.get(slug) is not None:
            raise ValueError(f"Provider '{slug}' already exists")
        row = ProviderConfig(
            slug=slug,
            type=type,
            label=label or slug,
            api_key_encrypted=encrypt(api_key) if api_key else None,
            base_url=base_url or None,
            models=models or [],
            enabled=enabled,
            priority=priority,
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def update(
        self,
        slug: str,
        *,
        label: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
        enabled: bool | None = None,
        priority: int | None = None,
    ) -> ProviderConfig:
        row = self.get(slug)
        if row is None:
            raise ValueError(f"Provider '{slug}' not found")
        if label is not None:
            row.label = label
        if api_key is not None and api_key != "***":
            row.api_key_encrypted = encrypt(api_key) if api_key else None
        if base_url is not None:
            row.base_url = base_url or None
        if models is not None:
            row.models = models
        if enabled is not None:
            row.enabled = enabled
        if priority is not None:
            row.priority = priority
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def delete(self, slug: str) -> bool:
        row = self.get(slug)
        if row is None:
            return False
        self.session.delete(row)
        self.session.commit()
        return True

    @staticmethod
    def decrypt_key(row: ProviderConfig) -> str | None:
        return decrypt(row.api_key_encrypted) if row.api_key_encrypted else None

    @staticmethod
    def to_public_dict(row: ProviderConfig) -> dict[str, Any]:
        """Serializable shape for the frontend — never includes the raw key."""
        return {
            "slug": row.slug,
            "type": row.type,
            "label": row.label,
            "hasApiKey": row.api_key_encrypted is not None,
            "baseUrl": row.base_url or "",
            "models": list(row.models or []),
            "enabled": row.enabled,
            "priority": row.priority,
        }
