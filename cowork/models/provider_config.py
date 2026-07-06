from typing import Any

from sqlalchemy import JSON
from sqlmodel import Column, Field

from cowork.models.base import BaseSQLModel


class ProviderConfig(BaseSQLModel, table=True):
    """A user-configured LLM connection: one API key + base URL + model list.

    Distinct rows let the same UI provider `type` (e.g. "gemini") appear
    twice with different keys — this is how two free-tier Gemini accounts
    become two independent, round-robin-able failover candidates instead
    of fighting over the single-slot fields on `UserSettings`.
    """

    __tablename__ = "provider_configs"

    slug: str = Field(
        max_length=64,
        unique=True,
        index=True,
        description="Stable id used in model strings as '{slug}/{model_id}' and in API requests.",
    )
    type: str = Field(
        max_length=32,
        description="anthropic | openai | gemini | openai-compatible | minds-cloud",
    )
    label: str = Field(max_length=128, description="Display name in the model picker.")
    api_key_encrypted: str | None = Field(default=None, description="Fernet-encrypted API key.")
    base_url: str | None = Field(default=None)
    models: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Model ids this connection can serve, offered in the picker.",
    )
    enabled: bool = Field(default=True)
    priority: int = Field(
        default=100,
        description="Lower priority runs first in the failover chain when multiple candidates match.",
    )
