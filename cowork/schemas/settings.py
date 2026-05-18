from enum import Enum
from typing import ClassVar, get_args

from pydantic import BaseModel, Field, SecretStr, model_validator

from cowork.common.settings import Settings


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class UserSettings(Settings):
    PLANNING_MODEL_DEFAULTS: ClassVar[dict[Provider, str]] = {
        Provider.ANTHROPIC: "claude-sonnet-4-6",
        Provider.OPENAI: "gpt-4o",
    }
    CODING_MODEL_DEFAULTS: ClassVar[dict[Provider, str]] = {
        Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
        Provider.OPENAI: "gpt-5.3-codex",
    }

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        title="Anthropic API Key",
        description="API key for Anthropic Claude models. Required if not using OpenAI.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        title="OpenAI API Key",
        description="API key for OpenAI models. Required if not using Anthropic.",
    )
    planning_provider: Provider = Field(
        default=Provider.ANTHROPIC,
        title="Planning Provider",
        description="The provider to use for the reasoning/planning model.",
    )
    planning_model: str | None = Field(
        default=None,
        title="Planning Model",
        description="The reasoning model used for planning. Defaults to the recommended model for the selected provider.",
    )
    coding_provider: Provider = Field(
        default=Provider.ANTHROPIC,
        title="Coding Provider",
        description="The provider to use for the coding model.",
    )
    coding_model: str | None = Field(
        default=None,
        title="Coding Model",
        description="The coding model. Defaults to the recommended model for the selected provider.",
    )

    @model_validator(mode='after')
    def apply_model_defaults(self) -> 'UserSettings':
        if self.planning_model is None:
            self.planning_model = self.PLANNING_MODEL_DEFAULTS.get(self.planning_provider)
        if self.coding_model is None:
            self.coding_model = self.CODING_MODEL_DEFAULTS.get(self.coding_provider)
        return self


def field_is_sensitive(field_name: str) -> bool:
    annotation = UserSettings.model_fields[field_name].annotation
    return SecretStr in get_args(annotation)


def field_options(field_name: str) -> list[str] | None:
    annotation = UserSettings.model_fields[field_name].annotation
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return [e.value for e in annotation]
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, Enum):
            return [e.value for e in arg]
    return None


class SettingUpsertRequest(BaseModel):
    value: str


class SettingResponse(BaseModel):
    key: str
    label: str
    description: str
    is_sensitive: bool
    is_set: bool
    value: str | None
    options: list[str] | None = None
