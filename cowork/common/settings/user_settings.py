from enum import Enum
from typing import Annotated, Callable, ClassVar, get_args

from pydantic import Field, SecretStr, field_validator, model_validator

from cowork.common.settings.app_settings import Settings


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MINDS_CLOUD = "minds_cloud"


class _DynamicOptions:
    """Annotated metadata marker for fields whose valid options are resolved lazily from a callable."""

    def __init__(self, fn: Callable[[], list[str]]) -> None:
        self._fn = fn

    def get(self) -> list[str]:
        return self._fn()


def _harness_options() -> list[str]:
    from cowork.harnesses.base import _registry
    return list(_registry.keys())


class UserSettings(Settings):
    PLANNING_MODEL_DEFAULTS: ClassVar[dict[Provider, str]] = {
        Provider.ANTHROPIC: "claude-sonnet-4-6",
        Provider.OPENAI: "gpt-4o",
        Provider.MINDS_CLOUD: "_reason_",
    }
    CODING_MODEL_DEFAULTS: ClassVar[dict[Provider, str]] = {
        Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
        Provider.OPENAI: "gpt-5.3-codex",
        Provider.MINDS_CLOUD: "_code_",
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
    minds_api_key: SecretStr | None = Field(
        default=None,
        title="MindsHub API Key",
        description="API key for MindsHub. Required if using MindsHub as a provider.",
    )
    minds_url: str = Field(
        default="https://api.mindshub.ai/v1",
        title="MindsHub URL",
        description="Base URL for the MindsHub API.",
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
    harness: Annotated[str, _DynamicOptions(_harness_options)] = Field(
        default="anton",
        title="Harness",
        description="The AI harness used to generate responses.",
    )

    @field_validator("harness")
    @classmethod
    def validate_harness(cls, v: str) -> str:
        options = _harness_options()
        if v not in options:
            available = ", ".join(options) or "none"
            raise ValueError(f"Unknown harness '{v}'. Available: {available}")
        return v

    @model_validator(mode='after')
    def apply_model_defaults(self) -> 'UserSettings':
        if self.planning_model is None:
            self.planning_model = self.PLANNING_MODEL_DEFAULTS.get(self.planning_provider)
        if self.coding_model is None:
            self.coding_model = self.CODING_MODEL_DEFAULTS.get(self.coding_provider)
        return self

    @staticmethod
    def field_is_sensitive(field_name: str) -> bool:
        annotation = UserSettings.model_fields[field_name].annotation
        return SecretStr in get_args(annotation)

    @staticmethod
    def field_options(field_name: str) -> list[str] | None:
        field_info = UserSettings.model_fields[field_name]
        for meta in field_info.metadata:
            if isinstance(meta, _DynamicOptions):
                return meta.get()
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return [e.value for e in annotation]
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, Enum):
                return [e.value for e in arg]
        return None


_cache: UserSettings | None = None


def get_user_settings() -> UserSettings:
    global _cache
    if _cache is None:
        _cache = _load_from_db()
    return _cache


def invalidate_user_settings_cache() -> None:
    global _cache
    _cache = None


def _load_from_db() -> UserSettings:
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    session = get_open_session()
    try:
        return SettingService(session).load()
    finally:
        session.close()
