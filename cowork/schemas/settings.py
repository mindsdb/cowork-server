from typing import Literal

from pydantic import BaseModel, Field

from cowork.schemas.base import CamelResponse


class SettingUpsertRequest(BaseModel):
    value: str


class SettingsBulkUpsertRequest(BaseModel):
    # `None` joins "***" as a skip sentinel — the client's write-diff sends it
    # for untouched fields, and the service already skips both.
    values: dict[str, str | None]


class SettingResponse(BaseModel):
    key: str
    label: str
    description: str
    is_sensitive: bool
    is_set: bool
    value: str | None
    options: list[str] | None = None


# The gateway reasons a failed MindsHub health probe can report, verbatim. The
# classifier's allowlist is `PROBE_DENIAL_REASONS` in cowork/handlers/turn_errors.py.
ProbeDenialCode = Literal[
    "wallet_empty",
    "included_allowance_exhausted",
    "free_air_daily_spend_fuse_exceeded",
    "rate_limited",
    "policy_unavailable",
]


class ProviderProbeDenial(CamelResponse):
    """Why the MindsHub gateway refused a Settings health probe."""

    code: ProbeDenialCode
    # The gate's X-MindsHub-Reset-At instant, as the opaque ISO string it sent.
    # Set on the allowance and free-Air fuse denials; the renderer formats it.
    reset_at: str | None = None


class ProviderPingResponse(CamelResponse):
    """``POST /settings/test-providers``: each provider's probe, keyed by type."""

    # "ok" or "fail".
    provider_status: dict[str, str]
    # The dot tooltip's text, e.g. "HTTP 429: <gateway message>".
    provider_status_details: dict[str, str]
    # Only the types the MindsHub gateway refused with a named reason. Additive:
    # a renderer that predates it keeps reading the two maps above.
    provider_status_reasons: dict[str, ProviderProbeDenial] = Field(default_factory=dict)
