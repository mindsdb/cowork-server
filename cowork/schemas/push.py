from typing import Literal

from pydantic import Field

from cowork.schemas.base import CamelRequest


class PushDeviceRegistration(CamelRequest):
    """A phone the mobile app registers for push (POST /push/devices)."""

    token: str = Field(min_length=1, max_length=512)
    platform: Literal["ios", "android"]
    app_version: str | None = Field(default=None, max_length=64)
    # APNs only. Null tries production first and learns sandbox on its own.
    environment: Literal["production", "sandbox"] | None = None


class PushDeviceRemoval(CamelRequest):
    """The token to stop pushing to (DELETE /push/devices), on sign-out."""

    token: str = Field(min_length=1, max_length=512)
