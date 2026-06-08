from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass


class LifecycleError(Exception):
    """Raised by a plugin's setup/teardown to surface a specific HTTP status
    (e.g. 400 missing bot_token, 409 no public base URL, 502 platform error)."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class LifecycleResult:
    active: bool
    detail: str


@dataclass
class LifecycleContext:
    """Host-provided capabilities for one setup/teardown call."""

    channel_type: str
    webhook_url: str | None
    credentials: Mapping[str, str]
    persist_credentials: Callable[[dict[str, str]], None]
    refresh_adapter: Callable[[], Awaitable[bool]]
    remove_adapter: Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class ChannelLifecycle:
    setup: Callable[[LifecycleContext], Awaitable[LifecycleResult]]
    teardown: Callable[[LifecycleContext], Awaitable[LifecycleResult]]
