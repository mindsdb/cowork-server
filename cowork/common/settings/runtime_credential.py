"""The MindsHub credential the desktop app hands over while it runs.

The desktop app holds the user's session credential and pushes it here after
sign-in, after every token refresh, and every time the sidecar starts. Nothing
here persists it, which is the whole point: an access token that lives ten
minutes must not outlive the process on disk, and a user-supplied API key is
kept out of the settings table for the same reason the desktop keeps it out of
``.env``.

``SettingService._raw_data`` overlays whatever is held here onto the stored
rows, so every reader of ``get_user_settings()`` picks it up with no further
wiring: both harnesses, the publish and comments proxies, and the readiness
gate behind ``/health``'s ``config_ready``.

**Local mode only.** An org deployment mints a per-turn credential in the turn
producer and its pods are never handed one, so a value set here would be one
tenant's credential answering for every tenant. The setter refuses in org mode
rather than trusting the route guard to be the only thing standing there.

Held in a module global with no lock. Uvicorn runs sync endpoints on a thread
pool, so the write and the reads land on different threads, but rebinding a
single name is atomic under the GIL and there is no read-modify-write here to
tear.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cowork.common.settings.app_settings import get_app_settings

_minds_credential: str | None = None
# The organization the desktop has mounted, and the turn key pinned to it.
# Set only by a desktop that sends its organization; older ones leave both None.
_organization_id: str | None = None
_inference_key: InferenceKey | None = None


@dataclass(frozen=True)
class InferenceKey:
    value: str
    organization_id: str
    instance_id: str
    expires_at: datetime


def _org_mode() -> bool:
    return get_app_settings().tenancy_mode == "org"


def set_minds_credential(value: str) -> None:
    """Hold ``value`` as the MindsHub credential for the life of this process.

    A blank value is a clear rather than a stored empty string, matching how
    ``SettingService._raw_data`` already treats a blank credential: no
    credential, so the provider reads as unconfigured instead of configured
    with something unusable.
    """
    global _minds_credential
    if _org_mode():
        return
    _minds_credential = value or None


def clear_minds_credential() -> None:
    """Drop the held credential. Sign-out and a failed hand-over both land here."""
    global _minds_credential, _organization_id, _inference_key
    _minds_credential = None
    _organization_id = None
    _inference_key = None


def set_organization(organization_id: str | None) -> None:
    global _organization_id
    if not _org_mode():
        _organization_id = organization_id


def get_organization() -> str | None:
    return None if _org_mode() else _organization_id


def set_inference_key(key: InferenceKey | None) -> None:
    global _inference_key
    if not _org_mode():
        _inference_key = key


def get_inference_key() -> InferenceKey | None:
    return None if _org_mode() else _inference_key


def get_inference_credential() -> str | None:
    """The credential LLM calls bill with.

    With a mounted organization, only a live key pinned to it: the session
    token would bill whatever organization Keycloak has active right now.
    """
    if _org_mode():
        return None
    if _organization_id is None:
        return get_minds_credential()
    key = _inference_key
    if (
        key
        and key.organization_id == _organization_id
        and key.expires_at > datetime.now(timezone.utc)
    ):
        return key.value
    return None


def get_minds_credential() -> str | None:
    """The held credential, or ``None`` when nothing has been handed over."""
    if _org_mode():
        return None
    return _minds_credential
