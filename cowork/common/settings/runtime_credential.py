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

The desktop app's own browser is handed over the same way (LocalBrowserHost):
the app runs it inside its window and serves anton's browser API on loopback
with a key it makes at each launch, then hands this process the address and
the key. Both stay in memory, like the credential, and reach anton only as the
in-process turn's browser config (cowork.services.browser.desktop_browser_config).
The holder refuses wherever more than one person may share this process: in
org tenancy, and on a local deployment that trusts identity headers
(cowork.db.scoped.tenant_scope_required), so one person's browser can never
answer another person's turn.

Held in a module global with no lock. Uvicorn runs sync endpoints on a thread
pool, so the write and the reads land on different threads, but rebinding a
single name is atomic under the GIL and there is no read-modify-write here to
tear.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr

from cowork.common.settings.app_settings import get_app_settings

_minds_credential: str | None = None


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
    global _minds_credential
    _minds_credential = None


def get_minds_credential() -> str | None:
    """The held credential, or ``None`` when nothing has been handed over."""
    if _org_mode():
        return None
    return _minds_credential


class LocalBrowserHost(BaseModel):
    """The desktop app's own browser for this launch: where it listens, and
    the key that opens it. Built only from the checked hand-over body
    (cowork.api.v1.endpoints.runtime_credential.BrowserHostBody)."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    #: ``http://127.0.0.1:<port>``, exactly.
    url: str
    token: SecretStr


_browser_host: LocalBrowserHost | None = None


def _shared_process() -> bool:
    from cowork.db.scoped import tenant_scope_required

    return tenant_scope_required()


def set_browser_host(host: LocalBrowserHost) -> None:
    """Hold ``host`` as the desktop app's browser for the life of this process.
    Ignored where more than one person may share this process."""
    global _browser_host
    if _shared_process():
        return
    _browser_host = host


def clear_browser_host() -> None:
    """Drop the held browser. The app sends a blank hand-over when it quits its browser."""
    global _browser_host
    _browser_host = None


def get_browser_host() -> LocalBrowserHost | None:
    """The held browser, or ``None`` when the app has registered none or more
    than one person may share this process."""
    if _shared_process():
        return None
    return _browser_host
