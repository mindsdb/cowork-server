from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OAuthServiceConfig:
    engine: str
    # Whether this service's credentials response should include the Google
    # Picker API key — a property of the service, not something callers
    # should re-derive by string-comparing engine names.
    uses_picker: bool = False


# Provider-agnostic registry: service id (the slug used in OAuth routes,
# e.g. "google-drive") -> engine + any provider-specific extras. Scopes,
# endpoints, and capability flags live in each connector's spec JSON (the
# `browser_oauth_builtin` method's `oauth` block) — the canonical
# description of a connector's OAuth shape — not here, to avoid a second
# copy that can silently drift out of sync with it. See
# OAuthService._oauth_config_for().
OAUTH_SERVICES: dict[str, OAuthServiceConfig] = {
    "google-drive": OAuthServiceConfig(engine="google_drive", uses_picker=True),
    "google-calendar": OAuthServiceConfig(engine="google_calendar"),
    "gmail": OAuthServiceConfig(engine="gmail"),
    "google-ads": OAuthServiceConfig(engine="google_ads"),
    "google-analytics": OAuthServiceConfig(engine="google_analytics_4"),
    "linear": OAuthServiceConfig(engine="linear"),
    "github": OAuthServiceConfig(engine="github"),
    "supabase": OAuthServiceConfig(engine="supabase"),
    "posthog": OAuthServiceConfig(engine="posthog"),
    # No `browser_oauth_builtin` method exists for hubspot (its method id is
    # "mcp" — see hubspot.json), so `_oauth_config_for()` will find nothing
    # and `start()`/`callback()` cleanly 500 if ever called for this service.
    # Registered here anyway because `/credentials` is engine-keyed and
    # shared by every connector, including HubSpot's Electron-native PKCE
    # flow (see cowork's oauth-identity.ts / DataVaultFormPanel.jsx) — it
    # still needs this app's client_id/secret even though it never goes
    # through OAuthService.start()/.callback() to get them.
    "hubspot": OAuthServiceConfig(engine="hubspot"),
}
