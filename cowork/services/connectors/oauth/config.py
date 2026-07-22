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
}
