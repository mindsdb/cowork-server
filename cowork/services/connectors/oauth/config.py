from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoogleServiceConfig:
    engine: str
    scopes: list[str] = field(default_factory=list)


GOOGLE_SERVICES: dict[str, GoogleServiceConfig] = {
    "google-drive": GoogleServiceConfig(
        engine="google_drive",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/drive.file",
        ],
    ),
    "google-calendar": GoogleServiceConfig(
        engine="google_calendar",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            # Sensitive
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/calendar.events.readonly",
        ],
    ),
    "gmail": GoogleServiceConfig(
        engine="gmail",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            # Non-sensitive
            "https://www.googleapis.com/auth/gmail.labels",
            # Sensitive
            "https://www.googleapis.com/auth/gmail.send",
            # Restricted
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.metadata",
        ],
    ),
    "google-ads": GoogleServiceConfig(
        engine="google_ads",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/adwords",
        ],
    ),
    "google-analytics": GoogleServiceConfig(
        engine="google_analytics_4",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/analytics.readonly",
        ],
    ),
}
