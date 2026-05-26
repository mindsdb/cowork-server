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
            "email",
            "profile",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ],
    ),
    "google-calendar": GoogleServiceConfig(
        engine="google_calendar",
        scopes=[
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/calendar.events.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar",
        ],
    ),
    "gmail": GoogleServiceConfig(
        engine="gmail",
        scopes=[
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.modify",
        ],
    ),
    "google-ads": GoogleServiceConfig(
        engine="google_ads",
        scopes=[
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/adwords",
        ],
    ),
    "google-analytics": GoogleServiceConfig(
        engine="google_analytics_4",
        scopes=[
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/analytics.readonly",
        ],
    ),
    "gcp": GoogleServiceConfig(
        engine="gcp",
        scopes=[
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    ),
}
