"""What a push says, and how each platform's request carries it.

The app reads three custom fields from every notification: ``kind`` (one of
the PushKind values), ``path`` (an in-app route, for a task) and ``url`` (the
shared view of an artifact). Each field is left out when it has no value.
That contract is shared with the mobile app (mindsdb/cowork), so a change
here is a change there.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Long enough for any title a lock screen shows, short enough that no request
# comes near APNs' 4 KB payload limit.
_MAX_TITLE_CHARS = 120


class PushKind(StrEnum):
    TASK_FINISHED = "task_finished"
    SCHEDULED_RUN_FINISHED = "scheduled_run_finished"
    ARTIFACT_SHARED = "artifact_shared"


@dataclass(frozen=True)
class PushPayload:
    kind: PushKind
    title: str
    body: str
    path: str | None = None
    url: str | None = None

    def data(self) -> dict[str, str]:
        """The custom fields, every value a string, empty ones left out."""
        fields = {"kind": str(self.kind), "path": self.path or "", "url": self.url or ""}
        return {key: value for key, value in fields.items() if value}

    def apns_body(self) -> dict[str, Any]:
        return {
            "aps": {
                "alert": {"title": self.title, "body": self.body},
                "sound": "default",
                "thread-id": str(self.kind),
            },
            **self.data(),
        }

    def fcm_message(self, token: str) -> dict[str, Any]:
        return {
            "message": {
                "token": token,
                "notification": {"title": self.title, "body": self.body},
                "data": self.data(),
                "android": {"priority": "HIGH"},
            }
        }


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _conversation_path(conversation_id: str) -> str:
    return f"/c/{conversation_id}"


def task_finished(*, conversation_id: str, title: str | None, succeeded: bool) -> PushPayload:
    return PushPayload(
        kind=PushKind.TASK_FINISHED,
        title=_clip(title or "", _MAX_TITLE_CHARS) or "Your task",
        body="Finished." if succeeded else "Didn't finish. Tap to see what happened.",
        path=_conversation_path(conversation_id),
    )


def scheduled_run_finished(*, conversation_id: str | None, schedule_name: str | None, succeeded: bool) -> PushPayload:
    return PushPayload(
        kind=PushKind.SCHEDULED_RUN_FINISHED,
        title=_clip(schedule_name or "", _MAX_TITLE_CHARS) or "Scheduled task",
        body="Scheduled run finished." if succeeded else "Scheduled run didn't finish.",
        path=_conversation_path(conversation_id) if conversation_id else None,
    )


def artifact_shared(*, sharer: str | None, artifact_title: str | None, url: str | None) -> PushPayload:
    who = _clip(sharer or "", 60) or "Someone"
    what = _clip(artifact_title or "", 80)
    shared = f"“{what}”" if what else "an artifact"
    return PushPayload(
        kind=PushKind.ARTIFACT_SHARED,
        title=_clip(f"{who} shared {shared} with you", _MAX_TITLE_CHARS),
        body="Tap to open.",
        url=url or None,
    )
