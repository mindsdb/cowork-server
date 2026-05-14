"""Filesystem-backed Cowork conversation store primitives."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cowork.runtime.inference import profile_for_storage
from cowork.runtime.schemas import (
    CoworkConversation,
    CoworkEvent,
    CoworkMessage,
    CoworkTurn,
    ResolvedInferenceProfile,
    new_id,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class CoworkConversationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _store_dir(self, project_id: str) -> Path:
        path = self.root / project_id / ".cowork" / "conversations"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_path(self, conversation_id: str) -> Path | None:
        for path in self.root.glob("*/.cowork/conversations/*.json"):
            if path.stem == conversation_id:
                return path
        return None

    def create(
        self,
        *,
        project_id: str,
        harness: str,
        inference: ResolvedInferenceProfile,
        conversation_id: str | None = None,
        title: str = "",
        disabled_connections: list[dict[str, Any]] | None = None,
    ) -> CoworkConversation:
        now = utc_now_iso()
        conv = CoworkConversation(
            id=conversation_id or new_id("conv"),
            project_id=project_id,
            harness=harness,
            inference_profile=profile_for_storage(inference),
            title=title or "New task",
            preview=title[:80] if title else "",
            disabled_connections=disabled_connections or [],
            created_at=now,
            updated_at=now,
        )
        self.save(conv)
        return conv

    def save(self, conv: CoworkConversation) -> None:
        _atomic_write(self._store_dir(conv.project_id) / f"{conv.id}.json", conv.model_dump())

    def get(self, conversation_id: str) -> CoworkConversation | None:
        path = self._find_path(conversation_id)
        if path is None:
            return None
        try:
            return CoworkConversation.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list(self, *, limit: int = 200, project_id: str | None = None) -> list[dict[str, Any]]:
        pattern = f"{project_id}/.cowork/conversations/*.json" if project_id else "*/.cowork/conversations/*.json"
        conversations: list[CoworkConversation] = []
        for path in self.root.glob(pattern):
            try:
                conversations.append(CoworkConversation.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        conversations.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        return [self.meta(item) for item in conversations[:limit]]

    def meta(self, conv: CoworkConversation) -> dict[str, Any]:
        return {
            "id": conv.id,
            "title": conv.title or "Untitled task",
            "turns": len([m for m in conv.messages if m.role == "user"]),
            "preview": conv.preview,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "project": conv.project_id,
            "harness": conv.harness,
            "inferenceProfile": conv.inference_profile,
            "disabled_connections": conv.disabled_connections,
        }

    def append_message(self, conv: CoworkConversation, message: CoworkMessage) -> CoworkConversation:
        now = utc_now_iso()
        message.created_at = message.created_at or now
        message.updated_at = now
        conv.messages.append(message)
        if message.role == "user":
            text = message.content.strip()
            if text:
                conv.preview = conv.preview or text[:80]
                if not conv.title or conv.title == "New task":
                    conv.title = text[:80]
        conv.updated_at = now
        self.save(conv)
        return conv

    def start_turn(self, conv: CoworkConversation, user_message_id: str) -> tuple[CoworkConversation, CoworkTurn, CoworkMessage]:
        now = utc_now_iso()
        assistant = CoworkMessage(role="assistant", content="", created_at=now, updated_at=now)
        turn = CoworkTurn(user_message_id=user_message_id, assistant_message_id=assistant.id, started_at=now)
        assistant.turn_id = turn.id
        for msg in conv.messages:
            if msg.id == user_message_id:
                msg.turn_id = turn.id
                break
        conv.messages.append(assistant)
        conv.turns.append(turn)
        conv.updated_at = now
        self.save(conv)
        return conv, turn, assistant

    def append_event(self, conv: CoworkConversation, turn_id: str, event: CoworkEvent) -> CoworkConversation:
        for turn in conv.turns:
            if turn.id == turn_id:
                turn.events.append(event)
                break
        if event.type == "response.delta":
            delta = str(event.payload.get("delta") or "")
            if delta:
                self.append_assistant_delta(conv, turn_id, delta, save=False)
        elif event.type == "response.failed":
            text = str(event.payload.get("message") or event.payload.get("error") or "")
            if text:
                self.append_assistant_delta(conv, turn_id, text, save=False)
        elif event.type == "artifact.created":
            artifact = event.payload.get("artifact")
            if isinstance(artifact, dict):
                conv.artifacts.append(artifact)
        conv.updated_at = utc_now_iso()
        self.save(conv)
        return conv

    def append_assistant_delta(self, conv: CoworkConversation, turn_id: str, delta: str, *, save: bool = True) -> None:
        now = utc_now_iso()
        for msg in conv.messages:
            if msg.role == "assistant" and msg.turn_id == turn_id:
                msg.content += delta
                msg.updated_at = now
                break
        if save:
            conv.updated_at = now
            self.save(conv)

    def finish_turn(self, conv: CoworkConversation, turn_id: str, status: str, error: str | None = None) -> CoworkConversation:
        now = utc_now_iso()
        for turn in conv.turns:
            if turn.id == turn_id:
                if status in {"completed", "failed", "cancelled", "partial"}:
                    turn.status = status  # type: ignore[assignment]
                turn.completed_at = now
                turn.error = error
                break
        conv.updated_at = now
        self.save(conv)
        return conv

