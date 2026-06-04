from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from cowork.common.settings.user_settings import get_user_settings
from cowork.harnesses.base import get_harness
from cowork.models.message import Message as DBMessage
from cowork.schemas.responses import (
    Content,
    ContentType,
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
    Role,
)
from cowork.services.conversations import ConversationService
from cowork.services.files import FileService
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService, artifact_root_for_project
from cowork.services.skills import SkillService


import logging

logger = logging.getLogger(__name__)


# ── Artifact reconciliation helpers ───────────────────────────────

def _snapshot_artifacts(artifacts_root: Path) -> dict:
    """Take a pre-turn snapshot of the artifacts directory."""
    try:
        from anton.core.artifacts import snapshot_dir
        return snapshot_dir(artifacts_root)
    except Exception:
        logger.debug("Could not snapshot artifacts dir", exc_info=True)
        return {}


def _reconcile_artifacts(
    artifacts_root: Path,
    snapshot_before: dict,
    conversation_id: str,
    conversation_title: str | None,
    turn_index: int,
    user_prompt: str,
) -> None:
    """Diff artifacts and record provenance for any that changed."""
    try:
        from anton.core.artifacts import ArtifactStore, diff_snapshots, snapshot_dir
        from anton.core.artifacts.snapshot import _files_by_artifact
    except Exception:
        logger.debug("Anton artifact module unavailable; skipping reconcile", exc_info=True)
        return
    if not artifacts_root.exists():
        return
    after = snapshot_dir(artifacts_root)
    changed = diff_snapshots(snapshot_before, after)
    if not changed:
        return
    grouped = _files_by_artifact(changed)
    store = ArtifactStore(artifacts_root)
    for slug, files in grouped.items():
        try:
            store.record_turn(
                slug,
                conversation_id=conversation_id,
                conversation_title=conversation_title,
                turn_index=turn_index,
                summary=user_prompt[:240] if user_prompt else "",
                files_touched=files,
            )
            store.rescan_files(slug)
        except Exception:
            logger.debug("Failed to record artifact provenance for slug=%s", slug, exc_info=True)


class ResponsesHandler:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.harness = get_harness(get_user_settings().harness)
        self.last_conversation_id: str | None = None

    async def handle(self, request: ResponsesRequest) -> AsyncGenerator[str, None] | Response:
        logger.info("[responses] handle() called — conversation=%s, stream=%s", request.conversation, request.stream)
        await self.harness.sync_skills(SkillService(self.session).list_skills())

        conversation_service = ConversationService(self.session)
        project_id = self._resolve_project_id(request)

        harness_input = self._build_harness_input(request)
        original_content = self._extract_original_content(request)

        if request.conversation:
            try:
                conv_id = UUID(request.conversation)
            except ValueError:
                conv_id = None
            if conv_id is not None:
                conversation = conversation_service.get_conversation(conv_id)
            else:
                # Client sent a non-UUID id (e.g. legacy name-based format) —
                # treat as a new conversation since it won't exist in the DB.
                conversation = conversation_service.create_conversation(
                    topic=self._prompt_text(harness_input)[:80],
                    project_id=project_id,
                )
        else:
            conversation = conversation_service.create_conversation(
                topic=self._prompt_text(harness_input)[:80],
                project_id=project_id,
            )

        self.last_conversation_id = str(conversation.id)

        # Pre-load messages before committing the new user message so the ORM
        # cache doesn't include it when _build_chat_session reads history.
        _ = conversation.messages

        user_message = DBMessage(
            conversation_id=conversation.id,
            role=Role.user,
            content=original_content,
        )
        self.session.add(user_message)
        self.session.commit()
        self.session.refresh(user_message)

        # Pre-turn artifact snapshot so post-turn reconciliation can detect changes.
        artifacts_root: Path | None = None
        artifact_snapshot: dict = {}
        artifacts_root = artifact_root_for_project(getattr(conversation, "project", None))
        if artifacts_root is not None:
            artifact_snapshot = _snapshot_artifacts(artifacts_root)

        turn_index = sum(1 for m in conversation.messages if m.role == "user")
        user_prompt = self._prompt_text(harness_input)

        # The model provided as part of the request is ignored for now, because the Cowork
        # UI does not currently provide a way to specify it when making each request.
        # It is only extracted from the values specified in the settings.
        stream = self.harness.stream_response(
            conversation=conversation,
            input=harness_input,
            # model=request.model,
            disabled_connections=[dc.model_dump() for dc in request.disabled_connections]
            if request.disabled_connections else None,
        )

        reconcile_ctx = {
            "artifacts_root": artifacts_root,
            "snapshot_before": artifact_snapshot,
            "conversation_id": str(conversation.id),
            "conversation_title": getattr(conversation, "topic", None),
            "turn_index": turn_index,
            "user_prompt": user_prompt,
        }

        if request.stream:
            return self._stream(stream, conversation.id, request.model, reconcile_ctx)

        return await self._collect(stream, conversation.id, request.model, str(user_message.id), reconcile_ctx)

    async def _stream(
        self,
        stream,
        conversation_id: UUID,
        model: str,
        reconcile_ctx: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        collected_text: list[str] = []
        collected_events: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        event_count = 0
        async for sse_string in self.harness.formatter(stream, model, event_sink):
            event_count += 1
            if event_count <= 3 or "response.completed" in sse_string:
                logger.info("[responses] SSE event #%d (first 120 chars): %s", event_count, sse_string[:120].replace('\n', '\\n'))
            # Inject conversation_id into the response.created event so the
            # client learns the canonical (UUID) id for this conversation.
            if "response.created" in sse_string and "conversation_id" not in sse_string:
                try:
                    lines = sse_string.strip().split("\n")
                    data_line = next(l for l in lines if l.startswith("data:"))
                    payload = json.loads(data_line[5:])
                    payload["conversation_id"] = str(conversation_id)
                    sse_string = f"event: response.created\ndata: {json.dumps(payload)}\n\n"
                except Exception:
                    pass
            yield sse_string

        logger.info("[responses] stream finished — %d events, %d chars of text", event_count, len("".join(collected_text)))
        self._save_assistant_turn(conversation_id, "".join(collected_text), collected_events)
        self._maybe_reconcile_artifacts(reconcile_ctx)

    async def _collect(
        self,
        stream,
        conversation_id: UUID,
        model: str,
        output_item_id: str,
        reconcile_ctx: dict | None = None,
    ) -> Response:
        collected_text: list[str] = []
        collected_events: list[dict] = []

        def event_sink(event_type: str, data: dict) -> None:
            collected_events.append(data)
            if event_type == "response.output_text.delta":
                collected_text.append(data.get("delta", ""))

        async for _ in self.harness.formatter(stream, model, event_sink):
            pass

        assistant_text = "".join(collected_text)
        self._save_assistant_turn(conversation_id, assistant_text, collected_events)
        self._maybe_reconcile_artifacts(reconcile_ctx)

        return Response(
            status=ResponseStatus.completed,
            model=model,
            output=[self._build_output(output_item_id, assistant_text)],
        )

    def _save_assistant_turn(
        self,
        conversation_id: UUID,
        text: str,
        events: list[dict],
    ) -> None:
        ConversationService(self.session).save_assistant_turn(conversation_id, text, events)

    @staticmethod
    def _maybe_reconcile_artifacts(ctx: dict | None) -> None:
        if ctx is None or ctx.get("artifacts_root") is None:
            return
        try:
            _reconcile_artifacts(
                artifacts_root=ctx["artifacts_root"],
                snapshot_before=ctx["snapshot_before"],
                conversation_id=ctx["conversation_id"],
                conversation_title=ctx.get("conversation_title"),
                turn_index=ctx["turn_index"],
                user_prompt=ctx.get("user_prompt", ""),
            )
        except Exception:
            logger.debug("Post-turn artifact reconciliation failed", exc_info=True)

    def _build_harness_input(self, request: ResponsesRequest) -> list[dict]:
        blocks: list[dict] = []

        # Resolve attachment_ids to image/file blocks
        if request.attachment_ids:
            file_svc = FileService(self.session)
            for aid in request.attachment_ids:
                try:
                    content_type, filename, filepath = file_svc.get_file_content(UUID(aid))
                except (ValueError, Exception):
                    continue
                if content_type and content_type.startswith("image/"):
                    blocks.append(self._image_block(filepath, content_type))
                else:
                    blocks.append({"type": "file", "path": str(filepath), "filename": filename})

        # Extract text input
        if isinstance(request.input, str):
            blocks.append({"type": "text", "text": request.input})
        elif isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    if isinstance(msg.content, str):
                        blocks.append({"type": "text", "text": msg.content})
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, Content):
                                if item.type == ContentType.text and item.text:
                                    blocks.append({"type": "text", "text": item.text})
                                elif item.type == ContentType.file and item.file_id:
                                    try:
                                        content_type, filename, filepath = FileService(self.session).get_file_content(UUID(item.file_id))
                                    except ValueError:
                                        raise HTTPException(status_code=404, detail=f"File {item.file_id!r} not found")
                                    if content_type and content_type.startswith("image/"):
                                        blocks.append(self._image_block(filepath, content_type))
                                    else:
                                        blocks.append({"type": "file", "path": str(filepath), "filename": filename})
                    break

        return blocks or [{"type": "text", "text": ""}]

    def _resolve_project_id(self, request: ResponsesRequest) -> UUID:
        if request.project_id is not None:
            return request.project_id
        if request.project:
            try:
                return ProjectService(self.session).get_project_by_name(request.project).id
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=f"Project not found: {request.project}") from exc
        return GENERAL_PROJECT_ID

    @staticmethod
    def _image_block(filepath: Path, media_type: str) -> dict:
        data = filepath.read_bytes()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        }

    @staticmethod
    def _prompt_text(harness_input: list[dict]) -> str:
        return " ".join(b["text"] for b in harness_input if b.get("type") == "text")

    @staticmethod
    def _extract_original_content(request: ResponsesRequest) -> str | list:
        if isinstance(request.input, str):
            return request.input
        if isinstance(request.input, list):
            for msg in reversed(request.input):
                if msg.role == Role.user and msg.content:
                    if isinstance(msg.content, str):
                        return msg.content
                    if isinstance(msg.content, list):
                        return [item.model_dump() if isinstance(item, Content) else item for item in msg.content]
        return ""

    @staticmethod
    def _build_output(item_id: str, text: str) -> ResponseOutput:
        return ResponseOutput(
            id=item_id,
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=text)],
        )
