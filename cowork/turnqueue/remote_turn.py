"""Channel turns on the remote worker: the same stream_remote_replies
kind-dispatch _produce_remote's own inner generator does for browser turns
(cowork/handlers/responses.py), built separately for channels rather than
shared — sharing it would mean migrating the monkeypatch target of nearly
every one of that file's ~20 existing tests across a module boundary, for
one bounded, rarely-changing translation loop. Not worth the risk.

The five DB-touching helpers below ARE reused unchanged, via ResponsesHandler
directly — plain @staticmethods with no other class dependency, so calling
them through the class from here carries none of that risk: a class
attribute lookup always sees the class's current (possibly monkeypatched)
state, regardless of which module is asking."""
from __future__ import annotations

from uuid import UUID

from cowork.db.scoped import ScopedSession
from cowork.handlers.responses import ResponsesHandler
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE
from cowork.turnqueue.producer import step_stream_events, stream_remote_replies


class RemoteTurnFailed(Exception):
    """A remote channel turn ended in turn_failed. Carries (code, message)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


async def remote_turn_events(
    *,
    session: ScopedSession,
    conv_id: UUID,
    org_id: str | None,
    user_id: str | None,
    input_text: str,
    model: str | None,
    turn_rows: list,
    turn_id: int = 0,
    correlation_id: str | None = None,
    llm: dict | None = None,
):
    """Run one channel turn on the remote worker, yielding the same
    StreamTextDelta / StreamTaskProgress / ArtifactCreated / SkillCreated
    stream format_responses_stream already knows how to render — identical
    output shape to a local-mode turn, so callers downstream are unchanged.
    Mutates turn_rows in place on a turn_history reply (mirrors
    _produce_remote's own closure-captured turn_rows, as an explicit param
    since this isn't a closure). Raises RemoteTurnFailed on a worker-reported
    failure; the caller decides how to surface that."""
    from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
    from cowork.handlers._turn_history import sanitize_turn_history_rows
    from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated, SkillCreated
    from cowork.services.task_objects import (
        index_turn_artifacts,
        publish_and_card_turn_artifacts,
        remote_skill_draft_result,
        snapshot_artifact_state,
    )

    artifacts = ResponsesHandler._remote_artifacts_context(session, conv_id)
    before_slugs, before_mtimes = (
        snapshot_artifact_state(artifacts[1]) if artifacts else (set(), {})
    )
    new_slugs: list[str] = []
    touched_slugs: set[str] = set()
    turn_scope = None

    try:
        async for kind, data in stream_remote_replies(
            conversation_id=str(conv_id),
            org_id=org_id,
            user_id=user_id,
            input_text=input_text,
            model=model,
            turn_id=turn_id,
            history=ResponsesHandler._remote_history(session, conv_id),
            **ResponsesHandler._remote_workspace(session, conv_id),
            correlation_id=correlation_id,
            llm=llm,
        ):
            if kind == "turn_delta":
                yield StreamTextDelta(text=data.get("text", ""))
            elif kind == "turn_step":
                for event in step_stream_events(data):
                    yield event
            elif kind == "turn_memory":
                ResponsesHandler._persist_turn_memory(session, conv_id, data.get("entries") or [])
            elif kind == "turn_history":
                turn_rows[:] = sanitize_turn_history_rows(data.get("rows"))
            elif kind == "turn_skill":
                for entry in data.get("entries") or []:
                    payload, reasons = remote_skill_draft_result(entry)
                    if payload is not None:
                        yield SkillCreated(payload)
                    for reason in reasons:
                        yield StreamTaskProgress(phase="skill_draft_dropped", message=reason)
            elif kind == "turn_completed":
                break
            elif kind == "turn_failed":
                message = data.get("message") or GENERIC_TURN_ERROR_MESSAGE
                code = data.get("code") or GENERIC_TURN_ERROR_CODE
                raise RemoteTurnFailed(code, message)
    finally:
        if artifacts is not None:
            new_slugs, touched_slugs, turn_scope = index_turn_artifacts(
                artifacts[0], conv_id, artifacts[2], artifacts[1],
                before_slugs, before_mtimes,
            )

    if artifacts is not None:
        for card in await publish_and_card_turn_artifacts(
            artifacts[1],
            new_slugs=new_slugs,
            touched_slugs=touched_slugs,
            scope=turn_scope,
            project_id=artifacts[2],
            project_name=artifacts[3],
        ):
            yield ArtifactCreated(card)
