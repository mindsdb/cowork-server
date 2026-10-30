"""Channel turns on the remote worker, mirroring _produce_remote's own generator
but built separately, since sharing it would move ~20 tests' patch targets."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from uuid import UUID

from cowork.common.logger import log_context
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.units import run_db, run_efs
from cowork.handlers.responses import ResponsesHandler, _RemoteArtifactsDir
from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE
from cowork.services.connectors.vault_secrets import register_vault_secrets
from cowork.services.conversations import read_earlier_answers
from cowork.streaming.cancellation import stop_controller_turn
from cowork.turnqueue import producer
from cowork.turnqueue.producer import RemoteWorkspace, step_stream_events, stream_remote_replies


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ChannelTurnInputs:
    """What a channel turn's pod is seeded with, read as one unit."""

    history: list[dict]
    seed_info: dict | None
    workspace: RemoteWorkspace | None
    files: AgentFiles | None
    artifacts: _RemoteArtifactsDir | None
    started_at: str | None


def _read_channel_turn_inputs(session: ScopedSession, *, conv_id: UUID) -> _ChannelTurnInputs:
    """The scrubbed history, the conversation's workspace and files, where the
    project's artifacts live and who created the conversation, and when it
    started. The same reads as a browser turn's (_read_remote_turn_inputs),
    less the project memory a channel turn has never sent."""
    history, seed_info = ResponsesHandler._remote_seed_history(session, conv_id)
    workspace = ResponsesHandler._remote_workspace(session, conv_id)
    artifacts = ResponsesHandler._remote_artifacts_context(session, conv_id)
    return _ChannelTurnInputs(
        history=history,
        seed_info=seed_info,
        workspace=workspace,
        files=ResponsesHandler._remote_agent_files(session, conv_id, workspace=workspace),
        artifacts=None if artifacts is None else _RemoteArtifactsDir(
            base=artifacts[1],
            project_id=artifacts[2],
            project_name=artifacts[3],
            creator=getattr(artifacts[0], "created_by", None),
        ),
        started_at=ResponsesHandler._remote_started_at(session, conv_id),
    )


class RemoteTurnFailed(Exception):
    """A remote channel turn ended in turn_failed. Carries (code, message)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def remote_turn_events(
    *,
    scope: TenantScope,
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
    """Yields the same stream format_responses_stream renders for local turns.
    Mutates turn_rows in place; raises RemoteTurnFailed on a worker failure.

    Cancelled, or closed by its reader, before the turn's terminal reply, it
    writes the controller's cancel flag, so the sandbox turn stops too. At
    shutdown that is the drain of channel tasks (drain_background_tasks)
    cancelling it; no registry handle covers a channel turn.

    Each read and write is a unit of its own, so the turn holds no pooled
    connection while the pod runs."""
    from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
    from cowork.handlers._turn_history import remote_turn_history
    from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated, SkillCreated, TurnContext
    from cowork.services.task_objects import (
        publish_and_card_turn_artifacts,
        remote_skill_draft_result,
        snapshot_artifact_state,
    )

    # Minted here rather than inside stream_remote_replies, so a cancel below
    # can name the turn to the controller.
    correlation_id = correlation_id or producer._new_correlation_id()
    # A channel turn never goes through handle(), which registers these for
    # a web turn; the seed history below is scrubbed against them.
    await register_vault_secrets(scope)
    inputs = await run_db(partial(_read_channel_turn_inputs, conv_id=conv_id), scope=scope)
    seeded_history, seed_info = inputs.history, inputs.seed_info
    yield TurnContext(
        question=input_text,
        files=inputs.files,
        history=seeded_history,
        earlier_answers=await read_earlier_answers(conversation_id=conv_id, scope=scope),
    )
    artifacts = inputs.artifacts
    before_slugs, before_mtimes = (
        await run_efs(
            partial(snapshot_artifact_state, artifacts.base), scope=scope, deadline=False
        )
        if artifacts else (set(), {})
    )
    new_slugs: list[str] = []
    touched_slugs: set[str] = set()
    artifact_writes_allowed = False
    # Set only on turn_completed; see `turn_created_slugs(accept_unattributed=)`.
    completed_cleanly = False
    # Whether the pod compacted or hard-truncated the turn's context: its
    # compacted step comes for both, its compaction reply for a summary. Both
    # arrive before the history frame.
    compacted = False

    try:
        async for kind, data in stream_remote_replies(
            conversation_id=str(conv_id),
            org_id=org_id,
            user_id=user_id,
            input_text=input_text,
            model=model,
            turn_id=turn_id,
            history=seeded_history,
            workspace=inputs.workspace,
            started_at=inputs.started_at,
            correlation_id=correlation_id,
            llm=llm,
        ):
            if kind == "progress" and data.get("phase") == "workspace_authorized":
                artifact_writes_allowed = data.get("workspace_mode") == "persistent"
            elif kind == "turn_delta":
                yield StreamTextDelta(text=data.get("text", ""))
            elif kind == "turn_step":
                if data.get("step") == "compacted":
                    compacted = True
                for event in step_stream_events(data):
                    yield event
            elif kind == "turn_memory":
                # A channel turn runs for the org, not for one member, so it
                # writes memory with no principal.
                await run_db(
                    partial(
                        ResponsesHandler._persist_turn_memory,
                        conv_id=conv_id, entries=data.get("entries") or [], principal=None,
                    ),
                    scope=scope,
                )
            elif kind == "turn_history":
                history = remote_turn_history(pod_rows=data.get("rows"), compacted=compacted)
                turn_rows[:] = [] if history is None else history.rows
                if history is not None:
                    yield history
            elif kind == "turn_compaction":
                compacted = True
                await run_db(
                    partial(
                        ResponsesHandler._persist_remote_compaction,
                        conv_id=conv_id, data=data, seed_info=seed_info,
                    ),
                    scope=scope,
                )
            elif kind == "turn_skill":
                for entry in data.get("entries") or []:
                    payload, reasons = remote_skill_draft_result(entry)
                    if payload is not None:
                        yield SkillCreated(payload)
                    for reason in reasons:
                        yield StreamTaskProgress(phase="skill_draft_dropped", message=reason)
            elif kind == "turn_completed":
                completed_cleanly = True
                break
            elif kind == "turn_failed":
                message = data.get("message") or GENERIC_TURN_ERROR_MESSAGE
                code = data.get("code") or GENERIC_TURN_ERROR_CODE
                raise RemoteTurnFailed(code, message)
    except (asyncio.CancelledError, GeneratorExit):
        # Both terminals leave this block (break, raise), so the controller is
        # still running the turn here.
        await stop_controller_turn(correlation_id, turn_id=correlation_id, reason="channel_turn_ended")
        raise
    finally:
        if artifacts is not None and artifact_writes_allowed:
            new_slugs, touched_slugs = await _record_turn_artifacts(
                artifacts,
                conv_id=conv_id,
                scope=scope,
                before_slugs=before_slugs,
                before_mtimes=before_mtimes,
                completed_cleanly=completed_cleanly,
            )

    if artifacts is not None and artifact_writes_allowed:
        for card in await publish_and_card_turn_artifacts(
            artifacts.base,
            new_slugs=new_slugs,
            touched_slugs=touched_slugs,
            scope=scope,
            project_id=artifacts.project_id,
            project_name=artifacts.project_name,
        ):
            yield ArtifactCreated(card)


async def _record_turn_artifacts(
    artifacts: _RemoteArtifactsDir,
    *,
    conv_id: UUID,
    scope: TenantScope,
    before_slugs: set[str],
    before_mtimes: dict[str, int],
    completed_cleanly: bool,
) -> tuple[list[str], set[str]]:
    """The folders the turn created and touched. Those it created are indexed
    as this conversation's (and, in org mode, its creator's), in a unit.

    Never raises: this runs in the turn's ``finally``, so a failure here
    would replace the turn's real outcome. A failed diff records nothing, and
    outside org mode the next turn in the project reconciles it; in org mode
    nothing does."""
    from cowork.services.task_objects import record_new_artifacts, turn_artifact_changes

    try:
        # Waits for a file store slot rather than failing: in org mode
        # nothing reconciles a turn whose diff was refused.
        changes = await run_efs(
            partial(
                turn_artifact_changes,
                conversation_id=conv_id,
                artifacts_base=artifacts.base,
                before=before_slugs,
                before_mtimes=before_mtimes,
                # The project base is shared, so only folders whose
                # provenance names this conversation are its own.
                attribute_by_provenance=True,
                completed_cleanly=completed_cleanly,
            ),
            scope=scope,
            deadline=False,
        )
    except Exception:
        logger.warning("could not diff the artifacts of channel turn %s", conv_id, exc_info=True)
        return [], set()
    if changes.created:
        try:
            await run_db(
                partial(
                    record_new_artifacts,
                    conversation_id=conv_id,
                    project_id=UUID(artifacts.project_id) if artifacts.project_id else None,
                    slugs=changes.created,
                    creator=artifacts.creator,
                    artifact_ids=changes.created_ids,
                ),
                scope=scope,
            )
        except Exception:
            # record_new_artifacts logs each slug whose write fails itself;
            # this covers the unit's own refusal or a failed commit.
            logger.error(
                "could not record the artifacts a channel turn created",
                exc_info=True,
                extra=log_context(
                    project_id=artifacts.project_id,
                    conversation_id=conv_id,
                    artifact_slugs=changes.created,
                ),
            )
    return changes.created, changes.touched
