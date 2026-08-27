from collections.abc import AsyncIterator
import inspect
import os
from pathlib import Path
import shutil
import tempfile

from cowork.build_info import surface_kwarg
from cowork.common.chat_session import build_chat_session
from cowork.common.logger import get_logger
from cowork.common.paths import cowork_home, pod_local_only
from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.base import ChannelContext, FileInputBlock, TextInputBlock, register
from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated, SkillCreated, TurnHistory, format_responses_stream
from cowork.models.conversation import Conversation
from cowork.models.skill import Skill
from cowork.harnesses.anton_harness.scratchpad_cell_replay import extract_scratchpad_cells_from_message_events
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings
from cowork.services.connectors.connections import service


logger = get_logger(__name__)


def _vault_scratch_dir() -> Path:
    """Where the temporary filtered data-vault directory is staged when a
    turn disables one or more connections (see ``_build_chat_session``).

    Local mode: cowork_home()/tmp, unchanged. Org mode: relocated off shared
    EFS storage by pod_local_only (see its docstring), because this directory
    carries no org_id segment: left on cowork_home() it would put every
    organization's temporary vault contents under the same shared, readable
    location.
    """
    return pod_local_only(cowork_home() / "tmp", "tmp")


#: Settings copied from the Cowork DB onto anton's own settings object, in
#: order. The last three are non-nullable ints with defaults — unlike the
#: entries above them, ``db_val`` is never None, so they ALWAYS override anton's
#: own 25/3 defaults (and any ANTON_* env value) for Cowork sessions.
#: ``max_turn_tokens`` is the per-turn spend ceiling (ENG-1286); it overlays the
#: SAME value anton defaults to rather than a looser one, because the
#: distribution that sized it was measured on this traffic.
_OVERLAID_SETTINGS: tuple[str, ...] = (
    "planning_provider", "planning_model",
    "coding_provider", "coding_model",
    "memory_enabled", "memory_mode",
    "episodic_memory", "proactive_dashboards", "act_first",
    "max_tool_rounds", "max_continuations", "max_turn_tokens",
)


def _overlay_user_settings(anton_settings, user) -> list[str]:
    """Copy the Cowork DB's settings onto anton's settings object.

    Extracted from ``_build_chat_session`` so it can be tested against a REAL
    ``AntonSettings``. It previously lived inline, and the only test of it
    re-implemented this loop in the test body — so dropping a key from the tuple
    (silently disabling a user-facing setting) or removing the skew guard below
    (a total agent outage) both shipped green.

    Returns the attrs actually applied, so a caller or test can assert on it.

    **The skew guard is the load-bearing part.** anton is pinned as a git dep on
    ``branch = "main"`` (see ``[tool.uv.sources]``), so cowork-server can ship a
    setting whose field has only reached anton's ``staging`` — it arrives on
    main at the weekly release, not when the anton PR merges. pydantic raises
    ``ValueError: "AntonSettings" object has no field "x"`` on setattr of an
    unknown field, and this runs on EVERY session build, so an unguarded overlay
    turns a one-week ordering gap into a total agent outage rather than a
    missing setting.
    """
    applied: list[str] = []
    for attr in _OVERLAID_SETTINGS:
        db_val = getattr(user, attr, None)
        if db_val is None:
            continue
        if not hasattr(anton_settings, attr):
            logger.warning(
                "anton settings has no field %r — skipping overlay; the pinned "
                "anton predates this setting (harmless: anton falls back to its "
                "own default until the next release)", attr,
            )
            continue
        # Provider enum -> string value for AntonSettings. The DB enum uses
        # snake_case (openai_compatible, minds_cloud) but AntonSettings /
        # LLMClient expect kebab-case (openai-compatible, minds-cloud).
        if hasattr(db_val, "value"):
            db_val = db_val.value.replace("_", "-")
        setattr(anton_settings, attr, db_val)
        applied.append(attr)
    return applied


def _apply_model_override(anton_settings, model: str | None) -> list[str]:
    """A per-conversation model pick (the composer's dropdown) overrides
    planning/coding/router for THIS call only — the account-wide
    planning_model/coding_model/router_model settings applied by
    ``_overlay_user_settings`` above are left untouched for every other
    conversation. Provider is deliberately NOT overridden: the composer's
    model list is itself scoped to whichever provider is already configured,
    so the existing planning/coding/router providers stay correct for the
    picked model.

    No-op (returns []) when ``model`` is falsy, so the account-wide defaults
    keep governing conversations with no per-conversation pick.

    Same hasattr skew guard as ``_overlay_user_settings`` — see its docstring.
    """
    if not model:
        return []
    applied: list[str] = []
    for attr in ("planning_model", "coding_model", "router_model"):
        if hasattr(anton_settings, attr):
            setattr(anton_settings, attr, model)
            applied.append(attr)
    return applied


def _apply_workspace_env_if_safe(workspace) -> bool:
    """Load `<project>/.anton/.env` into this process's environment, unless
    org mode. Returns whether it applied.

    `workspace` is an `anton.workspace.Workspace`; left unannotated since that
    type is only ever imported locally in `_build_chat_session`, not at
    module scope.

    Extracted from `_build_chat_session` so the guard is testable without
    constructing a full ChatSession.

    `Workspace.apply_env_to_process` (anton's workspace.py) loads every key
    from that file that isn't already set into THIS PROCESS's os.environ,
    not a child process's, cowork-server's own, for the rest of its life. In
    org mode that .env lives on shared EFS and any org's agent can write it;
    a PYTHONPATH or LD_PRELOAD entry there would turn the next subprocess
    this pod spawns into arbitrary code execution, and the mutation outlives
    this turn, reaching every later request from every tenant this pod
    serves.
    """
    if get_app_settings().tenancy_mode == "org":
        return False
    workspace.apply_env_to_process()
    return True


settings = AntonHarnessSettings()


def build_elicitor(conversation_id: str):
    """The question strategy for this conversation, or None when disabled.

    Returning None is the kill switch: anton only registers `ask_user` when
    an elicitor supports "choice", so the model reverts to asking in plain
    text with no silent-failure window. This holds only as long as the
    ChatSessionConfig built from this value is not also given a `console`:
    with elicitor=None and a console present, anton constructs its own
    CLIElicitor (which does support "choice"), silently reopening the
    switch. See tests/test_ask_user_flag.py for the guard.
    """
    if not get_app_settings().ask_user_enabled:
        return None
    from cowork.harnesses.anton_harness.elicitor import CoworkElicitor
    from cowork.streaming.answers import broker

    # timeout_s deliberately not passed: elicitor.DEFAULT_TIMEOUT_S is the one
    # place the number lives, so this call site does not restate it.
    return CoworkElicitor(conversation_id, broker)


def _split_turn_into_rows(history_slice: list) -> list[dict]:
    """Extract this turn's tool block-rows from anton's raw history slice.

    Delegates to anton rather than reimplementing: the pod entrypoint
    (`anton.cloud_turn.__main__`) builds the same rows for a cloud turn, and the
    two must agree byte for byte. Desktop and SaaS read the same `messages`
    table, so a divergence would not fail here — it would surface as an invalid
    tool_use -> tool_result sequence many turns later, in whichever path did not
    write the row. anton owns the implementation (ENG-1808); the slicing and the
    compaction/cancel guards stay on this side, which is the only side that
    knows whether a turn ended cleanly.

    Imported inside the function, like every other anton import in this file.
    `anton.cloud_turn.__init__` re-exports the pod entrypoint's contract and
    session builder, so a module-level import here would pull
    `anton.cloud_turn.session` into every process that merely loads this
    harness — the pod's session machinery, which the desktop path never uses.
    The function itself has no imports of its own beyond `__future__`.
    """
    from anton.cloud_turn.history_rows import split_turn_into_rows

    return split_turn_into_rows(history_slice)


def _build_filtered_vault(source_vault, disabled_connections: list[dict], temp_dir: Path, LocalDataVault):
    disabled_keys = {(d["engine"], d["name"]) for d in disabled_connections}
    filtered = LocalDataVault(temp_dir)
    for conn in source_vault.list_connections():
        if (conn["engine"], conn["name"]) not in disabled_keys:
            creds = source_vault.load(conn["engine"], conn["name"]) or {}
            filtered.save(conn["engine"], conn["name"], creds)
    return filtered


def _turn_style_context(channel: ChannelContext | None) -> str:
    """Lead block of the system-prompt suffix: desktop guidance for UI turns,
    support-chat guidance for channel turns.

    Both branches name how a finished file reaches the user: the channel branch
    says "I'm sending the file", the desktop branch points at the Live Artifacts
    panel. Without the desktop half, anton pasted the artifact's local path as a
    markdown link — inert in chat (ENG-1636).

    The desktop branch is asserted byte-for-byte by test_channel_context.py
    (cache-stable prompt prefix) — change both together.
    """
    if channel is None:
        return (
            "The Anton CoWork desktop UI displays progress, tool usage, and actions "
            "as separate structured activity rows. Keep assistant text focused on the "
            "user-facing answer; do not narrate internal work with status phrases like "
            "\"I'll check\", \"let me query\", or \"I have access\" unless that wording "
            "is itself the final answer the user needs. "
            "Files you create as artifacts appear automatically in the Live Artifacts "
            "panel beside the chat, where the user previews them and uses the Download "
            "control (and Open, on desktop). When a file is ready, tell the user it is "
            "in the Live Artifacts panel and can be downloaded there — do NOT hand them "
            "its location on disk. Never put a file's local path (for example "
            "C:\\Users\\... or /Users/...) into your reply as a markdown link or as "
            "text: such a link does nothing when clicked in chat, and the bare path "
            "only exposes the user's machine layout. Never invent a download URL such "
            "as sandbox:/mnt/data/...; no link of that form exists. If the user says "
            "they cannot find or download the file, point them again at the Live "
            "Artifacts panel's Download control (and Open, on desktop) — never repeat "
            "the path."
        )
    setting = (
        "a group chat with multiple participants" if channel.is_group
        else "a one-on-one direct chat"
    )
    name = f" ({channel.display_name})" if channel.display_name else ""
    operator = (
        f"\nOperator instructions for this chat:\n{channel.instructions.strip()}\n"
        if channel.instructions and channel.instructions.strip()
        else ""
    )
    return (
        f"You are replying inside {setting}{name} on {channel.channel_type} — a live "
        "messaging conversation, not the Anton CoWork desktop UI. Act as a concise, "
        "friendly support agent:\n"
        "- Write short plain-text replies. Avoid headings, tables, code blocks, and heavy "
        "Markdown — chat apps render them poorly.\n"
        "- Answer only what was asked; offer to go deeper rather than sending long explanations.\n"
        "- If a request is ambiguous, ask one short clarifying question instead of guessing.\n"
        "- Do not narrate internal work (\"let me check\", \"I'll query\"), and never mention "
        "the scratchpad, tools, internal file paths, or the desktop UI.\n"
        + (
            "- Several people can read your replies — keep them professional and "
            "self-contained.\n"
            "- Incoming messages are prefixed with the sender's name — use it to "
            "address the person who asked.\n"
            if channel.is_group else ""
        )
        + "- Files you create as artifacts are sent into this chat automatically right after "
        "your reply — tell the user you're sending the file rather than describing where it "
        "lives.\n"
        f"{operator}"
    )


# How a file the agent cannot reach gets to it. Stated on EVERY turn, not
# just turns that have attachments — the project context above tells the agent
# what it may not access and used to say nothing about how access is granted,
# so an agent asked to work on an unattached file had no legitimate move left.
# It cannot ask for a path (forbidden two lines above, and by the file-access
# policy), it cannot guess (forbidden by select_path's prompt), and the picker
# cannot render in cowork — which is how one session ended up inventing the
# user's sales data and reporting a forecast from it (ENG-1357).
_ATTACHMENT_AFFORDANCE = (
    " Files reach you in exactly two ways: they are already in the project "
    "directory above, or the user attaches them to this conversation. If the "
    "user refers to a file you cannot find in either place, ask them to attach "
    "it to the conversation — that is how they grant you access. Do not ask "
    "them to type or paste a filesystem path; you are not permitted to read "
    "files that are neither in the project nor attached."
)


def _conversation_attachment_context(conversation) -> str:
    """Prompt fragment telling the agent which files are attached to this
    conversation, and how the user can attach more.

    Uploads are stored under the files dir (``.cowork/files/<uuid>/<name>``),
    which is OUTSIDE the project workspace. An agent that only scans the
    project root therefore never sees them and wrongly tells the user no
    files were uploaded (the Cyberdeck bug). Handing it the exact paths lets
    it read them on demand on any turn — not just the turn they arrived on.

    Never returns "": every turn carries ``_ATTACHMENT_AFFORDANCE`` so the
    agent always knows attaching is the route to a file it cannot reach
    (ENG-1357). Two distinct outcomes, and the distinction is load-bearing:

    * **Known-empty** (no rows, or none still on disk) — say so, and name
      attachment as the remedy.
    * **Unknown** (detached session, no tenant scope, org mismatch, or a
      swallowed exception) — emit the affordance ALONE. Asserting "no files
      are attached" here would turn a failure to look into a confident false
      negative, which is the Cyberdeck bug in the other direction: the user
      attached a file and the agent flatly denies it exists.
    """
    try:
        from sqlalchemy.orm import object_session
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, scope_of_session
        from cowork.services.files import FileService, attachment_purpose

        db_session = object_session(conversation)
        if db_session is None:
            return _ATTACHMENT_AFFORDANCE
        # Re-wrap with the ORIGINAL scope the conversation was loaded under —
        # never derived from the row itself. No recorded scope in org mode is
        # an invariant violation: log it and list nothing.
        scope = scope_of_session(db_session)
        if scope is None:
            if get_app_settings().tenancy_mode == "org":
                logger.warning(
                    "attachments: session carries no tenant scope in org mode — "
                    "listing skipped (conversation %s)", conversation.id,
                )
                return _ATTACHMENT_AFFORDANCE
            scope = LOCAL_SCOPE
        if scope.org_mode and conversation.org_id != scope.org_id:
            logger.warning(
                "attachments: conversation %s org %r does not match scope org %r — listing skipped",
                conversation.id, conversation.org_id, scope.org_id,
            )
            return _ATTACHMENT_AFFORDANCE
        rows = FileService(ScopedSession(db_session, scope)).list_file_rows(
            purpose=attachment_purpose(str(conversation.id))
        )
        # Only list files that still exist on disk — a row whose file was
        # deleted would otherwise hand the agent a dead path to chase.
        # Resolve one row at a time: a single bad row (e.g. a path the OS
        # rejects) must not abort the whole list and hide every OTHER
        # attachment — skip the bad one and keep going.
        attached: list[str] = []
        for r in rows:
            try:
                path = getattr(r, "path", "")
                if path and Path(path).exists():
                    attached.append(f"  - {r.path}  ({r.filename})")
            except Exception:
                logger.warning(
                    "Skipping unresolvable attachment row (file id=%s) while "
                    "building context for conversation %s",
                    getattr(r, "id", "<unknown>"),
                    getattr(conversation, "id", "<unknown>"),
                    exc_info=True,
                )
        if not attached:
            return (
                " No files are currently attached to this conversation."
                + _ATTACHMENT_AFFORDANCE
            )
        return (
            " The user has attached the following files to THIS conversation. "
            "They live OUTSIDE the project directory, so a project-only scan will "
            "miss them — read them directly from these absolute paths whenever the "
            "user refers to uploaded or reference materials, and never report them "
            "missing just because they aren't in the project folder:\n"
            + "\n".join(attached)
        )
    except Exception:
        # Never crash a turn over attachment context — but don't fail
        # silently either. A swallowed error here is indistinguishable from
        # "no attachments", which is exactly how the agent ends up telling
        # the user no files were uploaded (the Cyberdeck bug this helper
        # exists to fix). Log it so the failure is diagnosable; the agent
        # still degrades gracefully to "".
        try:
            # A broken session state can make even attribute access raise —
            # the log line must never re-crash the handler it protects.
            conv_id = getattr(conversation, "id", "<unknown>")
        except Exception:
            conv_id = "<unknown>"
        logger.warning(
            "Failed to build conversation attachment context for conversation %s; "
            "the agent will not see attached files this turn",
            conv_id,
            exc_info=True,
        )
        return _ATTACHMENT_AFFORDANCE


@register
class AntonHarness:
    id: str = "anton"
    label: str = "Anton"
    formatter = staticmethod(format_responses_stream)

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # Per-conversation model pick (the composer's dropdown) — overrides
        # planning/coding/router for this call only; see _build_chat_session.
        model: str | None = None,
        # Per-task reasoning-effort pick (the composer's Effort sub-picker) —
        # overrides planning/coding effort for this call only; see
        # _build_chat_session and providers.build_llm_client's effort_override.
        reasoning_effort: str | None = None,
        disabled_connections: list[dict] | None = None,
        # Observability pass-through (see ResponsesRequest / HarnessProvider):
        # forwarded to Anton's per-turn TraceContext so they land on the
        # Langfuse trace the LLM router records. Generic so new eval/telemetry
        # tags or metadata need no change here.
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
        channel_context: ChannelContext | None = None,
    ) -> AsyncIterator[str]:
        if get_app_settings().tenancy_mode == "org":
            # Org-mode turns must run on the remote worker, never in this
            # process: _build_chat_session below hands the LLM a `scratchpad`
            # tool (anton/core/session.py) that spawns a per-named-venv
            # subprocess and pipes agent-written Python into it, exactly the
            # code execution this whole EFS-hardening task exists to keep out
            # of cowork-server. The remote-turn producer normally routes
            # streaming requests to the worker over Redis (see
            # handlers/responses.py's _select_producer), but three callers
            # reach this method directly, bypassing that gate entirely: the
            # legacy non-streaming branch in handlers/responses.py.handle
            # (ResponsesRequest.stream defaults to False, and any client can
            # leave it unset), _produce/_run_turn's in-process fallback
            # whenever COWORK_TURN_BACKEND isn't "remote", and the
            # channel-ingress runtime (cowork/channels/runtime.py). This
            # refusal is the single point that closes all three.
            raise RuntimeError(
                "Turns must run on the remote worker in this deployment; "
                "in-process execution is disabled."
            )
        temp_vault_dir: Path | None = None
        # Attribute + surface any artifact this turn created or edited. Anton's
        # artifact tools record what they touch as they run
        # (ChatSession.artifacts_touched), and that set — not a bare diff of the
        # artifacts dir — is what bounds the turn's cards: every conversation in
        # a project shares one artifacts directory, so a concurrent sibling
        # turn's brand-new artifact would otherwise land in this turn's diff and
        # be carded onto the wrong conversation (ENG-1933). The dir snapshot is
        # still taken, to tell CREATED from EDITED within that set.
        from cowork.services.task_objects import (
            finalize_turn_skill_drafts,
            index_turn_artifacts,
            publish_and_card_turn_artifacts,
            snapshot_artifact_state,
            snapshot_skill_drafts,
            snapshot_stray_skills,
        )
        project_path = Path(conversation.project.path)
        artifacts_base = project_path / ".anton" / "artifacts"
        # Names AND content mtimes: a name diff only reveals artifacts the turn
        # CREATED, and the reconciler must also see the ones it EDITED.
        before_slugs, before_mtimes = snapshot_artifact_state(artifacts_base)
        # Capture ids while the conversation is unambiguously attached — the
        # end-of-turn finally must not depend on the session still being live.
        conv_id = conversation.id
        conv_project_id = conversation.project_id
        # Same reason: the card carries the project name to the client, and reading
        # the relation after the turn could hit an expired session.
        conv_project_name = conversation.project.name
        # Skill drafts surface as cards (never auto-saved). Anton has no
        # skill-draft tool (it runs anton-core's own registry), so routing is
        # prompt + dir-diff only — consistent with its artifact flow. The
        # stray-skills relocation in finalize_turn_skill_drafts is the backstop.
        skill_drafts_base = project_path / ".anton" / "skill_drafts"
        before_drafts = snapshot_skill_drafts(skill_drafts_base)
        before_strays = snapshot_stray_skills(project_path / "skills")
        cards: list[dict] = []
        skill_drafts: list[dict] = []
        new_slugs: list[str] = []
        touched_slugs: set[str] = set()
        turn_scope = None
        turn_rows: list[dict] | None = None
        session = None
        seed_info: dict | None = None
        try:
            session, temp_vault_dir, seed_info = await self._build_chat_session(
                conversation,
                model=model,
                reasoning_effort=reasoning_effort,
                disabled_connections=disabled_connections or [],
                channel_context=channel_context,
            )
            # Length of the seeded history — everything anton appends past this
            # index is this turn's block-messages (tool_use / tool_result / text).
            # Guarded: an anton build (or test double) without `.history` simply
            # skips capture and falls back to text-only replay.
            seed_len = len(session.history) if hasattr(session, "history") else None
            compacted = False  # set if anton summarizes history mid-turn
            # Forward trace annotations only if the installed anton's
            # turn_stream accepts them. Deployed cowork-server resolves anton
            # from PyPI/main, which may predate the trace-tags kwargs (anton
            # #218); gating keeps those builds working — tags simply don't flow
            # until the anton floor is bumped — instead of raising TypeError.
            turn_kwargs: dict = {}
            _turn_params = inspect.signature(session.turn_stream).parameters
            if "trace_tags" in _turn_params:
                turn_kwargs["trace_tags"] = trace_tags
            if "trace_metadata" in _turn_params:
                turn_kwargs["trace_metadata"] = trace_metadata
            async for event in session.turn_stream(self._to_anton_input(input), **turn_kwargs):
                # Under context pressure anton summarizes history mid-turn,
                # reassigning session.history and invalidating seed_len.
                if type(event).__name__ == "StreamContextCompacted":
                    compacted = True
                yield event
            # Turn completed cleanly: capture its tool block-rows for
            # persistence. Skipped on cancel/error (the block below never runs),
            # so a partial turn falls back to text-only replay and anton's own
            # dangling-tool_use sealing keeps its in-memory history valid.
            #
            # Also skipped after a mid-turn compaction: seed_len no longer marks
            # this turn's start, so slicing could drop rows or surface an orphan
            # tool_result (its tool_use summarized away) → invalid replay. Text-
            # only replay stays valid, so degrade to it.
            if seed_len is not None and not compacted:
                turn_slice = session.history[seed_len:]
                while turn_slice and (
                    not isinstance(turn_slice[0], dict)
                    or turn_slice[0].get("role") != "assistant"
                ):
                    turn_slice = turn_slice[1:]
                turn_rows = _split_turn_into_rows(turn_slice) or None
        finally:
            if temp_vault_dir:
                shutil.rmtree(temp_vault_dir, ignore_errors=True)
            if session is not None and seed_info is not None:
                # Best-effort — must never mask the turn's real outcome.
                try:
                    self._persist_history_compaction(conversation, session, seed_info)
                except Exception:
                    logger.exception(
                        "[anton_harness] failed to persist history compaction for conversation %s",
                        conv_id,
                    )
            # One dir diff → index the new artifacts and work out what this turn
            # touched. Runs on every exit (success, error, cancel), so an artifact
            # is always indexed. Synchronous by design: an `await` here would be
            # skipped on cancellation, so anything awaited would silently not run.
            new_slugs, touched_slugs, turn_scope = index_turn_artifacts(
                conversation, conv_id, conv_project_id, artifacts_base,
                before_slugs, before_mtimes,
                # Anton reports both: `create_artifact` and `open_artifact`
                # (the only way to get a path to write into) both record, so
                # the same set bounds created AND edited. Read defensively —
                # on an early failure `session` is None, and an anton build
                # predating the field has no `artifacts_touched`; both degrade
                # to the pre-existing diff-only behaviour rather than dropping
                # the turn's cards entirely.
                tracked_new=getattr(session, "artifacts_touched", None),
                tracked_edits=getattr(session, "artifacts_touched", None),
            )
            skill_drafts = finalize_turn_skill_drafts(
                project_path, before_drafts, before_strays,
            )
        # Autopublish and cards live in the normal-completion path, matching what
        # cards already did: on Stop/cancel neither runs, and the next turn in this
        # project heals it (if there is one — an abandoned conversation never does).
        # Inline, so the card carries its published URL rather than appearing
        # without one and needing a later refresh.
        cards = await publish_and_card_turn_artifacts(
            artifacts_base,
            new_slugs=new_slugs,
            touched_slugs=touched_slugs,
            scope=turn_scope,
            project_id=str(conv_project_id) if conv_project_id else None,
            project_name=conv_project_name,
        )
        for card in cards:
            yield ArtifactCreated(card)
        for draft in skill_drafts:
            yield SkillCreated(draft)
        if turn_rows:
            yield TurnHistory(turn_rows)

    @staticmethod
    def _stamp_message(m) -> dict:
        """Embed a USER message's created_at as a `[YYYY-MM-DD HH:MM] ` prefix
        so the agent always knows WHEN something was said (even resuming a
        conversation days/weeks later). Absolute stamps are fixed per
        message, so the history prefix stays byte-stable across turns
        (cache-safe).

        User-only, matching anton's own live-turn stamping
        (core_agent/anton/core/session.py's _stamp_user_content). An earlier
        version stamped assistant replies too, which meant Anton's own prior
        replies came back to it prefixed with a timestamp in its replayed
        history, and it would imitate that visible convention in new
        output — most visible on short turns like "hi"/"who are you?" with
        little else to anchor generation.

        Extracted (not an inline closure) so this can be unit-tested
        directly against fake messages, same reasoning as _seed_history.
        """
        om = m.to_openai_message().model_dump()
        ts = m.created_at.strftime("%Y-%m-%d %H:%M") if getattr(m, "created_at", None) else None
        if m.role == "user" and ts and isinstance(om.get("content"), str) and om["content"]:
            om["content"] = f"[{ts}] {om['content']}"
        return om

    @staticmethod
    def _seed_history(ordered_messages: list, history_summary: str | None, cutoff_id, stamp) -> tuple[list[dict], dict]:
        """Build initial_history as [summary] + [messages after cutoff] when
        the saved compaction is still valid, else full history.

        Returns `(initial_history, seed_info)` — `seed_info` is what
        `_persist_history_compaction` needs to map this turn's compaction
        result back onto `ordered_messages`.

        Anton's `_summarize_history` never cuts mid tool_use/tool_result pair,
        so `cutoff_id` always lands on a clean boundary: the tail's first
        message is a real user text or an assistant message, never an orphan
        tool_result. That makes the `role == "user"` separator check below
        both safe and sufficient.
        """
        tail_start = 0
        if history_summary and cutoff_id:
            for i, m in enumerate(ordered_messages):
                if m.id == cutoff_id:
                    tail_start = i + 1
                    break
            else:
                history_summary = None  # cutoff message is gone — stale

        tail = [stamp(m) for m in ordered_messages[tail_start:]]
        if history_summary:
            summary_msg = {"role": "user", "content": history_summary}
            if tail and tail[0].get("role") == "user":
                # Same fix anton's own _summarize_history applies: two
                # consecutive user messages break/degrade most providers.
                initial_history = [
                    summary_msg,
                    {"role": "assistant", "content": "Understood — using that as reference."},
                    *tail,
                ]
                synthetic_prefix_len = 2
            else:
                initial_history = [summary_msg, *tail]
                synthetic_prefix_len = 1
        else:
            initial_history = tail
            synthetic_prefix_len = 0

        seed_info = {
            "ordered_messages": ordered_messages,
            "tail_start": tail_start,
            "synthetic_prefix_len": synthetic_prefix_len,
        }
        return initial_history, seed_info

    @staticmethod
    def _persist_history_compaction(conversation: Conversation, session, seed_info: dict) -> None:
        """Save anton's compacted summary + cutoff if it compacted this turn.

        `seed_info["ordered_messages"]`/`["tail_start"]` are what this turn's
        `initial_history` was built from; `["synthetic_prefix_len"]` is how
        many non-real entries (summary, plus an assistant separator if one was
        needed) were prepended ahead of them — `covered_through` from
        `session.last_compaction` counts those too, so they must be subtracted
        before mapping onto `ordered_messages`.

        `getattr` (not `session.last_compaction` directly): an anton build
        predating this property must no-op here, not raise — cowork-server and
        anton ship and deploy independently.
        """
        compaction = getattr(session, "last_compaction", None)
        if compaction is None:
            return
        offset = seed_info["synthetic_prefix_len"]
        covered = compaction["covered_through"] - offset
        if covered <= 0:
            return
        ordered_messages = seed_info["ordered_messages"]
        idx = seed_info["tail_start"] + covered - 1
        if not (0 <= idx < len(ordered_messages)):
            return
        from sqlalchemy.orm import object_session
        from cowork.db.scoped import adopt_scoped_session
        from cowork.services.conversations import ConversationService

        db_session = object_session(conversation)
        if db_session is None:
            return
        ConversationService(adopt_scoped_session(db_session)).update_history_compaction(
            conversation.id, compaction["summary"], ordered_messages[idx].id,
        )

    @staticmethod
    def _to_anton_input(input_blocks: list[dict]) -> str | list[dict]:
        if len(input_blocks) == 1 and input_blocks[0].get("type") == "text":
            return input_blocks[0]["text"]
        anton_blocks = []
        for block in input_blocks:
            if block.get("type") == "text":
                anton_blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                anton_blocks.append(block)
            elif block.get("type") == "file":
                anton_blocks.append({
                    "type": "text",
                    "text": f"[Attached file '{block['filename']}' is at: {block['path']}]",
                })
        return anton_blocks
        
    async def _build_chat_session(
        self,
        conversation: Conversation,
        model: str | None = None,
        reasoning_effort: str | None = None,
        disabled_connections: list[dict] | None = None,
        channel_context: ChannelContext | None = None,
    ):
        """Build the same core runtime the Anton CLI uses, scoped to one project."""
        from anton.chat_session import build_runtime_context
        from anton.config.settings import AntonSettings
        from anton.context.self_awareness import SelfAwarenessContext
        from anton.core.memory.cortex import Cortex
        # from anton.core.memory.episodes import EpisodicMemory
        from anton.core.memory.hippocampus import Hippocampus
        from anton.core.session import ChatSessionConfig, SystemPromptContext
        # from anton.memory.history_store import HistoryStore
        from anton.tools import CONNECT_DATASOURCE_TOOL
        from anton.workspace import Workspace
        # Cowork override — anton's stock PUBLISH_TOOL prints to a Rich
        # Console and pops a webbrowser, both of which die in the FastAPI
        # process. The wrapper exposes the same schema to the LLM but
        # routes through a server-aware handler.
        from .tools import (
            build_cowork_publish_tool,
            build_cowork_lookup_connector_tool,
            build_cowork_label_connection_tool,
            build_cowork_request_credentials_tool,
            build_cowork_create_skill_draft_tool,
        )
        PUBLISH_TOOL = build_cowork_publish_tool()
        LOOKUP_CONNECTOR_TOOL = build_cowork_lookup_connector_tool()
        REQUEST_CREDENTIALS_TOOL = build_cowork_request_credentials_tool()
        LABEL_CONNECTION_TOOL = build_cowork_label_connection_tool()
        CREATE_SKILL_DRAFT_TOOL = build_cowork_create_skill_draft_tool()

        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception:  # pragma: no cover
            LocalDataVault = None

        base = Path(conversation.project.path)

        # Build AntonSettings for workspace/path resolution (fields only
        # in AntonSettings like memory_dir, context_dir, artifacts_dir).
        # Then overlay the DB-authoritative values for all fields that
        # overlap between AntonSettings and UserSettings (API keys,
        # provider, model, memory flags, etc.) so the DB is the single
        # source of truth — no .env reload needed.
        from cowork.common.settings.user_settings import get_user_settings
        from pydantic import SecretStr

        anton_settings = AntonSettings()
        anton_settings.resolve_workspace(str(base))

        # Per-project skills
        project_skills_dir = base / "skills"
        project_skills_dir.mkdir(parents=True, exist_ok=True)
        anton_settings.skills_root = project_skills_dir

        # Host skills: read-only, they back
        # the deferred tool bundles (e.g. `connect-datasource` unlocks the
        # interactive connection tools). Path relative to this module.
        host_skills_dir = Path(__file__).parent / "skills"
        anton_settings.skills_extra_roots = [host_skills_dir]

        user = get_user_settings()
        _overlay_user_settings(anton_settings, user)

        # API keys: UserSettings stores SecretStr, AntonSettings uses plain str
        for attr in ("anthropic_api_key", "openai_api_key", "minds_api_key"):
            db_val = getattr(user, attr, None)
            if db_val is not None:
                setattr(anton_settings, attr, db_val.get_secret_value() if isinstance(db_val, SecretStr) else db_val)

        # URLs (skip empty strings so AntonSettings.model_post_init derivations
        # and AntonSettings' own publish_url default are preserved)
        for attr in ("minds_url", "openai_base_url", "publish_url"):
            db_val = getattr(user, attr, None)
            if db_val:
                setattr(anton_settings, attr, db_val)

        # Routing & summarization role → anton's router_* fields. The LLM
        # client is actually built by build_llm_client (which reads the
        # resolved router model directly), so this only keeps AntonSettings
        # consistent for any path that reads it. Guarded so an anton build
        # predating ENG-648 (no router_* fields) doesn't raise.
        router_provider = getattr(user, "router_provider", None)
        if router_provider is not None and hasattr(anton_settings, "router_provider"):
            anton_settings.router_provider = (
                router_provider.value.replace("_", "-")
                if hasattr(router_provider, "value") else router_provider
            )
        router_model = getattr(user, "router_model", None)
        if router_model is not None and hasattr(anton_settings, "router_model"):
            anton_settings.router_model = router_model

        _apply_model_override(anton_settings, model)

        workspace = Workspace(base)
        workspace.initialize()
        _apply_workspace_env_if_safe(workspace)

        anton_dir = base / ".anton"

        def _settings_path(value: object, fallback: Path) -> Path:
            raw = str(value or "").strip()
            if not raw:
                return fallback
            path = Path(raw).expanduser()
            return path if path.is_absolute() else base / path

        artifacts_dir = anton_dir / "artifacts"
        # Skills the agent builds stage here (sibling of artifacts, under the
        # off-limits .anton/) — never the live skills store, so a built skill is
        # surfaced as a draft card rather than auto-saved.
        skill_drafts_dir = anton_dir / "skill_drafts"
        context_dir = _settings_path(getattr(anton_settings, "context_dir", None), anton_dir / "context")
        episodes_dir = anton_dir / "episodes"
        project_memory_dir = anton_dir / "memory"
        for directory in (artifacts_dir, skill_drafts_dir, context_dir, episodes_dir, project_memory_dir):
            directory.mkdir(parents=True, exist_ok=True)

        llm_client = self._build_llm_client(effort=reasoning_effort)
        self_awareness = SelfAwarenessContext(context_dir)

        from cowork.common.settings.app_settings import get_app_settings
        from cowork.common.settings.user_settings import current_settings_scope
        from cowork.db.scoped import scoped_user_storage_root

        # Per-(org, user) via the turn's ambient scope — the in-process harness
        # must read/write the same global-scope memory the /memory API serves.
        global_memory_dir = scoped_user_storage_root(
            Path(get_app_settings().memory.root_dir).expanduser(),
            current_settings_scope(),
            store="memory",
        )
        global_memory_dir.mkdir(parents=True, exist_ok=True)
        cortex = Cortex(
            global_hc=Hippocampus(global_memory_dir),
            project_hc=Hippocampus(project_memory_dir),
            mode=anton_settings.memory_mode if anton_settings.memory_enabled else "off",
            llm_client=llm_client,
        )
        # TODO: Is episodic memory required given that we are handling history outside of the harness?
        # episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
        # episodic.resume_session(conversation_id)
        # history_store = HistoryStore(episodes_dir)
        # initial_history = history_store.load(conversation_id)

        # Conversation-attached uploads land in the files dir
        # (.cowork/files/<uuid>/<name>), OUTSIDE the project directory — so
        # the agent must be told their exact paths or it scans only the
        # project root and wrongly reports "no files uploaded" (Cyberdeck bug).
        attachment_context = _conversation_attachment_context(conversation)

        project_context = (
            f"You are operating in the project {conversation.project.name}."
            f"You have access to all of the files in the project at {str(base)} except for the .anton/ directory."
            "They are off limits. Do not mention the .anton/ directory in your responses."
            "You can perform operations on these files via the scratchpad."
            "You can freely read any of these project files."
            "If you need to perform any actions on these files, ask the user for permission first."
            "The only other files that you are allowed to access are any items that are attached to the conversation."
            "Access to any files not attached to the conversation or located outside the project is strictly forbidden."
            "ALWAYS use the scratchpad to interact with files."
            f"Your scratchpad's working directory is {str(base)} — bare relative paths like `open('data.csv')` resolve from the project root."
            + attachment_context
        )
        output_context = (
            # Artifacts now live in their own visible folder at the
            # project root (`<base>/artifacts/<slug>/...`), one folder
            # per output. The agent never picks the folder name itself
            # — it calls `create_artifact` to claim one, then writes
            # files into the absolute path the tool returns. Provenance
            # (which conversation, which turns) is tracked server-side
            # and stamped into each folder's metadata.json + README.md
            # automatically.
            f"User-facing artifacts (HTML dashboards, CSVs, PDFs, datasets, fullstack apps, etc.) live under `{str(artifacts_dir)}/`. "
            "Workflow:\n"
            "  1. Call `create_artifact(name, description, type)` BEFORE writing any output. "
            "It returns `{slug, path, ...}` — write your files into the returned `path`.\n"
            "  2. To MODIFY an existing artifact, call `list_artifacts()` to find its slug, "
            "then `open_artifact(slug)` to get the path again.\n"
            "  3. Use absolute paths from a scratchpad cell so the file always lands in the right place: "
            "`with open(f\"{path}/dashboard.html\", \"w\") as f: ...`\n"
            "Never write to the legacy `.anton/output/` directory — it's no longer scanned by the artifacts view."
        )
        # A skill is NOT an artifact and is NOT auto-saved. When the agent builds
        # or improves a skill (e.g. via skill-creator), it stages a DRAFT via the
        # create_skill_draft tool — the turn-end diff surfaces it as a card the
        # user saves/downloads. Editing an existing skill goes through the same
        # tool (it pre-seeds the draft from the saved version), NEVER by editing
        # the live `skills/` directory in place.
        skill_output_context = (
            "Skills you build or improve for the user (e.g. while running the skill-creator skill) "
            "are DRAFTS — never auto-saved. Workflow:\n"
            "  1. Call `create_skill_draft(name)` FIRST to claim a folder; it returns `{slug, path, skill_file}`. "
            "If a skill of that name is already saved, the folder comes pre-filled with its current contents "
            "so you edit from the saved version.\n"
            "  2. Write the SKILL.md to `skill_file` and any sibling files into `path` (one folder per skill).\n"
            "  3. A skill is NOT an artifact: never call `create_artifact` for a skill, and NEVER write a skill "
            "into the project `skills/` directory (that is the live store — editing it there bypasses the draft).\n"
            "The staged skill surfaces as a card the user explicitly saves or downloads; it is not saved until they do."
        )

        from cowork.common.settings.app_settings import get_app_settings

        # When connections are disabled, a temporary data vault is created because within Anton,
        # the vault is used to inject a prompt related to the connected data sources.
        data_vault = None
        temp_vault_dir: Path | None = None
        if LocalDataVault is not None:
            source_vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
            if disabled_connections:
                _tmp_base = _vault_scratch_dir()
                _tmp_base.mkdir(parents=True, exist_ok=True)
                temp_vault_dir = Path(tempfile.mkdtemp(prefix="cowork-vault-", dir=_tmp_base))
                data_vault = _build_filtered_vault(source_vault, disabled_connections, temp_vault_dir, LocalDataVault)
            else:
                data_vault = source_vault
            # restore_namespaced_env (instead of a bare inject_env loop) also
            # registers each connection's DS_* var names for credential
            # scrubbing — without it, scrub_credentials treats every field as
            # unknown and redacts non-secret values like base_url into
            # [DS_*] markers in user-facing output (ENG-688). It also clears
            # stale DS_* vars a previous turn injected for now-disabled
            # connections.
            from anton.utils.datasources import restore_namespaced_env

            restore_namespaced_env(data_vault)

        # TODO: Add guidance for integrations

        # Google Drive's google_drive connector uses the drive.file OAuth
        # scope, which only covers files the app created itself, plus files
        # the user explicitly granted access to via the Google Picker
        # (persisted as a `_picked_files` vault field — see
        # cowork/services/connectors/connections.py). A plain
        # files.list()/files.search() call does NOT return the latter, so
        # without calling them out by name here the agent has no way to
        # know they're reachable at all — inject_env() below only puts the
        # raw JSON in an env var, which isn't enough on its own for the
        # agent to notice or act on.
        #
        # Parsing `_picked_files` and applying the project-scoping rule is
        # connector logic, not agent logic, so it lives in
        # ConnectionsService.picked_files_by_project(); this loop only
        # injects env vars and turns the result into agent-facing prompt text.
        integration_guidance = ""
        picked_by_connection: dict[str, list[dict]] = {}
        if data_vault is not None:
            for conn in data_vault.list_connections():
                data_vault.inject_env(conn["engine"], conn["name"])
            picked_by_connection = service.picked_files_by_project(data_vault, conversation.project.name)

            if picked_by_connection:
                def _describe(f: dict, conn_name: str) -> str:
                    line = f"- {f.get('name', 'untitled')} (id: {f.get('id')}, connection: {conn_name}"
                    resource_key = f.get("resourceKey") or f.get("resource_key")
                    if resource_key:
                        line += f", resourceKey: {resource_key}"
                    return line + ")"

                picked_lines = [
                    _describe(f, conn_name)
                    for conn_name, files in picked_by_connection.items()
                    for f in files
                ]
                integration_guidance = (
                    "\n\nIMPORTANT — additional Google Drive files the user has explicitly granted "
                    "access to via the Google Picker, which a plain files.list()/files.search() call "
                    "will NOT return (the google_drive connector's scope only covers files this app "
                    "created itself, plus these specifically granted ones):\n"
                    + "\n".join(picked_lines)
                    + "\nWhenever you list, search, or enumerate Drive files for the user, you MUST "
                    "include every file above IN ADDITION to whatever files.list()/files.search() "
                    "returns — do not report only the API call's results. To read one of these "
                    "files' content, call files.get(fileId=...) directly with its id above; do not "
                    "expect it to appear in a files.list() response first. If a file above has a "
                    "resourceKey listed, you MUST send it or the call will fail with a 404 notFound "
                    "even though access was actually granted — either add header "
                    "'X-Goog-Drive-Resource-Keys: <id>/<resourceKey>' to the request, or pass "
                    "resourceKey=<resourceKey> as a query parameter. Some of these files may live in "
                    "a Shared Drive rather than the user's My Drive — Drive API calls silently return "
                    "404 notFound for Shared Drive items unless you pass supportsAllDrives=true (and, "
                    "for files.list()/files.search(), also includeItemsFromAllDrives=true). Always "
                    "include both params on any Drive API call touching these files; they're no-ops "
                    "for regular files, so there's no downside to always sending them. CRITICAL: when "
                    "calling files.list()/files.search(), do NOT pass corpora='allDrives' — unlike "
                    "supportsAllDrives/includeItemsFromAllDrives, that parameter is NOT properly scoped "
                    "by this connector's restricted OAuth grant and will return files across the user's "
                    "entire Google account that this app was never actually given access to. Omit "
                    "corpora entirely (or use corpora='user') — combined with "
                    "includeItemsFromAllDrives=true and supportsAllDrives=true, that already correctly "
                    "surfaces every file this app can legitimately see, Shared Drive items included."
                )

        # Canonical order (created_at, seq, ...); the bare `conversation.messages`
        # relationship is unordered and would scramble a turn's tool_use/tool_result
        # block-rows. Ordering needs the DB session, so the conversation must be
        # attached — callers always pass an attached instance; fail fast rather
        # than silently fall back to a scrambled, replay-breaking history.
        from sqlalchemy.orm import object_session
        from cowork.db.scoped import adopt_scoped_session
        from cowork.services.conversations import ConversationService

        db_session = object_session(conversation)
        if db_session is None:
            raise RuntimeError(
                f"Conversation {conversation.id} is detached from its Session; "
                "cannot resolve ordered history for replay."
            )
        ordered_messages = ConversationService(
            adopt_scoped_session(db_session)
        ).get_ordered_messages(conversation.id)

        cells = extract_scratchpad_cells_from_message_events(ordered_messages)
        os.environ["ANTON_SCRATCHPAD_PERSIST_SESSION"] = "true"

        replayable = [m for m in ordered_messages if m.role in {"user", "assistant"}]
        # Replay [summary] + [messages after cutoff] instead of full history
        # when a saved compaction is still valid (ENG-664). Disabled → plain
        # full history and `seed_info = None`, which skips persistence too.
        if user.history_compaction_enabled:
            initial_history, seed_info = self._seed_history(
                replayable,
                conversation.history_summary,
                conversation.history_summary_cutoff_id,
                self._stamp_message,
            )
        else:
            initial_history = [self._stamp_message(m) for m in replayable]
            seed_info = None

        config = ChatSessionConfig(
            llm_client=llm_client,
            settings=anton_settings,
            self_awareness=self_awareness,
            cortex=cortex,
            # episodic=episodic,
            system_prompt_context=SystemPromptContext(
                runtime_context=build_runtime_context(anton_settings),
                suffix=(
                    _turn_style_context(channel_context)
                    + f"{project_context}"
                    + f"{output_context}"
                    + f"{skill_output_context}"
                    + f"{integration_guidance}"
                ),
            ),
            workspace=workspace,
            data_vault=data_vault,
            initial_history=initial_history,
            # history_store=history_store,
            session_id=str(conversation.id),
            elicitor=build_elicitor(str(conversation.id)),
            # Surfaced on langfuse traces (Langfuse-Tags / metadata) so calls
            # are attributed to the active harness. self.id == "anton".
            harness=self.id,
            # WHERE the user is, which `harness` cannot say: this one server
            # serves both the desktop sidecar and the multi-tenant web build,
            # and both report harness="anton" (ENG-1459). Only the deployment
            # knows which, so it is resolved here rather than by anton.
            **surface_kwarg(ChatSessionConfig),
            proactive_dashboards=anton_settings.proactive_dashboards,
            act_first=anton_settings.act_first,
            # "Conversation started" stamp for the cache-stable prompt prefix
            # (anton 2a). The live current time is rendered separately in the
            # volatile tail, so resuming days later still reports the real "now".
            started_at=conversation.created_at,
            tools=[
                CONNECT_DATASOURCE_TOOL,
                PUBLISH_TOOL,
                LOOKUP_CONNECTOR_TOOL,
                REQUEST_CREDENTIALS_TOOL,
                LABEL_CONNECTION_TOOL,
                CREATE_SKILL_DRAFT_TOOL,
                # FETCH_SUBMISSION_TOOL,
                # UPDATE_FORM_TOOL,
            ],
            cells=cells
        )
        # Not `ChatSession(config)` directly: every construction of anton's
        # executor inside cowork-server goes through build_chat_session, which
        # refuses in org mode. stream_response already refuses earlier on this
        # path, so this is the second of two gates rather than the only one,
        # but keeping the construction uniform is what lets the static test
        # (tests/test_no_subprocess_static.py) treat any other ChatSession(...)
        # call under cowork/ as a new, unreviewed execution site.
        return build_chat_session(config), temp_vault_dir, seed_info

    @staticmethod
    def _build_llm_client(effort: str | None = None):
        from cowork.services.providers import build_llm_client
        return build_llm_client(effort_override=effort)
