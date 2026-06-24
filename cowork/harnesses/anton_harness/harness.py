from collections.abc import AsyncIterator
import os
from pathlib import Path
import shutil
import tempfile

from cowork.common.logger import get_logger
from cowork.harnesses.base import FileInputBlock, TextInputBlock, register
from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated, format_responses_stream
from cowork.models.conversation import Conversation
from cowork.models.skill import Skill
from cowork.harnesses.anton_harness.scratchpad_cell_replay import extract_scratchpad_cells_from_message_events
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings


logger = get_logger(__name__)
settings = AntonHarnessSettings()


def _build_filtered_vault(source_vault, disabled_connections: list[dict], temp_dir: Path, LocalDataVault):
    disabled_keys = {(d["engine"], d["name"]) for d in disabled_connections}
    filtered = LocalDataVault(temp_dir)
    for conn in source_vault.list_connections():
        if (conn["engine"], conn["name"]) not in disabled_keys:
            creds = source_vault.load(conn["engine"], conn["name"]) or {}
            filtered.save(conn["engine"], conn["name"], creds)
    return filtered


def _conversation_attachment_context(conversation) -> str:
    """Prompt fragment listing the absolute paths of every file attached to
    this conversation.

    Uploads are stored under the files dir (``.cowork/files/<uuid>/<name>``),
    which is OUTSIDE the project workspace. An agent that only scans the
    project root therefore never sees them and wrongly tells the user no
    files were uploaded (the Cyberdeck bug). Handing it the exact paths lets
    it read them on demand on any turn — not just the turn they arrived on.

    Returns "" when there are no attachments or they can't be resolved
    (e.g. the conversation is detached from its session), so the caller can
    append it unconditionally.
    """
    try:
        from sqlalchemy.orm import object_session
        from cowork.services.files import FileService, attachment_purpose

        db_session = object_session(conversation)
        if db_session is None:
            return ""
        rows = FileService(db_session).list_file_rows(
            purpose=attachment_purpose(conversation.project.name, str(conversation.id))
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
            return ""
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
        conv_id = getattr(conversation, "id", "<unknown>")
        logger.warning(
            "Failed to build conversation attachment context for conversation %s; "
            "the agent will not see attached files this turn",
            conv_id,
            exc_info=True,
        )
        return ""


@register
class AntonHarness:
    id: str = "anton"
    label: str = "Anton"
    formatter = staticmethod(format_responses_stream)

    async def sync_skills(self, skills: list[Skill]) -> None:
        # No-op: Anton's skills dir is pointed to cowork's canonical (seedev_setup)
        return

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
        disabled_connections: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        temp_vault_dir: Path | None = None
        # Attribute + surface any artifact created during this turn. Anton runs
        # with its own session id and doesn't tag artifacts with the cowork
        # conversation_id, so we diff the project's artifacts dir around the run
        # (see services.task_objects.finalize_turn_artifacts).
        from cowork.services.task_objects import finalize_turn_artifacts, snapshot_artifact_slugs
        artifacts_base = Path(conversation.project.path) / ".anton" / "artifacts"
        before_slugs = snapshot_artifact_slugs(artifacts_base)
        cards: list[dict] = []
        try:
            session, temp_vault_dir = await self._build_chat_session(
                conversation, disabled_connections=disabled_connections or []
            )
            async for event in session.turn_stream(self._to_anton_input(input)):
                yield event
        finally:
            if temp_vault_dir:
                shutil.rmtree(temp_vault_dir, ignore_errors=True)
            # One dir diff → index the new artifacts AND build their cards.
            # Runs on every exit (success, error, cancel) so an artifact is
            # always indexed; cards are yielded just below on normal completion.
            cards = finalize_turn_artifacts(
                conversation.id, conversation.project_id, artifacts_base, before_slugs,
            )
        for card in cards:
            yield ArtifactCreated(card)

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
        # model: str,
        disabled_connections: list[dict] | None = None,
    ):
        """Build the same core runtime the Anton CLI uses, scoped to one project."""
        from anton.chat_session import build_runtime_context
        from anton.config.settings import AntonSettings
        from anton.context.self_awareness import SelfAwarenessContext
        from anton.core.memory.cortex import Cortex
        # from anton.core.memory.episodes import EpisodicMemory
        from anton.core.memory.hippocampus import Hippocampus
        from anton.core.session import ChatSession, ChatSessionConfig, SystemPromptContext
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
            build_cowork_request_credentials_tool,
            # build_cowork_fetch_submission_tool,
            # build_cowork_update_form_tool,
        )
        PUBLISH_TOOL = build_cowork_publish_tool()
        LOOKUP_CONNECTOR_TOOL = build_cowork_lookup_connector_tool()
        REQUEST_CREDENTIALS_TOOL = build_cowork_request_credentials_tool()
        # TODO: Determine if these two tools are really needed.
        # FETCH_SUBMISSION_TOOL = build_cowork_fetch_submission_tool()
        # UPDATE_FORM_TOOL = build_cowork_update_form_tool()

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

        from cowork.common.settings.app_settings import get_app_settings
        anton_settings.skills_root = Path(get_app_settings().skill.root_dir)

        user = get_user_settings()
        for attr in (
            "planning_provider", "planning_model",
            "coding_provider", "coding_model",
            "memory_enabled", "memory_mode",
            "episodic_memory", "proactive_dashboards", "act_first",
            "publish_url",
        ):
            db_val = getattr(user, attr, None)
            if db_val is None:
                continue
            # Provider enum -> string value for AntonSettings.
            # The DB enum uses snake_case (openai_compatible, minds_cloud)
            # but AntonSettings / LLMClient expect kebab-case
            # (openai-compatible, minds-cloud).
            if hasattr(db_val, "value"):
                db_val = db_val.value.replace("_", "-")
            setattr(anton_settings, attr, db_val)

        # API keys: UserSettings stores SecretStr, AntonSettings uses plain str
        for attr in ("anthropic_api_key", "openai_api_key", "minds_api_key"):
            db_val = getattr(user, attr, None)
            if db_val is not None:
                setattr(anton_settings, attr, db_val.get_secret_value() if isinstance(db_val, SecretStr) else db_val)

        # URLs (skip empty strings so AntonSettings.model_post_init derivations are preserved)
        for attr in ("minds_url", "openai_base_url"):
            db_val = getattr(user, attr, None)
            if db_val:
                setattr(anton_settings, attr, db_val)

        workspace = Workspace(base)
        workspace.initialize()
        workspace.apply_env_to_process()

        anton_dir = base / ".anton"

        def _settings_path(value: object, fallback: Path) -> Path:
            raw = str(value or "").strip()
            if not raw:
                return fallback
            path = Path(raw).expanduser()
            return path if path.is_absolute() else base / path

        artifacts_dir = anton_dir / "artifacts"
        context_dir = _settings_path(getattr(anton_settings, "context_dir", None), anton_dir / "context")
        episodes_dir = anton_dir / "episodes"
        project_memory_dir = anton_dir / "memory"
        for directory in (artifacts_dir, context_dir, episodes_dir, project_memory_dir):
            directory.mkdir(parents=True, exist_ok=True)

        llm_client = self._build_llm_client()
        self_awareness = SelfAwarenessContext(context_dir)

        from cowork.common.settings.app_settings import get_app_settings

        global_memory_dir = Path(get_app_settings().memory.root_dir).expanduser()
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

        from cowork.common.settings.app_settings import get_app_settings

        # When connections are disabled, a temporary data vault is created because within Anton,
        # the vault is used to inject a prompt related to the connected data sources.
        data_vault = None
        temp_vault_dir: Path | None = None
        if LocalDataVault is not None:
            source_vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
            if disabled_connections:
                _tmp_base = Path.home() / ".cowork" / "tmp"
                _tmp_base.mkdir(parents=True, exist_ok=True)
                temp_vault_dir = Path(tempfile.mkdtemp(prefix="cowork-vault-", dir=_tmp_base))
                data_vault = _build_filtered_vault(source_vault, disabled_connections, temp_vault_dir, LocalDataVault)
            else:
                data_vault = source_vault
            for conn in data_vault.list_connections():
                data_vault.inject_env(conn["engine"], conn["name"])

        # TODO: Add guidance for integrations

        cells = extract_scratchpad_cells_from_message_events(conversation.messages)
        os.environ["ANTON_SCRATCHPAD_PERSIST_SESSION"] = "true"

        # Per-message timestamps: embed each message's created_at so the agent
        # always knows WHEN something was said (even resuming a conversation
        # days/weeks later). Absolute stamps are fixed per message, so the
        # history prefix stays byte-stable across turns (cache-safe).
        def _stamped(m):
            om = m.to_openai_message().model_dump()
            ts = m.created_at.strftime("%Y-%m-%d %H:%M") if getattr(m, "created_at", None) else None
            if ts and isinstance(om.get("content"), str) and om["content"]:
                om["content"] = f"[{ts}] {om['content']}"
            return om

        initial_history = [
            _stamped(m) for m in conversation.messages
            if m.role in {"user", "assistant"}
        ]

        config = ChatSessionConfig(
            llm_client=llm_client,
            settings=anton_settings,
            self_awareness=self_awareness,
            cortex=cortex,
            # episodic=episodic,
            system_prompt_context=SystemPromptContext(
                runtime_context=build_runtime_context(anton_settings),
                suffix=(
                    "The Anton CoWork desktop UI displays progress, tool usage, and actions "
                    "as separate structured activity rows. Keep assistant text focused on the "
                    "user-facing answer; do not narrate internal work with status phrases like "
                    "\"I'll check\", \"let me query\", or \"I have access\" unless that wording "
                    "is itself the final answer the user needs."
                    f"{project_context}"
                    f"{output_context}"
                ),
            ),
            workspace=workspace,
            data_vault=data_vault,
            initial_history=initial_history,
            # history_store=history_store,
            session_id=str(conversation.id),
            # Surfaced on langfuse traces (Langfuse-Tags / metadata) so calls
            # are attributed to the active harness. self.id == "anton".
            harness=self.id,
            proactive_dashboards=anton_settings.proactive_dashboards,
            act_first=settings.act_first,
            # "Conversation started" stamp for the cache-stable prompt prefix
            # (anton 2a). The live current time is rendered separately in the
            # volatile tail, so resuming days later still reports the real "now".
            started_at=conversation.created_at,
            tools=[
                CONNECT_DATASOURCE_TOOL,
                PUBLISH_TOOL,
                LOOKUP_CONNECTOR_TOOL,
                REQUEST_CREDENTIALS_TOOL,
                # FETCH_SUBMISSION_TOOL,
                # UPDATE_FORM_TOOL,
            ],
            cells=cells
        )
        return ChatSession(config), temp_vault_dir

    @staticmethod
    def _build_llm_client():
        from cowork.services.providers import build_llm_client
        return build_llm_client()
