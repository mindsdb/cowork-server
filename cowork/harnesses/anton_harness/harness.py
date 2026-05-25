
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path

from cowork.common.logger import get_logger
from cowork.harnesses.base import (
    FileInputBlock, TextInputBlock, MemoryScope, register
)
from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
from cowork.models.conversation import Conversation
from cowork.models.skill import Skill
from cowork.models.project import Project
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings


logger = get_logger(__name__)
settings = AntonHarnessSettings()


# TODO: Handle topics.
class AntonMemoryCategory(str, Enum):
    lesson = "lesson"
    rule = "rule"
    topic = "topic"


@register
class AntonHarness:
    id: str = "anton"
    label: str = "Anton"
    formatter = staticmethod(format_responses_stream)

    async def sync_skills(self, skills: list[Skill]) -> None:
        from datetime import datetime, timezone
        from anton.core.memory.skills import Skill as AntonSkill, SkillStore
        
        from cowork.harnesses.anton_harness.settings import AntonHarnessSettings

        settings = AntonHarnessSettings()
        store = SkillStore(Path(settings.skills_root_dir))
        active_labels: set[str] = set()
        for skill in skills:
            anton_skill = AntonSkill(
                label=skill.label,
                name=skill.name,
                description=skill.description or "",
                when_to_use=skill.when_to_use or "",
                declarative_md=skill.instructions,
                created_at=skill.created_at.isoformat() if skill.created_at else datetime.now(timezone.utc).isoformat(),
                provenance="cowork",  # Helps track which skills originated from cowork.
            )
            store.save(anton_skill)
            active_labels.add(skill.label)

        # Delete any existing Anton skills that are not in the current list.
        for existing in store.list_all():
            if existing.provenance == "cowork" and existing.label not in active_labels:
                store.delete(existing.label)
    
    async def overwrite_memory(self, scope: MemoryScope, category: str, content: str, project: Project | None = None) -> None:
        # Validate provided category.
        # This is not done at the schema (request) level because each harness supports different categories.
        category_enum = AntonMemoryCategory(category)  # This will raise a ValueError if the category is not supported.

        if scope == MemoryScope.global_:
            await self._write_to_global_memory(category_enum, content)
        elif scope == MemoryScope.project:
            await self._write_to_project_memory(project, category_enum, content)

    async def _write_to_global_memory(self, category: AntonMemoryCategory, content: str) -> None:
        global_memory_dir = Path(settings.global_memory_root_dir)
        global_memory_dir.mkdir(parents=True, exist_ok=True)
        
        memory_file = self._resolve_memory_path(global_memory_dir, category)
        memory_file.write_text(content + "\n", encoding="utf-8")

    async def _write_to_project_memory(self, project: Project, category: AntonMemoryCategory, content: str) -> None:
        project_memory_dir = Path(project.path) / ".anton" / "memory"
        project_memory_dir.mkdir(parents=True, exist_ok=True)

        memory_file = self._resolve_memory_path(project_memory_dir, category)
        memory_file.write_text(content + "\n", encoding="utf-8")

    def _resolve_memory_path(self, root_dir: Path, category: AntonMemoryCategory) -> Path:
        # TODO: Topics are not handled at the moment because there are some discrepancies in
        # how they are handled in Cowork Vs what Anton actually expects.
        scope_to_path = {
            AntonMemoryCategory.lesson: root_dir / "lessons.md",
            AntonMemoryCategory.rule: root_dir / "rules.md",
        }
        return scope_to_path[category]
    
    async def retrieve_memory(self, scope: MemoryScope, category: str, project: Project | None = None) -> str:
        category_enum = AntonMemoryCategory(category)  # This will raise a ValueError if the category is not supported.

        if scope == MemoryScope.global_:
            return await self._read_from_global_memory(category_enum)
        elif scope == MemoryScope.project:
            return await self._read_from_project_memory(project, category_enum)
        
    async def _read_from_global_memory(self, category: AntonMemoryCategory) -> str:
        global_memory_dir = Path(settings.global_memory_root_dir)
        memory_file = self._resolve_memory_path(global_memory_dir, category)
        if not memory_file.is_file():
            return ""
        return memory_file.read_text(encoding="utf-8")
    
    async def _read_from_project_memory(self, project: Project, category: AntonMemoryCategory) -> str:
        project_memory_dir = Path(project.path) / ".anton" / "memory"
        memory_file = self._resolve_memory_path(project_memory_dir, category)
        if not memory_file.is_file():
            return ""
        return memory_file.read_text(encoding="utf-8")

    async def list_memory(self, projects: list[Project]) -> list:
        from cowork.harnesses.base import MemoryItem
        supported = [AntonMemoryCategory.lesson, AntonMemoryCategory.rule]
        results = []
        for category in supported:
            content = await self._read_from_global_memory(category)
            results.append(MemoryItem(scope=MemoryScope.global_, category=category.value, content=content, project=None))
        for project in projects:
            for category in supported:
                content = await self._read_from_project_memory(project, category)
                results.append(MemoryItem(scope=MemoryScope.project, category=category.value, content=content, project=project))
        return results

    async def delete_memory(self, scope: MemoryScope, category: str, project: Project | None = None) -> None:
        category_enum = AntonMemoryCategory(category)  # This will raise a ValueError if the category is not supported.

        if scope == MemoryScope.global_:
            await self._delete_global_memory(category_enum)
        elif scope == MemoryScope.project:
            await self._delete_project_memory(project, category_enum)

    async def _delete_global_memory(self, category: AntonMemoryCategory) -> None:
        global_memory_dir = Path(settings.global_memory_root_dir)
        memory_file = self._resolve_memory_path(global_memory_dir, category)
        if memory_file.is_file():
            memory_file.unlink()

    async def _delete_project_memory(self, project: Project, category: AntonMemoryCategory) -> None:
        project_memory_dir = Path(project.path) / ".anton" / "memory"
        memory_file = self._resolve_memory_path(project_memory_dir, category)
        if memory_file.is_file():
            memory_file.unlink()

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
    ) -> AsyncIterator[str]:
        session = await self._build_chat_session(conversation)

        async for event in session.turn_stream(self._to_anton_input(input)):
            yield event

    @staticmethod
    def _to_anton_input(input_blocks: list[dict]) -> str | list[dict]:
        if len(input_blocks) == 1 and input_blocks[0].get("type") == "text":
            return input_blocks[0]["text"]
        anton_blocks = []
        for block in input_blocks:
            if block.get("type") == "text":
                anton_blocks.append({"type": "text", "text": block["text"]})
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
            # build_list_conversation_datasources_tool,
        )
        PUBLISH_TOOL = build_cowork_publish_tool()
        LOOKUP_CONNECTOR_TOOL = build_cowork_lookup_connector_tool()
        REQUEST_CREDENTIALS_TOOL = build_cowork_request_credentials_tool()
        # FETCH_SUBMISSION_TOOL = build_cowork_fetch_submission_tool()
        # UPDATE_FORM_TOOL = build_cowork_update_form_tool()
        # LIST_CONVERSATION_DATASOURCES_TOOL = build_list_conversation_datasources_tool()

        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception:  # pragma: no cover
            LocalDataVault = None
            
        from cowork.harnesses.anton_harness.settings import AntonHarnessSettings

        base = Path(conversation.project.path)
        # Reload ~/.anton/.env into os.environ before building settings.
        # AntonSettings caches its env_file list at module import time — if the
        # server started before ~/.anton/.env existed (first-run onboarding),
        # the file is not in the cached list and planning_provider would fall
        # back to the "anthropic" default, causing a TypeError when no
        # ANTHROPIC_API_KEY is set. Loading the file here ensures settings
        # always reflect the current config, even after onboarding.
        # Skip server-operational vars that the Electron host controls.
        # TODO: Is all of this necessary?
        # _SERVER_MANAGED_KEYS = {"ANTON_SERVER_PORT", "ANTON_SERVER_HOST", "ANTON_PROJECTS_DIR"}
        # _user_env = Path.home() / ".anton" / ".env"
        # if _user_env.is_file():
        #     for _line in _user_env.read_text(encoding="utf-8").splitlines():
        #         _line = _line.strip()
        #         if _line and not _line.startswith("#") and "=" in _line:
        #             _k, _, _v = _line.partition("=")
        #             _k = _k.strip()
        #             if _k not in _SERVER_MANAGED_KEYS:
        #                 os.environ[_k] = _v.strip().strip('"').strip("'")

        settings = AntonSettings()
        settings.resolve_workspace(str(base))
        # if model:
        #     # Minds Cloud sentinels (`_reason_`, `_code_`) only resolve at
        #     # the openai-compatible router. If the active provider is
        #     # something else (e.g. anthropic, after the user switched off
        #     # Minds), an old cowork preference can keep sending these on
        #     # every request. Drop the override and stay with the env's
        #     # `ANTON_PLANNING_MODEL` instead of forwarding `_reason_` to
        #     # api.anthropic.com (which 404s).
        #     is_minds_sentinel = model.startswith("_") and model.endswith("_")
        #     if is_minds_sentinel and settings.planning_provider != "openai-compatible":
        #         logger.warning(
        #             "Ignoring Minds sentinel model %r — active planning_provider is %r. "
        #             "Falling back to env ANTON_PLANNING_MODEL=%r.",
        #             model, settings.planning_provider, settings.planning_model,
        #         )
        #     else:
        #         settings.planning_model = model

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
        context_dir = _settings_path(getattr(settings, "context_dir", None), anton_dir / "context")
        episodes_dir = anton_dir / "episodes"
        project_memory_dir = anton_dir / "memory"
        for directory in (artifacts_dir, context_dir, episodes_dir, project_memory_dir):
            directory.mkdir(parents=True, exist_ok=True)

        llm_client = self._build_llm_client()
        self_awareness = SelfAwarenessContext(context_dir)
        global_memory_dir = Path(AntonHarnessSettings().global_memory_root_dir)
        global_memory_dir.mkdir(parents=True, exist_ok=True)
        cortex = Cortex(
            global_hc=Hippocampus(global_memory_dir),
            project_hc=Hippocampus(project_memory_dir),
            mode=settings.memory_mode if settings.memory_enabled else "off",
            llm_client=llm_client,
        )
        # TODO: Is episodic memory required given that we are handling history outside of the harness?
        # episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
        # episodic.resume_session(conversation_id)
        # history_store = HistoryStore(episodes_dir)
        # initial_history = history_store.load(conversation_id)

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

        data_vault = LocalDataVault() if LocalDataVault is not None else None

        # TODO: Add guidance for integrations

        history = [message.to_openai_message() for message in conversation.messages if message.role in {"user", "assistant"}]
        config = ChatSessionConfig(
            llm_client=llm_client,
            settings=settings,
            self_awareness=self_awareness,
            cortex=cortex,
            # episodic=episodic,
            system_prompt_context=SystemPromptContext(
                runtime_context=build_runtime_context(settings),
                suffix=(
                    "The Anton CoWork desktop UI displays progress, tool usage, and actions "
                    "as separate structured activity rows. Keep assistant text focused on the "
                    "user-facing answer; do not narrate internal work with status phrases like "
                    "\"I'll check\", \"let me query\", or \"I have access\" unless that wording "
                    "is itself the final answer the user needs."
                    f"{project_context}"
                ),
                output_context=output_context,
            ),
            workspace=workspace,
            data_vault=data_vault,
            initial_history=[message.model_dump() for message in history],
            # history_store=history_store,
            session_id=conversation.id,
            proactive_dashboards=settings.proactive_dashboards,
            tools=[
                CONNECT_DATASOURCE_TOOL,
                PUBLISH_TOOL,
                LOOKUP_CONNECTOR_TOOL,
                REQUEST_CREDENTIALS_TOOL,
                # FETCH_SUBMISSION_TOOL,
                # UPDATE_FORM_TOOL,
                # LIST_CONVERSATION_DATASOURCES_TOOL,
            ],
        )
        return ChatSession(config)

    def _build_llm_client(self):
        from anton.core.llm.client import LLMClient
        from anton.core.llm.anthropic import AnthropicProvider
        from anton.core.llm.openai import OpenAIProvider

        from cowork.common.settings.user_settings import get_user_settings
        from cowork.schemas.settings import Provider

        settings = get_user_settings()

        def _make_provider(role: Provider):
            if role == Provider.MINDS_CLOUD:
                return OpenAIProvider(
                    api_key=settings.minds_api_key.get_secret_value(),
                    base_url=settings.minds_url,
                )
            provider_map = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}
            cls = provider_map[role.value]
            return cls(api_key=getattr(settings, f"{role.value}_api_key").get_secret_value())

        return LLMClient(
            planning_provider=_make_provider(settings.planning_provider),
            planning_model=settings.planning_model,
            coding_provider=_make_provider(settings.coding_provider),
            coding_model=settings.coding_model,
        )
