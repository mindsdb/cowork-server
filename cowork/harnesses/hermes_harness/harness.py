from __future__ import annotations

import asyncio
from enum import Enum
import os
from collections.abc import AsyncIterator
from pathlib import Path

from cowork.common.logger import get_logger
from cowork.harnesses.base import FileInputBlock, TextInputBlock, MemoryScope, register
from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings
from cowork.harnesses.hermes_harness.stream_formatter import format_hermes_stream
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.skill import Skill

# Redirect all Hermes data (skills, sessions, config) to ~/.cowork/hermes before
# run_agent is first imported, so its module-level get_hermes_home() call lands here.
os.environ.setdefault("HERMES_HOME", HermesHarnessSettings().root_dir)

logger = get_logger(__name__)

_VAULT_ENV_PROMPT = (
    "Connected datasource credentials are injected as namespaced environment "
    "variables in the form DS_<ENGINE>_<NAME>__<FIELD> "
    "(e.g. DS_POSTGRES_PROD_DB__HOST, DS_POSTGRES_PROD_DB__PASSWORD, "
    "DS_HUBSPOT_MAIN__ACCESS_TOKEN). Use those variables directly in scratchpad "
    "code and never read ~/.cowork/data-vault/ files directly. "
    "Flat variables like DS_HOST or DS_PASSWORD are used only temporarily "
    "during internal connection test snippets — do not assume they exist "
    "during normal chat/runtime execution."
)


class HermesMemoryCategory(str, Enum):
    user = "user"
    memory = "memory"


@register
class HermesHarness:
    id: str = "hermes"
    label: str = "Hermes"
    formatter = staticmethod(format_hermes_stream)

    async def sync_skills(self, skills: list[Skill]) -> None:
        import shutil
        import yaml

        settings = HermesHarnessSettings()
        skills_dir = Path(settings.root_dir) / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        active_labels: set[str] = set()
        for skill in skills:
            skill_dir = skills_dir / skill.label
            skill_dir.mkdir(parents=True, exist_ok=True)

            frontmatter: dict = {"name": skill.name or skill.label}
            if skill.description:
                frontmatter["description"] = skill.description
            if skill.when_to_use:
                frontmatter["when_to_use"] = [
                    line.strip()
                    for line in skill.when_to_use.splitlines()
                    if line.strip()
                ]

            content = (
                f"---\n"
                f"{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}"
                f"---\n\n"
                f"{skill.instructions or ''}"
            )
            (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
            active_labels.add(skill.label)

        # Delete skills that no longer exist in cowork.
        for existing_dir in skills_dir.iterdir():
            if existing_dir.is_dir() and existing_dir.name not in active_labels:
                shutil.rmtree(existing_dir)

    async def overwrite_memory(self, scope: MemoryScope, category: str, content: str, project: Project | None = None) -> None:
        if scope == MemoryScope.project:
            raise ValueError("Project-scoped memory is not supported for the Hermes harness.")
        category_enum = HermesMemoryCategory(category)
        memory_dir = Path(HermesHarnessSettings().root_dir) / "memories"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._resolve_memory_path(memory_dir, category_enum).write_text(content + "\n", encoding="utf-8")

    async def retrieve_memory(self, scope: MemoryScope, category: str, project: Project | None = None) -> str:
        if scope == MemoryScope.project:
            raise ValueError("Project-scoped memory is not supported for the Hermes harness.")
        category_enum = HermesMemoryCategory(category)
        memory_file = self._resolve_memory_path(Path(HermesHarnessSettings().root_dir) / "memories", category_enum)
        if not memory_file.is_file():
            return ""
        return memory_file.read_text(encoding="utf-8")

    async def delete_memory(self, scope: MemoryScope, category: str, project: Project | None = None) -> None:
        if scope == MemoryScope.project:
            raise ValueError("Project-scoped memory is not supported for the Hermes harness.")
        category_enum = HermesMemoryCategory(category)
        memory_file = self._resolve_memory_path(Path(HermesHarnessSettings().root_dir) / "memories", category_enum)
        if memory_file.is_file():
            memory_file.unlink()

    async def list_memory(self, projects: list[Project]) -> list:
        from cowork.harnesses.base import MemoryItem
        memory_dir = Path(HermesHarnessSettings().root_dir) / "memories"
        results = []
        for category in HermesMemoryCategory:
            memory_file = self._resolve_memory_path(memory_dir, category)
            content = memory_file.read_text(encoding="utf-8") if memory_file.is_file() else ""
            results.append(MemoryItem(scope=MemoryScope.global_, category=category.value, content=content, project=None))
        return results

    def _resolve_memory_path(self, root_dir: Path, category: HermesMemoryCategory) -> Path:
        return root_dir / {
            HermesMemoryCategory.user: "USER.md",
            HermesMemoryCategory.memory: "MEMORY.md",
        }[category]

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
    ) -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def _put(item: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def stream_callback(delta: str) -> None:
            _put({"type": "delta", "delta": delta})

        def tool_start_callback(tool_call_id: str, name: str, args: dict) -> None:
            _put({"type": "thought.tool_call.start", "tool_call_id": tool_call_id, "name": name, "args": args})

        def tool_complete_callback(tool_call_id: str, name: str, args: dict, result: str) -> None:
            _put({"type": "thought.tool_call.end", "tool_call_id": tool_call_id, "name": name, "result": result})

        def tool_progress_callback(event_type: str, name: str, preview=None, args=None, **kwargs) -> None:
            _put({"type": "thought.tool_call.progress", "event": event_type, "name": name, "preview": preview})

        def reasoning_callback(text: str) -> None:
            _put({"type": "thought.progress", "subtype": "reasoning", "content": text})

        def thinking_callback(text: str) -> None:
            _put({"type": "thought.progress", "subtype": "thinking", "content": text})

        history = [
            msg.to_openai_message().model_dump()
            for msg in conversation.messages
            if msg.role in {"user", "assistant"}
        ]

        def run_sync() -> dict:
            try:
                return self._run(
                    str(conversation.id),
                    self._to_prompt_string(input),
                    history,
                    stream_callback=stream_callback,
                    tool_start_callback=tool_start_callback,
                    tool_complete_callback=tool_complete_callback,
                    tool_progress_callback=tool_progress_callback,
                    reasoning_callback=reasoning_callback,
                    thinking_callback=thinking_callback,
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, run_sync)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        yield await task

    @staticmethod
    def _to_prompt_string(input_blocks: list[dict]) -> str:
        parts = []
        for block in input_blocks:
            if block.get("type") == "text":
                parts.append(block["text"])
            elif block.get("type") == "file":
                parts.append(f"[Attached file '{block['filename']}': {block['path']}]")
        return "\n\n".join(parts)

    @staticmethod
    def _run(
        session_id: str,
        prompt: str,
        history: list[dict],
        stream_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        tool_progress_callback=None,
        reasoning_callback=None,
        thinking_callback=None,
    ) -> dict:
        from pathlib import Path

        from run_agent import AIAgent
        from anton.core.datasources.data_vault import LocalDataVault

        from cowork.common.settings.app_settings import get_app_settings
        from cowork.common.settings.user_settings import get_user_settings
        from cowork.harnesses.hermes_harness.tools import register_connector_tools
        from cowork.schemas.settings import Provider

        register_connector_tools()

        vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
        for conn in vault.list_connections():
            vault.inject_env(conn["engine"], conn["name"])

        settings = get_user_settings()
        model = settings.planning_model

        if settings.planning_provider == Provider.MINDS_CLOUD:
            agent = AIAgent(
                session_id=session_id,
                provider="openai",
                base_url=settings.minds_url,
                model=model,
                api_key=settings.minds_api_key.get_secret_value(),
                quiet_mode=True,
                ephemeral_system_prompt=_VAULT_ENV_PROMPT,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                # tool_progress_callback=tool_progress_callback,  -- This seems to fire on start and end too.
                reasoning_callback=reasoning_callback,
                thinking_callback=thinking_callback,
            )
        else:
            provider = settings.planning_provider.value
            api_key = getattr(settings, f"{provider}_api_key").get_secret_value()
            agent = AIAgent(
                provider=provider,
                model=model,
                api_key=api_key,
                quiet_mode=True,
                ephemeral_system_prompt=_VAULT_ENV_PROMPT,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                # tool_progress_callback=tool_progress_callback,  -- This seems to fire on start and end too.
                reasoning_callback=reasoning_callback,
                thinking_callback=thinking_callback,
            )

        try:
            return agent.run_conversation(
                user_message=prompt,
                conversation_history=history,
                stream_callback=stream_callback,
            )
        finally:
            vault.clear_ds_env()
