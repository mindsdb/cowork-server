from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from cowork.common.logger import get_logger
from cowork.harnesses.base import FileInputBlock, TextInputBlock, register
from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings
from cowork.harnesses.hermes_harness.stream_formatter import format_hermes_stream
from cowork.models.conversation import Conversation
from cowork.models.skill import Skill

# Redirect all Hermes data (skills, sessions, config) to ~/.cowork/hermes before
# run_agent is first imported, so its module-level get_hermes_home() call lands here.
os.environ.setdefault("HERMES_HOME", HermesHarnessSettings().root_dir)

logger = get_logger(__name__)

# Map cowork provider names to the env var AIAgent checks at init time.
_PROVIDER_ENV_KEY = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _sync_hermes_config(provider: str, api_key: str | None) -> None:
    """Write ~/.cowork/hermes/config.yaml and set the env var that AIAgent expects."""
    import yaml

    hermes_home = Path(os.environ.get("HERMES_HOME", HermesHarnessSettings().root_dir))
    config_path = hermes_home / "config.yaml"

    # Read existing config to preserve other fields
    existing: dict = {}
    if config_path.is_file():
        try:
            existing = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            existing = {}

    if existing.get("provider") != provider:
        existing["provider"] = provider
        hermes_home.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(existing, default_flow_style=False))

    # Set the env var so AIAgent's init-time check passes
    env_var = _PROVIDER_ENV_KEY.get(provider)
    if env_var and api_key:
        os.environ[env_var] = api_key


def _build_datasource_context(vault, disabled_keys: set[tuple[str, str]]) -> str:
    """Build a system-prompt section listing connected data sources and their DS_* env var names."""
    try:
        conns = [c for c in vault.list_connections() if (c["engine"], c["name"]) not in disabled_keys]
    except Exception:
        conns = []

    lines = [
        "## Connected Data Sources",
        "Credentials are pre-injected as namespaced DS_<ENGINE>_<NAME>__<FIELD> environment variables "
        "(e.g. DS_POSTGRES_PROD_DB__HOST). Use them directly in scratchpad code and never read the "
        "data vault files directly. "
        "Flat variables like DS_HOST or DS_PASSWORD are only used temporarily during internal "
        "connection test snippets — do not assume they exist during normal execution.\n",
    ]

    for c in conns:
        fields = vault.load(c["engine"], c["name"]) or {}
        prefix = (
            "DS_"
            + c["engine"].upper().replace("-", "_")
            + "_"
            + c["name"].upper().replace("-", "_")
        )
        var_names = ", ".join(f"{prefix}__{k.upper()}" for k in fields) if fields else "(no fields)"
        lines.append(f"- `{c['engine']}-{c['name']}` ({c['engine']}) → {var_names}")

    return "\n".join(lines)


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

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
        disabled_connections: list[dict] | None = None,
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

        # Resolve project-derived values while the DB session is alive —
        # _run executes on an executor thread where lazy relationship
        # loads would fail.
        project_path = str(conversation.project.path)
        conversation_id = conversation.id
        project_id = conversation.project_id
        conversation_topic = conversation.topic
        prompt = self._to_prompt_string(input)

        # Snapshot the artifacts dir before the run so we can index + surface
        # any artifacts this turn produces — the same diff the Anton harness
        # uses, so both harnesses behave identically.
        from cowork.services.task_objects import finalize_turn_artifacts, snapshot_artifact_slugs
        artifacts_base = Path(project_path) / ".anton" / "artifacts"
        before_slugs = snapshot_artifact_slugs(artifacts_base)

        def run_sync() -> dict:
            try:
                return self._run(
                    str(conversation.id),
                    prompt,
                    history,
                    project_name=conversation.project.name,
                    project_path=project_path,
                    conversation_topic=conversation_topic,
                    stream_callback=stream_callback,
                    tool_start_callback=tool_start_callback,
                    tool_complete_callback=tool_complete_callback,
                    tool_progress_callback=tool_progress_callback,
                    reasoning_callback=reasoning_callback,
                    thinking_callback=thinking_callback,
                    disabled_connections=disabled_connections or [],
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, run_sync)

        cards: list[dict] = []
        result: dict
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            result = await task
        finally:
            # One dir diff → index the new artifacts AND build their cards.
            # Runs on every exit so an artifact is always indexed; cards are
            # yielded just below before the terminal result on normal
            # completion (mapped to response.artifact_created by the formatter,
            # same event the Anton harness produces).
            cards = finalize_turn_artifacts(
                conversation_id, project_id, artifacts_base, before_slugs,
            )
        for card in cards:
            yield {"type": "artifact_created", "artifact": card}

        yield result

    @staticmethod
    def _to_prompt_string(input_blocks: list[dict]) -> str:
        parts = []
        for block in input_blocks:
            if block.get("type") == "text":
                parts.append(block["text"])
            elif block.get("type") == "image":
                parts.append("[Attached image — vision not supported in this mode]")
            elif block.get("type") == "file":
                parts.append(f"[Attached file '{block['filename']}': {block['path']}]")
        return "\n\n".join(parts)

    @staticmethod
    def _run(
        session_id: str,
        prompt: str,
        history: list[dict],
        *,
        project_name: str,
        project_path: str,
        conversation_topic: str | None = None,
        stream_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        tool_progress_callback=None,
        reasoning_callback=None,
        thinking_callback=None,
        disabled_connections: list[dict] | None = None,
    ) -> dict:
        from pathlib import Path

        from run_agent import AIAgent
        from anton.core.datasources.data_vault import LocalDataVault

        from cowork.common.settings.app_settings import get_app_settings
        from cowork.common.settings.user_settings import get_user_settings
        from cowork.harnesses.hermes_harness.tools import (
            finalize_artifact_run_context,
            register_artifact_tools,
            register_connector_tools,
            set_artifact_run_context,
        )
        from cowork.common.settings.user_settings import Provider
        from cowork.harnesses.hermes_harness.memory_adapter import HermesMemoryAdapter

        register_connector_tools()
        register_artifact_tools()

        # Same folder-per-artifact convention as the Anton harness, so
        # Hermes outputs surface in the (harness-agnostic) Artifacts UI.
        artifacts_root = Path(project_path) / ".anton" / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifact_context = (
            "## Artifacts\n"
            f"User-facing outputs (HTML dashboards, reports, CSVs/datasets, images, apps) "
            f"live under `{artifacts_root}/`, one folder per artifact, and appear in the "
            "app's Artifacts UI where the user can view, publish, and manage them.\n"
            "Workflow:\n"
            "  1. Call `create_artifact(name, description, type)` BEFORE writing any output "
            "file. It returns `{slug, path}` — write your files into that absolute `path`.\n"
            "  2. To MODIFY an existing artifact, call `list_artifacts()` to find its slug "
            "and path, then write into that folder.\n"
            "  3. Never pick artifact folder names yourself, and never write user-facing "
            "outputs anywhere else in the project."
        )

        # Sync Hermes config.yaml with the user's cowork provider/key settings.
        # AIAgent validates config.yaml at init time (before using constructor
        # args), so the file must reflect the active provider and the matching
        # API key env var must be set.
        settings = get_user_settings()
        model = settings.planning_model
        provider_value = settings.planning_provider.value if settings.planning_provider != Provider.MINDS_CLOUD else "openai"
        api_key_value = (
            settings.minds_api_key.get_secret_value()
            if settings.planning_provider == Provider.MINDS_CLOUD
            else getattr(settings, f"{provider_value}_api_key", None)
        )
        if api_key_value and hasattr(api_key_value, "get_secret_value"):
            api_key_value = api_key_value.get_secret_value()

        _sync_hermes_config(provider_value, api_key_value)

        vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
        disabled_keys = {(d["engine"], d["name"]) for d in (disabled_connections or [])}
        for conn in vault.list_connections():
            if (conn["engine"], conn["name"]) not in disabled_keys:
                vault.inject_env(conn["engine"], conn["name"])

        project_context = (
            f"You are operating in the project {project_name}."
            f"You have access to all of the files in the project at {str(project_path)} except for the .anton/ directory."
            "They are off limits. Do not mention the .anton/ directory in your responses."
            "You can perform operations on these files by executing code."
            "You can freely read any of these project files."
            "If you need to perform any actions on these files, ask the user for permission first."
            "The only other files that you are allowed to access are any items that are attached to the conversation."
            "Access to any files not attached to the conversation or located outside the project is strictly forbidden."
        )
        datasource_context = _build_datasource_context(vault, disabled_keys)
        memory_context = HermesMemoryAdapter().build_prompt_context(Path(project_path))
        system_context = "\n\n".join(
            part for part in (project_context, memory_context, datasource_context, artifact_context) if part
        )

        if settings.planning_provider == Provider.MINDS_CLOUD:
            from cowork.services.providers import minds_chat_base_url
            agent = AIAgent(
                session_id=session_id,
                provider="openai",
                base_url=minds_chat_base_url(settings.minds_url),
                model=model,
                api_key=settings.minds_api_key.get_secret_value(),
                quiet_mode=True,
                ephemeral_system_prompt=system_context,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                # tool_progress_callback=tool_progress_callback,  -- This seems to fire on start and end too.
                reasoning_callback=reasoning_callback,
                thinking_callback=thinking_callback,
            )
        else:
            # The DB enum uses snake_case (openai_compatible) but AIAgent
            # expects kebab-case (openai-compatible).
            provider = settings.planning_provider.value.replace("_", "-")
            # Resolve key and base URL through the shared single-source helpers
            # (providers.provider_base_url / user_settings.provider_api_key) so
            # the hermes path can't drift from the anton path. provider_api_key
            # applies the gemini/openai-compatible → openai fallback (avoids a
            # None.get_secret_value() crash for a user on the shared key).
            from cowork.common.settings.user_settings import provider_api_key
            from cowork.services.providers import provider_base_url

            _key = provider_api_key(settings, settings.planning_provider)
            api_key = _key.get_secret_value() if _key else ""

            # AIAgent needs an explicit base_url to skip its config.yaml
            # provider-resolution path. provider_base_url returns None for
            # direct openai (SDK default) and anthropic; anthropic is handled
            # natively by AIAgent, but openai needs the explicit host here.
            base_url = provider_base_url(
                provider,
                openai_base_url=settings.openai_base_url or "",
                minds_url=settings.minds_url,
            )
            if base_url is None and provider == "openai":
                base_url = "https://api.openai.com/v1"

            kwargs = dict(
                provider=provider,
                model=model,
                api_key=api_key,
                quiet_mode=True,
                ephemeral_system_prompt=system_context,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                reasoning_callback=reasoning_callback,
                thinking_callback=thinking_callback,
            )
            if base_url:
                kwargs["base_url"] = base_url

            agent = AIAgent(**kwargs)

        # The conversation id doubles as run_agent's task_id: dispatch
        # forwards it to every tool handler (including on the concurrent
        # worker-thread path), which is how the artifact tools find this
        # run's project. Passing it also keeps terminal/VM state keyed
        # per conversation instead of per turn.
        set_artifact_run_context(
            session_id,
            artifacts_root=artifacts_root,
            conversation_id=session_id,
            conversation_title=conversation_topic,
            turn_summary=prompt,
        )
        try:
            return agent.run_conversation(
                user_message=prompt,
                conversation_history=history,
                stream_callback=stream_callback,
                task_id=session_id,
            )
        finally:
            finalize_artifact_run_context(session_id)
            vault.clear_ds_env()
