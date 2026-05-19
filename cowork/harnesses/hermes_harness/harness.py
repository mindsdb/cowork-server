from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os

from cowork.common.logger import get_logger
from cowork.harnesses.base import FileInputBlock, TextInputBlock, register
from cowork.harnesses.hermes_harness.stream_formatter import format_hermes_stream
from cowork.models.conversation import Conversation


logger = get_logger(__name__)


@register
class HermesHarness:
    id: str = "hermes"
    label: str = "Hermes"
    formatter = staticmethod(format_hermes_stream)

    async def stream_response(
        self,
        *,
        conversation: Conversation,
        input: list[TextInputBlock | FileInputBlock],
        # model: str,
    ) -> AsyncIterator[dict]:
        history = [
            msg.to_openai_message().model_dump()
            for msg in conversation.messages
            if msg.role in {"user", "assistant"}
        ]

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._run,
            self._to_prompt_string(input),
            history,
        )
        yield result

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
    def _run(prompt: str, history: list[dict]) -> dict:
        from run_agent import AIAgent

        from cowork.common.settings.user_settings import get_user_settings

        settings = get_user_settings()
        provider = settings.planning_provider.value
        model = settings.planning_model
        api_key = getattr(settings, f"{provider}_api_key").get_secret_value()

        agent = AIAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            quiet_mode=True,
        )
        return agent.run_conversation(
            user_message=prompt,
            conversation_history=history,
        )
