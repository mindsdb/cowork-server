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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def stream_callback(delta: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "delta", "delta": delta})

        history = [
            msg.to_openai_message().model_dump()
            for msg in conversation.messages
            if msg.role in {"user", "assistant"}
        ]

        def run_sync() -> dict:
            try:
                return self._run(self._to_prompt_string(input), history, stream_callback)
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
    def _run(prompt: str, history: list[dict], stream_callback=None) -> dict:
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
            stream_callback=stream_callback,
        )
