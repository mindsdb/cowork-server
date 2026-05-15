from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os

from cowork.common.logger import get_logger
from cowork.harnesses.base import register
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
        prompt: str,
        model: str,
    ) -> AsyncIterator[dict]:
        history = [
            msg.to_openai_message().model_dump()
            for msg in conversation.messages
            if msg.role in {"user", "assistant"}
        ]

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._run,
            prompt,
            model,
            history,
        )
        yield result

    @staticmethod
    def _run(prompt: str, model: str, history: list[dict]) -> dict:
        from run_agent import AIAgent

        agent = AIAgent(
            provider="anthropic",  # TODO: Fix this.
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),  # TODO: Fix this.
            quiet_mode=True,
        )
        return agent.run_conversation(
            user_message=prompt,
            conversation_history=history,
        )
