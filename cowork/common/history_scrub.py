"""Redact credentials from persisted history before it reaches an LLM.

`Message.to_openai_message()` replays stored content verbatim — only the
CURRENT turn's input is scrubbed automatically (anton's `_scrub_user_input`).
Every caller that replays history from storage must scrub it itself with
`scrubbed_openai_dump` below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env, scrub_credentials

from cowork.common.logger import get_logger
from cowork.services.connectors.persist import vault_for_scope

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

logger = get_logger(__name__)

__all__ = [
    "register_vault_secrets",
    "scrub_credentials",
    "scrub_message_dict",
    "scrubbed_openai_dump",
]


def scrub_message_dict(om: dict) -> dict:
    """Scrub credentials from one OpenAI-shaped message dict, in place.

    String content is scrubbed whole. List content (tool_use/tool_result
    rows) is left alone except sibling text/input_text blocks, which can
    carry a secret typed in the same message as a tool call.
    """
    content = om.get("content")
    if isinstance(content, str) and content:
        om["content"] = scrub_credentials(content)
    elif isinstance(content, list):
        om["content"] = [
            {**block, "text": scrub_credentials(block["text"])}
            if isinstance(block, dict) and block.get("type") in ("text", "input_text") and block.get("text")
            else block
            for block in content
        ]
    return om


def scrubbed_openai_dump(message, **dump_kwargs) -> dict:
    """`message.to_openai_message().model_dump(**dump_kwargs)`, scrubbed."""
    return scrub_message_dict(message.to_openai_message().model_dump(**dump_kwargs))


def register_vault_secrets(scope: TenantScope | None) -> None:
    """Register this request's DS_* secrets so `scrub_credentials` redacts
    them by exact value, not only by API-key shape.

    Call at request entry, before any history is scrubbed. The scope it opens
    is this request's own and reaches every task spawned from it. A failure
    must not fail the turn: scrubbing falls back to the shape-based regex.
    """
    begin_ds_turn_scope()
    try:
        restore_namespaced_env(vault_for_scope(scope))
    except Exception:
        logger.warning(
            "Could not register vault secrets; this turn's history is scrubbed "
            "by key shape only",
            exc_info=True,
        )
