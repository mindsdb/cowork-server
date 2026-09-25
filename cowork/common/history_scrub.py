"""Redact credentials from persisted history before it reaches an LLM.

`Message.to_openai_message()` replays stored content verbatim — only the
CURRENT turn's input is scrubbed automatically (anton's `_scrub_user_input`).
Every caller that replays history from storage must scrub it itself with
`scrubbed_openai_dump` below.
"""
from anton.utils.datasources import scrub_credentials

__all__ = ["scrub_credentials", "scrub_message_dict", "scrubbed_openai_dump"]


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
