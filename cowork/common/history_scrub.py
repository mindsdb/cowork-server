"""Redact credentials from persisted history before it reaches an LLM.

`Message.to_openai_message()` replays stored content verbatim — only the
CURRENT turn's input is scrubbed automatically (anton's `_scrub_user_input`).
Every caller that replays history from storage must scrub it itself with
`scrubbed_openai_dump` below.
"""
from anton.utils.datasources import scrub_credentials, scrub_output

__all__ = ["scrub_credentials", "scrub_message_dict", "scrubbed_openai_dump"]


def _scrub_tool_result(block: dict) -> dict:
    """Return a tool_result block with its output scrubbed.

    Tool output is scrubbed with `scrub_output`, which also redacts OAuth
    tokens a cell printed. Rows stored before that existed are fixed here on
    replay. Non-text inner blocks keep their shape.
    """
    content = block.get("content")
    if isinstance(content, str):
        return {**block, "content": scrub_output(content)}
    if isinstance(content, list):
        return {**block, "content": [
            {**inner, "text": scrub_output(inner["text"])}
            if isinstance(inner, dict) and inner.get("type") == "text" and inner.get("text")
            else inner
            for inner in content
        ]}
    return block


def scrub_message_dict(om: dict) -> dict:
    """Scrub credentials from one OpenAI-shaped message dict, in place.

    String content is scrubbed whole. In list content, text/input_text blocks
    get the input filter (they can carry a secret typed next to a tool call),
    tool_result blocks get the output filter, and tool_use blocks are left as
    the model wrote them.
    """
    content = om.get("content")
    if isinstance(content, str) and content:
        om["content"] = scrub_credentials(content)
    elif isinstance(content, list):
        scrubbed = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "input_text") and block.get("text"):
                block = {**block, "text": scrub_credentials(block["text"])}
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                block = _scrub_tool_result(block)
            scrubbed.append(block)
        om["content"] = scrubbed
    return om


def scrubbed_openai_dump(message, **dump_kwargs) -> dict:
    """`message.to_openai_message().model_dump(**dump_kwargs)`, scrubbed."""
    return scrub_message_dict(message.to_openai_message().model_dump(**dump_kwargs))
