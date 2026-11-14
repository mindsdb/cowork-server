"""The label of an agent turn's artifact revision: what changed, not
"Agent updated artifact" (ENG-4227).

A quick edit's revision already reads "Quick edit: Added a field of
twinkling stars". An edit made in a chat turn gets the same, from the best
source the turn has, in this order:

1. the artifact editor's own summary: anton's ``edit_artifact`` tool result
   says ``Edited “<name>” (`<slug>`): …`` and then ``What changed: <the
   editor's reply>`` for that artifact (anton/core/tools/edit_artifact.py);
2. the first sentence of the turn's final reply, when it reads like a
   description of a change (not a question);
3. the request that started the turn, as ``Asked: …``;
4. none, so the revision keeps the generic label.

Each label is prefixed ``Agent: ``. A turn that changed several artifacts
gives each its own editor summary where it has one; the reply and the
request are the turn's, shared by the others.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "AGENT_PREFIX",
    "MAX_SUMMARY",
    "TurnRevisionSummaries",
    "editor_summaries",
    "reply_summary",
    "request_summary",
    "summaries_from_messages",
]

AGENT_PREFIX = "Agent: "
#: About one line of the History list.
MAX_SUMMARY = 140

# The edit_artifact tool's result for an applied edit: its first line names
# the artifact, a later line carries the editor's reply.
_EDITED = re.compile(r"Edited “[^”\n]*” \(`(?P<slug>[^`\n]+)`\)")
_WHAT_CHANGED = re.compile(r"^What changed: (?P<summary>.+)$", re.MULTILINE)
_MARKDOWN = re.compile(r"[*_`#>]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _clip(text: str, limit: int = MAX_SUMMARY) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _texts(content: object) -> Iterable[str]:
    """Every text in a message's or a block's content."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    yield item["text"]
                elif "content" in item:
                    yield from _texts(item["content"])


def _tool_results(messages: Iterable[object]) -> Iterable[str]:
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield "\n".join(_texts(block.get("content")))


def editor_summaries(messages: Iterable[object]) -> dict[str, str]:
    """``{slug: what the artifact editor said it changed}`` from a turn's
    messages or tool rows. Several edits of one artifact in a turn read as
    one, in order."""
    found: dict[str, list[str]] = {}
    for text in _tool_results(messages):
        edited = _EDITED.search(text)
        changed = _WHAT_CHANGED.search(text)
        if edited is None or changed is None:
            continue
        summary = " ".join(changed.group("summary").split())
        if summary and summary not in found.setdefault(edited.group("slug"), []):
            found[edited.group("slug")].append(summary)
    return {slug: "; ".join(parts) for slug, parts in found.items() if parts}


def reply_summary(reply: str | None) -> str | None:
    """The reply's first real sentence, when it reads like a description of
    what was done. A short opener ("Done!", a "## Updated" heading) is
    skipped; a question or a lead-in to a list ("Here is what I changed:")
    means the reply doesn't say it in a sentence, so None."""
    text = _MARKDOWN.sub("", str(reply or ""))
    for line in text.splitlines():
        for sentence in _SENTENCE_END.split(line.strip()):
            sentence = sentence.strip()
            if len(sentence) < 12:
                continue
            if sentence.endswith(("?", ":")):
                return None
            return sentence
    return None


def request_summary(request: str | None) -> str | None:
    text = " ".join(str(request or "").split())
    return f"Asked: {text}" if text else None


def _last_assistant_text(messages: list[object]) -> str | None:
    """The text of the turn's last assistant message: its reply."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
            )
        else:
            text = ""
        if text.strip():
            return text
    return None


@dataclass(frozen=True)
class TurnRevisionSummaries:
    """What one turn can say about the artifacts it changed."""

    editor: dict[str, str] = field(default_factory=dict)
    reply: str | None = None
    request: str | None = None

    def summary_for(self, slug: str) -> str | None:
        """The label for ``slug``'s revision, or None for the generic one."""
        for text in (self.editor.get(slug), reply_summary(self.reply), request_summary(self.request)):
            if text:
                return _clip(AGENT_PREFIX + text)
        return None


def summaries_from_messages(
    messages: Iterable[object] | None, *, reply: str | None = None, request: str | None = None,
) -> TurnRevisionSummaries:
    """A turn's summaries from its messages (anton's session history since
    the turn began, or the tool rows a pod sent). ``reply`` overrides the
    last assistant text found there; the remote paths pass the answer text
    they collected."""
    listed = list(messages or [])
    return TurnRevisionSummaries(
        editor=editor_summaries(listed),
        reply=reply if reply is not None else _last_assistant_text(listed),
        request=request,
    )
