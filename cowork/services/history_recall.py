"""Ranked text search over the raw turns a conversation's summary replaced.

Pure functions only — no DB, no ORM, no anton. Callers pass messages as plain
dicts.

ponytail: lexical search only. It matches word forms ("deploy" → "deployment")
but not synonyms ("shipped to prod" won't answer "deploy"). Upgrade path if
evals show misses: embed the archive and rank by vector similarity instead of
`_rank_indices` — the rest of this module stays as is.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# A one-character token carries no signal and matches far too much.
_MIN_TOKEN_LEN = 2
# Shortest token where a prefix match means something: "code"/"codebase" is a
# real hit, "to"/"tonne" is noise.
_MIN_PREFIX_LEN = 4
# Typo tolerance is a tiebreaker, not the main mechanism: the queries come from
# a model quoting its own summary, so misspellings are rare.
_MIN_FUZZY_LEN = 5
_FUZZY_CUTOFF = 0.85

# Per-entry output caps. A tool result can be a megabyte-long dump, and one
# recall call must never cost more context than the compaction saved.
_USER_CAP = 500
_ASSISTANT_CAP = 2000
_TOTAL_CAP = 6000


@dataclass
class Entry:
    """One stored message, flattened to plain text."""

    role: str
    text: str
    is_tool: bool = False


@dataclass
class Turn:
    """A user message plus everything the agent did in reply to it."""

    number: int  # 1-based position in the archive
    entries: list[Entry] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(e.text for e in self.entries if e.text)


def _flatten(content: object) -> tuple[str, bool]:
    """Plain text of one stored message, and whether it is a tool block row.

    Stored content is either a string or a list of blocks; tool_use and
    tool_result blocks are searchable too — before compaction the model saw
    them in full.
    """
    if isinstance(content, str):
        return content, False
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content or ""), False

    parts: list[str] = []
    is_tool = False
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "tool_use":
            is_tool = True
            parts.append(f"{block.get('name', '')}({block.get('input', '')})")
        elif btype == "tool_result":
            is_tool = True
            parts.append(str(block.get("content", "")))
        else:
            parts.append(str(block.get("text") or ""))
    return "\n".join(p for p in parts if p), is_tool


def group_turns(messages: list[dict]) -> list[Turn]:
    """Group ordered messages into turns.

    A turn starts at a real user message. A user row carrying tool_result
    blocks is the agent's own loop, not a new turn, so it attaches to the
    current one — which is what makes a query whose words are split between
    the question and the answer still match (see `_rank_indices`).
    """
    turns: list[Turn] = []
    for message in messages:
        text, is_tool = _flatten(message.get("content"))
        role = str(message.get("role", ""))
        if not turns or (role == "user" and not is_tool):
            turns.append(Turn(number=len(turns) + 1))
        turns[-1].entries.append(Entry(role=role, text=text, is_tool=is_tool))
    return turns


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN}


def _query_tokens(query: str) -> list[str]:
    # dict.fromkeys: dedupe but keep order, so scoring is deterministic.
    return [
        t for t in dict.fromkeys(_TOKEN_RE.findall(query.lower()))
        if len(t) >= _MIN_TOKEN_LEN
    ]


def _matches(word: str, tokens: set[str]) -> bool:
    """Whether a query word hits any token of a turn.

    Three rules, cheapest first:
    - exact token match
    - prefix either way, so word forms match ("deploy" ↔ "deployment")
    - close spelling, as a tiebreaker for typos
    """
    if word in tokens:
        return True
    if len(word) >= _MIN_PREFIX_LEN:
        for token in tokens:
            if len(token) >= _MIN_PREFIX_LEN and (
                token.startswith(word) or word.startswith(token)
            ):
                return True
    if len(word) >= _MIN_FUZZY_LEN:
        candidates = [t for t in tokens if len(t) >= _MIN_FUZZY_LEN]
        return bool(
            difflib.get_close_matches(word, candidates, n=1, cutoff=_FUZZY_CUTOFF)
        )
    return False


def _rank_indices(turns: list[Turn], query: str) -> list[int]:
    """Indices of turns matching `query`, best first.

    Scoring, for a query of several words:
    - each word is weighted by how rare it is *in this conversation* — a word
      in every turn cannot tell turns apart, a word in one turn identifies it
    - weights are summed per turn, not per message, so a query split across the
      question and the answer still lands, and a turn matching more of the
      query outscores one matching less of it
    """
    words = _query_tokens(query)
    if not turns or not words:
        return []

    token_sets = [_tokens(t.text) for t in turns]
    total = len(turns)
    scores = [0.0] * total

    for word in words:
        matched = [i for i, tokens in enumerate(token_sets) if _matches(word, tokens)]
        if not matched:
            continue
        weight = math.log(1 + total / len(matched))
        for i in matched:
            scores[i] += weight

    ranked = [i for i in range(total) if scores[i]]
    # Ties resolve to the earlier turn, so repeated identical queries are stable.
    ranked.sort(key=lambda i: (-scores[i], i))
    return ranked


def search_turns(messages: list[dict], query: str, limit: int = 3) -> list[Turn]:
    """The `limit` turns of `messages` that best match `query`."""
    turns = group_turns(messages)
    return [turns[i] for i in _rank_indices(turns, query)[:limit]]


def _clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + "… (truncated)"


def format_turns(turns: list[Turn]) -> str:
    """Render matched turns for the model, within the output caps."""
    lines: list[str] = []
    used = 0
    for turn in turns:
        header = f"--- turn {turn.number} ---"
        lines.append(header)
        used += len(header)
        for entry in turn.entries:
            if not entry.text:
                continue
            cap = _USER_CAP if entry.role == "user" and not entry.is_tool else _ASSISTANT_CAP
            line = f"[{entry.role}] {_clip(entry.text, cap)}"
            if used + len(line) > _TOTAL_CAP:
                lines.append("… (more matches omitted — narrow the query)")
                return "\n".join(lines)
            lines.append(line)
            used += len(line)
    return "\n".join(lines)
