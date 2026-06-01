from __future__ import annotations


def split_for_limit(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` chars, preferring a
    newline boundary so messages don't break mid-line."""
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
