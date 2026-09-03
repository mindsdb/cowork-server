from __future__ import annotations

from pathlib import Path


def managed_key(value: str) -> Path:
    """Validate a deterministic task workspace key before joining it to a root."""
    parts = value.replace("\\", "/").split("/")
    if not parts or len(parts) > 2:
        raise ValueError("invalid managed workspace key")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if any(not part or any(character not in allowed for character in part) for part in parts):
        raise ValueError("invalid managed workspace key")
    return Path(*parts)
