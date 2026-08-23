from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"
MAX_SANITIZED_CHARS = 128 * 1024
MAX_REDACTION_INPUT_CHARS = 256 * 1024
_SENSITIVE_KEY = re.compile(r"(authorization|api.?key|token|password|secret|credential|cookie)", re.IGNORECASE)
_SENSITIVE_TEXT = re.compile(
    r"(?i)((?:authorization|api[_-]?key|access[_-]?token|password|secret|cookie)\s*[:=]\s*(?:bearer\s+)?)([^\s,;&]+)"
)


def redact_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    safe = value[:MAX_REDACTION_INPUT_CHARS]
    for secret in secrets:
        safe = safe.replace(secret, REDACTED)
    return _SENSITIVE_TEXT.sub(lambda match: match.group(1) + REDACTED, safe)


def sanitize(value: object) -> Any:
    """Bound and redact untrusted engine payloads before persistence or UI."""
    return _sanitize(value, 0, [MAX_SANITIZED_CHARS])


def _sanitize(value: object, depth: int, budget: list[int]) -> Any:
    if budget[0] <= 0:
        return "[truncated]"
    # Charge every node, not only leaf text. This keeps deeply nested arrays of
    # empty containers from bypassing the shared payload budget.
    budget[0] -= 16
    if depth > 5:
        return "[truncated]"
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            if budget[0] <= 0:
                break
            safe_key = str(key)[:256]
            budget[0] -= len(safe_key)
            output[safe_key] = REDACTED if _SENSITIVE_KEY.search(safe_key) else _sanitize(item, depth + 1, budget)
        return output
    if isinstance(value, list):
        output = []
        for item in value[:100]:
            if budget[0] <= 0:
                break
            output.append(_sanitize(item, depth + 1, budget))
        return output
    if isinstance(value, str):
        safe = redact_text(value)[: min(8_192, budget[0])]
        budget[0] -= len(safe)
        return safe
    if isinstance(value, (int, float, bool)) or value is None:
        budget[0] -= 8
        return value
    safe = str(value)[: min(8_192, budget[0])]
    budget[0] -= len(safe)
    return safe


def redact_secrets(value: object, secrets: tuple[str, ...]) -> Any:
    sanitized = sanitize(value)

    def replace(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace(child) for child in item]
        return redact_text(item, secrets) if isinstance(item, str) else item

    return replace(sanitized)
