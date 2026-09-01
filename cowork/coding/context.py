from __future__ import annotations

import os
import hashlib
from pathlib import Path

from cowork.coding.contracts import CodingSession, InputReference
from cowork.coding.engines.base import EngineCredentials, EngineInputReference
from cowork.coding.redaction import redact_text
from cowork.coding.workspace_files import WorkspaceFileBrowser

_EXCLUDED_WORKSPACE_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_MAX_WORKSPACE_FILES_SCANNED = 20_000


def workspace_files(session: CodingSession, query: str = "", limit: int = 40) -> list[dict[str, str]]:
    """Return bounded, deterministic file suggestions from a task workspace."""
    roots = [
        (item.folder_name, Path(item.workspace_path).resolve())
        for item in session.workspaces
    ] or [("", Path(session.workspace_path).resolve())]
    needle = query.casefold().strip()
    matches: list[dict[str, str]] = []
    visited = 0
    requested = max(1, min(limit, 100))
    for folder_name, root in roots:
        for current_root, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED_WORKSPACE_DIRECTORIES)
            entries = [(item, True) for item in dirs] + [(item, False) for item in sorted(files)]
            for name, is_directory in entries:
                visited += 1
                if visited > _MAX_WORKSPACE_FILES_SCANNED:
                    return matches
                path = Path(current_root, name)
                try:
                    relative = path.relative_to(root).as_posix() + ("/" if is_directory else "")
                except ValueError:
                    continue
                label = f"{folder_name}/{relative}" if folder_name else relative
                if needle and needle not in label.casefold():
                    continue
                matches.append({"name": label, "path": str(path), "kind": "mention"})
                if len(matches) >= requested:
                    return matches
    return matches


def validate_references(
    session: CodingSession,
    attachments: list[InputReference] | tuple[InputReference, ...],
) -> tuple[EngineInputReference, ...]:
    """Resolve attachments and map source paths into an isolated worktree."""
    resolved: list[EngineInputReference] = []
    mappings = [
        (Path(item.source_path).resolve(), Path(item.workspace_path).resolve())
        for item in session.workspaces
    ] or [(Path(session.source_path).resolve(), Path(session.workspace_path).resolve())]
    browser = WorkspaceFileBrowser(session)
    workspace = mappings[0][1]
    for item in attachments:
        path = (
            browser.absolute_path(item.resource_id, item.relative_path)
            if item.resource_id and item.relative_path
            else Path(item.path)
        )
        if not path.is_absolute():
            path = workspace / path
        try:
            path = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"Attached file is unavailable: {item.name}") from exc
        if item.kind == "local_image" and not path.is_file():
            raise ValueError(f"Image attachment is not a file: {item.name}")
        if item.kind == "mention" and not (path.is_file() or path.is_dir()):
            raise ValueError(f"Referenced path is not a file or folder: {item.name}")
        if item.content_hash and path.is_file():
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != item.content_hash:
                raise ValueError(f"Referenced file changed; select its lines again: {item.name}")
        for source, mapped_workspace in mappings:
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            workspace_file = mapped_workspace / relative
            if workspace_file.is_file() or workspace_file.is_dir():
                path = workspace_file
            break
        resolved.append(EngineInputReference(
            name=item.name,
            path=str(path),
            kind=item.kind,
            resource_id=item.resource_id,
            relative_path=item.relative_path,
            line_start=item.line_start,
            line_end=item.line_end,
            content_hash=item.content_hash,
        ))
    return tuple(resolved)


def validate_directories(directories: list[str]) -> list[str]:
    """Resolve existing folders once, preserving the user's order."""
    resolved: list[str] = []
    for value in directories:
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"Additional folder is unavailable: {value}") from exc
        if not path.is_dir():
            raise ValueError(f"Additional folder is not a directory: {value}")
        normalized = str(path)
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


def safe_engine_error(message: str, credentials: EngineCredentials) -> str:
    safe = message[:8_192] or "Unknown coding-agent error"
    secrets = (credentials.minds_api_key,) if credentials.minds_api_key else ()
    return redact_text(safe, secrets)


def is_context_exhaustion_error(message: str | None) -> bool:
    """Identify terminal model-context failures that cannot resume the same thread."""
    normalized = (message or "").casefold()
    return (
        "maximum context length" in normalized
        or "context length exceeded" in normalized
        or "context_length_exceeded" in normalized
        or "too many tokens" in normalized
        or (
            "context window" in normalized
            and any(
                marker in normalized
                for marker in ("exhaust", "ran out", "too long", "full", "limit reached")
            )
        )
    )


def slash_command(prompt: str) -> tuple[str, str]:
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return "", ""
    parts = stripped[1:].split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0].lower(), parts[1] if len(parts) > 1 else ""


def goal_directive(argument: str) -> tuple[str, str | None]:
    """Normalize the human-facing ``/goal`` grammar."""
    if not argument:
        return "view", None
    parts = argument.split(maxsplit=1)
    verb = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""
    lowered = verb.lower()
    if lowered in {"view", "status"}:
        if remainder.strip():
            raise ValueError(f"/goal {lowered} does not accept an objective")
        return "view", None
    if lowered in {"pause", "resume", "clear"}:
        if remainder.strip():
            raise ValueError(f"/goal {lowered} does not accept an objective")
        return lowered, None
    if lowered in {"set", "edit"}:
        objective = remainder.strip()
        if not objective:
            raise ValueError(f"Add an objective after /goal {lowered}")
        return lowered, objective
    return "set", argument.strip()


def goal_status_text(goal: dict) -> str:
    objective = str(goal.get("objective") or "Untitled goal")
    status = str(goal.get("status") or "unknown")
    tokens = goal.get("tokensUsed")
    budget = goal.get("tokenBudget")
    usage = ""
    if isinstance(tokens, int):
        usage = f" · {tokens:,} tokens"
        if isinstance(budget, int):
            usage += f" of {budget:,}"
    return f"Goal ({status}): {objective}{usage}"
