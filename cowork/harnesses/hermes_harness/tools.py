from __future__ import annotations

from contextvars import ContextVar, Token
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Context-local artifacts root, set by the harness for each Hermes turn.
_artifacts_root: ContextVar[Path | None] = ContextVar("hermes_artifacts_root", default=None)


def set_artifacts_root(root: Path | None) -> Token:
    """Called by HermesHarness.stream_response() before launching AIAgent."""
    return _artifacts_root.set(root)


def reset_artifacts_root(token: Token) -> None:
    _artifacts_root.reset(token)


def _get_artifact_store():
    """Return an ArtifactStore for the current project, or None."""
    root = _artifacts_root.get()
    if root is None:
        return None
    from anton.core.artifacts import ArtifactStore
    return ArtifactStore(root)


# ── Connector tool handlers ───────────────────────────────────────


async def _hermes_lookup_connector(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector
    return await _cowork_lookup_connector(None, args)


async def _hermes_request_credentials(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
    return await _cowork_request_credentials(None, args)


# ── Artifact tool handlers ────────────────────────────────────────


async def _hermes_create_artifact(args: dict, **kwargs) -> str:
    store = _get_artifact_store()
    if store is None:
        return "Artifact store unavailable (no project bound to this conversation)."

    name = (args.get("name") or "").strip()
    description = (args.get("description") or "").strip()
    artifact_type = (args.get("type") or "").strip()
    primary = args.get("primary")
    if not name:
        return "Error: `name` is required."
    if not description:
        return "Error: `description` is required."

    from anton.core.artifacts.models import ARTIFACT_TYPES

    if artifact_type not in ARTIFACT_TYPES:
        return f"Error: `type` must be one of {ARTIFACT_TYPES}. Got: {artifact_type!r}."

    artifact = store.create(
        name=name,
        description=description,
        type=artifact_type,
        primary=primary if isinstance(primary, str) else None,
    )
    folder = store.folder_for(artifact.slug)
    return json.dumps({
        "id": artifact.id,
        "slug": artifact.slug,
        "name": artifact.name,
        "type": artifact.type,
        "primary": artifact.primary,
        "path": str(folder),
    }, indent=2)


async def _hermes_set_artifact_primary(args: dict, **kwargs) -> str:
    store = _get_artifact_store()
    if store is None:
        return "Artifact store unavailable (no project bound to this conversation)."

    slug = (args.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    raw = args.get("primary")
    primary = raw if isinstance(raw, str) else None
    artifact = store.set_primary(slug, primary)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    return json.dumps({"slug": artifact.slug, "primary": artifact.primary}, indent=2)


async def _hermes_list_artifacts(args: dict, **kwargs) -> str:
    store = _get_artifact_store()
    if store is None:
        return "Artifact store unavailable (no project bound to this conversation)."

    artifacts = store.list()
    summaries = [
        {
            "slug": a.slug,
            "name": a.name,
            "type": a.type,
            "description": a.description,
            "file_count": len(a.files),
            "updatedAt": a.updatedAt,
        }
        for a in artifacts
    ]
    return json.dumps(summaries, indent=2)


async def _hermes_open_artifact(args: dict, **kwargs) -> str:
    store = _get_artifact_store()
    if store is None:
        return "Artifact store unavailable (no project bound to this conversation)."

    slug = (args.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    artifact = store.open(slug)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    folder = store.folder_for(artifact.slug)
    return json.dumps({
        "id": artifact.id,
        "slug": artifact.slug,
        "name": artifact.name,
        "type": artifact.type,
        "description": artifact.description,
        "primary": artifact.primary,
        "path": str(folder),
        "files": [
            {"path": f.path, "bytes": f.bytes, "modifiedAt": f.modifiedAt}
            for f in artifact.files
        ],
    }, indent=2)


# ── Tool registration ─────────────────────────────────────────────


def register_connector_tools() -> None:
    from tools.registry import registry
    from cowork.harnesses.anton_harness.tools import (
        _LOOKUP_CONNECTOR_SCHEMA,
        _LOOKUP_CONNECTOR_PROMPT,
        _REQUEST_CREDENTIALS_SCHEMA,
        _REQUEST_CREDENTIALS_PROMPT,
    )

    if registry.get_entry("lookup_connector") is None:
        registry.register(
            name="lookup_connector",
            toolset="connectors",
            schema={
                "name": "lookup_connector",
                "description": (
                    "Look up the canonical connector spec for a service by id or "
                    "natural-language query. Returns the same form blob the "
                    "in-app Connector Picker uses — pass it straight to "
                    "`request_credentials`.\n\n"
                    + _LOOKUP_CONNECTOR_PROMPT
                ),
                "parameters": _LOOKUP_CONNECTOR_SCHEMA,
            },
            handler=_hermes_lookup_connector,
            is_async=True,
            description="Look up a connector spec by id or natural-language query.",
            emoji="🔌",
        )

    if registry.get_entry("request_credentials") is None:
        registry.register(
            name="request_credentials",
            toolset="connectors",
            schema={
                "name": "request_credentials",
                "description": (
                    "Request credentials / configuration from the user via an interactive "
                    "form rendered in the side panel. Returns a markdown block you must "
                    "include verbatim in your next assistant message so the form appears.\n\n"
                    + _REQUEST_CREDENTIALS_PROMPT
                ),
                "parameters": _REQUEST_CREDENTIALS_SCHEMA,
            },
            handler=_hermes_request_credentials,
            is_async=True,
            description="Render a credential form in the side panel.",
            emoji="🔐",
        )


def register_artifact_tools() -> None:
    """Register artifact CRUD tools into the Hermes tool registry."""
    from tools.registry import registry

    if registry.get_entry("create_artifact") is not None:
        return

    registry.register(
        name="create_artifact",
        toolset="artifacts",
        schema={
            "name": "create_artifact",
            "description": (
                "Claim a folder for a user-facing output (HTML dashboard, document, "
                "dataset, image, fullstack app, etc.). Call this BEFORE writing the "
                "files — the tool returns the absolute folder path you should write "
                "into.\n\n"
                "Pick `type` from: html-app, document, dataset, image, mixed, "
                "fullstack-stateless-app, fullstack-stateful-app.\n\n"
                "To MODIFY an existing artifact, call `list_artifacts` first, then "
                "`open_artifact(slug)` to get the path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Human-readable artifact name."},
                    "description": {"type": "string", "description": "Short description of the artifact."},
                    "type": {
                        "type": "string",
                        "enum": [
                            "html-app", "document", "dataset", "image", "mixed",
                            "fullstack-stateless-app", "fullstack-stateful-app",
                        ],
                    },
                    "primary": {
                        "type": "string",
                        "description": "Relative path of the entry-point file (optional).",
                    },
                },
                "required": ["name", "description", "type"],
            },
        },
        handler=_hermes_create_artifact,
        is_async=True,
        description="Create a new artifact folder for user-facing output.",
        emoji="📦",
    )

    registry.register(
        name="set_artifact_primary",
        toolset="artifacts",
        schema={
            "name": "set_artifact_primary",
            "description": (
                "Update the primary-file pointer on an existing artifact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Folder slug of the artifact."},
                    "primary": {"type": "string", "description": "Relative path of the new entry-point file."},
                },
                "required": ["slug"],
            },
        },
        handler=_hermes_set_artifact_primary,
        is_async=True,
        description="Update the primary file of an artifact.",
        emoji="📦",
    )

    registry.register(
        name="list_artifacts",
        toolset="artifacts",
        schema={
            "name": "list_artifacts",
            "description": (
                "List every artifact in the current workspace (newest first)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_hermes_list_artifacts,
        is_async=True,
        description="List all artifacts in the workspace.",
        emoji="📦",
    )

    registry.register(
        name="open_artifact",
        toolset="artifacts",
        schema={
            "name": "open_artifact",
            "description": (
                "Load an existing artifact by slug. Returns the folder path and "
                "file list so you can decide what to edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Folder slug of the artifact."},
                },
                "required": ["slug"],
            },
        },
        handler=_hermes_open_artifact,
        is_async=True,
        description="Open an existing artifact by slug.",
        emoji="📦",
    )
