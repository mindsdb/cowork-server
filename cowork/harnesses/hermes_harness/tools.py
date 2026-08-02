from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Artifact run contexts ────────────────────────────────────────
#
# Hermes tool handlers run inside run_agent, which may execute tool
# calls on worker threads — so per-run state can't live in a
# thread-local or contextvar. Instead the harness passes the
# conversation id as run_conversation's task_id (run_agent forwards
# it to every registry dispatch as a kwarg), and this map resolves
# task_id → the run's artifact context.
_ARTIFACT_RUN_CONTEXTS: dict[str, dict] = {}
_ARTIFACT_CONTEXT_LOCK = threading.Lock()


def set_artifact_run_context(
    task_id: str,
    *,
    artifacts_root: Path,
    conversation_id: str,
    conversation_title: str | None,
    turn_summary: str,
    skill_drafts_root: Path | None = None,
) -> None:
    """Register the project context for one Hermes run (call before
    run_conversation; pair with finalize_artifact_run_context).

    ``skill_drafts_root`` is where ``create_skill_draft`` stages built skills —
    a sibling of the artifacts dir, never the live skills store."""
    with _ARTIFACT_CONTEXT_LOCK:
        _ARTIFACT_RUN_CONTEXTS[task_id] = {
            "artifacts_root": Path(artifacts_root),
            "conversation_id": conversation_id,
            "conversation_title": conversation_title,
            "turn_summary": turn_summary,
            "created_slugs": [],
            "skill_drafts_root": Path(skill_drafts_root) if skill_drafts_root else None,
        }


def finalize_artifact_run_context(task_id: str) -> None:
    """Drop the run context and refresh files[] in the metadata of every
    artifact this run created (Hermes writes files with its own file
    tools, so the store only learns about them by rescanning)."""
    with _ARTIFACT_CONTEXT_LOCK:
        ctx = _ARTIFACT_RUN_CONTEXTS.pop(task_id, None)
    if ctx is None or not ctx["created_slugs"]:
        return
    from anton.core.artifacts.store import ArtifactStore

    store = ArtifactStore(ctx["artifacts_root"])
    for slug in ctx["created_slugs"]:
        try:
            store.rescan_files(slug)
        except Exception:
            logger.warning("Could not rescan artifact %r after run", slug, exc_info=True)


def _artifact_context(kwargs: dict) -> dict | None:
    with _ARTIFACT_CONTEXT_LOCK:
        return _ARTIFACT_RUN_CONTEXTS.get(str(kwargs.get("task_id") or ""))


def _hermes_create_artifact(args: dict, **kwargs) -> str:
    from anton.core.artifacts.models import ARTIFACT_TYPES
    from anton.core.artifacts.store import ArtifactStore

    ctx = _artifact_context(kwargs)
    if ctx is None:
        return json.dumps({"error": "Artifact tools are unavailable: no project context for this run."})

    name = str(args.get("name") or "").strip()
    description = str(args.get("description") or "").strip()
    artifact_type = str(args.get("type") or "").strip()
    primary = args.get("primary")
    if not name:
        return json.dumps({"error": "`name` is required."})
    if artifact_type not in ARTIFACT_TYPES:
        return json.dumps({"error": f"`type` must be one of {sorted(ARTIFACT_TYPES)}."})

    store = ArtifactStore(ctx["artifacts_root"])
    artifact = store.create(
        name=name,
        description=description,
        type=artifact_type,
        primary=primary if isinstance(primary, str) else None,
    )
    store.record_turn(
        artifact.slug,
        conversation_id=ctx["conversation_id"],
        conversation_title=ctx["conversation_title"],
        turn_index=1,
        summary=ctx["turn_summary"],
        files_touched=[],
    )
    with _ARTIFACT_CONTEXT_LOCK:
        ctx["created_slugs"].append(artifact.slug)
    return json.dumps(
        {
            "slug": artifact.slug,
            "path": str(store.folder_for(artifact.slug)),
            "name": artifact.name,
            "type": artifact.type,
            "primary": artifact.primary,
        }
    )


def _hermes_list_artifacts(args: dict, **kwargs) -> str:
    from anton.core.artifacts.store import ArtifactStore

    ctx = _artifact_context(kwargs)
    if ctx is None:
        return json.dumps({"error": "Artifact tools are unavailable: no project context for this run."})

    store = ArtifactStore(ctx["artifacts_root"])
    return json.dumps(
        [
            {
                "slug": a.slug,
                "name": a.name,
                "description": (a.description or "")[:200],
                "type": a.type,
                "primary": a.primary,
                "files": len(a.files),
                "path": str(store.folder_for(a.slug)),
            }
            for a in store.list()
        ]
    )


def register_artifact_tools() -> None:
    """Register create_artifact / list_artifacts in run_agent's registry.

    Same folder-per-artifact convention as Anton (anton-core's
    ArtifactStore under `<project>/.anton/artifacts/`), so everything
    Hermes creates surfaces in the Artifacts UI with full
    preview/publish/delete behavior — that pipeline is filesystem-based
    and harness-agnostic.
    """
    from anton.core.artifacts.models import ARTIFACT_TYPES
    from tools.registry import registry

    if registry.get_entry("create_artifact") is None:
        registry.register(
            name="create_artifact",
            toolset="artifacts",
            schema={
                "name": "create_artifact",
                "description": (
                    "Claim a folder for a user-facing output (HTML dashboard, report, "
                    "CSV/dataset, image, app). Call this BEFORE writing any output file; "
                    "it returns {slug, path} — write your files into that absolute `path`. "
                    "Artifacts appear in the app's Artifacts UI where the user can view, "
                    "publish, and manage them. Never invent output folder names yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Human-readable artifact title shown in the UI.",
                        },
                        "description": {
                            "type": "string",
                            "description": "One-or-two-sentence summary shown in the Artifacts UI.",
                        },
                        "type": {
                            "type": "string",
                            "enum": sorted(ARTIFACT_TYPES),
                            "description": "Artifact shape; drives the preview affordance.",
                        },
                        "primary": {
                            "type": "string",
                            "description": (
                                "Optional relative path of the entry-point file you are about "
                                "to write (e.g. dashboard.html)."
                            ),
                        },
                    },
                    "required": ["name", "description", "type"],
                },
            },
            handler=_hermes_create_artifact,
            description="Claim a folder for a user-facing output artifact.",
            emoji="📦",
        )

    if registry.get_entry("list_artifacts") is None:
        registry.register(
            name="list_artifacts",
            toolset="artifacts",
            schema={
                "name": "list_artifacts",
                "description": (
                    "List this project's existing artifacts (slug, name, type, path). "
                    "Use it to find the folder of an artifact you want to modify "
                    "instead of creating a duplicate."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            handler=_hermes_list_artifacts,
            description="List the project's existing artifacts.",
            emoji="📦",
        )


def _hermes_create_skill_draft(args: dict, **kwargs) -> str:
    """Claim a staging folder for a skill the agent is building for the user.

    Returns ``{slug, path, skill_file}`` — the agent writes its SKILL.md into
    ``skill_file``. The folder lives under ``.anton/skill_drafts`` (off-limits to
    discovery, never the live skills store), so the skill is NOT auto-saved; the
    turn-end diff surfaces it as a `response.skill_created` card the user Saves
    or Downloads explicitly. Editing an existing skill pre-seeds the folder from
    the saved version (see ``stage_skill_draft``)."""
    from cowork.services.task_objects import stage_skill_draft

    ctx = _artifact_context(kwargs)
    if ctx is None or not ctx.get("skill_drafts_root"):
        return json.dumps({"error": "Skill-draft tools are unavailable: no project context for this run."})
    return json.dumps(stage_skill_draft(ctx["skill_drafts_root"], args.get("name")))


def register_skill_tools() -> None:
    """Register create_skill_draft in run_agent's registry.

    The sibling of create_artifact for skills: it stages a built skill in a
    draft folder instead of the canonical store, so skills are never auto-saved
    — the user decides via the in-chat card (Save / Download)."""
    from tools.registry import registry

    from cowork.services.task_objects import (
        CREATE_SKILL_DRAFT_DESCRIPTION,
        CREATE_SKILL_DRAFT_SCHEMA,
    )

    if registry.get_entry("create_skill_draft") is None:
        registry.register(
            name="create_skill_draft",
            toolset="skills",
            schema={
                "name": "create_skill_draft",
                "description": CREATE_SKILL_DRAFT_DESCRIPTION,
                "parameters": CREATE_SKILL_DRAFT_SCHEMA,
            },
            handler=_hermes_create_skill_draft,
            description="Stage a built skill as a draft for the user to save or download.",
            emoji="📜",
        )


async def _hermes_lookup_connector(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector
    return await _cowork_lookup_connector(None, args)


async def _hermes_request_credentials(args: dict, **kwargs) -> str:
    from cowork.harnesses.anton_harness.tools import _cowork_request_credentials
    return await _cowork_request_credentials(None, args)


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
