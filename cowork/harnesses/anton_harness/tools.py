"""Cowork-side tool wrappers.

Anton's stock tools (PUBLISH_TOOL, CONNECT_DATASOURCE_TOOL, …) are
written for the CLI: they assume a Rich Console attached to a TTY,
they pop the system browser, and they hold the user's gaze with
animated spinners. None of that works inside the FastAPI process the
desktop app spawns.

We build cowork-flavoured wrappers that share the LLM-facing schema
(name / description / input_schema) so the model uses them
identically, but whose handlers do the actual work in a way that
makes sense for a server process: no console.print, no Live spinner,
no webbrowser.open. Status flows back to the desktop UI through the
normal SSE event stream and the response string the LLM renders.

Right now we only override PUBLISH_TOOL — the only one users have hit.
Add more here as needed (CONNECT_DATASOURCE_TOOL is the next likely
candidate; same pattern).
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class PublishSnapshotError(RuntimeError):
    pass


def _get_env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _published_state(raw_path: str) -> dict:
    """Delegate to the publish service so the tool and GUI agree on where
    `.published.json` lives. Imported lazily to avoid a startup import cycle."""
    from cowork.services.publish import published_state
    return published_state(raw_path)


def _snapshot_publish_attempt(file_path: Path):
    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_open_session
        from cowork.services.artifact_versions import snapshot_artifact

        with get_open_session(get_app_settings().database.uri) as db_session:
            version = snapshot_artifact(db_session, str(file_path), operation_type="publish", label="Published version")
            return getattr(version, "id", None)
    except Exception:
        logger.warning("Could not create publish checkpoint for %s", file_path, exc_info=True)
        return None


def _publish_version_metadata(version_id) -> dict[str, Any]:
    if version_id is None:
        return {}
    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_open_session
        from cowork.models.artifact import ArtifactVersion

        with get_open_session(get_app_settings().database.uri) as db_session:
            version = db_session.get(ArtifactVersion, version_id)
            if version is None:
                return {}
            return {
                "artifact_id": str(version.artifact_id),
                "files_hash": version.files_hash,
                "manifest_hash": version.manifest_hash,
                "version_number": version.version_number,
            }
    except Exception:
        logger.warning("Could not read publish version metadata for %s", version_id, exc_info=True)
        return {}


def _record_publish_result(version_id, *, status: str, url: str | None = None, details: dict | None = None) -> None:
    if version_id is None:
        return
    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_open_session
        from cowork.models.artifact import Artifact, ArtifactVersion
        from cowork.services.artifact_versions import record_deployment

        with get_open_session(get_app_settings().database.uri) as db_session:
            version = db_session.get(ArtifactVersion, version_id)
            if version is None:
                return
            failed_deployment = record_deployment(
                db_session,
                version,
                target="publish",
                status=status,
                url=url,
                details=details or {},
            )
            if status == "failed":
                artifact = db_session.get(Artifact, version.artifact_id)
                if artifact is not None:
                    rollback_version_id = artifact.last_known_good_version_id or version.parent_version_id
                    rollback_error = None
                    if rollback_version_id is not None:
                        try:
                            from cowork.services.artifact_versions import ArtifactVersionService

                            ArtifactVersionService(db_session).replace_with_version(
                                rollback_version_id,
                                artifact.path,
                                preserve_published=True,
                            )
                        except Exception as exc:
                            rollback_error = str(exc) or exc.__class__.__name__
                            logger.warning(
                                "Could not roll back artifact files after failed publish for version %s",
                                version_id,
                                exc_info=True,
                            )
                    if rollback_error:
                        failed_details = dict(failed_deployment.details or {})
                        failed_details["rollbackError"] = rollback_error
                        failed_deployment.details = failed_details
                        db_session.add(failed_deployment)
                        db_session.commit()
                    elif rollback_version_id is not None:
                        artifact.current_version_id = rollback_version_id
                        db_session.add(artifact)
                        db_session.commit()
    except Exception:
        logger.warning("Could not record publish %s for version %s", status, version_id, exc_info=True)


@contextmanager
def _materialized_publish_source(file_path: Path, version_id):
    """Yield the immutable snapshot path that should be uploaded for a publish."""
    if version_id is None:
        yield file_path
        return

    temp_dir: TemporaryDirectory | None = None
    source = file_path
    try:
        from cowork.common.settings.app_settings import get_app_settings
        from cowork.db.session import get_open_session
        from cowork.models.artifact import Artifact, ArtifactVersion
        from cowork.services.artifact_versions import ArtifactVersionService

        temp_dir = TemporaryDirectory(prefix="cowork-chat-publish-version-")
        with get_open_session(get_app_settings().database.uri) as db_session:
            version = db_session.get(ArtifactVersion, version_id)
            if version is not None:
                artifact = db_session.get(Artifact, version.artifact_id)
                materialized_root = Path(temp_dir.name) / "artifact"
                version_service = ArtifactVersionService(db_session)
                version_service.materialize_version(
                    version.id,
                    materialized_root,
                    clean=True,
                )
                version_service.write_version_housekeeping(version.id, materialized_root)
                source = materialized_root
                if artifact is not None:
                    try:
                        rel = file_path.relative_to(Path(artifact.path))
                        candidate = materialized_root / rel
                        if candidate.is_file():
                            source = candidate
                    except ValueError:
                        pass
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
            temp_dir = None
        logger.exception("Could not materialize publish checkpoint for %s", file_path)
        raise PublishSnapshotError("Could not prepare the versioned artifact for publishing")

    try:
        yield source
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


# Cowork-flavoured description/prompt for the publish_or_preview tool.
# The CLI-flavoured copies inside anton-core's `tools.py` mention the
# legacy `.anton/output/` artifacts dir and reference a `/publish`
# slash command that doesn't exist in antontron — both confuse the
# LLM in the desktop context, so we override them in `build_cowork_publish_tool`.
COWORK_PUBLISH_DESCRIPTION = (
    "Preview, check, or publish an HTML dashboard / report. Files live "
    "under the project's `artifacts/<artifact-id>/<name>.html`. Actions: "
    "'ask' (default) and 'preview' check whether the file is already "
    "published and return the public URL if so — they DON'T publish; "
    "use them when generating a new file to confirm state. 'publish' "
    "actually publishes (or re-publishes if a report_id already exists) "
    "and returns the public URL. In the desktop app the user can also "
    "publish via the Live Artifacts panel — but you should call "
    "action='publish' directly whenever the user asks to publish, "
    "share, deploy, or make-public any artifact you generated. No "
    "slash command, no extra confirmation."
)
COWORK_PUBLISH_PROMPT = (
    "CONTENT SHARING POLICY (desktop chat):\n"
    "- Publishing dashboards or reports to the web is done ONLY via the `publish_or_preview` tool.\n"
    "- When the user asks to publish / share / deploy / make-public a generated artifact, call this\n"
    "  tool with `action: 'publish'` directly. Don't ask the user to use a slash command or a UI\n"
    "  panel — publishing works straight from chat when MindsHub is configured.\n"
    "- To check whether something is already published (e.g. the user asks 'is it live?'), call with\n"
    "  `action: 'ask'` or `action: 'preview'` — both return the public URL if one exists, without\n"
    "  publishing or re-publishing.\n"
    "- Re-running with `action: 'publish'` on a file that was already published reuses its\n"
    "  `report_id` so the public URL stays stable across edits.\n"
    "- Do NOT upload, post, or share generated files (HTML, data, images) to external hosting\n"
    "  services (paste sites, gists, CDNs, file hosts) via scratchpad code — unless the user\n"
    "  explicitly names the service and confirms. This rule applies only to sharing generated\n"
    "  output with the public internet; reading public APIs and writing to the user's connected\n"
    "  datasources (databases, CRMs, etc.) is fine."
)


async def _cowork_publish_or_preview(session: Any, tc_input: dict) -> str:
    """Server-side equivalent of anton.tools.handle_publish_or_preview.

    Mirrors the same `action` semantics:
      - 'ask' / 'preview' → return a string pointing the user at the
        Live Artifacts panel; the desktop UI already exposes preview
        and publish buttons there. We don't open a browser here.
      - 'publish' → call anton.publisher.publish directly, persist the
        result in `<output_dir>/.published.json`, return the view URL.
    """
    raw_path = tc_input.get("file_path", "")
    title = tc_input.get("title", "Dashboard")
    action = (tc_input.get("action") or "ask").lower()

    if not raw_path:
        return "publish_or_preview: missing file_path"

    file_path = Path(raw_path).expanduser()
    if not file_path.is_absolute():
        # Anton's session carries the active workspace base.
        workspace = getattr(session, "_workspace", None)
        if workspace is not None:
            base = getattr(workspace, "base", None)
            if base:
                file_path = Path(base) / raw_path
    file_path = file_path.resolve()

    if not file_path.exists():
        return f"File not found: {file_path}"

    abs_path = str(file_path)

    # 'ask'/'preview' are non-destructive: report current publish state so the
    # LLM can choose publish-vs-re-publish. State is resolved by the service so
    # fullstack (.published.json at the artifact root) and static (next to the
    # primary file) never disagree — the desync that gave fullstack a new URL
    # on every re-publish.
    #
    # Why the publish path is spelled out so directly: an earlier version told
    # the LLM "the user can publish from the Live Artifacts panel", which it
    # read as "publishing is the user's job" and never called action='publish'
    # even when explicitly asked. The wording below makes clear it CAN act.
    if action in ("ask", "preview"):
        state = _published_state(abs_path)
        if state.get("published") and state.get("url"):
            return (
                f"{title} is already published at {state['url']}. "
                f"It is also visible inline + in the Live Artifacts panel. "
                f"If the user asks to re-publish, call this tool again with "
                f"action='publish' — the same report_id is reused so the URL stays stable."
            )
        return (
            f"{title} is at {file_path} and visible in the Live Artifacts "
            f"panel. It has NOT been published to the public web yet. If the "
            f"user asks to publish / share / make it public, call this tool "
            f"again with action='publish' — MindsHub publishing works directly "
            f"from chat in the desktop app, no slash command needed."
        )

    if action != "publish":
        return f"publish_or_preview: unknown action '{action}'"

    # ── action == 'publish' ───────────────────────────────────────────
    # Read the API key the same way the cowork HTTP endpoint does so
    # both code paths agree on what's "configured".
    from .settings import AntonHarnessSettings

    api_key = _get_env("ANTON_MINDS_API_KEY")
    if not api_key:
        return (
            "STOP: No Minds API key configured. Tell the user to set their "
            "Minds API key in Settings (or in their .env) before publishing. "
            "Do NOT call this tool again until they confirm the key is set."
        )

    settings = AntonHarnessSettings()
    publish_url = settings.publish_url
    ssl_verify = settings.publish_ssl_verify

    try:
        from anton.publisher import publish
    except Exception as exc:
        logger.exception("Cowork publish tool could not import anton.publisher")
        return f"PUBLISH FAILED: anton.publisher unavailable ({exc})"

    output_dir = file_path.parent
    published_json_path = output_dir / ".published.json"
    published_map: dict[str, Any] = {}
    if published_json_path.is_file():
        try:
            published_map = json.loads(published_json_path.read_text(encoding="utf-8"))
        except Exception:
            published_map = {}

    file_key = file_path.name
    prev = published_map.get(file_key)
    report_id = prev.get("report_id") if isinstance(prev, dict) else None

    def _do_publish(source_path: Path, rid: str | None):
        return publish(
            source_path,
            api_key=api_key,
            report_id=rid,
            publish_url=publish_url,
            ssl_verify=ssl_verify,
        )

    publish_version_id = _snapshot_publish_attempt(file_path)
    if publish_version_id is None:
        return "PUBLISH FAILED: Could not create a publish checkpoint."
    try:
        with _materialized_publish_source(file_path, publish_version_id) as publish_source:
            result = _do_publish(publish_source, report_id)
    except PublishSnapshotError as exc:
        _record_publish_result(
            publish_version_id,
            status="failed",
            details={"error": str(exc)},
        )
        return f"PUBLISH FAILED: {exc}"
    except Exception as exc:
        # If we tried to update an existing report and the upstream
        # rejected it (e.g. report was deleted), retry as a fresh one
        # — same recovery path anton's CLI tool uses.
        if report_id:
            try:
                with _materialized_publish_source(file_path, publish_version_id) as publish_source:
                    result = _do_publish(publish_source, None)
            except PublishSnapshotError as snapshot_exc:
                _record_publish_result(
                    publish_version_id,
                    status="failed",
                    details={"error": str(snapshot_exc)},
                )
                return f"PUBLISH FAILED: {snapshot_exc}"
            except Exception as retry_exc:
                logger.exception("Cowork publish retry failed")
                _record_publish_result(
                    publish_version_id,
                    status="failed",
                    details={"error": str(retry_exc)},
                )
                return f"PUBLISH FAILED: {retry_exc}"
        else:
            logger.exception("Cowork publish failed")
            _record_publish_result(
                publish_version_id,
                status="failed",
                details={"error": str(exc)},
            )
            return f"PUBLISH FAILED: {exc}"

    view_url = result.get("view_url", "") if isinstance(result, dict) else ""
    returned_report_id = result.get("report_id", "") if isinstance(result, dict) else ""

    if returned_report_id:
        published_map[file_key] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", "") if isinstance(result, dict) else "",
            # Mirror the GUI publish path (services.publish.publish_artifact): a
            # record written by the chat tool must read back identically. Readers
            # gate liveness on this flag (a later unpublish flips it to False),
            # so omitting it would only work by relying on the .get(..., True)
            # default — set it explicitly.
            "published": True,
            "version_id": str(publish_version_id or ""),
            **_publish_version_metadata(publish_version_id),
        }
        try:
            published_json_path.write_text(
                json.dumps(published_map, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Could not persist .published.json", exc_info=True)

    _record_publish_result(
        publish_version_id,
        status="published",
        url=view_url,
        details={"reportId": returned_report_id},
    )
    if not view_url:
        return "Published, but no view URL was returned."
    return f"Published successfully! View URL: {view_url}"


def build_cowork_publish_tool():
    """Construct a ToolDef matching anton's PUBLISH_TOOL schema, but
    with the cowork-aware handler. Lazy-imports anton.tools so callers
    can build the session config without paying the import cost twice.

    The description and prompt are replaced with cowork-flavoured copy
    that names the right artifacts path (`artifacts/<id>/<file>.html`,
    not the legacy `.anton/output/`) and tells the LLM publishing
    works directly from chat — no slash command, no UI dance. Without
    these overrides the LLM defaults to CLI-era guidance and refuses
    to call `action: 'publish'` even when the user explicitly asked.
    """
    from anton.tools import PUBLISH_TOOL
    from anton.core.tools.tool_defs import ToolDef

    return ToolDef(
        name=PUBLISH_TOOL.name,
        description=COWORK_PUBLISH_DESCRIPTION,
        input_schema=PUBLISH_TOOL.input_schema,
        handler=_cowork_publish_or_preview,
        prompt=COWORK_PUBLISH_PROMPT,
    )


##################
# Connector Tools
##################

# Lookup Connector Tool
# The chat agent's "find the canonical spec for this service" path —
# same source of truth the in-app Connector Picker uses. Without this
# tool the LLM would hand-craft each form from training-data memory,
# producing slightly different shapes on each call (sometimes missing
# OAuth, sometimes bare username/password where the service has a
# proper OAuth path, etc.). Routing through the registry guarantees
# the chat-emitted form is byte-identical to what the picker emits.
#
# Mirrors POST /v1/connectors/specs/match:
# stage 1 exact match (id or alias, normalized) → stage 2 token-overlap.
# We don't reuse the route helpers directly to avoid a circular import
# (routes already pulls from anton_api), but the scoring stays in step.

def _lookup_normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _lookup_score(query: str, c: dict) -> float:
    q_tokens = set(_lookup_normalize(query).split())
    if not q_tokens:
        return 0.0
    label_tokens = set(_lookup_normalize(c.get("label", "")).split())
    alias_tokens: set[str] = set()
    for alias in c.get("aliases", []):
        alias_tokens.update(_lookup_normalize(alias).split())
    keyword_tokens = set(_lookup_normalize(" ".join(c.get("keywords", []))).split())
    desc_tokens = set(_lookup_normalize(c.get("description", "")).split())
    score = 0.0
    score += 3.0 * len(q_tokens & label_tokens)
    score += 2.5 * len(q_tokens & alias_tokens)
    score += 1.0 * len(q_tokens & keyword_tokens)
    score += 0.4 * len(q_tokens & desc_tokens)
    return score


async def _cowork_lookup_connector(session: Any, tc_input: dict) -> str:
    """Tool handler for `lookup_connector`.

    Pulls a registry-backed connector spec by id (slug) or by
    natural-language query. The returned `form` blob is what the LLM
    should pass to `request_credentials` verbatim — methods, OAuth
    blocks, how_to markdown, and help_url all already filled in. We
    also stamp `_connector_id` onto the form so submissions land via
    the registry-aware save path (POST /v1/connectors/{id}/save).
    """
    try:
        from cowork.services.connectors.specs._registry import registry
    except Exception as exc:
        logger.exception("Cowork lookup_connector could not import registry")
        return f"lookup_connector: registry unavailable ({exc})"

    cid = (tc_input.get("id") or "").strip()
    query = (tc_input.get("query") or "").strip()

    if not cid and not query:
        return "lookup_connector: provide either `id` (a connector slug) or `query` (a natural-language name)."

    def _present(connector: dict, confidence: float, stage: str) -> str:
        # Stamp `_connector_id` into the form blob so the LLM doesn't
        # have to remember to copy it across. The vault save endpoint
        # uses this to route through POST /v1/connectors/{id}/save
        # (which bypasses anton-core's built-in datasource registry).
        form_blob = dict(connector.get("form") or {})
        form_blob["_connector_id"] = connector.get("id")
        # Also copy the connector's logo onto the form if the form
        # itself doesn't carry one — the picker already does this for
        # the in-app picker path; mirroring it here keeps the chat-
        # emitted form visually identical.
        if "logo" not in form_blob and connector.get("logo"):
            form_blob["logo"] = connector["logo"]
        if "logo_color" not in form_blob and connector.get("logo_color"):
            form_blob["logo_color"] = connector["logo_color"]
        return json.dumps({
            "id": connector.get("id"),
            "label": connector.get("label"),
            "description": connector.get("description"),
            "category": connector.get("category"),
            "confidence": confidence,
            "stage": stage,
            "form": form_blob,
            "next_step": (
                "Pass this `form` blob to `request_credentials` verbatim "
                "(only tweak `selected_method` or `subtitle` if needed). "
                "It already includes `_connector_id`, methods[], OAuth "
                "blocks, how_to markdown, and help_url where applicable."
            ),
        })

    # Stage 0 — explicit id always wins.
    if cid:
        c = registry.get_connector(cid)
        if c:
            return _present(c.model_dump(), 1.0, "id")
        # Fall through to query-style lookup if the id didn't match.
        if not query:
            return json.dumps({
                "id": None,
                "match": "none",
                "message": (
                    f"No connector with id `{cid}` in the registry. "
                    f"Either retry with a `query` (natural-language) or "
                    f"handcraft the form spec — see the request_credentials "
                    f"schema for the OAuth/how_to/help_url fields."
                ),
                "available_ids": sorted(registry.get_connectors().keys()),
            })
        # If both were given and id missed, treat the id as part of the query.
        query = f"{cid} {query}".strip()

    # Stage 1 — exact match (id or alias) on the query.
    nq = _lookup_normalize(query)
    for c in registry.get_connectors().values():
        if _lookup_normalize(c.get("id", "")) == nq:
            return _present(c, 1.0, "exact-id")
        for alias in c.get("aliases", []):
            if _lookup_normalize(alias) == nq:
                return _present(c, 1.0, "exact-alias")

    # Stage 2 — token-overlap. Return up to 3 candidates with confidence.
    scored: list[tuple[float, dict]] = []
    for c in registry.get_connectors().values():
        s = _lookup_score(query, c)
        if s > 0:
            scored.append((s, c))
    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return json.dumps({
            "id": None,
            "match": "none",
            "message": (
                "No connector matched the query. Either ask the user to "
                "clarify, or handcraft the form spec — see the "
                "request_credentials schema for the OAuth/how_to/help_url "
                "fields you should fill in when you know the auth shape."
            ),
            "available_ids": sorted(registry.get_connectors().keys()),
        })

    top_score, top_c = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    # Top is dominant — single confident pick.
    if runner_up == 0.0 or top_score >= runner_up * 2:
        return _present(top_c, 0.85, "scored-single")

    # Otherwise return the top three with normalized confidence so the
    # LLM can either (a) pick one based on chat context or (b) ask the
    # user to clarify. We do NOT inline `form` for the runners-up to
    # keep the response small — the LLM can re-call lookup_connector
    # with an `id` once it picks.
    candidates = [
        {
            "id": c.get("id"),
            "label": c.get("label"),
            "description": c.get("description"),
            "confidence": round(s / top_score, 3),
        }
        for s, c in scored[:3]
    ]
    return json.dumps({
        "match": "ambiguous",
        "candidates": candidates,
        "message": (
            "Multiple connectors matched. Either ask the user to clarify, "
            "or call `lookup_connector` again with the chosen `id` to fetch "
            "the full form spec."
        ),
    })


_LOOKUP_CONNECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Connector slug (e.g. 'gmail', 'postgres', 'slack'). Use this when you already know the canonical id; matches exactly. Returns the full form spec including methods[], OAuth, how_to, help_url, and a stamped `_connector_id`.",
        },
        "query": {
            "type": "string",
            "description": "Natural-language query (e.g. 'google mail', 'my postgres database', 'send slack messages'). Runs id-or-alias exact match, then token-overlap scoring, returning a single confident hit or up to 3 candidates with confidence scores.",
        },
    },
}


_LOOKUP_CONNECTOR_PROMPT = (
    "Use `lookup_connector` to fetch the canonical form spec for any "
    "service the user names. The returned `form` blob is the SAME "
    "spec the in-app Connector Picker uses — pass it to "
    "`request_credentials` verbatim. The registry already encodes "
    "OAuth flows, how_to markdown, help URLs, and method-picker UI "
    "for services like Gmail, so handcrafting from training-data "
    "memory will produce a strictly worse form. Always lookup first; "
    "only handcraft when the registry returns no match."
)


def build_cowork_lookup_connector_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="lookup_connector",
        description=(
            "Look up the canonical connector spec for a service by id or "
            "natural-language query. Returns the same form blob the "
            "in-app Connector Picker uses — pass it straight to "
            "`request_credentials`."
        ),
        input_schema=_LOOKUP_CONNECTOR_SCHEMA,
        handler=_cowork_lookup_connector,
        prompt=_LOOKUP_CONNECTOR_PROMPT,
    )


# Request Credentials Tool
# After the lookup_connector tool surfaces the canonical form spec, the agent calls
# `request_credentials` to render the form for the user.

def _ensure_form_id(spec: dict) -> dict:
    """Normalize a form spec — generate a form_id if missing, fall back
    to a default title. Mutates a copy and returns it.
    """
    out = dict(spec)
    if not out.get("form_id"):
        out["form_id"] = "fm_" + uuid.uuid4().hex[:10]
    if not out.get("title"):
        out["title"] = "Connect"
    has_methods = isinstance(out.get("methods"), list) and out.get("methods")
    if not has_methods and (
        "fields" not in out or not isinstance(out.get("fields"), list)
    ):
        out["fields"] = []
    return out


async def _cowork_request_credentials(session: Any, tc_input: dict) -> str:
    """Tool handler for `request_credentials`.

    The LLM hands us a form spec; we wrap it in a `data-vault-form`
    markdown block (the renderer's MarkdownCode picks this up and
    publishes the spec into the per-conversation form store, which
    the right-rail DataVaultFormPanel mounts). The returned string
    instructs the LLM to relay the block verbatim.
    """
    spec = tc_input.get("spec") if isinstance(tc_input.get("spec"), dict) else tc_input
    if not isinstance(spec, dict):
        return "request_credentials: invalid spec — must be a JSON object with `title` and `fields`"

    spec = _ensure_form_id(spec)
    block = "```data-vault-form\n" + json.dumps(spec, indent=2) + "\n```"
    return (
        "Form ready. Include the following markdown block VERBATIM in your "
        "next message so it renders for the user in the side panel — do not "
        "summarize or paraphrase the JSON.\n\n"
        "FORMATTING (critical): the opening ``` and the closing ``` must "
        "each be on their own line, with a blank line BEFORE the opening "
        "fence and AFTER the closing fence. Do not concatenate the fence "
        "onto the end of a sentence — markdown parsers won't recognise it "
        "as a code block if it isn't at the start of a line.\n\n"
        "After the user submits, you'll receive a continuation message "
        "with `submission_id` (and any skipped field names). Call "
        "`fetch_submission(submission_id)` to retrieve the staged values "
        "when you need them.\n\n"
        f"{block}"
    )


_REQUEST_CREDENTIALS_SCHEMA = {
    "type": "object",
    "properties": {
        "form_id": {
            "type": "string",
            "description": "Stable identifier for this form. Generate a new one for a new question; reuse the same one when re-asking the same form (so the user's typed values persist).",
        },
        "engine": {
            "type": "string",
            "description": "REQUIRED. A short slug for the connector (e.g. 'postgres', 'mysql', 'snowflake', 'github', 'posthog', 'salesforce', 'gmail', 'google_calendar'). Use the closest convention; ANY value is accepted — engines not in anton's built-in registry are saved as 'custom' connections with whatever fields you list here. Don't gate on whether it's a known engine.",
        },
        "_connector_id": {
            "type": "string",
            "description": "OPTIONAL. The slug of the canonical connector spec (from `lookup_connector`) this form was built from. When set, submissions go to POST /v1/connectors/{id}/save (which bypasses anton-core's built-in datasource registry — required for OAuth-shaped saves). ALWAYS set this when the form spec came from `lookup_connector`; leave unset when handcrafting a one-off spec.",
        },
        "how_to": {
            "type": "string",
            "description": "OPTIONAL. Markdown-formatted setup instructions for SINGLE-method forms (use the per-method `how_to` for multi-method specs). The form panel shows a 'How to?' link in the actions row that opens a centered modal with this content.",
        },
        "help_url": {
            "type": "string",
            "description": "OPTIONAL. External help URL for SINGLE-method forms (use the per-method `help_url` for multi-method specs). Used as a fallback when no `how_to` markdown is provided.",
        },
        "logo": {
            "type": "string",
            "description": "Optional icon name from the app's palette — use one of: 'database', 'globe', 'cube', 'doc', 'code', 'image', 'folder', 'brain', 'sparkle', 'wifi', 'key', 'link', 'mindsdb'. URLs are NOT supported; pick the closest semantic match for the connector. Defaults to 'database' when omitted.",
        },
        "logo_color": {
            "type": "string",
            "description": "Optional CSS color for the icon (e.g. '#3b82f6', 'var(--accent)').",
        },
        "title": {
            "type": "string",
            "description": "Short headline (e.g. 'Connect to Postgres').",
        },
        "subtitle": {
            "type": "string",
            "description": "Optional one-liner under the title (e.g. 'Anton needs read-only access — credentials never leave your machine.').",
        },
        "form_warning": {
            "type": "string",
            "description": "Optional amber banner above the fields. Use for cautionary notes ('Last attempt timed out…').",
        },
        "form_error": {
            "type": "string",
            "description": "Optional red banner above the fields. Use when a previous attempt failed at the form level (e.g. wrong engine selected).",
        },
        "fields": {
            "type": "array",
            "description": "Field specs the user fills in (single-method form). Order matters — render top to bottom. For services with MULTIPLE auth options (Gmail = OAuth + app-password + service-account, etc.) prefer `methods` instead.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Field id (env-var-like)."},
                    "label": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["text", "password", "url", "select", "textarea", "boolean"],
                    },
                    "required": {"type": "boolean"},
                    "placeholder": {"type": "string"},
                    "default": {},
                    "value": {"description": "Pre-fill on re-render (e.g. preserve what the user typed last attempt)."},
                    "options": {
                        "type": "array",
                        "description": "For type=select.",
                        "items": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}, "label": {"type": "string"}},
                        },
                    },
                    "error": {"type": "string", "description": "Per-field red text under the input. Set on a retry to call out which field needs attention."},
                    "warning": {"type": "string", "description": "Per-field amber text under the input."},
                    "help": {"type": "string", "description": "Muted helper text under the input."},
                    "skipable": {"type": "boolean", "description": "Defaults to true. Pass false ONLY for absolute requirements where skipping makes no sense."},
                },
                "required": ["name", "label", "type"],
            },
        },
        "methods": {
            "type": "array",
            "description": (
                "Use INSTEAD of `fields` when the engine supports multiple auth methods "
                "(e.g. Gmail can be reached via OAuth, App Password, or Service Account). "
                "The user picks one method first, then fills in just that method's fields. "
                "Each method should have a clear label, a 1-2 sentence description, and "
                "its own fields. Mark the easiest one with `recommended: true`. If the "
                "user has already signalled a preference (\"I have an app password\"), "
                "set `selected_method` at the form's top level so we skip the picker."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable id for this method (e.g. 'oauth', 'app-password', 'service-account'). Anton's probe uses this id to decide which auth flow to test, and the vault stamps it onto the saved record as `_method`."},
                    "label": {"type": "string", "description": "Card title (e.g. 'App Password')."},
                    "description": {"type": "string", "description": "1-2 sentence description shown on the picker card. Mention prereqs and difficulty when relevant."},
                    "recommended": {"type": "boolean", "description": "Mark the easiest/most common method. Renders a 'Recommended' pill on the card."},
                    "how_to": {"type": "string", "description": "Optional markdown-formatted setup instructions for THIS method. Picker cards show a 'How to?' affordance that opens a centered modal with this content; the same link travels into the form-fill stage. Prefer this over `help_url` when you can write a good walkthrough."},
                    "help_url": {"type": "string", "description": "Optional external help URL for THIS method. Used when no `how_to` markdown is provided."},
                    "submit_action": {
                        "type": "string",
                        "enum": ["oauth_launch"],
                        "description": "Optional. When set to `oauth_launch`, the panel runs a PKCE OAuth flow on the user's machine when they click Submit (spawns a loopback HTTP server, opens the browser to the consent screen, exchanges the code for tokens, and saves the refresh_token to the vault). Required for any OAuth method. Pair with the `oauth` block.",
                    },
                    "oauth": {
                        "type": "object",
                        "description": "Required when `submit_action` is `oauth_launch`. Provides everything the desktop's PKCE helper needs to run the flow without the LLM in the loop. Anton spawns a loopback server, opens the browser, exchanges the code, and stores the refresh_token in the vault under this connector.",
                        "properties": {
                            "auth_url": {"type": "string", "description": "OAuth 2.0 authorization endpoint (e.g. https://accounts.google.com/o/oauth2/v2/auth)."},
                            "token_url": {"type": "string", "description": "OAuth 2.0 token endpoint (e.g. https://oauth2.googleapis.com/token)."},
                            "scopes": {
                                "type": "array",
                                "description": "Scopes to request. Provider-specific.",
                                "items": {"type": "string"},
                            },
                            "extra_auth_params": {
                                "type": "object",
                                "description": "Extra query params on the auth URL (e.g. {access_type: 'offline', prompt: 'consent'} for Google to force a refresh_token).",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["auth_url", "token_url", "scopes"],
                    },
                    "fields": {
                        "type": "array",
                        "description": "Fields specific to this method. Same shape as the top-level `fields` items. For `oauth_launch` methods, fields are typically `client_id` + `client_secret` (Pattern B — bring-your-own-OAuth-client) and may be empty when the connector ships a hosted client.",
                        "items": {"type": "object"},
                    },
                    "actions": {
                        "type": "array",
                        "description": "OPTIONAL — per-method action buttons. Falls back to the form's top-level actions, then to a default Submit + Cancel pair.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["id", "label", "fields"],
            },
        },
        "selected_method": {
            "type": "string",
            "description": "Pre-pick a method id from `methods[]` (skips the picker, jumps straight to that method's fields). Set when the user has clearly indicated a preference. The user can still hit 'change' to re-open the picker.",
        },
        "actions": {
            "type": "array",
            "description": "Optional. Defaults to a single primary 'Submit' action plus 'Cancel'. Use to surface custom actions like 'Try OAuth' or per-field skip shortcuts.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["primary", "skip", "cancel"]},
                    "field": {"type": "string", "description": "Only for kind='skip' — the field name to mark skipped."},
                },
                "required": ["id", "label"],
            },
        },
    },
    "required": ["engine", "title"],
}


_REQUEST_CREDENTIALS_PROMPT = (
    "DATA VAULT WORKFLOW — when the user asks to connect to a service or database:\n"
    "1. ALWAYS call `lookup_connector` FIRST with the user's wording (e.g. 'gmail', "
    "'google mail', 'postgres'). When it returns a `form` spec, that is the SAME "
    "spec the in-app Connector Picker uses — pass it to `request_credentials` "
    "VERBATIM (only tweak `selected_method` if the user has clearly signalled a "
    "preference, or `subtitle` to match the conversation context). Registry specs "
    "ship with `methods[]`, OAuth blocks (`submit_action: 'oauth_launch'` + "
    "`oauth: {...}`), `how_to` markdown, and `help_url` already filled in — do "
    "not strip those, do not paraphrase them. ALSO copy `_connector_id` (the "
    "lookup tool returns it) onto the form spec so submissions go to the "
    "registry-aware save endpoint.\n"
    "2. ONLY when `lookup_connector` returns no match, handcraft the spec. The "
    "schema documents OAuth, `how_to`, and `help_url` fields — use them when you "
    "know the auth shape (e.g. an OAuth provider). Better to write a good "
    "registry-grade spec than a bare username/password prompt.\n"
    "3. Call `request_credentials` with the (registry or handcrafted) spec the "
    "FIRST time. Include the returned markdown block VERBATIM (with blank lines "
    "around the fence) so the form renders in the side panel.\n"
    # "4. Wait for the user's submission. The follow-up message has a `submission_id` "
    # "and the names of any skipped fields. For OAuth methods, the desktop runs the "
    # "browser flow itself — you'll just receive the submission with the refresh "
    # "token already in the vault.\n"
    # "5. Call `fetch_submission(submission_id)` to retrieve the staged values. Test "
    # "the connection (`connect_datasource` or a scratchpad probe).\n"
    # "6. ON FAILURE — DO NOT re-emit the full form. Use `update_form` (which "
    # "returns a `data-vault-form-patch` block) to attach an `error` to the failing "
    # "field and tweak `subtitle` / `form_warning` if useful. The user's existing "
    # "inputs stay in the panel; only the changed bits update. NEVER include `value` "
    # "fields in a patch or full re-emit — that would echo credentials into chat "
    # "history. The user re-types what they want to fix.\n"
    # "7. On success, summarize what you connected and stop. Do NOT call "
    # "`request_credentials` again unless the user asks for another connection.\n\n"
    "4. Your job is done. The server tests the connection and saves credentials "
    "automatically once the user submits. Do NOT call `request_credentials` again "
    "unless the user asks to connect to a different service or explicitly requests "
    "a new form.\n\n"
    "MULTI-METHOD SHAPE — for engines with several auth options (Gmail can be "
    "reached via OAuth, App Password, or Service Account; Postgres might support "
    "password auth + IAM, etc.) emit a `methods[]` array INSTEAD of `fields[]`. "
    "Each method has its own id, label, description, fields, and (when relevant) "
    "`how_to` markdown / `help_url` / OAuth block. Mark the simplest one with "
    "`recommended: true`. The form panel renders a picker first; the user "
    "chooses, then types only the fields for the chosen method. If the user has "
    "CLEARLY signalled a preference (\"I already have an app password\"), "
    "pre-set `selected_method` to skip the picker.\n\n"
    "STRICT RULES:\n"
    "- Field VALUES never appear in chat. Don't echo them, don't include them in "
    "any form spec, don't paraphrase them.\n"
    # "- The fetch tool is the only read path.\n"
    # "- Use `update_form` for any retry / error / status change after the initial "
    # "form is up. Reserve `request_credentials` for first emission and for fully "
    # "switching to a different connector.\n"
    "- The chat-emitted form and the picker-emitted form must FEEL identical to "
    "the user — same methods, same OAuth flow, same how-to docs. The registry "
    "lookup is what guarantees that parity; use it."
)


def build_cowork_request_credentials_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="request_credentials",
        description=(
            "Request credentials / configuration from the user via an interactive "
            "form rendered in the side panel. Returns a markdown block you must "
            "include verbatim in your next assistant message so the form appears."
        ),
        input_schema=_REQUEST_CREDENTIALS_SCHEMA,
        handler=_cowork_request_credentials,
        prompt=_REQUEST_CREDENTIALS_PROMPT,
    )


# Fetch Submission Tool
# Pulls staged credential values after the user submits. 
# Anton uses these to test / save the connection, then either
# presents a new form (with errors) or moves on.

# async def _cowork_fetch_submission(session: Any, tc_input: dict) -> str:
#     """Tool handler for `fetch_submission` — return staged values for
#     a previously-submitted form, by submission_id.
#     """
#     sid = tc_input.get("submission_id") or tc_input.get("id")
#     if not sid:
#         return "fetch_submission: missing submission_id"
#     try:
#         from . import datavault_submissions
#     except Exception as exc:
#         logger.exception("Cowork fetch_submission could not import store")
#         return f"fetch_submission: store unavailable ({exc})"
#     entry = datavault_submissions.get_submission(sid)
#     if not entry:
#         return (
#             f"fetch_submission: submission `{sid}` not found or expired. "
#             f"Submissions TTL after 24h. Ask the user to resubmit the form."
#         )
#     return json.dumps({
#         "submission_id": entry.get("submission_id"),
#         "form_id": entry.get("form_id"),
#         "values": entry.get("values", {}),
#         "skipped": entry.get("skipped", []),
#     })


# _FETCH_SUBMISSION_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "submission_id": {
#             "type": "string",
#             "description": "The id from the user's continuation message after they submitted the form.",
#         },
#     },
#     "required": ["submission_id"],
# }


# def build_cowork_fetch_submission_tool():
#     from anton.core.tools.tool_defs import ToolDef
#     return ToolDef(
#         name="fetch_submission",
#         description=(
#             "Retrieve the staged values from a `data-vault-form` submission. "
#             "Returns JSON with `values`, `skipped`, and `form_id`. Field values "
#             "never appear in chat history — this tool is the only way to read them."
#         ),
#         input_schema=_FETCH_SUBMISSION_SCHEMA,
#         handler=_cowork_fetch_submission,
#         prompt=None,
#     )


# # ── update_form ───────────────────────────────────────────────────────
# # Patch dialect for in-place form updates. Anton uses this on retry
# # loops and any time the form needs a field-level error / warning /
# # label change without re-emitting the whole spec. The patch never
# # carries `value` fields — the user's existing inputs are preserved
# # client-side by the form panel.

# async def _cowork_update_form(session: Any, tc_input: dict) -> str:
#     """Tool handler for `update_form` — emit a patch dialect block
#     that the renderer merges into the active form for this
#     conversation.
#     """
#     patch = tc_input.get("patch") if isinstance(tc_input.get("patch"), dict) else tc_input
#     if not isinstance(patch, dict):
#         return "update_form: invalid patch — must be a JSON object with `form_id`"
#     if not patch.get("form_id"):
#         return "update_form: `form_id` is required (must match the form you previously emitted via request_credentials)"

#     # Strip any `value` keys that snuck in — patches must NEVER carry
#     # credential material. We log + drop rather than fail the call so
#     # an over-eager LLM doesn't get stuck retrying.
#     fields_obj = patch.get("fields")
#     if isinstance(fields_obj, dict):
#         sanitized_fields = {}
#         for name, fp in fields_obj.items():
#             if not isinstance(fp, dict):
#                 continue
#             cleaned = {k: v for k, v in fp.items() if k != "value"}
#             if "value" in fp:
#                 logger.info(
#                     "update_form: stripped `value` from field %r — patches must not carry credentials",
#                     name,
#                 )
#             sanitized_fields[name] = cleaned
#         patch = {**patch, "fields": sanitized_fields}

#     block = "```data-vault-form-patch\n" + json.dumps(patch, indent=2) + "\n```"
#     return (
#         "Patch ready. Include the following markdown block VERBATIM in your "
#         "next message (with blank lines around the fence). The form panel "
#         "will merge it into the existing form — the user's typed values are "
#         "preserved.\n\n"
#         f"{block}"
#     )


# _UPDATE_FORM_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "form_id": {
#             "type": "string",
#             "description": "Must match the `form_id` of the form currently shown in the side panel.",
#         },
#         "title": {"type": "string", "description": "Optional. Replace the form title."},
#         "subtitle": {"type": "string", "description": "Optional. Replace the subtitle."},
#         "form_warning": {"type": "string", "description": "Optional. Set the amber form-level banner. Pass null to clear."},
#         "form_error": {"type": "string", "description": "Optional. Set the red form-level banner. Pass null to clear."},
#         "fields": {
#             "type": "object",
#             "description": (
#                 "Map of field NAME → partial field spec. Only the keys you "
#                 "include override the existing field's properties. Pass `null` "
#                 "for a key to clear that property (e.g. `error: null` to dismiss "
#                 "an error). Add a brand-new field by including its full spec "
#                 "under a name not already in the form. NEVER include `value` — "
#                 "the user's input is preserved client-side."
#             ),
#             "additionalProperties": {
#                 "type": "object",
#                 "properties": {
#                     "label": {"type": "string"},
#                     "error": {"type": ["string", "null"], "description": "Per-field red text. Set on retry."},
#                     "warning": {"type": ["string", "null"], "description": "Per-field amber text."},
#                     "help": {"type": ["string", "null"]},
#                     "placeholder": {"type": ["string", "null"]},
#                     "required": {"type": "boolean"},
#                     "skipable": {"type": "boolean"},
#                 },
#             },
#         },
#         "actions": {
#             "type": "array",
#             "description": "Optional. Replace the actions list.",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "id": {"type": "string"},
#                     "label": {"type": "string"},
#                     "kind": {"type": "string", "enum": ["primary", "skip", "cancel"]},
#                     "field": {"type": "string"},
#                 },
#                 "required": ["id", "label"],
#             },
#         },
#     },
#     "required": ["form_id"],
# }


# _UPDATE_FORM_PROMPT = (
#     "Use `update_form` for ANY change to a form already shown by "
#     "`request_credentials`. Common cases:\n"
#     "  • Connection failed → set `fields: { <name>: { error: 'message' } }` "
#     "and `subtitle` to explain.\n"
#     "  • Need an extra field → add it under a new key in `fields`.\n"
#     "  • Need to clear a previous error → `fields: { <name>: { error: null } }`.\n"
#     "Never include `value` — the user's typed input is preserved. The patch "
#     "is far cheaper than a full re-emit and avoids leaking credentials into "
#     "chat history."
# )


# def build_cowork_update_form_tool():
#     from anton.core.tools.tool_defs import ToolDef
#     return ToolDef(
#         name="update_form",
#         description=(
#             "Patch the active data-vault-form for this conversation in place. "
#             "Use this for retry loops, error messages, and any field-level "
#             "tweak — the user's typed values are preserved client-side."
#         ),
#         input_schema=_UPDATE_FORM_SCHEMA,
#         handler=_cowork_update_form,
#         prompt=_UPDATE_FORM_PROMPT,
#     )
