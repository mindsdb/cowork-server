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
import re
from pathlib import Path
from typing import Any
import uuid

logger = logging.getLogger(__name__)


def _published_state(raw_path: str) -> dict:
    """Delegate to the publish service so the tool and GUI agree on where
    `.published.json` lives. Imported lazily to avoid a startup import cycle."""
    from cowork.services.publish import published_state
    return published_state(raw_path)


def _publish_artifact(raw_path: str) -> dict:
    from cowork.services.publish import publish_artifact
    return publish_artifact(raw_path)


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
    "- NEVER attempt to publish via scratchpad code. There is no `_get_env`, `publish_or_preview`,\n"
    "  or publish helper available in the scratchpad namespace — calling them will raise NameError.\n"
    "  The tool handles credential resolution internally; you do not need to fetch or pass an API key.\n"
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

    # action == 'publish' — delegate to the service (single source of truth for
    # target resolution, fullstack bundling, vault secrets, access, history,
    # and report_id reuse). The tool always publishes public.
    try:
        result = _publish_artifact(abs_path)
    except ValueError as exc:
        # The service raises ValueError for several distinct reasons. Only the
        # missing-API-key case warrants the STOP/"go configure a key" directive;
        # others (unsupported file type, unresolvable path) are real publish
        # failures — surfacing them as STOP would send the user chasing a key
        # they already have and block a legitimate retry.
        if "api key" in str(exc).lower():
            logger.info("Cowork publish blocked: %s", exc)
            return (
                "STOP: No Minds API key configured. Tell the user to set their "
                "Minds API key in Settings (or in their .env) before publishing. "
                "Do NOT call this tool again until they confirm the key is set."
            )
        logger.info("Cowork publish rejected: %s", exc)
        return f"PUBLISH FAILED: {exc}"
    except Exception as exc:
        logger.exception("Cowork publish tool failed")
        return f"PUBLISH FAILED: {exc}"

    view_url = result.get("url", "") if isinstance(result, dict) else ""
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


def _scrub_secret_values(spec: dict) -> dict:
    """Strip pre-filled `value`s from secret fields before the spec is
    serialized into the chat-visible ```data-vault-form``` block.

    `value` legitimately preserves non-secret inputs across re-renders, but a
    password value in the spec would persist a plaintext credential in chat
    history — guarantee in code what the prompt asks for in prose.
    """

    def scrub_fields(fields):
        out = []
        for f in fields:
            if isinstance(f, dict) and f.get("type") == "password" and "value" in f:
                f = {k: v for k, v in f.items() if k != "value"}
            out.append(f)
        return out

    out = dict(spec)
    if isinstance(out.get("fields"), list):
        out["fields"] = scrub_fields(out["fields"])
    if isinstance(out.get("methods"), list):
        methods = []
        for m in out["methods"]:
            if isinstance(m, dict) and isinstance(m.get("fields"), list):
                m = {**m, "fields": scrub_fields(m["fields"])}
            methods.append(m)
        out["methods"] = methods
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

    spec = _scrub_secret_values(_ensure_form_id(spec))
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
        "After the user submits, the server tests the connection and saves "
        "the credentials automatically — your job is done. Do not call any "
        "tool to retrieve the submitted values, and do not re-emit the form "
        "unless the user asks for a new one.\n\n"
        f"{block}"
    )


_REQUEST_CREDENTIALS_SCHEMA = {
    "type": "object",
    "properties": {
        "form_id": {
            "type": "string",
            "description": "Stable form identifier. Reuse the same id when re-asking the same form so the user's typed values persist.",
        },
        "engine": {
            "type": "string",
            "description": "REQUIRED. Connector slug (e.g. 'postgres', 'gmail'). Any value is accepted — unknown engines are saved as 'custom' connections.",
        },
        "_connector_id": {
            "type": "string",
            "description": "Slug of the canonical spec returned by `lookup_connector`. ALWAYS copy it over when the spec came from lookup (routes submissions to the registry-aware save endpoint); omit for handcrafted specs.",
        },
        "title": {"type": "string", "description": "Short headline (e.g. 'Connect to Postgres')."},
        "subtitle": {"type": "string"},
        "how_to": {
            "type": "string",
            "description": "Markdown setup walkthrough, shown behind a 'How to?' link (single-method forms; multi-method forms use per-method `how_to`).",
        },
        "help_url": {"type": "string", "description": "External help URL fallback when no `how_to` is given."},
        "logo": {
            "type": "string",
            "description": "Icon name: database|globe|cube|doc|code|image|folder|brain|sparkle|wifi|key|link|mindsdb. No URLs. Default 'database'.",
        },
        "logo_color": {"type": "string", "description": "CSS color for the icon."},
        "form_warning": {"type": "string", "description": "Amber banner above the fields."},
        "form_error": {"type": "string", "description": "Red banner above the fields (previous attempt failed)."},
        "fields": {
            "type": "array",
            "description": "Input fields, rendered in order (single-method form). Prefer `methods` when the service has several auth options.",
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
                    "value": {"description": "Pre-fill on re-render. NEVER for secrets — password values are scrubbed server-side."},
                    "options": {
                        "type": "array",
                        "description": "For type=select: [{value, label}].",
                        "items": {"type": "object"},
                    },
                    "error": {"type": "string", "description": "Red per-field text — set on a retry to flag the failing field."},
                    "warning": {"type": "string"},
                    "help": {"type": "string"},
                    "skipable": {"type": "boolean", "description": "Default true; false only for absolute requirements."},
                },
                "required": ["name", "label", "type"],
            },
        },
        "methods": {
            "type": "array",
            "description": (
                "Use INSTEAD of `fields` when the engine supports multiple auth methods "
                "(e.g. Gmail: OAuth / App Password / Service Account). The user picks a "
                "method, then fills only that method's fields. Mark the easiest one "
                "`recommended: true`."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable method id (e.g. 'oauth', 'app-password'); drives the auth probe and is saved as `_method`."},
                    "label": {"type": "string"},
                    "description": {"type": "string", "description": "1-2 sentences for the picker card."},
                    "recommended": {"type": "boolean"},
                    "how_to": {"type": "string", "description": "Markdown walkthrough for THIS method."},
                    "help_url": {"type": "string"},
                    "submit_action": {
                        "type": "string",
                        "enum": ["oauth_launch"],
                        "description": "Set for OAuth methods — Submit runs a local PKCE browser flow and saves the refresh_token to the vault. Pair with `oauth`.",
                    },
                    "oauth": {
                        "type": "object",
                        "description": "Required when submit_action=oauth_launch.",
                        "properties": {
                            "auth_url": {"type": "string"},
                            "token_url": {"type": "string"},
                            "scopes": {"type": "array", "items": {"type": "string"}},
                            "extra_auth_params": {
                                "type": "object",
                                "description": "Extra auth-URL query params (e.g. {access_type: 'offline', prompt: 'consent'} for Google).",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["auth_url", "token_url", "scopes"],
                    },
                    "fields": {
                        "type": "array",
                        "description": "Same shape as top-level `fields` items. Often just client_id/client_secret for OAuth; may be empty when the connector ships a hosted client.",
                        "items": {"type": "object"},
                    },
                    "actions": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["id", "label", "fields"],
            },
        },
        "selected_method": {
            "type": "string",
            "description": "Pre-pick a method id from `methods[]` (skips the picker). Set only when the user clearly signalled a preference.",
        },
        "actions": {
            "type": "array",
            "description": "Optional custom buttons; defaults to Submit + Cancel.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["primary", "skip", "cancel"]},
                    "field": {"type": "string", "description": "Only for kind='skip'."},
                },
                "required": ["id", "label"],
            },
        },
    },
    "required": ["engine", "title"],
}


_REQUEST_CREDENTIALS_PROMPT = (
    "DATA VAULT WORKFLOW — when the user asks to connect to a service or database:\n"
    "1. Call `lookup_connector` FIRST with the user's wording. When it returns a "
    "`form` spec, pass it to `request_credentials` VERBATIM (tweak only "
    "`selected_method` or `subtitle`), and copy `_connector_id` onto the spec. "
    "Registry specs already carry `methods[]`, OAuth blocks, `how_to`, and "
    "`help_url` — never strip or paraphrase them.\n"
    "2. Handcraft a spec ONLY when the registry has no match, using your own "
    "knowledge of the service's auth shape (host/port/user/password, API key, or "
    "OAuth). For engines with several auth options emit `methods[]` instead of "
    "`fields[]` and mark the simplest `recommended: true`; pre-set "
    "`selected_method` if the user already signalled a preference.\n"
    "3. Include the markdown block the tool returns VERBATIM in your next message "
    "(blank lines around the fence) so the form renders in the side panel.\n"
    "4. Your job is done — the server tests the connection and saves credentials "
    "on submit. Do NOT call `request_credentials` again unless the user asks to "
    "connect a different service or explicitly requests a new form.\n"
    "STRICT: field VALUES never appear in chat — don't echo them, don't include "
    "them in any form spec, don't paraphrase them. The chat-emitted form must "
    "FEEL identical to the in-app Connector Picker; the registry lookup is what "
    "guarantees that parity."
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


# Label Connection Tool
# Lets the agent persist a human role label ("Support", "Personal") onto a saved
# connection — the learn-and-persist half of multi-account identity. The label
# shows up next to the connection (here and in Connected Data Sources) so the
# right account can be selected later; it does not change the connection's
# identity/slug or its secrets.

_LABEL_CONNECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "engine": {"type": "string", "description": "The connection's engine, e.g. 'gmail'."},
        "name": {
            "type": "string",
            "description": "The connection's name/slug exactly as shown in Connected Data Sources.",
        },
        "label": {
            "type": "string",
            "description": "Human role label to assign, e.g. 'Support' or 'Personal'.",
        },
    },
    "required": ["engine", "name", "label"],
}

_LABEL_CONNECTION_PROMPT = (
    "Use `label_connection` to give a saved connection a human role label once "
    "the user tells you which is which — e.g. when two Gmail accounts are "
    "connected and the user says `regtr@mail.com` is their support address, call "
    "`label_connection(engine='gmail', name='<slug>', label='Support')`. The label "
    "is shown beside the connection in Connected Data Sources so you can pick the "
    "right account later. Never guess a label — ask the user first, then persist it."
)


async def _cowork_label_connection(session: Any, tc_input: dict) -> str:
    """Tool handler for `label_connection` — persist a human label on a saved
    connection (learn-and-persist)."""
    engine = str(tc_input.get("engine", "")).strip()
    name = str(tc_input.get("name", "")).strip()
    label = str(tc_input.get("label", "")).strip()
    if not engine or not name or not label:
        return "label_connection: `engine`, `name`, and `label` are all required."
    try:
        from cowork.services.connectors.persist import set_connection_label

        ok = set_connection_label(engine, name, label)
    except Exception as exc:
        logger.exception("Cowork label_connection failed")
        return f"label_connection: could not set label ({exc})"
    if not ok:
        return f"label_connection: no connection `{engine}/{name}` found."
    return f"Labeled `{engine}/{name}` as “{label}”."


def build_cowork_label_connection_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="label_connection",
        description=(
            "Assign a human role label (e.g. 'Support', 'Personal') to a saved "
            "connection so it can be identified and selected later. Use after the "
            "user clarifies which account is which — never guess."
        ),
        input_schema=_LABEL_CONNECTION_SCHEMA,
        handler=_cowork_label_connection,
        prompt=_LABEL_CONNECTION_PROMPT,
    )


# Browser Control Tool (Milestone 1 — read-only)
# A single `browser_control` tool the agent uses to drive an already-approved
# browser tab through four READ-ONLY primitives: inspect / follow_link /
# scroll / wait. There is deliberately NO click / type / submit / download /
# upload anywhere in M1. The handler is total (never raises): every failure
# path returns a JSON envelope whose `status` is one of the five canonical
# `BrowserErrorKind`s, and it maps the richer WS4-internal `result_code`s down
# to that vocabulary. An action is NEVER reported as success unless its
# transient `observed` blob is populated (the "no false success" rule).

_BROWSER_CONTROL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["inspect", "follow_link", "scroll", "wait"],
            "description": (
                "The read-only browser primitive to run. `inspect` reads the "
                "current tab; `follow_link` navigates to a link that appeared "
                "in the last `inspect` (same site only); `scroll` reveals more "
                "of the page; `wait` lets a slow page settle. No clicking, "
                "typing, form submission, or downloads exist in this tool."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                "Why a connector/integration cannot do this and the browser is "
                "needed instead. Required on every call — this is the "
                "connector-first justification that gets recorded."
            ),
        },
        "progress_message": {
            "type": "string",
            "description": (
                "One short human-readable line shown live to the user while "
                "this action runs, e.g. 'Reading the account list'."
            ),
        },
        "href": {
            "type": "string",
            "description": (
                "follow_link only — the link to open. Must be a link that "
                "appeared in the last `inspect` result, on the same site as "
                "the approved tab."
            ),
        },
        "direction": {
            "type": "string",
            "enum": ["up", "down"],
            "description": "scroll only — which way to scroll (default 'down').",
        },
    },
    "required": ["action", "reason", "progress_message"],
}

_BROWSER_CONTROL_PROMPT = (
    "`browser_control` drives an already-approved browser tab, READ-ONLY. "
    "Prefer a connector first: always try `lookup_connector` and a native "
    "integration before reaching for the browser — only use `browser_control` "
    "when no connector can satisfy the task, and always pass `reason` "
    "explaining why the connector could not do it. Available actions: "
    "`inspect` (read the current tab), `follow_link` (open a link that "
    "appeared in the last inspect, same site only — pass `href`), `scroll` "
    "(reveal more, pass `direction`), `wait` (let a slow page settle). There "
    "is NO clicking, typing, form submission, or downloading. Every call needs "
    "a one-line `progress_message` for the user. NEVER claim you saw or did "
    "something unless the tool returns `status: \"ok\"` with a populated "
    "`observed` — an unobserved action is not a success. On any error status "
    "(permission_denied / bridge_disconnected / tab_closed / navigation_failed "
    "/ unsupported_action) tell the user plainly and do not fabricate results."
)


def _get_bridge_client():
    """Thin indirection over WS4's `send_browser_command`.

    Isolated so tests can monkeypatch a fake bridge without importing the
    real broker, and so WS3 could be developed before WS4 landed. Returns the
    module-level async `send_browser_command(conversation_id, action, ...)`
    callable.
    """
    from cowork.services.browser.client import send_browser_command

    return send_browser_command


async def _cowork_browser_control(session: Any, tc_input: dict) -> str:
    """Tool handler for `browser_control` — total and non-raising.

    Validates inputs (a missing required field returns an actionable JSON
    error WITHOUT touching the bridge), dispatches through
    `send_browser_command`, maps the WS4-internal `result_code` to the five
    canonical `BrowserErrorKind`s, enforces the observed-result guard (an
    `ok` with no `observed` is downgraded — never surfaced as success), and
    returns a `json.dumps` envelope. Hard failures are prefixed `ERROR:` so
    anton's circuit-breaker sees them.
    """
    import time as _time

    from cowork.schemas.browser import (
        LLM_ACTION_TO_TYPE,
        BrowserActionType,
        BrowserErrorKind,
        ResultCode,
        result_code_to_error_kind,
    )
    from .browser_telemetry import (
        build_browser_span,
        emit_browser_span,
        get_browser_ids,
    )

    _t0 = _time.monotonic()

    def _err(kind: str, message: str, *, hard: bool = False) -> str:
        body = {"status": kind, "error": message}
        payload = json.dumps(body)
        # Hard failures get the ERROR: prefix so the circuit-breaker counts
        # them; the JSON follows so the LLM can still parse the status.
        return f"ERROR: {payload}" if hard else payload

    action = tc_input.get("action")
    reason = tc_input.get("reason")
    progress_message = tc_input.get("progress_message")

    # ── input validation (bridge NOT touched on any failure here) ────
    valid_actions = set(LLM_ACTION_TO_TYPE)
    if not action or not isinstance(action, str):
        return _err(
            BrowserErrorKind.unsupported_action.value,
            "`action` is required and must be one of "
            "inspect / follow_link / scroll / wait.",
        )
    if action not in valid_actions:
        return _err(
            BrowserErrorKind.unsupported_action.value,
            f"unsupported action '{action}'. "
            "M1 is read-only: only inspect / follow_link / scroll / wait "
            "are supported (no click / type / submit / download / upload).",
        )
    if not reason or not isinstance(reason, str) or not reason.strip():
        return _err(
            BrowserErrorKind.unsupported_action.value,
            "`reason` is required — explain why a connector cannot do this "
            "and the browser is needed instead.",
        )
    if (
        not progress_message
        or not isinstance(progress_message, str)
        or not progress_message.strip()
    ):
        return _err(
            BrowserErrorKind.unsupported_action.value,
            "`progress_message` is required — one short human-readable line "
            "describing what this action is doing.",
        )

    href = tc_input.get("href")
    direction = tc_input.get("direction")
    if action == "follow_link" and (not href or not isinstance(href, str)):
        return _err(
            BrowserErrorKind.navigation_failed.value,
            "`follow_link` requires `href` — a link from the last inspect "
            "result on the same site.",
        )

    # Resolve the conversation defensively — the anton session stamps the
    # cowork conversation id onto `_session_id`.
    conversation_id = getattr(session, "_session_id", None)
    if not conversation_id:
        return _err(
            BrowserErrorKind.bridge_disconnected.value,
            "no active conversation — the browser bridge is unavailable.",
            hard=True,
        )

    # ── dispatch to WS4's bridge ─────────────────────────────────────
    try:
        send_browser_command = _get_bridge_client()
        verdict = await send_browser_command(
            conversation_id,
            action,
            href=href if action == "follow_link" else None,
            direction=direction if action == "scroll" else None,
        )
    except Exception as exc:  # never let the bridge raise into the loop
        logger.exception("Cowork browser_control dispatch failed")
        return _err(
            BrowserErrorKind.bridge_disconnected.value,
            f"browser bridge error ({exc}).",
            hard=True,
        )

    # ── map WS4 result_code → canonical BrowserErrorKind ─────────────
    result_code = getattr(verdict, "result_code", ResultCode.error)
    action_type = getattr(verdict, "action_type", None) or LLM_ACTION_TO_TYPE.get(
        action, BrowserActionType.inspect
    )
    kind = result_code_to_error_kind(result_code, action_type)

    # ── observed-result guard (no false success) ─────────────────────
    # WS4's BridgeClient already downgrades an unobserved `ok` before it
    # persists/returns, but the guard is repeated here as defence-in-depth
    # (e.g. a bridge seam that returns `ok` with no observed) so the tool
    # NEVER surfaces an unobserved action as success.
    observed = getattr(verdict, "observed", None)
    effective_kind = kind
    if kind == BrowserErrorKind.ok and not observed:
        # An `ok` with nothing observed is not a real success. Downgrade to
        # navigation_failed for a navigate, else bridge_disconnected.
        effective_kind = (
            BrowserErrorKind.navigation_failed
            if action_type == BrowserActionType.navigate
            else BrowserErrorKind.bridge_disconnected
        )

    # ── content-free Langfuse tool span (WS5-T3) ────────────────────
    # Emitted for every DISPATCHED action (validation failures above never
    # touch the bridge and so never produce a span). The span carries only
    # the action class, a host-only domain, timing, the EFFECTIVE result
    # code (post no-false-success downgrade), and the shared IDs — no page
    # content. Emitting the effective code keeps the trace from recording a
    # false `ok` for an unobserved action.
    try:
        _ids = get_browser_ids()
        emit_browser_span(
            build_browser_span(
                command_type=action_type.value,
                result_code=effective_kind.value,
                duration_ms=int((_time.monotonic() - _t0) * 1000),
                domain=getattr(verdict, "domain", None),
                installation_id=_ids.get("installation_id"),
                session_id=_ids.get("session_id") or str(conversation_id),
                task_id=_ids.get("task_id"),
                action_id=getattr(verdict, "action_id", None),
            )
        )
    except Exception:
        logger.debug("browser_control span emit failed", exc_info=True)

    if effective_kind != BrowserErrorKind.ok:
        detail = getattr(verdict, "detail", None)
        # `stopped` / `taken_over` are CONTROL terminal states, not error
        # kinds. Surface them distinctly (alongside the canonical error
        # kind) so the UI can render a stopped / taken-over terminal state
        # rather than a generic permission denial.
        control_state = getattr(verdict, "control_state", None)
        if kind == BrowserErrorKind.ok and not observed:
            detail = (
                "the action completed but returned no observable result; "
                "treating it as a failure rather than claiming success."
            )
        body = {"status": effective_kind.value, "error": detail or f"browser action {effective_kind.value}."}
        if control_state is not None:
            cs = control_state.value if hasattr(control_state, "value") else str(control_state)
            body["control_state"] = cs
        return json.dumps(body)

    # Success — surface the TRANSIENT observed blob + citations. This is
    # never persisted here; WS4 owns the content-free digest.
    envelope: dict[str, Any] = {
        "status": BrowserErrorKind.ok.value,
        "action": action,
        "observed": observed,
        "citations": getattr(verdict, "citations", None) or [],
    }
    domain = getattr(verdict, "domain", None)
    if domain:
        envelope["domain"] = domain
    return json.dumps(envelope)


def build_cowork_browser_tool():
    from anton.core.tools.tool_defs import ToolDef

    return ToolDef(
        name="browser_control",
        description=(
            "Drive an already-approved browser tab, READ-ONLY, through four "
            "primitives: inspect (read the tab), follow_link (open a link from "
            "the last inspect, same site), scroll, and wait. No clicking, "
            "typing, form submission, or downloads. Use only when no connector "
            "can satisfy the task; always pass `reason`. Returns a JSON "
            "envelope with `status` (ok or an error kind) and, on ok, "
            "`observed` + `citations`."
        ),
        input_schema=_BROWSER_CONTROL_SCHEMA,
        handler=_cowork_browser_control,
        prompt=_BROWSER_CONTROL_PROMPT,
    )
