"""Publish service — publish HTML artifacts to MindsHub.

Ported from cowork/server/routes/utilities.py (publish section).
Uses a local JSON state file for publish history tracking.
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import logging
import os
import tempfile
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import urlparse

from fastapi import HTTPException
from pydantic import SecretStr

from cowork.code_sandbox import refuse_planted_links
from cowork.common.logger import log_context
from cowork.common.paths import cowork_home, pod_local_only
# Imported for its side effect on the module namespace as well as its use:
# tests monkeypatch `publish.get_app_settings`, so removing it because a
# linter sees no local call breaks them. noqa keeps that from recurring.
from cowork.common.settings.app_settings import get_app_settings  # noqa: F401

from cowork.services.connectors.persist import vault_for_scope
from cowork.services.providers import (
    is_mindshub_publish_url,
    normalize_publish_url,
    publish_url_for_endpoint,
)
from cowork.common.settings.user_settings import Provider, get_user_settings, provider_api_key
from anton.minds_client import describe_minds_connection_error
from anton.publish_access import access_from_owner_side
from anton.publish_access import normalize_emails as _normalize_emails
from anton.publish_access import resolve_access as _resolve_access
from anton.publish_access import resolve_publish_target as _anton_resolve_publish_target
from cowork.services.artifacts import (
    _content_mtime,
    _load_published_map,
    PUBLISHABLE_STATIC_SUFFIXES,
    _scan_artifact_dirs,
    html_artifacts,
    is_non_content_path,
    resolve_artifact_path,
)

# Type-only, matching connectors/persist.py: cowork.db.scoped drags in the
# session factory and the FastAPI dependency graph, and this module is imported
# from the harness turn path where that is dead weight.
if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope
    from cowork.services.artifact_access import ArtifactAudience, ArtifactRule

logger = logging.getLogger(__name__)


class PublisherUnavailable(RuntimeError):
    """A local publish dependency (anton.publisher, markdown) failed to import.

    Distinct from RuntimeError so the endpoint layer can map it to 503
    without parsing message text — see the "unavailable" substring sentinel
    this replaced, which broke once upstream error text could itself contain
    that word (e.g. an HTTP 503 reason phrase or a timeout advice string).
    """


class _CodedRefusal(HTTPException):
    """An HTTPException whose detail is `{"code", "message"}`: clients match on the code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code, detail={"code": code, "message": message})


# A local ownership check or the upload service refuses another account's page; the
# renderer matches on the code.
OWNED_BY_OTHER_USER_CODE = "artifact_owned_by_other_user"
OWNED_BY_OTHER_USER_MESSAGE = "This artifact was published by another owner."


class PublishOwnedByOtherUser(_CodedRefusal):
    """A local check or the upload service refused another account's report.

    An HTTPException, like ProductPermissionDenied, so any route that lets it
    through answers 409 with the code. Nobody retries it with a fresh
    report_id, which would publish a copy under a new URL.
    """

    def __init__(self) -> None:
        super().__init__(409, OWNED_BY_OTHER_USER_CODE, OWNED_BY_OTHER_USER_MESSAGE)


# The upload service's 413: the bundle is over its request size limit.
ARTIFACT_TOO_LARGE_CODE = "artifact_too_large"
ARTIFACT_TOO_LARGE_MESSAGE = (
    "This artifact is too large to publish. "
    "Remove or shrink large files in it, such as PDFs or images."
)
# Any 503 from the upload service other than an auth outage.
UPLOAD_UNAVAILABLE_CODE = "publish_unavailable"
UPLOAD_UNAVAILABLE_MESSAGE = "The publishing service is temporarily unavailable. Try again in a moment."


class PublishTooLarge(_CodedRefusal):
    """The upload service refused the bundle as too large.

    The same files are refused on every retry, so callers report it instead
    of retrying.
    """

    def __init__(self) -> None:
        super().__init__(413, ARTIFACT_TOO_LARGE_CODE, ARTIFACT_TOO_LARGE_MESSAGE)


class UploadServiceUnavailable(_CodedRefusal):
    """The upload service answered 503 without marking it as an auth outage."""

    def __init__(self) -> None:
        super().__init__(503, UPLOAD_UNAVAILABLE_CODE, UPLOAD_UNAVAILABLE_MESSAGE)


class PublishRejected(RuntimeError):
    """A publish failure whose text is safe to show the user.

    A plain RuntimeError from `publish_artifact` can carry local detail such as
    a shared-storage path, so org routes show this type's text and hide the
    other's.
    """


class AccessModeChanged(PublishRejected):
    """A re-publish in hosted org mode that isn't a person's save (live
    sync, autopublish) found a .published.json entry that isn't restricted
    while the auth service still holds a restricted rule for the artifact,
    under its identity or under a key the entry recorded. Only a restricted
    publish leaves a rule there, and a public or password one deletes it, so
    the entry's mode changed after the owner's last save: the pod can write
    that file, and metadata.json too. A re-publish of a page the entry names
    no key for is refused the same way, since no rule can be read for it.
    Refused before anything is uploaded, so the artifact never goes public
    or password-protected over the rule (_access_mode_refusal).

    No route shows its text. Live sync and autopublish log the refusal and
    report a failed publish, and the owner's own save (``owner_save``)
    never raises it; that save is what ends it."""

    def __init__(self) -> None:
        super().__init__("The artifact's sharing changed outside the Share dialog")


class UploadError(NamedTuple):
    """An upload-service answer `classify_upload_error` left to its caller."""

    status: int
    code: str | None
    error: str | None


def _read_http_error_body(exc: Exception) -> tuple[str, str | None, str | None]:
    """(body, code, error) of an HTTPError, from a single read.

    A second `read()` returns nothing, so every consumer goes through here once.
    Any failure to read gives an empty body rather than replacing `exc`.
    """
    try:
        body = exc.read(65_536).decode("utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        return "", None, None
    try:
        payload = json.loads(body)
    except ValueError:
        return body, None, None
    if not isinstance(payload, dict):
        return body, None, None
    code = payload.get("code")
    error = payload.get("error")
    if not isinstance(error, str) and getattr(exc, "code", None) == 429:
        # API Gateway's throttling text. Its other `message` answers, such as
        # "Missing Authentication Token" for a wrong URL, would mislead.
        error = payload.get("message")
    return (
        body,
        code if isinstance(code, str) else None,
        error if isinstance(error, str) else None,
    )


def classify_upload_error(exc: Exception) -> UploadError | None:
    """Raise the typed error for an upload-service refusal, or return it for wording.

    413 is too large. A 503 is an auth outage only when the service says so
    (`auth_unavailable`); any other 503 is the upload service itself. A 403 is
    a denial only with `permission_denied`. Every other HTTP answer comes back
    as an UploadError; a failure that got no HTTP answer returns None.
    """
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    if isinstance(
        exc,
        (ProductPermissionDenied, ProductPermissionUnavailable, PublishTooLarge, UploadServiceUnavailable),
    ):
        raise exc
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    status = exc.code
    body, code, error = _read_http_error_body(exc)
    typed: HTTPException | None = None
    if status == 413:
        typed = PublishTooLarge()
    elif status == 503:
        typed = ProductPermissionUnavailable() if code == "auth_unavailable" else UploadServiceUnavailable()
    elif status == 403 and code == "permission_denied":
        typed = ProductPermissionDenied()
    if typed is None:
        return UploadError(status, code, error)
    # A typed refusal leaves publish_artifact before its own failure log.
    logger.warning("Publish service answered HTTP %s code=%s body=%r", status, code, body[:200])
    raise typed from exc


def _clip(text: str) -> str:
    """Server text short enough to show, keeping the end where pip names the cause."""
    head, tail, gap = 100, 400, " … "
    if len(text) <= head + len(gap) + tail:
        return text
    return f"{text[:head]}{gap}{text[-tail:]}"


#: The files the publish store holds: its state, and the temporary file a
#: save writes and moves over it.
_STATE_FILES = ("state.json", "state.json.tmp")


def _cowork_state_dir(*, scope: TenantScope | None) -> Path:
    base = os.environ.get("ANTON_COWORK_STATE_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        # Consolidated under the cowork data root (was ~/.anton/cowork); the
        # desktop app migrates the existing state.json on first run. Org mode
        # relocates this off shared EFS storage via pod_local_only (see its
        # docstring): state.json holds publish_history below, which carries
        # no org_id segment, so left on cowork_home() every organization
        # would read every other organization's publish history.
        path = pod_local_only(cowork_home(), "publish")
    if get_app_settings().tenancy_mode != "org":
        # Publish history is part of the projects resource: a person whose
        # projects are personal reads and writes `<shared>/<partition>/publish`,
        # and LOCAL_SCOPE keeps the path above. scoped_storage_root refuses
        # `scope=None` on a deployment that trusts identity headers.
        from cowork.db.scoped import scoped_storage_root

        path = scoped_storage_root(path, scope, store="publish")
    # Under the code sandbox, a link or hard-linked file that agent code left
    # on the way to state.json, or at it, fails the read or write instead of
    # being followed. Only the store's own files are checked: the operator
    # view's store is cowork-home itself.
    refuse_planted_links(path, only=_STATE_FILES)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(*, scope: TenantScope | None) -> Path:
    return _cowork_state_dir(scope=scope) / "state.json"


def _load_state(*, scope: TenantScope | None) -> dict[str, Any]:
    path = _state_path(scope=scope)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, Any], *, scope: TenantScope | None) -> None:
    path = _state_path(scope=scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _secret_str(val: SecretStr | str | None) -> str:
    """Unwrap a SecretStr (or plain string) to a plain string, defaulting to ''."""
    if val is None:
        return ""
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    return str(val)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lambda_artifact_key(result: dict) -> str:
    """The upload lambda's own `artifact_key` echo, or "" when it sent none.

    Kept apart from the `artifact_key` entry field, which falls back to cowork's
    canonical key: that fallback is indistinguishable from an echo, and the two
    imply DIFFERENT cloud comment scopes (stable key vs. composite), so mixing
    them routes comments to a scope with no access rule behind it.
    """
    return str(result.get("artifact_key") or "").strip()


def _write_published_map(published_json: Path, published_map: dict[str, Any]) -> None:
    """Write `.published.json` atomically.

    Two turns in one project (different conversations) can reconcile the same
    slug, and a publish thread abandoned by a timeout can still land here after
    the turn closed. A partially written record would lose `report_id` - and with
    it the ability to reuse or revoke the published URL - so the write goes to a
    sibling temp file and is swapped in with os.replace.
    """
    tmp = published_json.with_name(f"{published_json.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(published_map, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, published_json)
    except Exception:
        logger.warning("Could not write %s", published_json, exc_info=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class DesktopArtifact(NamedTuple):
    """An artifact and the artifacts root it sits under (desktop_artifact_and_base)."""

    artifact: Path
    artifacts_base: Path


class ResolvedReport(NamedTuple):
    """A published artifact's state file, its key, and its report id (_resolve_report_id)."""

    published_json: Path
    published_key: str
    report_id: str


def desktop_artifact_and_base(
    raw_path: str, *, scope: TenantScope | None
) -> DesktopArtifact:
    """(artifact, artifacts_base) for a desktop request path, resolved only
    inside ``scope``'s artifacts roots.

    Split from credential resolution so callers can validate cheap, local things
    first: `update_artifact` must still answer "no published version to update"
    for an unpublished artifact rather than "configure your API key".

    The loop only IDENTIFIES which root the artifact sits under;
    `resolve_artifact_path` has already rejected anything outside them.
    """
    artifact = resolve_artifact_path(raw_path, scope=scope, allow_dir=True)
    if artifact is None:
        raise FileNotFoundError("Artifact not found")
    for root in _scan_artifact_dirs(scope=scope):
        try:
            artifact.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        return DesktopArtifact(artifact, root)
    raise FileNotFoundError("Artifact is not in a known artifacts directory")


def desktop_publish_credential() -> tuple[str, str]:
    """(api_key, publish_url) from the active provider, for the desktop path."""
    publish_url, api_key = _resolve_publish_endpoint(get_user_settings())
    if not api_key:
        raise ValueError("Configure your provider API key in Settings before publishing")
    return api_key, publish_url


class PublishContext(NamedTuple):
    """What a desktop publish needs (desktop_publish_context)."""

    artifact: Path
    artifacts_base: Path
    api_key: str
    publish_url: str


def desktop_publish_context(
    raw_path: str, *, scope: TenantScope | None
) -> PublishContext:
    """The artifact, its artifacts root and the credential, for the desktop path.

    Keeps the pre-existing behavior - path resolution against the registered
    artifact roots of ``scope``, credential resolution from the active
    provider - in one place, so the publish functions themselves stay free of
    both. The org path supplies these four values from the DB and a minted turn
    key instead.
    """
    found = desktop_artifact_and_base(raw_path, scope=scope)
    api_key, publish_url = desktop_publish_credential()
    return PublishContext(
        artifact=found.artifact,
        artifacts_base=found.artifacts_base,
        api_key=api_key,
        publish_url=publish_url,
    )


def _resolve_publish_target(
    artifact: Path, *, container_dirs: list[Path]
) -> tuple[Path, Path, str, bool]:
    """Decide what to publish + where `.published.json` lives + its key.

    Thin wrapper over the single source of truth in anton
    (`anton.publish_access.resolve_publish_target`); cowork supplies its own
    container dirs so the metadata-climb is bounded to `.anton/artifacts/` roots.

    `container_dirs` is required: the org path passes the artifact's own root,
    and a desktop path passes the scan of its scope
    (`_scan_artifact_dirs(scope=...)`), which only sees the desktop layout.

    Returns (publish_target, published_dir, published_key, is_fullstack).
    """
    return _anton_resolve_publish_target(artifact, container_dirs)


def _resolve_publish_endpoint(settings) -> tuple[str, str]:
    """Resolve the (publish base URL, api key) for the active provider's env.

    Publishing follows the env the provider points at: a custom OpenAI-compatible
    MindsHub endpoint (dev/staging) wins over the default minds_url (prod) and
    authenticates with that provider's own key — so pointing the provider at
    dev/staging publishes there too. api key is "" when the chosen provider has none.

    A non-blank ``COWORK_PUBLISH_TOKEN`` (AppSettings.publish_token) replaces
    that key on every provider. A self-hosted publisher then never takes the
    model key, which the agent's code holds as its own OPENAI_API_KEY on a
    MindsHub endpoint. Under COWORK_CODE_SANDBOX=landlock, anton's launcher
    passes no COWORK_* name to the agent's code (`anton.core.backends.sandbox`),
    so the token stays out of it too.

    Resolution order for the publish base URL (first non-empty wins):
      1. ``ANTON_PUBLISH_URL`` env var — an operator/dev override that trumps
         the stored setting (the DB row would otherwise shadow env; see
         SettingService, which passes DB rows as init kwargs);
      2. the ``publish_url`` setting;
      3. the host derived from the active provider endpoint.

    A result on the retired 4nton.ai host becomes the prod publish host
    (`normalize_publish_url`): a saved setting may still name it.
    """
    oai_host = (urlparse(settings.openai_base_url or "").hostname or "").lower()
    if oai_host.startswith("api") and oai_host.endswith(".mindshub.ai"):
        endpoint = settings.openai_base_url
        api_key = _secret_str(provider_api_key(settings, Provider.OPENAI_COMPATIBLE))
    else:
        endpoint, api_key = settings.minds_url, _secret_str(settings.minds_api_key)
    api_key = _secret_str(get_app_settings().publish_token).strip() or api_key
    env_publish_url = os.environ.get("ANTON_PUBLISH_URL", "").strip()
    publish_url = normalize_publish_url(
        env_publish_url or settings.publish_url or publish_url_for_endpoint(endpoint)
    )
    return publish_url, api_key


def list_publishable(*, scope: TenantScope | None) -> dict:
    settings = get_user_settings()
    state = _load_state(scope=scope)
    publish_url, api_key = _resolve_publish_endpoint(settings)
    return {
        "artifacts": html_artifacts(scope=scope),
        "publishReady": bool(api_key),
        "publishUrl": publish_url,
        "history": state.get("publish_history", [])[:40],
    }


# _normalize_emails / _resolve_access now live in anton.publish_access (single
# source of truth) and are imported at the top of this module.


def is_publishable_static(target: Path, artifacts_base: Path) -> bool:
    """A non-fullstack target publishes only when it is an HTML/Markdown file
    that is artifact content (not README, a generation input or a revision copy)."""
    return target.suffix.lower() in PUBLISHABLE_STATIC_SUFFIXES and not is_non_content_path(
        target, artifacts_base
    )

# What Share and the agent's publish tool report for a full-stack artifact when
# the publish URL is not MindsHub's service (`is_mindshub_publish_url`).
FULLSTACK_PUBLISH_UNSUPPORTED = (
    "Full-stack apps can't be shared from this deployment. Its publishing "
    "service accepts HTML and Markdown artifacts only."
)

# Self-contained page wrapper for rendered Markdown. No external assets so
# the published bundle is a single index.html the viewer serves standalone.
# Styled to match Anton's dashboards (GitHub-dark palette + system fonts —
# see anton's generated reports) so a published doc looks of-a-piece with
# the dashboards/reports Anton produces, just tuned for long-form reading
# (comfortable column width + line-height).
_MD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.7; font-size: 16px; -webkit-font-smoothing: antialiased;
  }}
  .doc {{ max-width: 820px; margin: 0 auto; padding: 56px 24px 96px; }}
  h1, h2, h3, h4, h5 {{ line-height: 1.3; margin: 1.8em 0 0.6em; font-weight: 600; }}
  h1 {{ font-size: 2em; margin-top: 0; padding-bottom: 0.3em; border-bottom: 1px solid var(--border); letter-spacing: -0.4px; }}
  h2 {{ font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1.25em; }}
  p, ul, ol, blockquote, table, pre {{ margin: 0 0 1.1em; }}
  ul, ol {{ padding-left: 1.5em; }}
  li {{ margin: 0.3em 0; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  strong {{ color: #fff; }}
  code {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
          font-size: 0.88em; background: var(--bg3); padding: 0.2em 0.4em; border-radius: 6px; }}
  pre {{ background: var(--bg2); border: 1px solid var(--border); padding: 16px; border-radius: 10px; overflow: auto; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ margin-left: 0; padding: 0.2em 1em; color: var(--muted); border-left: 3px solid var(--accent); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95em; }}
  th, td {{ border: 1px solid var(--border); padding: 8px 13px; text-align: left; }}
  th {{ background: var(--bg2); font-weight: 600; }}
  tr:nth-child(even) td {{ background: rgba(255, 255, 255, 0.02); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2em 0; }}
  img {{ max-width: 100%; border-radius: 8px; }}
</style>
</head>
<body>
<main class="doc">
{body}
</main>
</body>
</html>
"""


def _markdown_title(md_path: Path, md_text: str) -> str:
    """Page <title>: the first ATX `# ` heading if present, else the filename."""
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or md_path.stem
    return md_path.stem


def _render_markdown_to_html(md_path: Path, out_dir: Path) -> Path:
    """Render a Markdown file to a standalone ``index.html`` in ``out_dir``.

    Returns the path to the generated file, which is what we hand to
    ``anton.publisher.publish`` (it zips it as index.html and serves it as a
    web page). The original ``.md`` is never modified — the registry and
    publish history still key off it, not this temp file.
    """
    # `markdown` is a direct dependency (pyproject.toml); the guard stays
    # defensive so a broken install reports a clear error.
    try:
        import markdown
    except Exception as exc:  # pragma: no cover - dependency guard
        raise PublisherUnavailable("Markdown renderer is unavailable") from exc

    md_text = md_path.read_text(encoding="utf-8", errors="replace")
    body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        output_format="html5",
    )
    page = _MD_HTML_TEMPLATE.format(title=_html.escape(_markdown_title(md_path, md_text)), body=body)
    out_path = out_dir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


# A restricted rule's org audience beyond emails and the org flag (ENG-3787):
# teams, and people in the org by Keycloak user id, under the auth service's
# ArtifactAccess names. anton's publisher knows neither, so this server writes
# a person's save of them to the auth service itself (_sync_team_audience).
_ORG_AUDIENCE_KEYS = ("allowed_team_ids", "allowed_user_ids")
# What a restricted entry of .published.json leaves out in hosted org mode:
# who the rule shares with, which the auth service's rule alone records
# (cowork.services.artifact_access.ArtifactRule). The entry keeps the mode,
# the password and the versions.
_ENTRY_AUDIENCE_KEYS = ("emails", "org_allowed", "owner_only")

# The app's names for a restricted rule's audience (ENG-3787), the auth
# service's ArtifactAccess ones: org_allowed, allowed_team_ids,
# allowed_user_ids and allowed_emails. anton's publisher reads emails by its own older name.
_APP_EMAILS_KEY = "allowed_emails"
# Names the app no longer sends: refused rather than silently dropped, so a
# mismatched client can't shrink an audience without noticing.
_RETIRED_ACCESS_KEYS = {
    "emails": "allowed_emails",
    "team_ids": "allowed_team_ids",
    "org_wide": "org_allowed",
}

# The app's "Notify people" choice, sent inside the access object (ENG-3787):
# ``notify: {user_ids: [...], emails: [...]}``, the individuals the person just
# added and chose to tell. Never part of the access anton's publisher or the
# stored entry sees: the caller acting on a person's save lifts it out and
# passes it to publish_artifact on its own, so a re-publish from the stored
# access (live sync, autopublish, a retry) can never carry it.
_NOTIFY_KEY = "notify"

# The auth service's bounds on what the share email names and links.
_MAX_NOTIFY_TITLE_CHARS = 200
_MAX_NOTIFY_URL_CHARS = 2048
# The viewer hosts the auth service's share email links to (dev and PR envs
# share dev's); the publish lambda's view_url is on one of them.
_NOTIFY_URL_HOSTS = frozenset({"view.dev.mindshub.ai", "view.staging.mindshub.ai", "view.mindshub.ai"})


class ShareNotify(NamedTuple):
    """Who a person's save asks the auth service to email: people in the org
    by Keycloak id, and anyone else by address."""

    user_ids: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.user_ids or self.emails)


def _notify_list(raw: Any, field: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"notify.{field} must be a list.")
    return [str(v).strip() for v in raw if isinstance(v, str) and v.strip()]


def pop_notify(access: dict | None) -> tuple[dict | None, ShareNotify]:
    """``(access without its notify field, who to notify)``. Raises
    ValueError for a ``notify`` that isn't ``{user_ids?: [], emails?: []}``,
    or that names more than the auth service emails from one save:
    MAX_SHARED_PEOPLE ids and MAX_NOTIFY_EMAILS addresses
    (cowork.services.skill_sharing). The access route calls it before
    anything is published."""
    from cowork.services.skill_sharing import MAX_NOTIFY_EMAILS, MAX_SHARED_PEOPLE

    if not isinstance(access, dict) or _NOTIFY_KEY not in access:
        return access, ShareNotify()
    out = {k: v for k, v in access.items() if k != _NOTIFY_KEY}
    raw = access.get(_NOTIFY_KEY)
    if raw is None:
        return out, ShareNotify()
    if not isinstance(raw, dict):
        raise ValueError("notify must be an object with user_ids and emails.")
    user_ids = tuple(dict.fromkeys(v.lower() for v in _notify_list(raw.get("user_ids"), "user_ids")))
    emails = tuple(dict.fromkeys(_normalize_emails(_notify_list(raw.get("emails"), "emails"))))
    if len(user_ids) > MAX_SHARED_PEOPLE:
        raise ValueError(f"notify can name at most {MAX_SHARED_PEOPLE} people.")
    if len(emails) > MAX_NOTIFY_EMAILS:
        raise ValueError(f"notify can name at most {MAX_NOTIFY_EMAILS} email addresses.")
    return out, ShareNotify(user_ids=user_ids, emails=emails)


def access_from_app(access: dict) -> dict:
    """The app's access object (ENG-3787) as this module and anton's publisher
    read it: ``allowed_emails`` (people outside the org, by address) becomes
    anton's ``emails``; ``org_allowed``, ``allowed_team_ids`` and
    ``allowed_user_ids`` pass through as they are. ValueError naming the
    field to use for a retired name (``emails``, ``team_ids``, ``org_wide``)."""
    for retired, name in _RETIRED_ACCESS_KEYS.items():
        if retired in access:
            raise ValueError(f"{retired} is not an access field; send {name}.")
    out = dict(access)
    if _APP_EMAILS_KEY in out:
        out["emails"] = out.pop(_APP_EMAILS_KEY)
    return out


def _notify_within(notify: ShareNotify, audience: ArtifactRule) -> ShareNotify:
    """Only people the saved rule actually names, by id or by address: a
    notify list never reaches anyone the share doesn't."""
    ids = {str(v).lower() for v in audience.allowed_user_ids}
    emails = {str(v).lower() for v in audience.allowed_emails}
    return ShareNotify(
        user_ids=tuple(v for v in notify.user_ids if v in ids),
        emails=tuple(v for v in notify.emails if v.lower() in emails),
    )


def _with_team_audience(access: dict | None) -> dict | None:
    """``access`` as anton's publisher reads it, which knows no teams and no
    people by id. A selection of teams or people alone is owner-only there,
    so the lambda writes a restricted rule that sync_artifact_team_access
    then widens; teams or people named where they can't apply (outside org
    mode) stay owner-only too, never the public rule anton falls back to for
    an empty selection.
    """
    if not isinstance(access, dict) or access.get("mode") != "restricted":
        return access
    out = {k: v for k, v in access.items() if k not in _ORG_AUDIENCE_KEYS}
    named = any(access.get(key) for key in _ORG_AUDIENCE_KEYS)
    if named and not out.get("emails") and not out.get("org_allowed"):
        out["owner_only"] = True
    return out


def audience_fields(audience: ArtifactRule | None) -> dict[str, Any]:
    """Who a restricted rule shares with, under the app's names, for the
    publish response and GET /access. None (a publish that isn't restricted)
    names nobody; ``ownerOnly`` is a restricted rule that names nobody."""
    if audience is None:
        return {
            "accessEmails": [], "orgAllowed": False, "ownerOnly": False,
            "allowedTeamIds": [], "allowedUserIds": [],
        }
    return {
        "accessEmails": list(audience.allowed_emails),
        "orgAllowed": audience.org_allowed,
        "ownerOnly": audience.names_nobody,
        "allowedTeamIds": list(audience.allowed_team_ids),
        "allowedUserIds": list(audience.allowed_user_ids),
    }


def kept_access(entry: Any, *, org_mode: bool) -> dict:
    """The access a re-publish that isn't a person's save keeps (live sync,
    autopublish, an update): the entry's mode and password. In hosted org
    mode a restricted entry doesn't say who the rule shares with, so this
    names nobody and publish_artifact takes all of it from the auth
    service's rule (_plan_access); ``owner_only`` keeps a rule that names
    nobody restricted to its owner, never public. A mode that isn't
    restricted is refused there while the auth service still holds a rule
    (AccessModeChanged): the pod can write the entry. Elsewhere the entry is
    the record (anton's access_from_owner_side)."""
    access = access_from_owner_side(entry)
    if org_mode and access.get("mode") == "restricted":
        return {"mode": "restricted", "owner_only": True}
    return access


class AudienceChanges(NamedTuple):
    """The team and people lists a restricted publish changes on the
    artifact's rule (_plan_access), for _sync_team_audience to write. None
    leaves that list as the auth service holds it; false when neither
    changes."""

    allowed_team_ids: list[str] | None = None
    allowed_user_ids: list[str] | None = None

    def __bool__(self) -> bool:
        return self.allowed_team_ids is not None or self.allowed_user_ids is not None


class AccessPlan(NamedTuple):
    """One publish's access (_plan_access)."""

    # What anton's publisher is given, and the versions it sends.
    effective_access: dict
    pwd_version: int
    access_version: int
    # What the entry stores: the mode, the password and the versions, and
    # outside hosted org mode who a restricted rule shares with.
    owner_side: dict
    # Who the restricted rule shares with after this publish, as far as this
    # server knows; None for a publish that isn't restricted.
    audience: ArtifactRule | None
    # The team and people lists this publish changes, for
    # _sync_team_audience to write.
    changes: AudienceChanges


def _plan_access(
    password: str | None, access: dict | None, previous: Any, *, stored: ArtifactRule | None
) -> AccessPlan:
    """anton's resolve_access, with the org audience (ENG-3787) around it.

    In hosted org mode ``stored`` is the artifact's rule as the auth service
    stores it, read for every restricted publish (publish_artifact): the one
    record of who the artifact is shared with. A restricted publish takes
    each of the emails, the org flag, the teams and the people from
    ``access`` when it names it, else from ``stored``. So a re-publish that
    isn't a person's save (kept_access) sends what the rule holds and writes
    no team or person, and a team the auth service dropped from the rule
    when it was deleted is never sent back. Only a team or people list that
    differs from the stored one is written (``changes``), and that bumps
    the version, as a changed email list does. The version is judged against
    what the rule holds, never against the entry, which keeps the mode, the
    password and the versions only.

    Outside hosted org mode, and for a publish that isn't restricted,
    ``stored`` is None: the entry is the record anton's publisher keeps, and
    teams and people don't apply. Teams or people alone are restricted to
    them, never public (_with_team_audience).
    """
    from cowork.services.artifact_access import ArtifactRule

    if isinstance(access, dict):
        # Only the person's own save carries notify, and its caller lifted it
        # out already; anything else just loses the field.
        access = {k: v for k, v in access.items() if k != _NOTIFY_KEY}
    prev_restricted = isinstance(previous, dict) and previous.get("mode") == "restricted"
    if stored is not None and isinstance(access, dict) and access.get("mode") == "restricted":
        access = {
            "emails": stored.allowed_emails,
            "org_allowed": stored.org_allowed,
            "allowed_team_ids": stored.allowed_team_ids,
            "allowed_user_ids": stored.allowed_user_ids,
            **access,
        }
        if prev_restricted:
            previous = {
                **previous,
                "emails": stored.allowed_emails,
                "org_allowed": stored.org_allowed,
                "owner_only": not stored.allowed_emails and not stored.org_allowed,
            }
    effective_access, pwd_version, access_version, owner_side = _resolve_access(
        password, _with_team_audience(access), previous
    )
    if owner_side.get("mode") != "restricted":
        return AccessPlan(
            effective_access, pwd_version, access_version, owner_side, None, AudienceChanges()
        )
    if stored is None or not isinstance(access, dict):
        audience = ArtifactRule(
            org_allowed=bool(owner_side.get("org_allowed")),
            allowed_emails=list(owner_side.get("emails") or []),
            allowed_team_ids=[],
            allowed_user_ids=[],
        )
        return AccessPlan(
            effective_access, pwd_version, access_version, owner_side, audience, AudienceChanges()
        )
    named = {key: [str(v) for v in access.get(key) or []] for key in _ORG_AUDIENCE_KEYS}
    changes = AudienceChanges(**{
        key: values for key, values in named.items() if set(values) != set(getattr(stored, key))
    })
    if changes and prev_restricted:
        # A changed audience invalidates the viewer grants already issued, as
        # a changed email list does.
        access_version = max(access_version, (previous.get("access_version", 0) or 0) + 1)
    audience = ArtifactRule(
        org_allowed=bool(owner_side.get("org_allowed")),
        allowed_emails=list(owner_side.get("emails") or []),
        **named,
    )
    owner_side = {
        **{k: v for k, v in owner_side.items() if k not in _ENTRY_AUDIENCE_KEYS},
        "access_version": access_version,
    }
    return AccessPlan(effective_access, pwd_version, access_version, owner_side, audience, changes)


def _rule_key(canonical_key: str | None, entry: Any) -> str:
    """The key an artifact's rule is stored under: its identity, else, for a
    legacy loose file with none, the key the upload service gave the publish
    ``entry`` records. The one source of that key, for the read before a
    publish and for every write after it."""
    if canonical_key:
        return canonical_key
    return str(entry.get("artifact_key") or "") if isinstance(entry, dict) else ""


def _prior_rule_keys(canonical_key: str | None, entry: Any) -> list[str]:
    """Every key a rule the last publish left could sit under: _rule_key's
    pick, then each key ``entry`` recorded for that publish (``artifact_key``
    and the lambda's ``lambda_artifact_key``) when it differs. The pod can
    write metadata.json, which moves _rule_key's pick, but not the rule the
    auth service already holds. Empty when none is known."""
    recorded = (
        [str(entry.get(field) or "") for field in ("artifact_key", "lambda_artifact_key")]
        if isinstance(entry, dict)
        else []
    )
    return [key for key in dict.fromkeys([_rule_key(canonical_key, entry), *recorded]) if key]


def _access_mode_refusal(
    canonical_key: str | None, entry: Any, *, reuses_report: bool, scope: TenantScope
) -> Literal["rule_held", "no_rule_key"] | None:
    """Why a re-publish in hosted org mode that isn't a person's save and
    isn't restricted must not go ahead (AccessModeChanged), or None. The
    auth service holds a rule under one of the keys the last publish could
    have left it under (_prior_rule_keys): ``rule_held``. Or the publish
    reuses a page the entry names no key for, so no rule can be read:
    ``no_rule_key``. A failed read raises."""
    keys = _prior_rule_keys(canonical_key, entry)
    if not keys:
        return "no_rule_key" if reuses_report else None
    if _held_rule(keys, scope) is not None:
        return "rule_held"
    return None


def _held_rule(keys: list[str], scope: TenantScope) -> ArtifactRule | None:
    """The first rule the auth service holds under ``keys``, in order, or
    None when it holds none. A failed read raises."""
    for key in keys:
        rule = _read_rule(key, scope)
        if rule is not None:
            return rule
    return None


def _authorization_key(
    published_dir: Path, artifacts_base: Path, project_id: str | None, scope: TenantScope | None
) -> str | None:
    """One identity spans private drafts, published versions, and comments:
    the artifact's, from its metadata. A legacy loose file has no metadata
    and keeps the service-generated key (None)."""
    if not (published_dir / "metadata.json").is_file():
        return None
    from cowork.services.artifact_identity import artifact_key, ensure_full_id

    artifact_id, _metadata = ensure_full_id(published_dir)
    if scope is not None and scope.org_mode:
        from cowork.services.artifact_authorization_identity import publish_authorization_key

        return publish_authorization_key(
            artifact_id, artifacts_base, published_dir.name, project_id, scope
        )
    return artifact_key(artifact_id)


def _read_rule(rule_key: str, scope: TenantScope) -> ArtifactRule | None:
    """The artifact's rule as the auth service stores it under ``rule_key``
    (read_artifact_access), or None when it holds none, as for an artifact
    with no key yet. A failed read raises, so a publish never guesses who it
    shares with."""
    from cowork.services.artifact_access import read_artifact_access

    if not rule_key:
        return None
    return read_artifact_access(
        artifact_id=rule_key, scope=scope, owner_user_id=str(scope.user_id or "")
    )


def _stored_rule(rule_key: str, scope: TenantScope) -> ArtifactRule:
    """Who the artifact is shared with (_read_rule): nobody but the owner
    when the auth service holds no rule for it."""
    from cowork.services.artifact_access import ArtifactRule

    rule = _read_rule(rule_key, scope)
    return ArtifactRule.nobody() if rule is None else rule


def stored_artifact_rule(
    artifact: Path, *, artifacts_base: Path, scope: TenantScope, project_id: str | None
) -> ArtifactRule:
    """Who ``artifact`` is shared with, as the auth service's rule stores it
    (ENG-3787), read under the key publish_artifact publishes it with. What
    the access route checks a save against and shows; .published.json never
    says it in hosted org mode."""
    _target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    previous = _load_published_map(published_dir).get(published_key)
    canonical_key = _authorization_key(published_dir, artifacts_base, project_id, scope)
    return _stored_rule(_rule_key(canonical_key, previous), scope)


def rule_held_over_entry(
    artifact: Path, *, artifacts_base: Path, scope: TenantScope, project_id: str | None
) -> ArtifactRule | None:
    """The rule the auth service holds for ``artifact`` under a key its live
    .published.json entry's last publish could have left it under
    (_prior_rule_keys, the keys _access_mode_refusal reads before a
    re-publish), or None: for no live entry, or no rule. The access route
    reads it for an entry that isn't restricted. Only a restricted publish
    leaves a rule in hosted org mode, so a rule held there means the pod
    rewrote the entry's mode after the owner's last save, and the rule, not
    the entry, says who the artifact is shared with. A failed read raises."""
    _target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    entry = _load_published_map(published_dir).get(published_key)
    if not isinstance(entry, dict) or not entry.get("published", True):
        return None
    canonical_key = _authorization_key(published_dir, artifacts_base, project_id, scope)
    return _held_rule(_prior_rule_keys(canonical_key, entry), scope)


def publish_artifact(
    artifact: Path,
    *,
    artifacts_base: Path,
    api_key: str,
    publish_url: str,
    password: str | None = None,
    access: dict | None = None,
    scope: TenantScope | None = None,
    project_id: str | None = None,
    notify: ShareNotify | None = None,
    stored: ArtifactRule | None = None,
    owner_save: bool = False,
) -> dict:
    """Zip an artifact and upload it, returning its public URL.

    `artifact` is the artifact folder or a single legacy loose file (loose-HTML /
    chat-bubble / the Utilities per-page list); `_resolve_publish_target`
    normalizes both. `artifacts_base` is the container root that bounds its
    metadata-climb.

    `api_key`/`publish_url` come from the caller: desktop resolves them from the
    active provider (see `desktop_publish_context`), the org path mints a
    per-reconciliation turn key. That distinction is load-bearing - the upload
    lambda takes `owner_keycloak_id` from the token and folds md5(user_id)[:9]
    into the URL, so the key decides who owns the published artifact.

    `scope` selects the connector vault the publisher reads datasource secrets
    from; it is org-keyed, so an unscoped lookup would resolve to the shared
    namespace root. Optional because the loose-file and Markdown paths carry no
    datasources at all, and because `vault_for_scope` fail-closes on an org
    deployment when it is missing - a caller that forgets it gets an error, not
    another org's secrets.

    `project_id` is the artifact's project; required in organization mode,
    where the owner lookup is keyed by it.

    `notify` (ENG-3787) is set only for a person's own save of the Share
    dialog with "Notify people" chosen: the individuals it added, by id and by
    address. The auth service emails them (those the saved rule names); the
    write that carries the list is this server's post-publish write of the
    rule, made for a restricted org-mode share even without teams or people
    by id, so people added by address alone are emailed too. Every other
    publish (live sync, autopublish, a retry) leaves it None.

    `stored` (ENG-3787) is the artifact's rule the access route already read
    under the publish lock to check this save, so the save reads the rule
    once and is planned against the rule it was checked against. Every other
    caller leaves it None, and a restricted publish in hosted org mode reads
    the rule itself.

    `owner_save` is True only for a person's own save of the Share dialog
    (PUT /access), whose access is what they chose. Any other publish in
    hosted org mode (live sync, autopublish) takes its mode from the
    .published.json entry, which the pod can write. One that isn't
    restricted reads the rule first, under the artifact's identity and under
    each key the entry recorded. While the auth service still holds one,
    the entry's mode changed after the owner's last save, and a re-publish
    of a page the entry names no key for can't be checked: either is
    refused (AccessModeChanged) and logged before anything is uploaded.
    """
    if scope is not None and scope.org_mode and project_id is None:
        raise ValueError("publish_artifact requires project_id in organization mode")
    if not api_key:
        raise ValueError("Publishing requires an API key")

    publish_target, published_dir, published_key, is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    if not is_fullstack and not is_publishable_static(publish_target, artifacts_base):
        raise ValueError("Only HTML and Markdown artifact content can be published")
    # Refuse before anything is built: `anton.publisher.publish` puts the
    # artifact's datasource credentials, in plaintext, in the body of a
    # full-stack upload, and a service that can't run the app would receive
    # them only to answer 400.
    if is_fullstack and not is_mindshub_publish_url(publish_url):
        raise ValueError(FULLSTACK_PUBLISH_UNSUPPORTED)

    try:
        from anton.publisher import ArtifactOwnedByOtherUserError, StatePublishBlocked, publish
    except Exception as exc:
        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    from cowork.services.published_page_ownership import publication_owner

    with publication_owner(
        publish_target, published_dir=published_dir, publish_url=publish_url, scope=scope,
    ) as ownership:
        published_json = published_dir / ".published.json"
        published_map: dict[str, Any] = {}
        if published_json.is_file():
            try:
                published_map = json.loads(published_json.read_text(encoding="utf-8"))
            except Exception:
                published_map = {}
        previous = published_map.get(published_key)
        report_id = previous.get("report_id") if isinstance(previous, dict) else None
        if ownership is not None:
            report_id = ownership.checked_report_id(
                report_id, url=previous.get("url") if isinstance(previous, dict) else "",
            )
            if report_id is None:
                previous = None

        canonical_artifact_key = _authorization_key(published_dir, artifacts_base, project_id, scope)
        org_scope = scope if scope is not None and scope.org_mode else None
        restricted = isinstance(access, dict) and access.get("mode") == "restricted"
        refusal = (
            _access_mode_refusal(
                canonical_artifact_key, previous, reuses_report=report_id is not None, scope=org_scope
            )
            if org_scope is not None and not owner_save and not restricted
            else None
        )
        if refusal is not None:
            logger.warning(
                "Refused a re-publish whose .published.json entry isn't restricted: "
                "the auth service still holds a restricted rule for the artifact, "
                "or the entry names no key to read one under",
                extra={
                    "event": "artifact_access_mode_changed",
                    "reason": refusal,
                    **log_context(project_id=project_id, artifact_slug=published_dir.name),
                },
            )
            raise AccessModeChanged()

        # Resolve the effective access (mode + version) from the request and the
        # prior publish. Versions bump only when the password / restricted list
        # changed so previously issued viewer grants invalidate. In hosted org
        # mode who a restricted publish shares with starts from the artifact's
        # rule as the auth service stores it: the one the caller read for this
        # save, else read here. A failed read fails the publish.
        if org_scope is None or not restricted:
            stored = None
        elif stored is None:
            stored = _stored_rule(_rule_key(canonical_artifact_key, previous), org_scope)
        plan = _plan_access(password, access, previous, stored=stored)
        owner_side = plan.owner_side
        # Who the rule shares with: the plan's, then the auth service's answer
        # to this publish's write, if it makes one.
        audience = plan.audience

        # Markdown is rendered to a throwaway index.html that we hand to the
        # publisher; `.html` and fullstack publish their real target directly.
        # `publish_target` stays the original artifact so the registry, history,
        # and unpublish all key off the file the user actually sees.
        publish_source = publish_target
        md_tmp_dir: tempfile.TemporaryDirectory | None = None
        if not is_fullstack and publish_target.suffix.lower() == ".md":
            md_tmp_dir = tempfile.TemporaryDirectory(prefix="cowork-md-publish-")
            publish_source = _render_markdown_to_html(publish_target, Path(md_tmp_dir.name))

        ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
        try:
            result = publish(
                publish_source,
                api_key=api_key,
                report_id=report_id,
                publish_url=publish_url,
                ssl_verify=ssl_verify,
                access=plan.effective_access,
                access_version=plan.access_version,
                pwd_version=plan.pwd_version,
                artifact_key=canonical_artifact_key,
                # Resolve datasource secrets from cowork's own vault
                # (`~/.cowork/data-vault`), not anton's default
                # (`~/.anton/data_vault`) — otherwise secrets are missed and
                # the published artifact has no DB connection in the cloud.
                # Org-keyed: the persisted vault is per organization, and an
                # unscoped lookup would resolve to the shared namespace root.
                vault=vault_for_scope(scope),
                # While the password is unchanged anton re-sends the hash stored
                # here, so a content-only re-publish keeps viewers' password grants.
                previous_access=previous if isinstance(previous, dict) else None,
            )
        except ArtifactOwnedByOtherUserError as exc:
            # Not a failure to log or wrap: the upload service holds this
            # report_id for another account.
            raise PublishOwnedByOtherUser() from exc
        except Exception as exc:
            upload_error = classify_upload_error(exc)
            logger.exception("Publishing failed")
            # A 401's text ("Invalid token") says less than the credentials advice below.
            if (
                upload_error is not None
                and upload_error.error
                and 400 <= upload_error.status < 500
                and upload_error.status != 401
            ):
                raise PublishRejected(f"Publishing failed: {_clip(upload_error.error)}") from exc
            # Only network/HTTP failures get the "Connection failed" framing — a
            # gateway timeout (e.g. a fullstack artifact whose deps take too long
            # to install remotely) or a server-side 5xx reads very differently to
            # the user than an auth rejection, but neither applies to a local
            # failure (e.g. reading the artifact's files to zip it) that happened
            # before any request went out. urllib.error HTTPError is a URLError
            # subclass, so this covers both.
            if isinstance(exc, urllib.error.URLError):
                headline, advice = describe_minds_connection_error(exc)
                raise PublishRejected(f"Publishing failed. {headline} {advice}") from exc
            if isinstance(exc, StatePublishBlocked):
                raise PublishRejected(f"Publishing failed: {exc}") from exc
            raise RuntimeError(f"Publishing failed: {exc}") from exc
        finally:
            if md_tmp_dir is not None:
                md_tmp_dir.cleanup()

        # Owner-side state for the next publish, stored in the entry below and
        # never part of the response.
        password_hash = result.pop("password_hash", None)
        view_url = result.get("view_url", "")
        returned_report_id = result.get("report_id", "")
        if ownership is not None:
            try:
                ownership.record(returned_report_id)
            except Exception:
                if report_id is None:
                    # A new page went live with no record and no local entry;
                    # keep the orphaned_publish prefix.
                    logger.warning(
                        "orphaned_publish identifier=%s url=%s reason=owner_record_failed",
                        returned_report_id, view_url,
                    )
                raise
        echoed_key = _lambda_artifact_key(result)
        if returned_report_id:
            history_item = {
                "artifact": str(publish_target),
                "artifactName": published_key,
                "url": view_url,
                "reportId": returned_report_id,
                "publishedAt": _utc_now_iso(),
            }
            # Owner-side only — .published.json never enters the bundle. `owner_side`
            # carries `mode` (+ `access_password`/`pwd_version` for password, or
            # `emails`/`org_allowed`/`access_version` for restricted; in hosted
            # org mode `access_version` alone, since who a restricted rule shares
            # with is the auth service's rule's to say), and always
            # `requires_password` for back-compat with older readers. Password mode
            # also keeps the `password_hash` anton sent, handed back as
            # `previous_access` on the next publish.
            entry: dict[str, Any] = {
                "report_id": returned_report_id,
                "url": view_url,
                # The lambda's echo, else the key cowork sent. Comment routing reads
                # `lambda_artifact_key` below, not this field.
                "artifact_key": result.get("artifact_key") or canonical_artifact_key or "",
                # The lambda's raw echo, or "" from an older lambda that sends none.
                # Decides which scope the published page uses for comments; see
                # services/comments_scope. Deliberately NOT merged with the field
                # above, whose canonical fallback would read as an echo.
                "lambda_artifact_key": echoed_key,
                "last_md5": result.get("md5", ""),
                # Snapshot of the artifact's content mtime at publish time — the
                # cheap gate for the `modified` badge (see card_for_folder). Uses
                # the SAME basis as the card's `mtime` so the comparison is exact.
                "published_mtime": _content_mtime(published_dir),
                "published": True,
                **owner_side,
            }
            if password_hash:
                entry["password_hash"] = password_hash
            published_map[published_key] = entry
            _write_published_map(published_json, published_map)
            recipients = (
                _notify_within(notify, audience)
                if notify and org_scope is not None and audience is not None
                else ShareNotify()
            )
            notice = (
                {**_share_notice(published_dir, published_key, view_url), "recipients": recipients}
                if recipients
                else None
            )
            if plan.changes:
                # Only a restricted publish in hosted org mode plans changes.
                assert org_scope is not None and audience is not None
                echoed = _sync_team_audience(
                    artifact_id=_rule_key(canonical_artifact_key, entry), scope=org_scope,
                    access_version=plan.access_version, changes=plan.changes, notify=notice,
                )
                audience = audience.model_copy(update=echoed.model_dump())
            elif notice is not None:
                # No teams or people changed, so the lambda's write was the
                # whole change; this one only asks the auth service to email
                # the people the save added. Best effort: the share itself
                # already stands.
                assert org_scope is not None and audience is not None
                try:
                    echoed = _sync_team_audience(
                        artifact_id=_rule_key(canonical_artifact_key, entry), scope=org_scope,
                        access_version=plan.access_version, changes=AudienceChanges(), notify=notice,
                    )
                    audience = audience.model_copy(update=echoed.model_dump())
                except Exception:
                    logger.warning(
                        "Could not ask the auth service to notify the people an artifact was shared with",
                        exc_info=True,
                    )
            state = _load_state(scope=scope)
            state["publish_history"] = [history_item, *state.get("publish_history", [])][:100]
            _save_state(state, scope=scope)

        if scope is not None and scope.org_mode:
            # Org comment routing never reads the record, so only the echo can
            # point the renderer at the scope the published page uses.
            response_key = echoed_key or canonical_artifact_key or ""
        else:
            # The card's key: desktop routing finds the cloud scope in the record.
            response_key = canonical_artifact_key or echoed_key
        return {
            "status": "ok",
            "url": view_url,
            "accessMode": owner_side.get("mode", "public"),
            "accessProtected": bool(owner_side.get("requires_password")),
            **audience_fields(audience),
            # The renderer addresses comment requests by this key.
            "artifactKey": response_key,
            "result": {k: v for k, v in result.items() if k != "file_payload"},
        }


def _share_notice(published_dir: Path, published_key: str, view_url: str) -> dict[str, str]:
    """What the auth service's share email names and links: the artifact's
    title (its metadata name, else its folder or file name) and the published
    URL the recipient opens, within the auth service's bounds. A URL that
    isn't an https link on one of the viewer hosts it links to is left out."""
    title = ""
    try:
        meta = json.loads((published_dir / "metadata.json").read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            title = str(meta.get("title") or meta.get("name") or "")
    except (OSError, ValueError):
        pass
    title = " ".join((title or published_dir.name or published_key or "").split())
    url = view_url if isinstance(view_url, str) else ""
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = None
    if parsed is None or parsed.scheme != "https" or parsed.hostname not in _NOTIFY_URL_HOSTS:
        url = ""
    return {
        "artifact_title": title[:_MAX_NOTIFY_TITLE_CHARS],
        "artifact_url": url if len(url) <= _MAX_NOTIFY_URL_CHARS else "",
    }


def _sync_team_audience(
    *,
    artifact_id: str,
    scope: TenantScope,
    access_version: int,
    changes: AudienceChanges,
    notify: dict[str, Any] | None = None,
) -> ArtifactAudience:
    """Write the teams and people a person's save changed (``changes``) to
    the auth service, after the lambda wrote the rest of the rule, at the
    version it wrote (sync_artifact_team_access), and return the org flag,
    teams and people the rule holds now. A list ``changes`` leaves None is
    the auth service's to keep. ``notify`` (a person's save with "Notify
    people" chosen) carries the ``recipients`` (ShareNotify) and the title
    and URL the share email uses; None never notifies."""
    from cowork.services.artifact_access import ArtifactAccessUnavailable, sync_artifact_team_access
    from cowork.services.artifact_autopublish import _active_workspace_id

    if not artifact_id:
        raise ArtifactAccessUnavailable("The artifact has no identity to share with teams")
    return sync_artifact_team_access(
        artifact_id=artifact_id,
        scope=scope,
        owner_user_id=str(scope.user_id or ""),
        workspace_id=_active_workspace_id(scope),
        access_version=access_version or 1,
        allowed_team_ids=changes.allowed_team_ids,
        allowed_user_ids=changes.allowed_user_ids,
        notify_user_ids=list(notify["recipients"].user_ids) if notify else [],
        notify_emails=list(notify["recipients"].emails) if notify else [],
        artifact_title=(notify or {}).get("artifact_title", ""),
        artifact_url=(notify or {}).get("artifact_url", ""),
    )


def compute_publish_md5(artifact: Path, *, artifacts_base: Path) -> str | None:
    """Recompute the md5 of the publish bundle for an artifact.

    Matches the `last_md5` the lambda stores at publish time (md5 of the zip
    bytes `anton.publisher` produces). Mirrors `publish_artifact`'s source
    resolution exactly: markdown renders to a throwaway index.html first;
    static publishes the single primary file; fullstack publishes the
    artifact directory.

    `artifact` is the folder (or the single legacy loose file) and
    `artifacts_base` is its container root — both supplied by the caller, so this
    works identically on desktop and on the org layout the module-level FS scan
    cannot see. Without that, the "changed since publish" gate would silently
    never fire in an org deployment.

    Returns None when the artifact can't be resolved or bundled — the caller
    treats None as "can't tell" and does not flag the artifact as modified.
    """
    try:
        publish_target, _published_dir, _key, is_fullstack = _resolve_publish_target(
            artifact, container_dirs=[artifacts_base]
        )
    except Exception:
        return None
    if not is_fullstack and not is_publishable_static(publish_target, artifacts_base):
        return None
    try:
        from anton.publisher import _zip_fullstack, _zip_html
    except Exception:
        return None

    md_tmp_dir: tempfile.TemporaryDirectory | None = None
    try:
        publish_source = publish_target
        if not is_fullstack and publish_target.suffix.lower() == ".md":
            md_tmp_dir = tempfile.TemporaryDirectory(prefix="cowork-md-md5-")
            publish_source = _render_markdown_to_html(publish_target, Path(md_tmp_dir.name))
        if is_fullstack:
            zipped, _included = _zip_fullstack(publish_source)
        else:
            zipped = _zip_html(publish_source)
    except Exception:
        return None
    finally:
        if md_tmp_dir is not None:
            md_tmp_dir.cleanup()
    return hashlib.md5(zipped).hexdigest()


def unpublish_artifact(
    artifact: Path, *, artifacts_base: Path, api_key: str, publish_url: str,
    scope: TenantScope | None = None,
) -> dict:
    """Soft-delete a published artifact: drop it upstream, keep `report_id`.

    Same explicit-input contract as `publish_artifact`. A 404 from upstream is
    treated as "already gone" (pre-existing behavior) - but it can also mean the
    remote object is alive under a DIFFERENT owner prefix, because the delete
    lambda scopes by the token's user_dir. Once the local record is cleared there
    is nothing left to find it by, so that case is logged as `orphaned_publish`.
    """
    if not api_key:
        raise ValueError("Unpublishing requires an API key")

    # Mirror publish: resolve the same .published.json location + key
    # (primary file name) whether a folder or a file was passed.
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    from cowork.services.published_page_ownership import publication_owner

    with publication_owner(
        _publish_target, published_dir=published_dir, publish_url=publish_url, scope=scope,
    ) as ownership:
        published_json = published_dir / ".published.json"
        if not published_json.is_file():
            raise FileNotFoundError("Artifact has no publish record")

        try:
            published_map: dict[str, Any] = json.loads(published_json.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("Could not read publish record") from exc

        entry = published_map.get(published_key)
        identifier = None
        if isinstance(entry, dict):
            identifier = entry.get("report_id") or entry.get("last_md5") or None
        if ownership is not None:
            identifier = ownership.checked_report_id(
                entry.get("report_id") if isinstance(entry, dict) else None,
                url=entry.get("url", "") if isinstance(entry, dict) else "",
            )
            if identifier is None:
                published_map.pop(published_key, None)
                _write_published_map(published_json, published_map)
                return {"status": "ok"}
        if not identifier:
            raise FileNotFoundError("No published version on file")

        try:
            from anton.publisher import unpublish
        except Exception as exc:
            raise PublisherUnavailable("Anton publisher is unavailable") from exc

        ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
        try:
            unpublish(
                identifier,
                api_key=api_key,
                publish_url=publish_url,
                ssl_verify=ssl_verify,
            )
        except Exception as exc:
            classify_upload_error(exc)
            msg = str(exc) or "Unpublishing failed."
            if "404" in msg or "not found" in msg.lower():
                # Already gone upstream, OR still alive under another owner's prefix
                # and now unreachable — see the docstring. cowork-server has no
                # metrics backend, so this structured log line IS the metric; keep
                # the `orphaned_publish` prefix stable.
                logger.warning(
                    "orphaned_publish identifier=%s url=%s reason=unpublish_404",
                    identifier,
                    entry.get("url", "") if isinstance(entry, dict) else "",
                )
            else:
                logger.exception("Unpublishing failed (identifier=%s)", identifier)
                raise RuntimeError(f"Unpublishing failed: {msg}") from exc

        # Soft-delete: keep report_id (and url) so a later re-publish reuses the
        # same public URL. Only flip `published` off so readers stop showing it as
        # live. The backend object is gone, but lambda re-mints at the same id when
        # we resend report_id on the next publish.
        if isinstance(entry, dict):
            entry["published"] = False
            published_map[published_key] = entry
            _write_published_map(published_json, published_map)
        return {"status": "ok"}


def published_artifact_access(artifact: Path, *, artifacts_base: Path, org_mode: bool) -> dict:
    """Return the access a re-publish of a live artifact keeps (kept_access),
    or raise when it is not live.

    Credential resolution deliberately lives outside this helper. Callers can
    therefore distinguish an unpublished draft from a publish failure before
    minting an org turn key or asking Desktop for configured provider settings.
    """
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    published_json = published_dir / ".published.json"
    if not published_json.is_file():
        raise FileNotFoundError("Artifact has no publish record")
    try:
        published_map: dict[str, Any] = json.loads(published_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Could not read publish record") from exc

    entry = published_map.get(published_key)
    if not isinstance(entry, dict) or not entry.get("published", True) or not entry.get("report_id"):
        raise FileNotFoundError("No published version to update")
    return kept_access(entry, org_mode=org_mode)


def update_artifact(raw_path: str, *, scope: TenantScope | None) -> dict:
    """Re-publish an already-published artifact, preserving its URL and access.

    `publish_artifact` reuses the stored report_id (→ the lambda mints a new
    version at the same URL), but it would reset access to public if handed no
    access object. So reconstruct the prior access from `.published.json` and
    pass it through. The publish also refreshes `published_mtime` under its
    folder lock. Raises FileNotFoundError when there's nothing published to update.
    """
    # Path first, credential later: an artifact with nothing published must
    # report that, not "configure your API key".
    found = desktop_artifact_and_base(raw_path, scope=scope)
    artifact, artifacts_base = found.artifact, found.artifacts_base
    access = published_artifact_access(
        artifact, artifacts_base=artifacts_base, org_mode=scope is not None and scope.org_mode
    )

    # Delegates: reuses report_id (read from .published.json) + refreshes last_md5.
    api_key, publish_url = desktop_publish_credential()
    return publish_artifact(
        artifact, artifacts_base=artifacts_base, api_key=api_key,
        publish_url=publish_url, access=access, scope=scope,
    )


def _resolve_report_id(raw_path: str, *, scope: TenantScope | None) -> ResolvedReport:
    """Resolve (published_json_path, published_key, report_id) for an artifact
    of ``scope``.

    Raises FileNotFoundError when the artifact has no live publish record.
    """
    artifact = resolve_artifact_path(raw_path, scope=scope, allow_dir=True)
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=_scan_artifact_dirs(scope=scope)
    )
    published_json = published_dir / ".published.json"
    if not published_json.is_file():
        raise FileNotFoundError("Artifact has no publish record")
    try:
        published_map: dict[str, Any] = json.loads(published_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Could not read publish record") from exc
    entry = published_map.get(published_key)
    report_id = entry.get("report_id") if isinstance(entry, dict) else None
    if not report_id:
        raise FileNotFoundError("Artifact is not published")
    return ResolvedReport(published_json, published_key, report_id)


def _publisher_http_detail(exc: Exception) -> str:
    """Best-effort human message from an upstream HTTPError JSON body."""
    body, _code, error = _read_http_error_body(exc)
    return error or body or str(exc)


def list_versions(raw_path: str, *, scope: TenantScope | None) -> dict:
    """List the publish history (versions) of a live artifact.

    Returns {reportId, currentMd5, artifactType, versions: [...]} with versions
    newest-first, each tagged `isCurrent`.
    """
    settings = get_user_settings()
    publish_url, api_key = _resolve_publish_endpoint(settings)
    if not api_key:
        raise ValueError("Configure your provider API key in Settings to view versions")

    report_id = _resolve_report_id(raw_path, scope=scope).report_id

    try:
        from anton.publisher import list_versions as _list_versions
    except Exception as exc:
        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
    from urllib.error import HTTPError
    try:
        data = _list_versions(report_id, api_key=api_key, publish_url=publish_url, ssl_verify=ssl_verify)
    except HTTPError as exc:
        detail = _publisher_http_detail(exc)
        if exc.code == 404:
            raise FileNotFoundError(detail or "Report not found")
        raise RuntimeError(detail or "Could not fetch versions")
    except Exception as exc:
        logger.exception("Listing versions failed (report_id=%s)", report_id)
        raise RuntimeError("Could not fetch versions") from exc

    current = data.get("current_md5") or ""
    versions = [
        {
            "md5": v.get("md5", ""),
            "publishedAt": v.get("published_at", ""),
            "title": v.get("title", ""),
            "isCurrent": v.get("md5") == current,
        }
        for v in (data.get("versions") or [])
    ]
    versions.reverse()  # newest publish first
    return {
        "reportId": data.get("report_id", report_id),
        "currentMd5": current,
        "artifactType": data.get("artifact_type"),
        "versions": versions,
    }


def activate_version(raw_path: str, md5: str, *, scope: TenantScope | None) -> dict:
    """Roll the live URL back to an existing version (flips current_md5).

    On success, rewrites `.published.json` so the `modified` badge reflects the
    new reality: `last_md5` becomes the now-live version, and `published_mtime`
    is invalidated (the live content no longer matches the on-disk workspace, so
    the cheap mtime gate must fall through to the exact md5 comparison).
    """
    md5 = (md5 or "").strip()
    if not md5:
        raise ValueError("Missing target version (md5)")

    settings = get_user_settings()
    publish_url, api_key = _resolve_publish_endpoint(settings)
    if not api_key:
        raise ValueError("Configure your provider API key in Settings before rolling back")

    report = _resolve_report_id(raw_path, scope=scope)
    published_json, published_key, report_id = report.published_json, report.published_key, report.report_id

    try:
        from anton.publisher import activate_version as _activate_version
    except Exception as exc:
        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    ssl_verify = os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"
    from urllib.error import HTTPError
    try:
        result = _activate_version(report_id, md5, api_key=api_key, publish_url=publish_url, ssl_verify=ssl_verify)
    except HTTPError as exc:
        detail = _publisher_http_detail(exc)
        # 404 (unknown/pruned version), 409 (fullstack not supported), 400 (bad
        # request) are all caller-correctable → surface the message verbatim.
        if exc.code in (400, 404, 409):
            raise ValueError(detail or "Could not roll back to that version")
        raise RuntimeError(detail or "Roll back failed")
    except Exception as exc:
        logger.exception("Activate version failed (report_id=%s md5=%s)", report_id, md5)
        raise RuntimeError("Roll back failed") from exc

    # Reflect the rollback locally so the `modified` badge recomputes correctly.
    try:
        fresh_map = json.loads(published_json.read_text(encoding="utf-8"))
        fresh_entry = fresh_map.get(published_key)
        if isinstance(fresh_entry, dict):
            fresh_entry["last_md5"] = md5
            fresh_entry["published_mtime"] = 0  # force exact md5 check
            fresh_map[published_key] = fresh_entry
            published_json.write_text(json.dumps(fresh_map, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    return {
        "status": "ok",
        "currentMd5": result.get("current_md5", md5),
        "url": result.get("view_url", ""),
        "unchanged": bool(result.get("unchanged")),
    }


def published_state(raw_path: str, *, scope: TenantScope | None) -> dict:
    """Owner-side publish state for an artifact path, resolved exactly the way
    `publish_artifact` resolves it (so the chat tool and the GUI never disagree
    on where `.published.json` lives).

    Returns ``{"report_id", "url", "published"}``. `url` is blank unless the
    record is currently live (`published is True` and a url exists). `report_id`
    is returned even for soft-deleted records so a re-publish can reuse it.
    """
    blank = {"report_id": "", "url": "", "published": False}
    # resolve_artifact_path raises (not returns None) for paths outside a known
    # artifacts dir, so guard the whole resolution — the documented contract is
    # to return the blank default for any unresolvable path, never to raise.
    try:
        artifact = resolve_artifact_path(raw_path, scope=scope, allow_dir=True)
    except Exception:
        return dict(blank)
    if artifact is None:
        return dict(blank)
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=_scan_artifact_dirs(scope=scope)
    )
    entry = _load_published_map(published_dir).get(published_key)
    if not isinstance(entry, dict):
        return dict(blank)
    live = bool(entry.get("published", True)) and bool(entry.get("url"))
    return {
        "report_id": str(entry.get("report_id") or ""),
        "url": str(entry.get("url") or "") if live else "",
        "published": live,
    }


def published_owner_state(raw_path: str, *, scope: TenantScope | None) -> dict:
    """Raw owner-side `.published.json` entry, resolved exactly like
    `publish_artifact`. Returns {} for any unresolvable/absent record. Unlike
    `published_state`, exposes the access fields (mode/access_password/emails/
    org_allowed) needed to preserve access on re-publish."""
    try:
        artifact = resolve_artifact_path(raw_path, scope=scope, allow_dir=True)
    except Exception:
        return {}
    if artifact is None:
        return {}
    _t, published_dir, published_key, _fs = _resolve_publish_target(
        artifact, container_dirs=_scan_artifact_dirs(scope=scope)
    )
    entry = _load_published_map(published_dir).get(published_key)
    return entry if isinstance(entry, dict) else {}
