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
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import SecretStr

from cowork.common.paths import cowork_home, pod_local_only
# Imported for its side effect on the module namespace as well as its use:
# tests monkeypatch `publish.get_app_settings`, so removing it because a
# linter sees no local call breaks them. noqa keeps that from recurring.
from cowork.common.settings.app_settings import get_app_settings  # noqa: F401

from cowork.services.connectors.persist import vault_for_scope
from cowork.services.providers import publish_url_for_endpoint
from cowork.common.settings.user_settings import Provider, get_user_settings, provider_api_key
from anton.minds_client import describe_minds_connection_error
from anton.publish_access import access_from_owner_side
from anton.publish_access import normalize_emails as _normalize_emails
from anton.publish_access import resolve_access as _resolve_access
from anton.publish_access import resolve_publish_target as _anton_resolve_publish_target
from cowork.services.artifacts import (
    _content_mtime,
    _load_published_map,
    _scan_artifact_dirs,
    html_artifacts,
    resolve_artifact_path,
)

# Type-only, matching connectors/persist.py: cowork.db.scoped drags in the
# session factory and the FastAPI dependency graph, and this module is imported
# from the harness turn path where that is dead weight.
if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

logger = logging.getLogger(__name__)


class PublisherUnavailable(RuntimeError):
    """A local publish dependency (anton.publisher, markdown) failed to import.

    Distinct from RuntimeError so the endpoint layer can map it to 503
    without parsing message text — see the "unavailable" substring sentinel
    this replaced, which broke once upstream error text could itself contain
    that word (e.g. an HTTP 503 reason phrase or a timeout advice string).
    """


def _cowork_state_dir() -> Path:
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
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return _cowork_state_dir() / "state.json"


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
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


def desktop_artifact_and_base(raw_path: str) -> tuple[Path, Path]:
    """(artifact, artifacts_base) for a desktop request path.

    Split from credential resolution so callers can validate cheap, local things
    first: `update_artifact` must still answer "no published version to update"
    for an unpublished artifact rather than "configure your API key".

    The loop only IDENTIFIES which root the artifact sits under;
    `resolve_artifact_path` has already rejected anything outside them.
    """
    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    if artifact is None:
        raise FileNotFoundError("Artifact not found")
    for root in _scan_artifact_dirs():
        try:
            artifact.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        return artifact, root
    raise FileNotFoundError("Artifact is not in a known artifacts directory")


def desktop_publish_credential() -> tuple[str, str]:
    """(api_key, publish_url) from the active provider, for the desktop path."""
    publish_url, api_key = _resolve_publish_endpoint(get_user_settings())
    if not api_key:
        raise ValueError("Configure your provider API key in Settings before publishing")
    return api_key, publish_url


def desktop_publish_context(raw_path: str) -> tuple[Path, Path, str, str]:
    """(artifact, artifacts_base, api_key, publish_url) for the desktop path.

    Keeps the pre-existing behavior - path resolution against the registered
    artifact roots, credential resolution from the active provider - in one
    place, so the publish functions themselves stay free of both. The org path
    supplies these four values from the DB and a minted turn key instead.
    """
    artifact, root = desktop_artifact_and_base(raw_path)
    api_key, publish_url = desktop_publish_credential()
    return artifact, root, api_key, publish_url


def _resolve_publish_target(
    artifact: Path, container_dirs: list[Path] | None = None
) -> tuple[Path, Path, str, bool]:
    """Decide what to publish + where `.published.json` lives + its key.

    Thin wrapper over the single source of truth in anton
    (`anton.publish_access.resolve_publish_target`); cowork supplies its own
    container dirs so the metadata-climb is bounded to `.anton/artifacts/` roots.

    `container_dirs` MUST be passed on the org path: the default
    (`_scan_artifact_dirs()`) only sees the desktop layout, because org projects
    live one level deeper under `<root>/<org_id>/`, and it would return nothing
    there. Omitting it keeps the desktop behavior unchanged.

    Returns (publish_target, published_dir, published_key, is_fullstack).
    """
    return _anton_resolve_publish_target(
        artifact, container_dirs if container_dirs is not None else _scan_artifact_dirs()
    )


def _resolve_publish_endpoint(settings) -> tuple[str, str]:
    """Resolve the (publish base URL, api key) for the active provider's env.

    Publishing follows the env the provider points at: a custom OpenAI-compatible
    MindsHub endpoint (dev/staging) wins over the default minds_url (prod) and
    authenticates with that provider's own key — so pointing the provider at
    dev/staging publishes there too. api key is "" when the chosen provider has none.

    Resolution order for the publish base URL (first non-empty wins):
      1. ``ANTON_PUBLISH_URL`` env var — an operator/dev override that trumps
         the stored setting (the DB row would otherwise shadow env; see
         SettingService, which passes DB rows as init kwargs);
      2. the ``publish_url`` setting;
      3. the host derived from the active provider endpoint.
    """
    oai_host = (urlparse(settings.openai_base_url or "").hostname or "").lower()
    if oai_host.startswith("api") and oai_host.endswith(".mindshub.ai"):
        endpoint = settings.openai_base_url
        api_key = _secret_str(provider_api_key(settings, Provider.OPENAI_COMPATIBLE))
    else:
        endpoint, api_key = settings.minds_url, _secret_str(settings.minds_api_key)
    env_publish_url = os.environ.get("ANTON_PUBLISH_URL", "").strip()
    publish_url = env_publish_url or settings.publish_url or publish_url_for_endpoint(endpoint)
    return publish_url, api_key


def list_publishable() -> dict:
    settings = get_user_settings()
    state = _load_state()
    publish_url, api_key = _resolve_publish_endpoint(settings)
    return {
        "artifacts": html_artifacts(),
        "publishReady": bool(api_key),
        "publishUrl": publish_url,
        "history": state.get("publish_history", [])[:40],
    }


# _normalize_emails / _resolve_access now live in anton.publish_access (single
# source of truth) and are imported at the top of this module.


# Static artifact extensions a user can publish to a MindsHub web page.
# `.html` is served as-is; `.md` is rendered to a styled HTML page first
# (see `_render_markdown_to_html`). Fullstack artifacts bypass this — they
# publish their directory regardless of the primary file's suffix.
PUBLISHABLE_STATIC_SUFFIXES = (".html", ".md")

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
    # `markdown` ships transitively via hermes-agent (a pinned core
    # dependency), so it's always present in the resolved environment. The
    # guard stays defensive in case that ever changes; promote markdown to a
    # direct dependency in pyproject.toml when the lockfile is next
    # regenerated with the canonical uv version.
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


def publish_artifact(
    artifact: Path,
    *,
    artifacts_base: Path,
    api_key: str,
    publish_url: str,
    password: str | None = None,
    access: dict | None = None,
    scope: TenantScope | None = None,
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
    """
    if not api_key:
        raise ValueError("Publishing requires an API key")

    publish_target, published_dir, published_key, is_fullstack = _resolve_publish_target(
        artifact, container_dirs=[artifacts_base]
    )
    if not is_fullstack and publish_target.suffix.lower() not in PUBLISHABLE_STATIC_SUFFIXES:
        raise ValueError("Only HTML and Markdown artifacts can be published")

    try:
        from anton.publisher import publish
    except Exception as exc:
        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    published_json = published_dir / ".published.json"
    published_map: dict[str, Any] = {}
    if published_json.is_file():
        try:
            published_map = json.loads(published_json.read_text(encoding="utf-8"))
        except Exception:
            published_map = {}
    previous = published_map.get(published_key)
    report_id = previous.get("report_id") if isinstance(previous, dict) else None

    # Resolve the effective access (mode + version) from the request and the
    # prior publish. Versions bump only when the password / restricted list
    # changed so previously issued viewer grants invalidate.
    effective_access, pwd_version, access_version, owner_side = _resolve_access(password, access, previous)

    # One identity spans private drafts, published versions, and comments. A
    # legacy loose file has no metadata and keeps the service-generated key.
    canonical_artifact_key: str | None = None
    if (published_dir / "metadata.json").is_file():
        from cowork.services.artifact_identity import artifact_key, ensure_full_id

        artifact_id, _metadata = ensure_full_id(published_dir)
        canonical_artifact_key = artifact_key(artifact_id)

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
            access=effective_access,
            access_version=access_version,
            pwd_version=pwd_version,
            artifact_key=canonical_artifact_key,
            # Resolve datasource secrets from cowork's own vault
            # (`~/.cowork/data-vault`), not anton's default
            # (`~/.anton/data_vault`) — otherwise secrets are missed and
            # the published artifact has no DB connection in the cloud.
            # Org-keyed: the persisted vault is per organization, and an
            # unscoped lookup would resolve to the shared namespace root.
            vault=vault_for_scope(scope),
        )
    except Exception as exc:
        logger.exception("Publishing failed")
        # Only network/HTTP failures get the "Connection failed" framing — a
        # gateway timeout (e.g. a fullstack artifact whose deps take too long
        # to install remotely, ENG-1547/ENG-1580) or a server-side 5xx reads
        # very differently to the user than an auth rejection, but neither
        # applies to a local failure (e.g. reading the artifact's files to
        # zip it) that happened before any request went out. urllib.error
        # HTTPError is a URLError subclass, so this covers both.
        if isinstance(exc, urllib.error.URLError):
            headline, advice = describe_minds_connection_error(exc)
            raise RuntimeError(f"Publishing failed. {headline} {advice}") from exc
        raise RuntimeError(f"Publishing failed: {exc}") from exc
    finally:
        if md_tmp_dir is not None:
            md_tmp_dir.cleanup()

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
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
        # `emails`/`org_allowed`/`access_version` for restricted), and always
        # `requires_password` for back-compat with older readers.
        entry: dict[str, Any] = {
            "report_id": returned_report_id,
            "url": view_url,
            # Composite comments scope {user_dir}/{report_id} (Plan 4/5); persisted
            # so the comments panel can key threads after an app restart.
            "artifact_key": result.get("artifact_key") or canonical_artifact_key or "",
            # The lambda's raw echo, or "" from an older lambda that sends none.
            # Decides which scope the published page uses for comments; see
            # services/comments_scope. Deliberately NOT merged with the field
            # above, whose canonical fallback would read as an echo.
            "lambda_artifact_key": _lambda_artifact_key(result),
            "last_md5": result.get("md5", ""),
            # Snapshot of the artifact's content mtime at publish time — the
            # cheap gate for the `modified` badge (see card_for_folder). Uses
            # the SAME basis as the card's `mtime` so the comparison is exact.
            "published_mtime": _content_mtime(published_dir),
            "published": True,
            **owner_side,
        }
        published_map[published_key] = entry
        _write_published_map(published_json, published_map)
        state = _load_state()
        state["publish_history"] = [history_item, *state.get("publish_history", [])][:100]
        _save_state(state)

    return {
        "status": "ok",
        "url": view_url,
        "accessMode": owner_side.get("mode", "public"),
        "accessProtected": bool(owner_side.get("requires_password")),
        "accessEmails": owner_side.get("emails", []),
        "orgAllowed": bool(owner_side.get("org_allowed")),
        "ownerOnly": bool(owner_side.get("owner_only")),
        # Composite comments scope for the panel (Plan 5).
        "artifactKey": result.get("artifact_key") or canonical_artifact_key or "",
        "result": {k: v for k, v in result.items() if k != "file_payload"},
    }


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
    if not is_fullstack and publish_target.suffix.lower() not in PUBLISHABLE_STATIC_SUFFIXES:
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
    artifact: Path, *, artifacts_base: Path, api_key: str, publish_url: str
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


def update_artifact(raw_path: str) -> dict:
    """Re-publish an already-published artifact, preserving its URL and access.

    `publish_artifact` reuses the stored report_id (→ the lambda mints a new
    version at the same URL), but it would reset access to public if handed no
    access object. So reconstruct the prior access from `.published.json` and
    pass it through. After republish, refresh `published_mtime` so the badge
    clears. Raises FileNotFoundError when there's nothing published to update.
    """
    # Path first, credential later: an artifact with nothing published must
    # report that, not "configure your API key".
    artifact, artifacts_base = desktop_artifact_and_base(raw_path)
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

    # Rebuild the cowork→publish access shape from the stored owner-side state
    # (single source of truth in anton.publish_access).
    access = access_from_owner_side(entry)

    # Delegates: reuses report_id (read from .published.json) + refreshes last_md5.
    api_key, publish_url = desktop_publish_credential()
    result = publish_artifact(
        artifact, artifacts_base=artifacts_base, api_key=api_key,
        publish_url=publish_url, access=access,
    )

    # Refresh the mtime snapshot so the cheap gate clears the `modified` badge.
    try:
        fresh_map = json.loads(published_json.read_text(encoding="utf-8"))
        fresh_entry = fresh_map.get(published_key)
        if isinstance(fresh_entry, dict):
            fresh_entry["published_mtime"] = _content_mtime(published_dir)
            fresh_map[published_key] = fresh_entry
            _write_published_map(published_json, fresh_map)
    except Exception:
        pass

    return result


def _resolve_report_id(raw_path: str) -> tuple[Path, str, str]:
    """Resolve (published_json_path, published_key, report_id) for an artifact.

    Raises FileNotFoundError when the artifact has no live publish record.
    """
    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(artifact)
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
    return published_json, published_key, report_id


def _publisher_http_detail(exc: Exception) -> str:
    """Best-effort human message from an upstream HTTPError JSON body."""
    body = ""
    try:
        body = exc.read().decode()  # type: ignore[attr-defined]
        return json.loads(body).get("error") or body or str(exc)
    except Exception:
        return body or str(exc)


def list_versions(raw_path: str) -> dict:
    """List the publish history (versions) of a live artifact.

    Returns {reportId, currentMd5, artifactType, versions: [...]} with versions
    newest-first, each tagged `isCurrent`.
    """
    settings = get_user_settings()
    publish_url, api_key = _resolve_publish_endpoint(settings)
    if not api_key:
        raise ValueError("Configure your provider API key in Settings to view versions")

    _published_json, _key, report_id = _resolve_report_id(raw_path)

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


def activate_version(raw_path: str, md5: str) -> dict:
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

    published_json, published_key, report_id = _resolve_report_id(raw_path)

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


def published_state(raw_path: str) -> dict:
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
        artifact = resolve_artifact_path(raw_path, allow_dir=True)
    except Exception:
        return dict(blank)
    if artifact is None:
        return dict(blank)
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(artifact)
    entry = _load_published_map(published_dir).get(published_key)
    if not isinstance(entry, dict):
        return dict(blank)
    live = bool(entry.get("published", True)) and bool(entry.get("url"))
    return {
        "report_id": str(entry.get("report_id") or ""),
        "url": str(entry.get("url") or "") if live else "",
        "published": live,
    }


def published_owner_state(raw_path: str) -> dict:
    """Raw owner-side `.published.json` entry, resolved exactly like
    `publish_artifact`. Returns {} for any unresolvable/absent record. Unlike
    `published_state`, exposes the access fields (mode/access_password/emails/
    org_allowed) needed to preserve access on re-publish."""
    try:
        artifact = resolve_artifact_path(raw_path, allow_dir=True)
    except Exception:
        return {}
    if artifact is None:
        return {}
    _t, published_dir, published_key, _fs = _resolve_publish_target(artifact)
    entry = _load_published_map(published_dir).get(published_key)
    return entry if isinstance(entry, dict) else {}
