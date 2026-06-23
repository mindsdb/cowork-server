"""Publish service — publish HTML artifacts to 4nton.ai.

Ported from cowork/server/routes/utilities.py (publish section).
Uses a local JSON state file for publish history tracking.
"""
from __future__ import annotations

import html as _html
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings
from cowork.services.artifacts import (
    _artifact_root_for,
    _fullstack_types,
    _load_metadata,
    _load_published_map,
    _pick_primary,
    _user_files,
    html_artifacts,
    resolve_artifact_path,
)

logger = logging.getLogger(__name__)


def _cowork_state_dir() -> Path:
    base = os.environ.get("ANTON_COWORK_STATE_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        path = Path.home() / ".anton" / "cowork"
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


def _write_publish_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            tmp = Path(handle.name)
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception as exc:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise RuntimeError("Could not persist publish record") from exc


def _secret_str(val: SecretStr | str | None) -> str:
    """Unwrap a SecretStr (or plain string) to a plain string, defaulting to ''."""
    if val is None:
        return ""
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    return str(val)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_publish_target(artifact: Path) -> tuple[Path, Path, str, bool]:
    """Decide what to publish given a resolved artifact path (file OR dir).

    Folder-based artifacts are addressed by their folder; legacy loose-HTML
    (and chat-bubble / Utilities-list) artifacts by their file. In both cases:
      - fullstack (metadata.json type ∈ FULLSTACK_ARTIFACT_TYPES) → publish
        the artifact *directory* (anton bundles backend.py + static/ +
        requirements.txt only when handed a dir), `.published.json` at root;
      - static → publish the single primary *file* (anton's `_zip_html`
        renames it to index.html + pulls referenced siblings; handing it a
        dir would over-bundle metadata.json/data and skip the rename),
        `.published.json` in that file's parent.
    The map is always keyed by the primary file name — matching how
    `_published_url_for` / `list_artifacts` read it back.

    Returns (publish_target, published_dir, published_key, is_fullstack).
    """
    if artifact.is_dir():
        # Folder addressed directly — its primary file lives inside.
        artifact_root = artifact
        meta = _load_metadata(artifact_root) or {}
        primary = _pick_primary(artifact_root, _user_files(artifact_root), primary_hint=meta.get("primary"))
    else:
        # File addressed directly — it *is* the primary; climb to its root.
        artifact_root = _artifact_root_for(artifact)
        meta = _load_metadata(artifact_root) if (artifact_root / "metadata.json").is_file() else None
        primary = artifact

    if (meta or {}).get("type") in _fullstack_types():
        key = primary.name if primary else "index.html"
        return artifact_root, artifact_root, key, True
    if primary:
        return primary, primary.parent, primary.name, False
    return artifact_root, artifact_root, "index.html", False


def list_publishable() -> dict:
    settings = get_user_settings()
    state = _load_state()
    return {
        "artifacts": html_artifacts(),
        "publishReady": bool(_secret_str(settings.minds_api_key)),
        "publishUrl": settings.publish_url or "https://4nton.ai",
        "history": state.get("publish_history", [])[:40],
    }


def _normalize_emails(values) -> list[str]:
    """Strip + lowercase + de-dupe, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values or []:
        email = str(raw).strip().lower()
        if email and email not in seen:
            seen.add(email)
            out.append(email)
    return out


def _resolve_access(
    password: str | None, access: dict | None, previous: Any
) -> tuple[dict, int, int, dict]:
    """Resolve the effective publish access from the request + prior state.

    Returns ``(effective_access, pwd_version, access_version, owner_side)``:

    - ``effective_access`` — the cowork→anton shape passed to ``publish()``
      (``{"mode": "public"}`` / ``{"mode": "password", "password": ...}`` /
      ``{"mode": "restricted", "emails": [...], "org_allowed": bool}``).
    - ``pwd_version`` / ``access_version`` — monotonic versions, bumped only
      when the password (resp. restricted list/org) actually changes vs the
      previous publish, so stale viewer grants invalidate (mirrors the prior
      ``pwd_version`` logic).
    - ``owner_side`` — fields to persist in ``.published.json`` (kept
      back-compatible: ``requires_password`` is always present).

    A request with no usable selection (empty password, or restricted with no
    emails and no org) degrades to ``public`` rather than publishing an
    artifact nobody can open.
    """
    prev = previous if isinstance(previous, dict) else {}
    password = (password or "").strip() or None

    mode = (access or {}).get("mode") if access else None
    if not mode:
        mode = "password" if password else "public"

    prev_pwd_version = prev.get("pwd_version", 0) or 0
    prev_access_version = prev.get("access_version", 0) or 0
    pwd_version = prev_pwd_version or 1
    access_version = prev_access_version or 1

    if mode == "password":
        pw = ((access or {}).get("password") or password or "").strip() or None
        if pw:
            prev_password = prev.get("access_password")
            pwd_version = (prev_pwd_version + 1) if pw != prev_password else (prev_pwd_version or 1)
            owner_side = {
                "mode": "password",
                "requires_password": True,
                "access_password": pw,
                "pwd_version": pwd_version,
            }
            return {"mode": "password", "password": pw}, pwd_version, access_version, owner_side
        mode = "public"  # empty password → public

    if mode == "restricted":
        emails = _normalize_emails((access or {}).get("emails"))
        org_allowed = bool((access or {}).get("org_allowed"))
        if emails or org_allowed:
            prev_restricted = prev.get("mode") == "restricted"
            prev_emails = prev.get("emails") if prev_restricted else None
            prev_org = prev.get("org_allowed") if prev_restricted else None
            changed = (emails != prev_emails) or (org_allowed != prev_org)
            access_version = (prev_access_version + 1) if changed else (prev_access_version or 1)
            owner_side = {
                "mode": "restricted",
                "requires_password": False,
                "emails": emails,
                "org_allowed": org_allowed,
                "access_version": access_version,
            }
            return (
                {"mode": "restricted", "emails": emails, "org_allowed": org_allowed},
                pwd_version,
                access_version,
                owner_side,
            )
        mode = "public"  # nothing selected → public

    return {"mode": "public"}, pwd_version, access_version, {"mode": "public", "requires_password": False}


# Static artifact extensions a user can publish to a 4nton.ai web page.
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
        raise RuntimeError("Markdown renderer is unavailable") from exc

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
    raw_path: str,
    password: str | None = None,
    access: dict | None = None,
    version_metadata: dict[str, Any] | None = None,
    publish_source_path: str | None = None,
) -> dict:
    settings = get_user_settings()
    api_key = _secret_str(settings.minds_api_key)
    if not api_key:
        raise ValueError("Configure your Minds API key in Settings before publishing")

    # The request path is either the artifact folder (folder-based
    # artifacts) or a single file (legacy loose-HTML / chat-bubble / the
    # Utilities per-page list). `_resolve_publish_target` normalizes both.
    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    live_publish_target, published_dir, published_key, _live_is_fullstack = _resolve_publish_target(artifact)
    source_artifact = (
        Path(publish_source_path).expanduser().resolve(strict=False)
        if publish_source_path
        else artifact
    )
    publish_target, _source_published_dir, _source_published_key, is_fullstack = _resolve_publish_target(source_artifact)
    if not is_fullstack and publish_target.suffix.lower() not in PUBLISHABLE_STATIC_SUFFIXES:
        raise ValueError("Only HTML and Markdown artifacts can be published")

    try:
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.publisher import publish
    except Exception as exc:
        raise RuntimeError("Anton publisher is unavailable") from exc

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

    # Markdown is rendered to a throwaway index.html that we hand to the
    # publisher; `.html` and fullstack publish their real target directly.
    # `publish_target` stays the original artifact so the registry, history,
    # and unpublish all key off the file the user actually sees.
    publish_source = publish_target
    md_tmp_dir: tempfile.TemporaryDirectory | None = None
    if not is_fullstack and publish_target.suffix.lower() == ".md":
        md_tmp_dir = tempfile.TemporaryDirectory(prefix="cowork-md-publish-")
        publish_source = _render_markdown_to_html(publish_target, Path(md_tmp_dir.name))

    publish_url = settings.publish_url or "https://4nton.ai"
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
            # Resolve datasource secrets from cowork's own vault
            # (`~/.cowork/data-vault`), not anton's default
            # (`~/.anton/data_vault`) — otherwise secrets are missed and
            # the published artifact has no DB connection in the cloud.
            vault=LocalDataVault(Path(get_app_settings().connector.vault_dir)),
        )
    except Exception as exc:
        logger.exception("Publishing failed")
        raise RuntimeError("Publishing failed. Check your Minds credentials and try again.") from exc
    finally:
        if md_tmp_dir is not None:
            md_tmp_dir.cleanup()

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
    if returned_report_id:
        history_item = {
            "artifact": str(live_publish_target),
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
            "last_md5": result.get("md5", ""),
            "published": True,
            **owner_side,
        }
        if isinstance(version_metadata, dict):
            version_id = version_metadata.get("id") or version_metadata.get("versionId")
            if version_id:
                entry["version_id"] = str(version_id)
            artifact_id = version_metadata.get("artifact_id") or version_metadata.get("artifactId")
            if artifact_id:
                entry["artifact_id"] = str(artifact_id)
            files_hash = version_metadata.get("files_hash") or version_metadata.get("filesHash")
            if files_hash:
                entry["files_hash"] = str(files_hash)
            manifest_hash = version_metadata.get("manifest_hash") or version_metadata.get("manifestHash")
            if manifest_hash:
                entry["manifest_hash"] = str(manifest_hash)
            version_number = version_metadata.get("version_number") or version_metadata.get("versionNumber")
            if version_number:
                entry["version_number"] = version_number
        published_map[published_key] = entry
        _write_publish_record(published_json, published_map)
        state = _load_state()
        state["publish_history"] = [history_item, *state.get("publish_history", [])][:100]
        _save_state(state)

    return {
        "status": "ok",
        "url": view_url,
        "publishedVersionId": str((version_metadata or {}).get("id") or (version_metadata or {}).get("versionId") or ""),
        "publishedFilesHash": str((version_metadata or {}).get("files_hash") or (version_metadata or {}).get("filesHash") or ""),
        "publishedManifestHash": str((version_metadata or {}).get("manifest_hash") or (version_metadata or {}).get("manifestHash") or ""),
        "publishedVersionNumber": (version_metadata or {}).get("version_number") or (version_metadata or {}).get("versionNumber"),
        "accessMode": owner_side.get("mode", "public"),
        "accessProtected": bool(owner_side.get("requires_password")),
        "accessEmails": owner_side.get("emails", []),
        "orgAllowed": bool(owner_side.get("org_allowed")),
        "result": {k: v for k, v in result.items() if k != "file_payload"},
    }


def unpublish_artifact(raw_path: str) -> dict:
    settings = get_user_settings()
    api_key = _secret_str(settings.minds_api_key)
    if not api_key:
        raise ValueError("Configure your Minds API key in Settings before unpublishing")

    artifact = resolve_artifact_path(raw_path, allow_dir=True)
    # Mirror publish: resolve the same .published.json location + key
    # (primary file name) whether a folder or a file was passed.
    _publish_target, published_dir, published_key, _is_fullstack = _resolve_publish_target(artifact)
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
        raise RuntimeError("Anton publisher is unavailable") from exc

    publish_url = settings.publish_url or "https://4nton.ai"
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
            pass  # Already gone upstream — clear local record below
        else:
            logger.exception("Unpublishing failed (identifier=%s)", identifier)
            raise RuntimeError(f"Unpublishing failed: {msg}") from exc

    entry_snapshot = dict(entry) if isinstance(entry, dict) else {}
    # Soft-delete: keep report_id (and url) so a later re-publish reuses the
    # same public URL. Only flip `published` off so readers stop showing it as
    # live. The backend object is gone, but lambda re-mints at the same id when
    # we resend report_id on the next publish.
    if isinstance(entry, dict):
        entry["published"] = False
        published_map[published_key] = entry
        _write_publish_record(published_json, published_map)
    return {
        "status": "ok",
        "publishedUrl": entry_snapshot.get("url") or "",
        "publishedVersionId": entry_snapshot.get("version_id") or "",
        "publishedFilesHash": entry_snapshot.get("files_hash") or "",
        "publishedManifestHash": entry_snapshot.get("manifest_hash") or "",
        "publishedVersionNumber": entry_snapshot.get("version_number"),
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
