"""View statistics of a published artifact, read from the publishing service.

The stats routes of the publishing service accept the same credential a
publish uses: the provider key on desktop, a per-request publish key in org
mode. The report id comes from the artifact's `.published.json` record.
"""
from __future__ import annotations

import http.client
import json
import re
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from fastapi import HTTPException

from cowork.common.minds_tls import minds_ssl_verify
from cowork.services import artifacts as _artifacts
from cowork.services.product_permissions import ProductPermissionDenied
from cowork.services.publish import PublisherUnavailable, _resolve_publish_target

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

NOT_PUBLISHED_MESSAGE = "Artifact is not published"
PUBLISH_CREDENTIAL_MISSING_CODE = "publish_credential_missing"
PUBLISH_CREDENTIAL_REJECTED_CODE = "publish_credential_rejected"

_REPORT_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_TIMEOUT_SECONDS = 15


class StatsError(Exception):
    """A stats failure with its own response; see `stats_http_exception`."""


class StatsOwnedByOtherAccount(StatsError):
    """The service has no such report under the caller's account."""


class StatsUnavailable(StatsError):
    """The artifact is published, but the service keeps no stats for it."""


class StatsForbidden(StatsError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class StatsUpgradeRequired(StatsError):
    """The service answered 402: a plan upgrade or credits are needed.

    The body passes through with its code, e.g. `upgrade_required`, or
    `credits_required` when turning viewer analytics on with no credits.
    """

    def __init__(self, body: Any) -> None:
        super().__init__("Upgrade required")
        self.body = body


class StatsRejected(StatsError):
    """The service answered 422 with a code, e.g. `tracking_off` when asking
    for viewer emails while analytics are off. The body passes through."""

    def __init__(self, body: Any) -> None:
        super().__init__("Stats request rejected")
        self.body = body


class StatsNotFound(StatsError):
    """The service answered 404 for something other than the report, e.g. an
    unknown viewer on the viewer detail route."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PublishCredentialRejected(StatsError):
    """The service answered 401 to the publish credential."""


class StatsTemporarilyUnavailable(StatsError):
    """The service answered 503: its auth, DynamoDB or S3 dependency failed."""


class TrackingConflict(StatsError):
    """A publish kept rewriting the artifact's metadata during the tracking write."""


STATS_ROUTE_ERRORS = (StatsError, FileNotFoundError, ValueError, RuntimeError)


def _stats_url(publish_url: str, report_id: str) -> str:
    # Under the artifact's own route, which every host already sends to the
    # publishing service, so stats need no routing of their own (ENG-4455).
    # The service still answers /stats/{id}.
    return f"{publish_url.rstrip('/')}/a/{quote(report_id, safe='')}/_stats"


def _json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _request(method: str, url: str, *, key: str, json_body: dict | None = None) -> Any:
    """The parsed JSON body of a successful reply, or None if it is not JSON."""
    try:
        from anton.minds_client import minds_request
    except ImportError as exc:
        raise PublisherUnavailable("Anton publisher is unavailable") from exc

    payload = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    try:
        raw = minds_request(
            url, key, method=method, payload=payload, verify=minds_ssl_verify(), timeout=_TIMEOUT_SECONDS,
        )
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read(65_536)
        except OSError:
            error_body = b""
        _raise_for_status(exc.code, _json_body(error_body))
        raise RuntimeError(f"Stats request failed (HTTP {exc.code})") from exc
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise RuntimeError("Could not reach the stats service") from exc
    return _json_body(raw)


def _raise_for_status(status: int, body: Any) -> None:
    code = body.get("code") if isinstance(body, dict) else None
    error = body.get("error") if isinstance(body, dict) else None
    message = error if isinstance(error, str) else ""
    if status == 404 and code == "artifact_not_found":
        raise StatsOwnedByOtherAccount()
    if status == 404 and code == "stats_unavailable":
        raise StatsUnavailable()
    # A 404 the service answers itself carries a code or an "error"; a bare
    # {"message": "Not Found"} is the gateway's, for a route it lacks.
    if status == 404 and ((isinstance(code, str) and code) or message):
        raise StatsNotFound(code if isinstance(code, str) and code else "not_found", message or "Not found.")
    if status == 409 and code == "tracking_conflict":
        raise TrackingConflict()
    if status == 403:
        if code == "permission_denied":
            raise ProductPermissionDenied()
        raise StatsForbidden(
            code if isinstance(code, str) and code else "forbidden",
            message or "Access to stats was refused.",
        )
    if status == 402:
        raise StatsUpgradeRequired(body)
    if status == 422 and isinstance(code, str) and code:
        raise StatsRejected(body)
    if status == 401:
        raise PublishCredentialRejected()
    if status == 400:
        raise ValueError(message or "The stats service rejected the request.")
    if status == 503:
        raise StatsTemporarilyUnavailable()
    raise RuntimeError(f"Stats request failed (HTTP {status})")


def fetch_stats(*, publish_url: str, key: str, report_id: str) -> dict:
    """The service's stats JSON for one report, unchanged."""
    body = _request("GET", _stats_url(publish_url, report_id), key=key)
    if not isinstance(body, dict):
        raise RuntimeError("The stats service returned an unreadable response")
    return body


def set_tracking(
    *,
    publish_url: str,
    key: str,
    report_id: str,
    enabled: bool | None = None,
    require_email: bool | None = None,
) -> dict:
    """Change viewer analytics and the email gate; None leaves a setting as is.

    Returns the stored settings, `{"tracking": bool, "require_email": bool}`.
    The flag goes as both `tracking` (the current name) and `enabled` (the name
    services released before viewer analytics read).
    """
    if enabled is None and require_email is None:
        raise ValueError("Nothing to change")
    payload: dict[str, bool] = {}
    if enabled is not None:
        payload["tracking"] = enabled
        payload["enabled"] = enabled
    if require_email is not None:
        payload["require_email"] = require_email
    body = _request(
        "PUT",
        f"{_stats_url(publish_url, report_id)}/tracking",
        key=key,
        json_body=payload,
    )
    tracking = body.get("tracking") if isinstance(body, dict) else None
    if not isinstance(tracking, bool):
        raise RuntimeError("The stats service returned an unreadable response")
    stored_email = body.get("require_email")
    return {"tracking": tracking, "require_email": stored_email if isinstance(stored_email, bool) else False}


def fetch_viewer(*, publish_url: str, key: str, report_id: str, viewer_id: str) -> dict:
    """One viewer's visits with their per-section times, unchanged."""
    # Any other id is quoted into one path segment; a dot-only one would be
    # normalized out of the URL path.
    if not viewer_id or len(viewer_id) > 128 or not viewer_id.strip("."):
        raise StatsNotFound("viewer_not_found", "Viewer not found.")
    body = _request(
        "GET",
        f"{_stats_url(publish_url, report_id)}/viewers/{quote(viewer_id, safe='')}",
        key=key,
    )
    if not isinstance(body, dict):
        raise RuntimeError("The stats service returned an unreadable response")
    return body


def delete_stats(*, publish_url: str, key: str, report_id: str) -> None:
    """Purge every view, visit and viewer the service holds for the report."""
    _request("DELETE", _stats_url(publish_url, report_id), key=key)


def _report_id_from_record(published_dir: Path, published_key: str) -> str:
    published_json = published_dir / ".published.json"
    if not published_json.is_file():
        raise FileNotFoundError(NOT_PUBLISHED_MESSAGE)
    try:
        published_map = json.loads(published_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not read publish record") from exc
    entry = published_map.get(published_key) if isinstance(published_map, dict) else None
    if not isinstance(entry, dict) or not entry.get("published", True):
        raise FileNotFoundError(NOT_PUBLISHED_MESSAGE)
    report_id = str(entry.get("report_id") or "").strip()
    if report_id:
        # A dot-only id would be normalized out of the URL path by the HTTP client.
        if _REPORT_ID_RE.fullmatch(report_id) is None:
            raise FileNotFoundError(NOT_PUBLISHED_MESSAGE)
        return report_id
    # A publish whose upload reply carried no report id: unpublish still finds
    # it by md5, but the service files no stats under an md5.
    if entry.get("last_md5"):
        raise StatsUnavailable()
    raise FileNotFoundError(NOT_PUBLISHED_MESSAGE)


def report_id_for_path(raw_path: str, *, scope: TenantScope | None) -> str:
    """Desktop: the report id of a published artifact path inside ``scope``'s roots."""
    artifact = _artifacts.resolve_artifact_path(raw_path, scope=scope, allow_dir=True)
    if artifact is None:
        raise FileNotFoundError("Artifact not found")
    _target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        artifact, container_dirs=_artifacts._scan_artifact_dirs(scope=scope)
    )
    return _report_id_from_record(published_dir, published_key)


def report_id_for_folder(folder: Path) -> str:
    """Org: the report id of a published artifact folder."""
    _target, published_dir, published_key, _is_fullstack = _resolve_publish_target(
        folder, container_dirs=[folder.parent]
    )
    return _report_id_from_record(published_dir, published_key)


def stats_http_exception(exc: Exception, *, desktop: bool) -> HTTPException:
    """The response both route families give for a stats failure.

    A 401 carries the reconnect code only on desktop. There it means the
    user's own key expired or was revoked; an org key is minted for this one
    request, so the user has nothing to reconnect.
    """
    if isinstance(exc, StatsOwnedByOtherAccount):
        return HTTPException(409, detail={
            "code": "stats_owned_by_other_account",
            "message": "Stats are not available for this account.",
        })
    if isinstance(exc, StatsUnavailable):
        return HTTPException(404, detail={
            "code": "stats_unavailable",
            "message": "Stats are not available for this artifact.",
        })
    if isinstance(exc, StatsForbidden):
        return HTTPException(403, detail={"code": exc.code, "message": exc.message})
    if isinstance(exc, StatsUpgradeRequired):
        # The service puts its text in "error"; the renderer reads "message".
        body = dict(exc.body) if isinstance(exc.body, dict) else {}
        body.setdefault("code", "upgrade_required")
        message, error = body.get("message"), body.get("error")
        if not (isinstance(message, str) and message):
            body["message"] = error if isinstance(error, str) and error else "Upgrade required"
        return HTTPException(402, detail=body)
    if isinstance(exc, StatsRejected):
        body = dict(exc.body) if isinstance(exc.body, dict) else {}
        message, error = body.get("message"), body.get("error")
        if not (isinstance(message, str) and message):
            body["message"] = error if isinstance(error, str) and error else (
                "Turn on viewer analytics first." if body.get("code") == "tracking_off"
                else "The stats service rejected the request."
            )
        return HTTPException(422, detail=body)
    if isinstance(exc, StatsNotFound):
        return HTTPException(404, detail={"code": exc.code, "message": exc.message})
    if isinstance(exc, TrackingConflict):
        return HTTPException(409, detail={
            "code": "tracking_conflict",
            "message": "The artifact is being updated. Try again in a moment.",
        })
    if isinstance(exc, StatsTemporarilyUnavailable):
        return HTTPException(503, detail={
            "code": "stats_temporarily_unavailable",
            "message": "Stats are temporarily unavailable.",
        })
    if isinstance(exc, PublishCredentialRejected):
        if desktop:
            return HTTPException(502, detail={
                "code": PUBLISH_CREDENTIAL_REJECTED_CODE,
                "message": "Reconnect MindsHub to see stats.",
            })
        return HTTPException(502, detail="The publishing service rejected the stats credential.")
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, detail=str(exc) or NOT_PUBLISHED_MESSAGE)
    if isinstance(exc, ValueError):
        return HTTPException(400, detail=str(exc))
    if isinstance(exc, PublisherUnavailable):
        return HTTPException(503, detail=str(exc))
    return HTTPException(502, detail=str(exc) or "Stats request failed")
