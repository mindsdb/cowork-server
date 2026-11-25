"""A published full-stack artifact's stored data, read from the publishing service.

A stateful artifact stores what people do on its page (sign-ups, records,
form answers) in the publishing service's state storage, which until now only
the artifact's own backend could reach. The service's /a/{id}/_state routes
(ENG-4686) give the artifact's owner the same access, with the same credential
a publish and the stats routes use. The report id comes from the artifact's
`.published.json` record (artifact_stats.report_id_for_path).
"""
from __future__ import annotations

import http.client
import json
import urllib.error
from typing import Any
from urllib.parse import quote, urlencode

from cowork.common.minds_tls import minds_ssl_verify
from cowork.services.publish import PublisherUnavailable

_TIMEOUT_SECONDS = 20

# The broker ops the owner route accepts on POST; listing is its own GET.
OPS = ("get", "query", "put", "update", "delete")
WRITE_OPS = ("put", "update", "delete")


class ArtifactDataError(Exception):
    """A failure the agent can act on; the message says what happened."""


class NotOwnedOrMissing(ArtifactDataError):
    def __init__(self) -> None:
        super().__init__(
            "The publishing service has no such artifact under this account. "
            "Only the person who published it can read its data."
        )


class NoState(ArtifactDataError):
    def __init__(self) -> None:
        super().__init__(
            "This artifact keeps no stored data: it isn't a stateful full-stack app, "
            "so everything it shows comes from its code or its connected data sources."
        )


class Rejected(ArtifactDataError):
    """400: the request didn't fit the artifact's state schema or the broker's rules."""


class Conflict(ArtifactDataError):
    """409: a conditional write lost to a newer version of the item."""


class TemporarilyUnavailable(ArtifactDataError):
    def __init__(self, detail: str = "") -> None:
        super().__init__("The artifact's data storage is temporarily unavailable. Try again shortly." + (
            f" ({detail})" if detail else ""
        ))


class CredentialRejected(ArtifactDataError):
    def __init__(self) -> None:
        super().__init__("The publishing service rejected the MindsHub credential. Ask the user to sign in again.")


def _url(publish_url: str, report_id: str) -> str:
    # Under the artifact's own route, like the stats routes.
    return f"{publish_url.rstrip('/')}/a/{quote(report_id, safe='')}/_state"


def _json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _raise_for_status(status: int, body: Any) -> None:
    code = body.get("code") if isinstance(body, dict) else None
    error = body.get("error") if isinstance(body, dict) else None
    message = error if isinstance(error, str) and error else ""
    if status == 404 and code == "no_state":
        raise NoState()
    if status == 404:
        raise NotOwnedOrMissing()
    if status == 400:
        raise Rejected(message or "The data storage rejected the request.")
    if status == 409:
        raise Conflict(message or "The item changed since it was read. Read it again and retry.")
    if status == 401:
        raise CredentialRejected()
    if status == 403:
        raise ArtifactDataError(message or "Access to this artifact's data was refused.")
    if status in (429, 502, 503):
        raise TemporarilyUnavailable()
    raise ArtifactDataError(f"The data request failed (HTTP {status}).")


def _request(method: str, url: str, *, key: str, json_body: dict | None = None) -> dict:
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
        raise ArtifactDataError(f"The data request failed (HTTP {exc.code}).") from exc
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise TemporarilyUnavailable("could not reach the publishing service") from exc
    body = _json_body(raw)
    if not isinstance(body, dict):
        raise ArtifactDataError("The publishing service returned an unreadable response.")
    return body


def list_items(
    *, publish_url: str, key: str, report_id: str, cursor: str | None = None, limit: int | None = None,
) -> dict:
    """One page of every stored item: {"state_manifest", "items", "cursor"}.
    `cursor` is None on the last page."""
    query = {k: v for k, v in (("cursor", cursor), ("limit", limit)) if v}
    url = _url(publish_url, report_id) + (f"?{urlencode(query)}" if query else "")
    return _request("GET", url, key=key)


def run_op(*, publish_url: str, key: str, report_id: str, body: dict) -> dict:
    """One broker op (get/query/put/update/delete); the broker's answer."""
    if body.get("op") not in OPS:
        raise Rejected(f"op must be one of {', '.join(OPS)}")
    return _request("POST", _url(publish_url, report_id), key=key, json_body=body)
