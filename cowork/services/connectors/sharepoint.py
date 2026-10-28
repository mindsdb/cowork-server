"""SharePoint access through Microsoft Graph with an Entra app key.

The key is an app registered in the customer's Entra (tenant id, client id,
client secret) holding the ``Sites.Selected`` application permission, so it
reaches only the sites an admin granted it. Every Graph URL is built here from
validated parts and always sits under the site a connection names: a library
or item id from the caller never changes which site a request goes to.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx

GRAPH_URL = "https://graph.microsoft.com/v1.0"
LOGIN_URL = "https://login.microsoftonline.com"
MAX_SITES = 20
CHECK_SECONDS = 60.0
BROWSE_SECONDS = 30.0
PAGE_SIZE = 200

_MAX_TRIES = 3
_MAX_DRIVE_PAGES = 10
_TOKEN_MARGIN_SECONDS = 60.0
_DEFAULT_RETRY_SECONDS = 1.0
_CHILD_FIELDS = "id,name,size,file,folder,lastModifiedDateTime"
_STALE_CURSOR = "This folder listing is out of date. Open the folder again."

_GUID = re.compile(r"[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")
_DOMAIN = re.compile(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")
_SHAREPOINT_HOST = re.compile(r"[a-z0-9-]+\.sharepoint\.com")
_SITE_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")
_DRIVE_OR_ITEM_ID = re.compile(r"[A-Za-z0-9!_-]+")
_SITE_ID = re.compile(r"[A-Za-z0-9.,_-]+")

# Entra's AADSTS codes in a failed token response. The HTTP status alone does
# not separate them: an unknown tenant and an unknown app both answer 400.
_TENANT_NOT_FOUND_CODES = frozenset({90002})
_APP_NOT_FOUND_CODES = frozenset({700016, 700038})
_KEY_REJECTED_CODES = frozenset({7000215, 7000222})


class SharePointErrorKind(str, Enum):
    INVALID = "invalid"
    TENANT_NOT_FOUND = "tenant_not_found"
    APP_NOT_FOUND = "app_not_found"
    KEY_REJECTED = "key_rejected"
    SITE_NOT_GRANTED = "site_not_granted"
    ITEM_NOT_FOUND = "item_not_found"
    THROTTLED = "throttled"
    UNREACHABLE = "unreachable"


_MESSAGES = {
    SharePointErrorKind.INVALID: "That value isn't valid.",
    SharePointErrorKind.TENANT_NOT_FOUND: "Microsoft doesn't know this tenant. Check the tenant id.",
    SharePointErrorKind.APP_NOT_FOUND: "This app isn't registered in the tenant. Check the client id.",
    SharePointErrorKind.KEY_REJECTED: "Microsoft rejected the client secret. Check that it is right and hasn't expired.",
    SharePointErrorKind.SITE_NOT_GRANTED: "The app can't open this site. Ask your admin to grant it.",
    SharePointErrorKind.ITEM_NOT_FOUND: "That library or folder wasn't found on this site.",
    SharePointErrorKind.THROTTLED: "SharePoint is busy right now. Try again in a minute.",
    SharePointErrorKind.UNREACHABLE: "SharePoint couldn't be reached. Try again in a moment.",
}


class SharePointError(Exception):
    """A user-safe SharePoint failure. Its message never carries a token, the secret, or a Graph URL."""

    def __init__(self, kind: SharePointErrorKind, message: str | None = None) -> None:
        """Build the error from its kind; ``message`` overrides the kind's standard text."""
        super().__init__(message or _MESSAGES[kind])
        self.kind = kind


@dataclass(frozen=True)
class AppKey:
    tenant_id: str
    client_id: str
    client_secret: str

    def cache_key(self) -> tuple[str, str, str]:
        """Identify this key in the token cache without holding the secret itself."""
        digest = hashlib.sha256(self.client_secret.encode()).hexdigest()
        return (self.tenant_id.casefold(), self.client_id.casefold(), digest)


@dataclass(frozen=True)
class Site:
    url: str
    graph_id: str


@dataclass(frozen=True)
class Library:
    id: str
    name: str


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    size: int
    kind: str
    modified: str | None


@dataclass(frozen=True)
class Page:
    items: list[Item]
    cursor: str | None


_TOKENS: dict[tuple[str, str, str], tuple[str, float]] = {}
_TOKENS_LOCK = threading.Lock()


def validate_tenant_id(value: str) -> str:
    """Return ``value`` stripped if it is a tenant GUID or domain; raise INVALID otherwise."""
    tenant = value.strip()
    if _GUID.fullmatch(tenant) or _DOMAIN.fullmatch(tenant):
        return tenant
    raise SharePointError(SharePointErrorKind.INVALID, "Enter the tenant id as a GUID or a domain such as contoso.onmicrosoft.com.")


def normalise_site_url(value: str) -> str:
    """Return the canonical form of a SharePoint site URL, or raise INVALID.

    Only ``https://<name>.sharepoint.com`` with a path of ``/``, ``/sites/<name>``
    or ``/teams/<name>`` is accepted. OneDrive hosts (``-my``) are refused. The
    result is lower-cased with no trailing slash, so two spellings of one site
    compare equal.
    """
    refused = SharePointError(
        SharePointErrorKind.INVALID,
        "Enter a SharePoint site address such as https://contoso.sharepoint.com/sites/team.",
    )
    try:
        parsed = urlparse(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise refused from exc
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username
        or parsed.password
        or port is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not _SHAREPOINT_HOST.fullmatch(host)
        or host.endswith("-my.sharepoint.com")
    ):
        raise refused
    path = parsed.path.rstrip("/")
    if not path:
        return f"https://{host}"
    segments = path.split("/")[1:]
    if (
        len(segments) != 2
        or segments[0].casefold() not in {"sites", "teams"}
        or not _SITE_SEGMENT.fullmatch(segments[1])
        or segments[1] in {".", ".."}
    ):
        raise refused
    return f"https://{host}/{segments[0]}/{segments[1]}".casefold()


def parse_site_urls(text: str) -> list[str]:
    """Split one URL per line, normalise each and drop duplicates; at least 1 and at most ``MAX_SITES``."""
    sites: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        site = normalise_site_url(line)
        if site not in sites:
            sites.append(site)
    if not sites:
        raise SharePointError(SharePointErrorKind.INVALID, "Enter at least one SharePoint site address.")
    if len(sites) > MAX_SITES:
        raise SharePointError(SharePointErrorKind.INVALID, f"A connection can name at most {MAX_SITES} sites.")
    return sites


def _require_id(value: str, pattern: re.Pattern[str]) -> str:
    """Return ``value`` if it fits ``pattern`` and is not a dot segment; raise INVALID otherwise."""
    if not pattern.fullmatch(value) or value in {".", ".."}:
        raise SharePointError(SharePointErrorKind.INVALID, "That library or folder id isn't valid.")
    return value


def _skiptoken(next_link: Any) -> str | None:
    """Return the ``$skiptoken`` of an ``@odata.nextLink``, or None when there is no next page.

    The link itself is never requested: only its token goes back into a URL
    this module builds, so a response cannot choose where the next call goes.
    """
    if next_link is None:
        return None
    query = parse_qs(urlparse(str(next_link)).query)
    values = query.get("$skiptoken")
    if not values or not values[0]:
        raise SharePointError(SharePointErrorKind.UNREACHABLE, "SharePoint returned a page link this client can't follow.")
    return values[0]


def _retry_after_seconds(response: httpx.Response) -> float:
    """Seconds a 429 or 503 asks to wait; a missing or date-form header waits the default."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return _DEFAULT_RETRY_SECONDS
    return max(seconds, 0.0) if math.isfinite(seconds) else _DEFAULT_RETRY_SECONDS


class _Deadline:
    """One operation's time budget, shared by every request and retry wait in it.

    It is checked before each request and caps each request's timeout, but
    httpx applies that timeout per phase and restarts the read timeout on
    every chunk, so a slow reply can run past it: a soft budget, not a hard cap.
    """

    def __init__(self, clock: Callable[[], float], seconds: float) -> None:
        """Start the budget of ``seconds`` now, by ``clock``."""
        self._clock = clock
        self._ends_at = clock() + seconds

    def remaining(self) -> float:
        """Seconds left before the operation's budget runs out."""
        return self._ends_at - self._clock()


class SharePointClient:
    """Calls Entra and Graph for one app key. Not shared across keys; cheap to build per request."""

    def __init__(
        self,
        key: AppKey,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Hold the key and an HTTP client; ``transport``, ``clock`` and ``sleep`` are seams for tests."""
        self._key = key
        self._client = httpx.Client(transport=transport, follow_redirects=False)
        self._clock = clock
        self._sleep = sleep
        self._token: str | None = None
        self._fresh_only = False

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()

    def __enter__(self) -> SharePointClient:
        """Use the client as a context manager that closes it on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the HTTP client."""
        self.close()

    def check(self, site_urls: list[str]) -> list[Site]:
        """Prove the key works for every site: a fresh token, each site resolved, its libraries listed.

        The cached token is never used here, so a secret revoked in Entra fails
        the check at once. The fresh token is cached only after every site passed.
        """
        deadline = _Deadline(self._clock, CHECK_SECONDS)
        self._fresh_only = True
        token, expires_at = self._fetch_token(deadline)
        self._token = token
        sites = []
        for url in site_urls:
            site = self._resolve_site(url, deadline)
            self._list_drives(site, deadline)
            sites.append(site)
        with _TOKENS_LOCK:
            _TOKENS[self._key.cache_key()] = (self._token, expires_at)
        return sites

    def resolve_site(self, url: str) -> Site:
        """Resolve one site URL to its Graph site id."""
        return self._resolve_site(url, _Deadline(self._clock, BROWSE_SECONDS))

    def list_drives(self, site_url: str) -> list[Library]:
        """List a site's document libraries, the site lookup included, within one browse budget."""
        deadline = _Deadline(self._clock, BROWSE_SECONDS)
        return self._list_drives(self._resolve_site(site_url, deadline), deadline)

    def list_children(self, site_url: str, drive_id: str, item_id: str = "root", cursor: str | None = None) -> Page:
        """List one page of a folder's children; ``cursor`` is the previous page's skiptoken.

        Ids are checked before any request, and the library must be one of the
        site's own, so an id from another site is refused before any request
        names it. The site lookup, the library list and the page share one
        browse budget.
        """
        _require_id(drive_id, _DRIVE_OR_ITEM_ID)
        _require_id(item_id, _DRIVE_OR_ITEM_ID)
        deadline = _Deadline(self._clock, BROWSE_SECONDS)
        site = self._resolve_site(site_url, deadline)
        if drive_id not in {library.id for library in self._list_drives(site, deadline)}:
            raise SharePointError(SharePointErrorKind.ITEM_NOT_FOUND)
        url = (
            f"{GRAPH_URL}/sites/{quote(site.graph_id, safe='!,')}"
            f"/drives/{quote(drive_id, safe='!')}/items/{quote(item_id, safe='!')}/children"
        )
        params = {"$select": _CHILD_FIELDS, "$top": str(PAGE_SIZE)}
        if cursor:
            params["$skiptoken"] = cursor
        response = self._graph_get(url, params, deadline)
        if cursor and response.status_code == 400:
            raise SharePointError(SharePointErrorKind.INVALID, _STALE_CURSOR)
        if response.status_code in {400, 403, 404}:
            raise SharePointError(SharePointErrorKind.ITEM_NOT_FOUND)
        payload = self._json(response)
        items = [self._item(entry) for entry in payload.get("value", [])]
        return Page(items=items, cursor=_skiptoken(payload.get("@odata.nextLink")))

    def _resolve_site(self, url: str, deadline: _Deadline) -> Site:
        """Look the site up by host and path; 403 or 404 means the app was not granted it."""
        canonical = normalise_site_url(url)
        parsed = urlparse(canonical)
        address = f"{GRAPH_URL}/sites/{parsed.hostname}"
        if parsed.path:
            address += ":" + "/".join(quote(segment, safe="") for segment in parsed.path.split("/"))
        response = self._graph_get(address, None, deadline)
        if response.status_code in {403, 404}:
            raise SharePointError(
                SharePointErrorKind.SITE_NOT_GRANTED,
                f"The app can't open {canonical}. Ask your admin to grant it this site, and check the address.",
            )
        graph_id = str(self._json(response).get("id") or "")
        try:
            _require_id(graph_id, _SITE_ID)
        except SharePointError as exc:
            raise SharePointError(SharePointErrorKind.UNREACHABLE, "SharePoint returned an unexpected site id.") from exc
        return Site(url=canonical, graph_id=graph_id)

    def _list_drives(self, site: Site, deadline: _Deadline) -> list[Library]:
        """Read the site's libraries page by page, at most ``_MAX_DRIVE_PAGES`` pages."""
        base = f"{GRAPH_URL}/sites/{quote(site.graph_id, safe='!,')}/drives"
        libraries: list[Library] = []
        params: dict[str, str] = {"$select": "id,name"}
        for _ in range(_MAX_DRIVE_PAGES):
            response = self._graph_get(base, params, deadline)
            if response.status_code in {403, 404}:
                raise SharePointError(
                    SharePointErrorKind.SITE_NOT_GRANTED,
                    f"The app can't open {site.url}. Ask your admin to grant it this site.",
                )
            payload = self._json(response)
            for entry in payload.get("value", []):
                drive_id = str(entry.get("id") or "")
                try:
                    _require_id(drive_id, _DRIVE_OR_ITEM_ID)
                except SharePointError as exc:
                    raise SharePointError(SharePointErrorKind.UNREACHABLE, "SharePoint returned an unexpected library id.") from exc
                libraries.append(Library(id=drive_id, name=str(entry.get("name") or "")))
            token = _skiptoken(payload.get("@odata.nextLink"))
            if token is None:
                return libraries
            params = {"$select": "id,name", "$skiptoken": token}
        return libraries

    def _graph_get(self, url: str, params: dict[str, str] | None, deadline: _Deadline) -> httpx.Response:
        """GET from Graph with the app token, refetching it once on a 401."""
        response = self._send("GET", url, deadline, params=params, token=self._current_token(deadline))
        if response.status_code == 401:
            if self._fresh_only:
                # The check never reads the cache; it writes it only after every site passed.
                self._token, _ = self._fetch_token(deadline)
            else:
                self._evict_token()
            response = self._send("GET", url, deadline, params=params, token=self._current_token(deadline))
        if response.status_code == 401:
            raise SharePointError(SharePointErrorKind.KEY_REJECTED)
        return response

    def _current_token(self, deadline: _Deadline) -> str:
        """This client's token, else a cached one still in date, else a new one, which is cached."""
        if self._token is not None:
            return self._token
        key = self._key.cache_key()
        with _TOKENS_LOCK:
            cached = _TOKENS.get(key)
        if cached is not None and cached[1] > self._clock():
            self._token = cached[0]
            return self._token
        token, expires_at = self._fetch_token(deadline)
        with _TOKENS_LOCK:
            _TOKENS[key] = (token, expires_at)
        self._token = token
        return token

    def _evict_token(self) -> None:
        """Forget this key's token here and in the cache, after Graph refused it."""
        self._token = None
        with _TOKENS_LOCK:
            _TOKENS.pop(self._key.cache_key(), None)

    def _fetch_token(self, deadline: _Deadline) -> tuple[str, float]:
        """Run the client-credentials flow; return the token and when to stop reusing it."""
        tenant = validate_tenant_id(self._key.tenant_id)
        response = self._send(
            "POST",
            f"{LOGIN_URL}/{quote(tenant, safe='')}/oauth2/v2.0/token",
            deadline,
            data={
                "client_id": self._key.client_id,
                "client_secret": self._key.client_secret,
                "grant_type": "client_credentials",
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        payload = self._json_or_empty(response)
        token = payload.get("access_token")
        if response.status_code == 200 and isinstance(token, str) and token:
            try:
                lifetime = float(payload.get("expires_in", 0))
            except (TypeError, ValueError):
                lifetime = 0.0
            return token, self._clock() + max(lifetime - _TOKEN_MARGIN_SECONDS, 0.0)
        raw_codes = payload.get("error_codes")
        codes = {code for code in raw_codes if isinstance(code, int)} if isinstance(raw_codes, list) else set()
        if codes & _TENANT_NOT_FOUND_CODES:
            raise SharePointError(SharePointErrorKind.TENANT_NOT_FOUND)
        if codes & _APP_NOT_FOUND_CODES:
            raise SharePointError(SharePointErrorKind.APP_NOT_FOUND)
        if codes & _KEY_REJECTED_CODES:
            raise SharePointError(SharePointErrorKind.KEY_REJECTED)
        raise SharePointError(SharePointErrorKind.UNREACHABLE)

    def _send(
        self,
        method: str,
        url: str,
        deadline: _Deadline,
        *,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        token: str | None = None,
    ) -> httpx.Response:
        """Send one request inside the deadline, waiting out 429 and 503 while time is left."""
        headers = {"Authorization": f"Bearer {token}"} if token else None
        for attempt in range(_MAX_TRIES):
            remaining = deadline.remaining()
            if remaining <= 0:
                raise SharePointError(SharePointErrorKind.UNREACHABLE)
            try:
                response = self._client.request(
                    method, url, params=params, data=data, headers=headers, timeout=remaining,
                )
            except httpx.HTTPError as exc:
                raise SharePointError(SharePointErrorKind.UNREACHABLE) from exc
            if response.status_code not in {429, 503}:
                return response
            wait = _retry_after_seconds(response)
            if attempt == _MAX_TRIES - 1 or wait >= deadline.remaining():
                raise SharePointError(SharePointErrorKind.THROTTLED)
            self._sleep(wait)
        raise SharePointError(SharePointErrorKind.THROTTLED)

    @staticmethod
    def _item(entry: dict[str, Any]) -> Item:
        """Keep only the fields browse returns; an item with neither facet is ``other``."""
        if "folder" in entry:
            kind = "folder"
        elif "file" in entry:
            kind = "file"
        else:
            kind = "other"
        item_id = str(entry.get("id") or "")
        try:
            _require_id(item_id, _DRIVE_OR_ITEM_ID)
        except SharePointError as exc:
            raise SharePointError(SharePointErrorKind.UNREACHABLE, "SharePoint returned an unexpected item id.") from exc
        size = entry.get("size")
        modified = entry.get("lastModifiedDateTime")
        return Item(
            id=item_id,
            name=str(entry.get("name") or ""),
            size=size if isinstance(size, int) else 0,
            kind=kind,
            modified=modified if isinstance(modified, str) else None,
        )

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        """The JSON object of a successful response; anything else is UNREACHABLE."""
        if response.is_error:
            raise SharePointError(SharePointErrorKind.UNREACHABLE)
        payload = SharePointClient._json_or_empty(response)
        if not payload:
            raise SharePointError(SharePointErrorKind.UNREACHABLE)
        return payload

    @staticmethod
    def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
        """The response's JSON object, or an empty dict when the body is not one."""
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
