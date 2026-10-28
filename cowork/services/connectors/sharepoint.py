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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx

GRAPH_URL = "https://graph.microsoft.com/v1.0"
ENGINE = "sharepoint"
LOGIN_URL = "https://login.microsoftonline.com"
MAX_SITES = 20
CHECK_SECONDS = 60.0
BROWSE_SECONDS = 30.0
PAGE_SIZE = 200
PLAN_SECONDS = 60.0
IMPORT_SECONDS = 15 * 60.0
MAX_IMPORT_FILES = 100
MAX_IMPORT_BYTES = 2**30
MAX_IMPORT_DEPTH = 32

_MAX_TRIES = 3
_MAX_LIBRARY_PAGES = 10
_MAX_IMPORT_FOLDERS = 200
_MAX_IMPORT_PAGES = 200
_MAX_NAME_BYTES = 255
_TOKEN_MARGIN_SECONDS = 60.0
_DEFAULT_RETRY_SECONDS = 1.0
_READ_TIMEOUT_CAP_SECONDS = 60.0
_REFUSED_REDIRECT = "SharePoint sent the download to an address Cowork won't follow."
# An UNREACHABLE with less than this left of an import's own budget is
# reported as the import being too slow, not as SharePoint being down.
_SLOW_MARGIN_SECONDS = 1.0
_CHILD_FIELDS = "id,name,size,file,folder,lastModifiedDateTime"
_PICK_FIELDS = "id,name,size,file,folder"
_STALE_CURSOR = "This folder listing is out of date. Open the folder again."
_FOLDER_CHANGED = "The folder changed while it was read. Try the import again."
_WINDOWS_UNSAFE_CHARS = frozenset('<>:"|?*')
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{n}" for n in range(1, 10)} | {f"LPT{n}" for n in range(1, 10)}
)

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
    TOO_MANY_FILES = "too_many_files"
    TOO_LARGE = "too_large"
    BAD_NAME = "bad_name"
    PLAN_TOO_SLOW = "plan_too_slow"
    IMPORT_TOO_SLOW = "import_too_slow"


_MESSAGES = {
    SharePointErrorKind.INVALID: "That value isn't valid.",
    SharePointErrorKind.TENANT_NOT_FOUND: "Microsoft doesn't know this tenant. Check the tenant id.",
    SharePointErrorKind.APP_NOT_FOUND: "This app isn't registered in the tenant. Check the client id.",
    SharePointErrorKind.KEY_REJECTED: "Microsoft rejected the client secret. Check that it is right and hasn't expired.",
    SharePointErrorKind.SITE_NOT_GRANTED: "The app can't open this site. Ask your admin to grant it.",
    SharePointErrorKind.ITEM_NOT_FOUND: "That library or folder wasn't found on this site.",
    SharePointErrorKind.THROTTLED: "SharePoint is busy right now. Try again in a minute.",
    SharePointErrorKind.UNREACHABLE: "SharePoint couldn't be reached. Try again in a moment.",
    SharePointErrorKind.TOO_MANY_FILES: f"This import is over the {MAX_IMPORT_FILES}-file limit. Pick fewer files or folders.",
    SharePointErrorKind.TOO_LARGE: "This import is over the 1 GB limit. Pick fewer or smaller files.",
    SharePointErrorKind.BAD_NAME: "SharePoint returned a file name Cowork can't use.",
    SharePointErrorKind.PLAN_TOO_SLOW: "Listing the picked folders took longer than a minute. Pick fewer folders.",
    SharePointErrorKind.IMPORT_TOO_SLOW: "The import took longer than 15 minutes. Pick fewer or smaller files.",
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
    client_secret: str = field(repr=False)

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


@dataclass(frozen=True)
class PlannedFile:
    parts: tuple[str, ...]
    item_id: str
    size: int


@dataclass(frozen=True)
class SkippedItem:
    path: str
    reason: str


@dataclass(frozen=True)
class ImportPlan:
    site: Site
    drive_id: str
    files: list[PlannedFile]
    skipped: list[SkippedItem]


@dataclass
class _ImportBudget:
    """What one import plan has spent; each method refuses as soon as a limit is crossed."""

    files: int = 0
    bytes: int = 0
    folders: int = 0
    pages: int = 0
    skipped: list[SkippedItem] = field(default_factory=list)

    def add_file(self, size: int) -> None:
        """Count one planned file of ``size`` listed bytes."""
        self.files += 1
        self.bytes += max(size, 0)
        if self.files > MAX_IMPORT_FILES:
            raise SharePointError(SharePointErrorKind.TOO_MANY_FILES)
        if self.bytes > MAX_IMPORT_BYTES:
            raise SharePointError(SharePointErrorKind.TOO_LARGE)

    def visit_folder(self) -> None:
        """Count one folder whose children are listed."""
        self.folders += 1
        if self.folders > _MAX_IMPORT_FOLDERS:
            raise SharePointError(
                SharePointErrorKind.TOO_LARGE,
                f"This import is over the {_MAX_IMPORT_FOLDERS}-folder limit. Pick fewer folders.",
            )

    def read_page(self) -> None:
        """Count one page of children read."""
        self.pages += 1
        if self.pages > _MAX_IMPORT_PAGES:
            raise SharePointError(
                SharePointErrorKind.TOO_LARGE,
                "The picked folders hold too many items to list. Pick fewer folders.",
            )


class DownloadBudget:
    """Bytes one import may still write, shared by every file it downloads."""

    def __init__(self, limit: int = MAX_IMPORT_BYTES) -> None:
        """Start with ``limit`` bytes left."""
        self._left = limit

    def spend(self, size: int) -> None:
        """Take ``size`` bytes; refuse once the import would pass its limit."""
        self._left -= size
        if self._left < 0:
            raise SharePointError(SharePointErrorKind.TOO_LARGE)


def _within_depth(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``parts`` if the path is at most ``MAX_IMPORT_DEPTH`` levels; refuse it otherwise."""
    if len(parts) > MAX_IMPORT_DEPTH:
        raise SharePointError(
            SharePointErrorKind.TOO_LARGE,
            f"A picked file is more than {MAX_IMPORT_DEPTH} folders deep.",
        )
    return parts


def _skip_reason(item: Item) -> str | None:
    """Why an item is left out of an import, or None to import it.

    A name only a hostile or broken response could send (a dot segment, a
    separator, NUL or a control character) fails the whole import. A name
    upload would refuse or a desktop can't hold is skipped, as upload skips
    a bad file name and carries on.
    """
    name = item.name
    if name in {".", ".."} or "/" in name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise SharePointError(SharePointErrorKind.BAD_NAME, f"SharePoint returned a file name Cowork can't use: {name!r}.")
    if not name:
        return "it has no name"
    if name.startswith("."):
        return "its name starts with a dot"
    if any(c in _WINDOWS_UNSAFE_CHARS for c in name) or name != name.strip(" ") or name.endswith("."):
        return "its name can't be saved on every computer"
    if name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICE_NAMES:
        return "its name is reserved on Windows"
    if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        return "its name is too long"
    if item.kind == "other":
        return "it is not a file or a folder"
    return None


_TOKENS: dict[tuple[str, str, str], tuple[str, float]] = {}
_TOKENS_LOCK = threading.Lock()


def generic_setup_refusal(org_mode: bool) -> str:
    """What a generic connector path says instead of collecting a SharePoint key.

    SharePoint is saved only by its own routes, which check the key with
    Microsoft first; those routes are desktop and self-hosted only.
    """
    if org_mode:
        return "SharePoint is available in the desktop app."
    return "Update Cowork, then set up SharePoint in Customize, under Connections, where its key is checked."


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


def _graph_id(value: str, pattern: re.Pattern[str], what: str) -> str:
    """An id Graph returned, if it fits ``pattern``; a response that breaks the shape is UNREACHABLE."""
    try:
        return _require_id(value, pattern)
    except SharePointError as exc:
        raise SharePointError(SharePointErrorKind.UNREACHABLE, f"SharePoint returned an unexpected {what} id.") from exc


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
            self._list_libraries(site, deadline)
            sites.append(site)
        with _TOKENS_LOCK:
            _TOKENS[self._key.cache_key()] = (self._token, expires_at)
        return sites

    def list_libraries(self, site_url: str) -> list[Library]:
        """List a site's document libraries, the site lookup included, within one browse budget."""
        deadline = _Deadline(self._clock, BROWSE_SECONDS)
        return self._list_libraries(self._resolve_site(site_url, deadline), deadline)

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
        self._require_library(site, drive_id, deadline)
        return self._children_page(site, drive_id, item_id, cursor, deadline)

    def plan_import(self, site_url: str, drive_id: str, item_ids: list[str]) -> ImportPlan:
        """Work out which files picked items bring in, and where, before anything is downloaded.

        Picked folders are walked breadth-first in the order given. Overlap is
        decided by the ids Graph returns, never by path strings: a pick met
        inside another picked folder lands under that folder, once. Every
        limit is refused as soon as it is crossed, within one planning budget.
        """
        _require_id(drive_id, _DRIVE_OR_ITEM_ID)
        requested = list(dict.fromkeys(item_ids))
        if not requested:
            raise SharePointError(SharePointErrorKind.INVALID, "Pick at least one file or folder.")
        if len(requested) > MAX_IMPORT_FILES:
            raise SharePointError(SharePointErrorKind.INVALID, f"Pick at most {MAX_IMPORT_FILES} files or folders.")
        for item_id in requested:
            _require_id(item_id, _DRIVE_OR_ITEM_ID)
            if item_id.casefold() == "root":
                raise SharePointError(SharePointErrorKind.INVALID, "Pick files or folders, not the whole library.")
        deadline = _Deadline(self._clock, PLAN_SECONDS)
        try:
            site = self._resolve_site(site_url, deadline)
            self._require_library(site, drive_id, deadline)
            return self._plan(site, drive_id, requested, deadline)
        except SharePointError as exc:
            if exc.kind is SharePointErrorKind.UNREACHABLE and deadline.remaining() < _SLOW_MARGIN_SECONDS:
                raise SharePointError(SharePointErrorKind.PLAN_TOO_SLOW) from exc
            raise

    def import_deadline(self) -> _Deadline:
        """Start the budget one whole import shares: its plan and every download."""
        return _Deadline(self._clock, IMPORT_SECONDS)

    def download(
        self,
        site: Site,
        drive_id: str,
        item_id: str,
        sink: Callable[[bytes], None],
        budget: DownloadBudget,
        deadline: _Deadline,
    ) -> int:
        """Stream one file's bytes into ``sink`` and return how many were written.

        ``site`` and ``drive_id`` come from a plan, so no site or library call
        is repeated per file. Graph answers ``/content`` with a redirect to a
        pre-authenticated URL; it is followed only to the connection's own site
        host, without the token, and never appears in an error. Errors raised
        by ``sink`` propagate unchanged.
        """
        _require_id(drive_id, _DRIVE_OR_ITEM_ID)
        _require_id(item_id, _DRIVE_OR_ITEM_ID)
        try:
            return self._download(site, drive_id, item_id, sink, budget, deadline)
        except SharePointError as exc:
            if exc.kind is SharePointErrorKind.UNREACHABLE and deadline.remaining() < _SLOW_MARGIN_SECONDS:
                raise SharePointError(SharePointErrorKind.IMPORT_TOO_SLOW) from exc
            raise

    def _download(
        self,
        site: Site,
        drive_id: str,
        item_id: str,
        sink: Callable[[bytes], None],
        budget: DownloadBudget,
        deadline: _Deadline,
    ) -> int:
        """Ask Graph for the content, then read it from Graph or from the one redirect allowed."""
        url = (
            f"{GRAPH_URL}/sites/{quote(site.graph_id, safe='!,')}"
            f"/drives/{quote(drive_id, safe='!')}/items/{quote(item_id, safe='!')}/content"
        )
        response = self._stream_graph(url, deadline)
        try:
            if response.status_code == 200:
                return self._drain(response, sink, budget, deadline)
            if response.status_code in {400, 403, 404}:
                raise SharePointError(SharePointErrorKind.ITEM_NOT_FOUND, "A picked file wasn't found on this site.")
            if response.status_code not in {302, 303}:
                raise SharePointError(SharePointErrorKind.UNREACHABLE)
            target = self._download_url(response.headers.get("location"), site)
        finally:
            response.close()
        response = self._send_stream(target, deadline, token=None)
        try:
            if response.status_code != 200:
                raise SharePointError(SharePointErrorKind.UNREACHABLE)
            return self._drain(response, sink, budget, deadline)
        finally:
            response.close()

    @staticmethod
    def _download_url(location: str | None, site: Site) -> httpx.URL:
        """The redirect's URL, parsed once, if it is https on the site's own host; refuse anything else."""
        if not location:
            raise SharePointError(SharePointErrorKind.UNREACHABLE, _REFUSED_REDIRECT)
        try:
            url = httpx.URL(location)
        except httpx.InvalidURL as exc:
            raise SharePointError(SharePointErrorKind.UNREACHABLE, _REFUSED_REDIRECT) from exc
        allowed_host = urlparse(site.url).hostname
        if (
            not url.is_absolute_url
            or url.scheme != "https"
            or url.userinfo
            or url.port is not None
            or url.host.casefold() != allowed_host
        ):
            raise SharePointError(SharePointErrorKind.UNREACHABLE, _REFUSED_REDIRECT)
        return url

    def _drain(
        self, response: httpx.Response, sink: Callable[[bytes], None], budget: DownloadBudget, deadline: _Deadline,
    ) -> int:
        """Hand each raw read to ``sink`` as it arrives, checking the deadline and byte budget first."""
        written = 0
        try:
            for chunk in response.iter_bytes():
                if deadline.remaining() <= 0:
                    raise SharePointError(SharePointErrorKind.IMPORT_TOO_SLOW)
                budget.spend(len(chunk))
                sink(chunk)
                written += len(chunk)
        except httpx.HTTPError as exc:
            raise SharePointError(SharePointErrorKind.UNREACHABLE) from exc
        return written

    def _stream_graph(self, url: str, deadline: _Deadline) -> httpx.Response:
        """Open a streamed GET to Graph with the app token, refetching it once on a 401."""
        response = self._send_stream(url, deadline, token=self._current_token(deadline))
        if response.status_code == 401:
            response.close()
            self._evict_token()
            response = self._send_stream(url, deadline, token=self._current_token(deadline))
        if response.status_code == 401:
            response.close()
            raise SharePointError(SharePointErrorKind.KEY_REJECTED)
        return response

    def _send_stream(self, url: str | httpx.URL, deadline: _Deadline, *, token: str | None) -> httpx.Response:
        """Open one streamed GET inside the deadline, waiting out 429 and 503 like ``_send``.

        httpx applies one read timeout to every read of a body, so it is fixed
        here at the time left, capped. The caller closes the returned response.
        """
        headers = {"Authorization": f"Bearer {token}"} if token else None
        for attempt in range(_MAX_TRIES):
            remaining = deadline.remaining()
            if remaining <= 0:
                raise SharePointError(SharePointErrorKind.UNREACHABLE)
            request = self._client.build_request(
                "GET", url, headers=headers, timeout=min(remaining, _READ_TIMEOUT_CAP_SECONDS),
            )
            try:
                response = self._client.send(request, stream=True, follow_redirects=False)
            except httpx.HTTPError as exc:
                raise SharePointError(SharePointErrorKind.UNREACHABLE) from exc
            if response.status_code not in {429, 503}:
                return response
            wait = _retry_after_seconds(response)
            response.close()
            if attempt == _MAX_TRIES - 1 or wait >= deadline.remaining():
                raise SharePointError(SharePointErrorKind.THROTTLED)
            self._sleep(wait)
        raise SharePointError(SharePointErrorKind.THROTTLED)

    def _plan(self, site: Site, drive_id: str, requested: list[str], deadline: _Deadline) -> ImportPlan:
        """Read each pick, walk the picked folders, then place the picked files no folder covered."""
        budget = _ImportBudget()
        picked_ids: set[str] = set()
        picks: list[Item] = []
        skipped_picks: dict[str, SkippedItem] = {}
        for item_id in requested:
            item = self._get_item(site, drive_id, item_id, deadline)
            if item.id in picked_ids:
                continue
            picked_ids.add(item.id)
            reason = _skip_reason(item)
            if reason is not None:
                skipped_picks[item.id] = SkippedItem(path=item.name, reason=reason)
                continue
            picks.append(item)
        covered: set[str] = set()
        walked: dict[str, list[PlannedFile]] = {}
        for pick in picks:
            if pick.kind == "folder" and pick.id not in covered:
                walked[pick.id] = self._walk(site, drive_id, pick, deadline, budget, walked, picked_ids, covered)
        files = [planned for folder_files in walked.values() for planned in folder_files]
        for pick in picks:
            if pick.kind == "file" and pick.id not in covered:
                budget.add_file(pick.size)
                files.append(PlannedFile(parts=(pick.name,), item_id=pick.id, size=pick.size))
        # A skipped pick a walk also reached is already listed under the walk's path.
        budget.skipped.extend(item for item_id, item in skipped_picks.items() if item_id not in covered)
        return ImportPlan(site=site, drive_id=drive_id, files=files, skipped=budget.skipped)

    def _walk(
        self,
        site: Site,
        drive_id: str,
        folder: Item,
        deadline: _Deadline,
        budget: _ImportBudget,
        walked: dict[str, list[PlannedFile]],
        picked_ids: set[str],
        covered: set[str],
    ) -> list[PlannedFile]:
        """Plan every file under one picked folder, with paths starting at the folder's own name.

        A picked folder already walked is taken over, not walked again: its
        files move under this folder and it leaves ``walked``. Any pick met
        here is marked covered so it is not placed on its own.
        """
        files: list[PlannedFile] = []
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(folder.id, (folder.name,))])
        while queue:
            folder_id, prefix = queue.popleft()
            budget.visit_folder()
            cursor: str | None = None
            while True:
                budget.read_page()
                try:
                    page = self._children_page(site, drive_id, folder_id, cursor, deadline)
                except SharePointError as exc:
                    # Graph refused its own next-page cursor; reopening a folder doesn't apply here.
                    if cursor and exc.kind is SharePointErrorKind.INVALID:
                        raise SharePointError(SharePointErrorKind.INVALID, _FOLDER_CHANGED) from exc
                    raise
                for child in page.items:
                    path = (*prefix, child.name)
                    if child.id in picked_ids:
                        covered.add(child.id)
                    reason = _skip_reason(child)
                    if reason is not None:
                        budget.skipped.append(SkippedItem(path="/".join(path), reason=reason))
                        continue
                    if child.kind == "folder" and child.id in walked:
                        for planned in walked.pop(child.id):
                            files.append(PlannedFile(_within_depth((*prefix, *planned.parts)), planned.item_id, planned.size))
                    elif child.kind == "folder":
                        queue.append((child.id, _within_depth(path)))
                    else:
                        budget.add_file(child.size)
                        files.append(PlannedFile(parts=_within_depth(path), item_id=child.id, size=child.size))
                cursor = page.cursor
                if cursor is None:
                    break
        return files

    def _get_item(self, site: Site, drive_id: str, item_id: str, deadline: _Deadline) -> Item:
        """Read one picked item of a library already checked against the site."""
        url = (
            f"{GRAPH_URL}/sites/{quote(site.graph_id, safe='!,')}"
            f"/drives/{quote(drive_id, safe='!')}/items/{quote(item_id, safe='!')}"
        )
        response = self._graph_get(url, {"$select": _PICK_FIELDS}, deadline)
        if response.status_code in {400, 403, 404}:
            raise SharePointError(SharePointErrorKind.ITEM_NOT_FOUND)
        return self._item(self._json(response))

    def _require_library(self, site: Site, drive_id: str, deadline: _Deadline) -> None:
        """Refuse a library id that is not one of the site's own."""
        if drive_id not in {library.id for library in self._list_libraries(site, deadline)}:
            raise SharePointError(SharePointErrorKind.ITEM_NOT_FOUND)

    def _children_page(
        self, site: Site, drive_id: str, item_id: str, cursor: str | None, deadline: _Deadline,
    ) -> Page:
        """One page of a folder's children in a library already checked against the site."""
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
        graph_id = _graph_id(str(self._json(response).get("id") or ""), _SITE_ID, "site")
        return Site(url=canonical, graph_id=graph_id)

    def _list_libraries(self, site: Site, deadline: _Deadline) -> list[Library]:
        """Read the site's libraries page by page, at most ``_MAX_LIBRARY_PAGES`` pages."""
        base = f"{GRAPH_URL}/sites/{quote(site.graph_id, safe='!,')}/drives"
        libraries: list[Library] = []
        params: dict[str, str] = {"$select": "id,name"}
        for _ in range(_MAX_LIBRARY_PAGES):
            response = self._graph_get(base, params, deadline)
            if response.status_code in {403, 404}:
                raise SharePointError(
                    SharePointErrorKind.SITE_NOT_GRANTED,
                    f"The app can't open {site.url}. Ask your admin to grant it this site.",
                )
            payload = self._json(response)
            for entry in payload.get("value", []):
                drive_id = _graph_id(str(entry.get("id") or ""), _DRIVE_OR_ITEM_ID, "library")
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
        item_id = _graph_id(str(entry.get("id") or ""), _DRIVE_OR_ITEM_ID, "item")
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
