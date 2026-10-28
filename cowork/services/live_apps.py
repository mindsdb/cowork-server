"""Live apps: full-stack apps that run in app slots outside this server, each
opened only by the person who launched it.

anton launches an app into a slot (anton.core.artifacts.live_app_slots) from a
turn in this process, and calls the turn's CoworkLiveAppHost around it: the
host stops the slot's running launch record before the launch and starts a new
one once the slot accepts. Before the edge serves an app's address it asks the
access route, and check_access allows only the visitor whose viewer sign-in
names the recorded launcher of the app the slot runs now. A launch stays
recorded after it stops, with why: the slot's own report (record_slot_stop),
anton's next claim of the slot, or a slot found no longer running it
(reconcile_running). The card (person_apps), the page at the old address
(stopped_page) and the person's next turn (turn_live_apps) read those records.

A person shares an app of theirs with a colleague's address or with everyone
(record_share). The colleague sees it in their list (shares_with) and
duplicates it into a project of their own (read_shared_app, write_app_copy),
where it runs with their own connections. The copy takes a fixed list of
files, the same one anton launches.

Off unless live apps are on (cowork.code_sandbox.live_apps_on), and create_app
refuses them without the sandbox, trusted identity headers and personal
projects (cowork.code_sandbox._check_live_apps_startup).
"""

from __future__ import annotations

import asyncio
import errno
import html
import logging
import os
import re
import ssl
import stat
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache, partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import UUID

import httpx
import sqlalchemy as sa
from anton.core.artifacts.internal_files import (
    METADATA_FILENAME,
    PRD_FILENAME,
    TECH_SPEC_FILENAME,
)
from anton.core.artifacts.models import DatasourceRef
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import IntegrityError

from cowork.code_sandbox import live_apps_on, refuse_a_linked_folder, refuse_a_linked_project
from cowork.common.datetime_utils import ensure_utc
from cowork.common.paths import (
    O_NOFOLLOW,
    PinnedDir,
    dir_mkdir,
    dir_open,
    dir_rmtree,
    dir_scandir,
    makedirs_below_nofollow,
    open_pinned_child,
    opened_subdir_nofollow,
    read_bounded,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.units import run_db, run_to_completion
from cowork.models.live_app import (
    EMAIL_MAX_LENGTH,
    LiveAppCopy,
    LiveAppLaunchRecord,
    LiveAppShare,
    StopReason,
)
from cowork.principal import Principal
from cowork.services.artifact_roots import _ARTIFACTS_SUBPATH, project_artifacts_base

if TYPE_CHECKING:
    from anton.core.artifacts.live_app_slots import LiveAppLaunch, SlotStatus

logger = logging.getLogger(__name__)

#: The full-stack artifact types, the only ones a live app can be.
LIVE_APP_TYPES = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})

#: How long the access check waits on viewer-login and on the slot.
ACCESS_TIMEOUT_SECONDS = 2.0

#: The copy's cap on each of the PRD and the spec, which anton never launches.
COPY_DOC_MAX_BYTES = 1 << 20

#: The deepest folder under ``static/`` the copy follows.
COPY_MAX_DEPTH = 32

#: The documents the copy takes beside anton's launch files, for the
#: recipient's agent to read before the first launch.
COPY_DOCS = (PRD_FILENAME, TECH_SPEC_FILENAME)

#: The longest name or description "Shared with you" shows.
SHARED_TEXT_MAX_LENGTH = 200

#: The shortest email anton's LiveAppLauncher takes.
EMAIL_MIN_LENGTH = 3

# An address a turn's prompt may show as is: no backtick, space or other
# character that could end its code span or read as an instruction.
_PROMPT_SAFE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")


def normalized_email(value: str) -> str:
    """``value`` as live apps compare addresses: trimmed and lowercased."""
    return value.strip().lower()


def is_email(value: str) -> bool:
    if len(value) > EMAIL_MAX_LENGTH:
        return False
    local, separator, domain = value.partition("@")
    # A separator dot needs at least one domain character on each side.
    return (
        bool(local)
        and bool(separator)
        and "@" not in domain
        and "." in domain[1:-1]
        and not any(character.isspace() for character in value)
    )


def prompt_email(value: str) -> str:
    """Another person's address as a turn's prompt names it: in backticks
    when it holds only plain address characters, else a generic phrase,
    since the address comes from that person's account."""
    if _PROMPT_SAFE_EMAIL.fullmatch(value):
        return f"`{value}`"
    return "another person on this stack"


# ── who launches ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveAppPerson:
    """One person as live apps know them: their ``sub``, their space (the
    partition id their principal carries) and their verified email, normalized."""

    user_id: str
    partition_id: str
    email: str


def live_app_person(principal: Principal | None) -> LiveAppPerson | None:
    """The person ``principal`` names, or None for a principal with no email:
    live apps open, share and copy by email, so such a principal can do none
    of it. An email anton's LiveAppLauncher wouldn't take (under 3 or over
    254 characters) counts as none, so its turn runs without a host instead of
    failing."""
    if principal is None or not principal.user_id or not principal.org_id:
        return None
    email = normalized_email(principal.email)
    if not EMAIL_MIN_LENGTH <= len(email) <= EMAIL_MAX_LENGTH:
        return None
    return LiveAppPerson(user_id=principal.user_id, partition_id=principal.org_id, email=email)


_turn_launcher: ContextVar[LiveAppPerson | None] = ContextVar("live_app_launcher", default=None)


@contextmanager
def use_live_app_launcher(principal: Principal | None) -> Iterator[None]:
    """Name the person whose turn runs in this context, for the turn's live-app
    host (live_app_host_for_turn)."""
    token = _turn_launcher.set(live_app_person(principal))
    try:
        yield
    finally:
        _turn_launcher.reset(token)


def current_live_app_launcher() -> LiveAppPerson | None:
    """The person use_live_app_launcher named for this context, if any."""
    return _turn_launcher.get()


class CoworkLiveAppHost:
    """The host anton calls around each launch into a slot (anton's
    LiveAppHost protocol).

    anton calls ``claim`` when a launch replaces an app the slot runs, before
    it resets the slot, and ``claim`` stops that app's running record; a
    launch that then fails leaves no running record, so nobody opens the slot
    until a later launch succeeds, and anton calls ``launch_failed``, which
    makes that stop one the person hears of. ``launched`` starts the new
    record once the slot has accepted, ending any record still running there
    (_start_record).
    """

    def __init__(self, launcher: LiveAppPerson, *, project_id: UUID, launch_token: str) -> None:
        from anton.core.artifacts.live_app_slots import LiveAppLauncher

        self.launcher = LiveAppLauncher(subject=launcher.user_id, email=launcher.email)
        self.launch_token = launch_token
        self._person = launcher
        self._project_id = project_id
        # The record each claim stopped, by slot, so launched can tell the
        # person's relaunch of the same app from a replacement.
        self._claimed: dict[int, UUID] = {}
        # The boot of each launch launched is recording, by slot, so a
        # launch_failed after a Stop that landed once its record was written
        # stops that record too.
        self._launching: dict[int, str] = {}

    async def claim(self, slot: int) -> None:
        lost_at = await _slot_boot_time(slot)
        stopped = await run_db(
            partial(_stop_for_claim, slot=slot, claimer=self._person, lost_at=lost_at), scope=SYSTEM_SCOPE
        )
        if stopped is None:
            self._claimed.pop(slot, None)
        else:
            self._claimed[slot] = stopped

    async def launched(self, launch: LiveAppLaunch) -> None:
        self._launching[launch.slot] = launch.boot_id
        lost_at = await _slot_boot_time(launch.slot, boot_id=launch.boot_id)
        try:
            await run_db(
                partial(
                    _start_record,
                    launch=launch,
                    launcher=self._person,
                    project_id=self._project_id,
                    claimed=self._claimed.get(launch.slot),
                    lost_at=lost_at,
                ),
                scope=SYSTEM_SCOPE,
            )
        except asyncio.CancelledError:
            # anton resets the slot whenever launched raises, but calls
            # launch_failed only after a claim. A first launch into a free
            # slot has none, so a Stop that lands once the record is written
            # would leave it running until a later check finds it lost.
            if launch.slot not in self._claimed:
                await run_to_completion(self.launch_failed(launch.slot))
            raise
        # Only once the record is written: a launched that raises, or that a
        # Stop ends, leaves both for launch_failed, which anton calls next
        # after a claim, and the handler above calls without one.
        self._claimed.pop(launch.slot, None)
        self._launching.pop(launch.slot, None)

    async def launch_failed(self, slot: int) -> None:
        """A launch that claimed ``slot`` didn't finish: anton reset the slot,
        so the app claim stopped is gone and nothing runs there. Its stop
        becomes ``launch_failed``, and so does a record launched wrote for
        this launch before a Stop, which the card, the stopped page and the
        next turn tell. Never raises into anton's launch."""
        claimed = self._claimed.pop(slot, None)
        launching = self._launching.pop(slot, None)
        if claimed is None and launching is None:
            return
        try:
            await run_db(
                partial(_launch_failed, record_id=claimed, slot=slot, boot_id=launching), scope=SYSTEM_SCOPE
            )
        except Exception:
            logger.warning(
                "Live app launch failure not recorded",
                exc_info=True,
                extra={"event": "live_app_launch_failure_unrecorded", "slot": slot},
            )


def live_app_host_for_turn(*, project_id: UUID) -> CoworkLiveAppHost | None:
    """The live-app host for the turn running in this context, in project
    ``project_id``: None where live apps are off, or where the turn names
    nobody (use_live_app_launcher)."""
    launcher = current_live_app_launcher()
    if launcher is None or not live_apps_on():
        return None
    token = get_app_settings().live_apps_launch_token.get_secret_value()
    return CoworkLiveAppHost(launcher, project_id=project_id, launch_token=token)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _running_record(session: ScopedSession, *, slot: int) -> LiveAppLaunchRecord | None:
    return session.exec(
        session.select(LiveAppLaunchRecord).where(
            LiveAppLaunchRecord.slot == slot, LiveAppLaunchRecord.state == "running"
        )
    ).first()


def _stop(
    session: ScopedSession,
    record: LiveAppLaunchRecord,
    *,
    reason: StopReason,
    limit_hours: float | None = None,
    noticed: bool = False,
    stopped_at: datetime | None = None,
) -> None:
    """Stop ``record``, at ``stopped_at`` when the stop happened earlier than
    now (a slot's report, a slot found restarted), else now."""
    record.state = "stopped"
    record.stopped_at = _when_stopped(record, stopped_at)
    record.stop_reason = reason
    record.limit_hours = limit_hours
    record.stop_noticed = noticed
    session.add(record)
    # Before any insert of the slot's next running record, which the unique
    # running-slot index would refuse beside this one.
    session.flush()


def _when_stopped(record: LiveAppLaunchRecord, at: datetime | None) -> datetime:
    """``at`` held between the record's start and now, since a slot's report
    and a slot's boot time come from outside this server; now for None."""
    now = _utc_now()
    if at is None:
        return now
    return min(max(ensure_utc(at), ensure_utc(record.started_at)), now)


def _same_person(record: LiveAppLaunchRecord, person: LiveAppPerson) -> bool:
    return record.launcher_partition_id == person.partition_id and record.launcher_user_id == person.user_id


def _stop_for_claim(
    session: ScopedSession, *, slot: int, claimer: LiveAppPerson, lost_at: datetime | None
) -> UUID | None:
    """Stop slot ``slot``'s running record before ``claimer``'s launch, and
    return its id. Another person's record ends ``lost`` at ``lost_at``, the
    slot's current boot (_slot_boot_time)."""
    record = _running_record(session, slot=slot)
    if record is None:
        return None
    if _same_person(record, claimer):
        # The person approved a card naming the app it replaces, so no turn
        # repeats it.
        _stop(session, record, reason="replaced", noticed=True)
    else:
        # anton claims only a slot whose status runs the claimer's own app
        # (its slot choice never replaces anyone else's), so another person's
        # running record here is one the slot no longer shows: their app had
        # already gone without a report.
        _stop(session, record, reason="lost", stopped_at=lost_at)
    return record.id


def _launch_failed(session: ScopedSession, *, record_id: UUID | None, slot: int, boot_id: str | None) -> None:
    """Make the stop claim recorded for ``record_id`` a failed launch the
    person hasn't heard of, and stop slot ``slot``'s running record of boot
    ``boot_id`` the same way: launched wrote it, and anton then reset the slot."""
    record = None if record_id is None else session.get(LiveAppLaunchRecord, record_id)
    # Only the person's own app the claim replaced: another person's record
    # a claim ended was already gone (lost), and stays theirs as it is.
    if record is not None and record.state == "stopped" and record.stop_reason == "replaced":
        record.stop_reason = "launch_failed"
        record.stop_noticed = False
        session.add(record)
    running = _running_record(session, slot=slot) if boot_id is not None else None
    if running is not None and running.boot_id == boot_id:
        _stop(session, running, reason="launch_failed")


def _start_record(
    session: ScopedSession,
    *,
    launch: LiveAppLaunch,
    launcher: LiveAppPerson,
    project_id: UUID,
    claimed: UUID | None,
    lost_at: datetime | None,
) -> None:
    if claimed is not None:
        previous = session.get(LiveAppLaunchRecord, claimed)
        if (
            previous is not None
            and previous.stop_reason == "replaced"
            and previous.slug == launch.slug
            and previous.project_id == project_id
        ):
            previous.stop_reason = "relaunched"
            session.add(previous)
    # A record still running on the slot belongs to an app that already
    # ended there with no report (anton claims only a slot that runs an app),
    # or to another process's launch since the claim: either way this launch
    # now holds the slot. The person's own same app is a relaunch.
    stale = _running_record(session, slot=launch.slot)
    if stale is not None:
        if _same_person(stale, launcher) and stale.slug == launch.slug and stale.project_id == project_id:
            _stop(session, stale, reason="relaunched", noticed=True)
        else:
            _stop(session, stale, reason="lost", stopped_at=lost_at)
    session.add(
        LiveAppLaunchRecord(
            slot=launch.slot,
            launcher_user_id=launcher.user_id,
            launcher_partition_id=launcher.partition_id,
            launcher_email=launcher.email,
            project_id=project_id,
            slug=launch.slug,
            artifact_dir=launch.artifact_dir,
            address=launch.address,
            boot_id=launch.boot_id,
            started_at=_utc_now(),
        )
    )


# ── who may open an app ──────────────────────────────────────────────────────


def live_app_slot_addresses() -> tuple[str, ...]:
    """Each slot's control address, slot 1 first (anton's reading of
    ANTON_LIVE_APP_SLOTS, which check_startup has already parsed once)."""
    from anton.core.artifacts.live_app_slots import live_app_slot_addresses as addresses

    return tuple(addresses())


def is_live_app_slot(slot: int) -> bool:
    """Whether live apps are on and ``slot`` is one of their slots."""
    return live_apps_on() and 1 <= slot <= len(live_app_slot_addresses())


#: The slot-facing paths under /api/v1/live-apps/<n>/ that the bearer and
#: identity layers let through: the edge's access check and stopped page carry
#: only the visitor's cookie, and a slot's stop report carries nothing. Each
#: route checks its caller itself.
_SLOT_PATHS = ("access", "stopped-page", "stopped", "preview/bootstrap", "preview/authorize", "preview/exchange", "preview/connect", "preview/revoke-viewer")


def live_app_slot_paths() -> frozenset[str]:
    """The slot-facing path of each slot (_SLOT_PATHS); none with live apps off."""
    if not live_apps_on():
        return frozenset()
    count = len(live_app_slot_addresses())
    return frozenset(
        f"/api/v1/live-apps/{n}/{path}" for n in range(1, count + 1) for path in _SLOT_PATHS
    )


class ViewerSession(BaseModel):
    """viewer-login's /oauth2/userinfo answer (oauth2-proxy's UserInfo).
    ``user`` is the ID token's ``sub`` (oauth2-proxy's user claim), the id the
    recording proxy sends this server as X-User-Id. It answers 200 with ``{}``
    when it finds no session and no error, so an empty email is a visitor who
    isn't signed in."""

    model_config = ConfigDict(extra="ignore")

    user: str = ""
    email: str = ""


class ViewerIdentity(NamedTuple):
    """The signed-in visitor: their ``sub`` and their email, normalized."""

    user_id: str
    email: str


class LiveAppAccessUnavailable(RuntimeError):
    """viewer-login or the slot didn't answer, so nobody's access can be told.
    ``reason`` says which and how, as a code the log keeps."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


AccessOutcome = Literal["allowed", "signed_out", "refused"]


class _LaunchedBy(NamedTuple):
    user_id: str
    email: str
    boot_id: str


def _launched_by(session: ScopedSession, *, slot: int) -> _LaunchedBy | None:
    record = _running_record(session, slot=slot)
    if record is None:
        return None
    return _LaunchedBy(user_id=record.launcher_user_id, email=record.launcher_email, boot_id=record.boot_id)


@cache
def _ssl_context() -> ssl.SSLContext:
    # trust_env=False on the client stops httpx reading SSL_CERT_FILE and
    # SSL_CERT_DIR, so the context reads them itself, for a viewer-login on a
    # private CA. Built once: loading the CA bundle blocks the event loop, and
    # the edge asks the access route on every request to an app.
    return httpx.create_ssl_context(trust_env=True)


def _http_client() -> httpx.AsyncClient:
    # trust_env=False: viewer-login and the slots are sibling containers on
    # internal networks, never behind a proxy a deployment may set, which
    # would otherwise get the visitor's cookie and the launch token. A client
    # per check keeps no viewer-login cookie from one visitor to the next.
    return httpx.AsyncClient(
        timeout=ACCESS_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        verify=_ssl_context(),
    )


class _Reply(NamedTuple):
    status_code: int
    content: bytes


class _ReplyTooLarge(Exception):
    pass


async def _get(client: httpx.AsyncClient, url: str, *, headers: dict[str, str | bytes]) -> _Reply:
    """GET ``url``: its status and body, read no further than anton reads a
    slot's reply (MAX_REPLY_BYTES). Raises httpx.HTTPError as httpx does, and
    _ReplyTooLarge past the cap."""
    from anton.core.artifacts.live_app_slots import MAX_REPLY_BYTES

    async with client.stream("GET", url, headers=headers) as response:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body += chunk
            if len(body) > MAX_REPLY_BYTES:
                raise _ReplyTooLarge
        return _Reply(response.status_code, bytes(body))


async def check_access(slot: int, *, cookie: str) -> AccessOutcome:
    """Whether the visitor whose viewer sign-in cookie is ``cookie`` launched
    the app slot ``slot`` runs now.

    The launcher comes from the slot's running record and from the slot's own
    status, never from the request: both must name the visitor, by ``sub``
    and by email, and the slot must still run the launch the record names
    (its boot id). Raises LiveAppAccessUnavailable when viewer-login or the
    slot can't answer.
    """
    if not cookie:
        return "signed_out"
    async with _http_client() as client:
        visitor = await viewer_identity(client, cookie)
        if visitor is None:
            return "signed_out"
        launched_by = await run_db(partial(_launched_by, slot=slot), scope=SYSTEM_SCOPE)
        if launched_by is None or (launched_by.user_id, launched_by.email) != visitor:
            return "refused"
        slot_status = await _slot_status(client, slot)
    app = slot_status.app
    if (
        slot_status.state != "running"
        or app is None
        or slot_status.boot_id != launched_by.boot_id
        or app.launcher.subject != visitor.user_id
        or normalized_email(app.launcher.email) != visitor.email
    ):
        return "refused"
    return "allowed"


async def viewer_identity(client: httpx.AsyncClient, cookie: str) -> ViewerIdentity | None:
    """The visitor signed in with ``cookie``, or None for nobody. Raises
    LiveAppAccessUnavailable when viewer-login can't say."""
    if not cookie:
        return None
    url = get_app_settings().live_apps_viewer_login_url.strip().rstrip("/") + "/oauth2/userinfo"
    try:
        # The header's own bytes: Starlette decoded them as latin-1, and a
        # cookie the app's page set may hold any byte, which httpx can't
        # encode back from a str.
        reply = await _get(client, url, headers={"Cookie": cookie.encode("latin-1")})
    except (httpx.HTTPError, UnicodeError) as error:
        raise LiveAppAccessUnavailable(
            f"viewer-login didn't answer ({type(error).__name__})", reason="viewer_login_unreachable"
        ) from error
    except _ReplyTooLarge:
        raise LiveAppAccessUnavailable(
            "viewer-login's answer is too large", reason="viewer_login_reply_too_large"
        ) from None
    if reply.status_code == 401:
        return None
    if reply.status_code != 200:
        raise LiveAppAccessUnavailable(
            f"viewer-login answered {reply.status_code}", reason=f"viewer_login_status_{reply.status_code}"
        )
    try:
        session = ViewerSession.model_validate_json(reply.content)
    except ValidationError as error:
        raise LiveAppAccessUnavailable(
            "viewer-login's answer doesn't parse", reason="viewer_login_reply_unparsed"
        ) from error
    email = normalized_email(session.email)
    if not email:
        return None
    return ViewerIdentity(user_id=_canonical_sub(session.user), email=email)


def _canonical_sub(sub: str) -> str:
    """``sub`` in the form the identity middleware gives X-User-Id (a UUID's
    lowercase, hyphenated text), so the two compare; as it is otherwise."""
    try:
        return str(UUID(sub.strip()))
    except ValueError:
        return sub


async def _slot_status(client: httpx.AsyncClient, slot: int) -> SlotStatus:
    from anton.core.artifacts.live_app_slots import SlotStatus

    address = live_app_slot_addresses()[slot - 1]
    token = get_app_settings().live_apps_launch_token.get_secret_value()
    try:
        reply = await _get(client, f"{address}/status", headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as error:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot} didn't answer ({type(error).__name__})", reason="slot_unreachable"
        ) from error
    except _ReplyTooLarge:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot}'s status is too large", reason="slot_reply_too_large"
        ) from None
    if reply.status_code != 200:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot} answered {reply.status_code}", reason=f"slot_status_{reply.status_code}"
        )
    try:
        return SlotStatus.model_validate_json(reply.content)
    except ValidationError as error:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot}'s status doesn't parse", reason="slot_reply_unparsed"
        ) from error


# ── stops ────────────────────────────────────────────────────────────────────


class SlotStopReport(BaseModel):
    """A slot program's report that it ended its app (POST
    /api/v1/live-apps/<n>/stopped). ``limit_hours`` is the idle or maximum time
    for an ``idle`` or ``max_time`` stop; ``stopped_at`` is when the slot ended
    it, which the record keeps within the launch's own lifetime."""

    model_config = ConfigDict(extra="ignore")

    boot_id: str = Field(min_length=1, max_length=64)
    artifact: str = Field(default="", max_length=255)
    reason: Literal["idle", "max_time", "app_exited", "reset", "terminated"]
    limit_hours: float | None = Field(default=None, ge=0, le=100_000)
    stopped_at: datetime | None = None


class SlotReportRefused(PermissionError):
    """A stop report from an address other than its slot's own. ``reason``
    says how, as a code the log keeps."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


#: Headers a proxy in front of the server adds. uvicorn takes the client
#: address from X-Forwarded-For when the peer is loopback, which agent code in
#: the cowork container is, so a report carrying one is refused, and the
#: address checked is the peer's own.
_FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")


async def _slot_ips(slot: int) -> frozenset[str]:
    """The addresses slot ``slot``'s host resolves to from this container."""
    host = httpx.URL(live_app_slot_addresses()[slot - 1]).host
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as error:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot}'s host doesn't resolve ({error})", reason="slot_host_unresolved"
        ) from error
    return frozenset(info[4][0] for info in infos)


async def record_slot_stop(
    slot: int, report: SlotStopReport, *, peer: str | None, headers: Iterable[str]
) -> bool:
    """Stop slot ``slot``'s running record for ``report``, and say whether it did.

    Believed only from the slot's own address, with no forwarding header, and
    only for the boot id of the slot's running record, which only anton and
    the slot know, or of a record the slot's status already stopped for the
    same reason (reconcile_running), whose hours and time it fills in. A
    ``reset`` changes nothing: after a claim the record is
    already stopped, and otherwise (anton resets an exited slot, or one after
    a launch that didn't finish) the slot's next boot marks a record it still
    runs lost. Raises SlotReportRefused for any other address.
    """
    if peer is None or any(name.lower() in _FORWARDING_HEADERS for name in headers):
        raise SlotReportRefused(f"a stop report for live-app slot {slot} came through a proxy", reason="forwarded")
    if peer not in await _slot_ips(slot):
        raise SlotReportRefused(
            f"a stop report for live-app slot {slot} came from another address", reason="other_address"
        )
    if report.reason == "reset":
        return False
    return await run_db(partial(_stop_reported, slot=slot, report=report), scope=SYSTEM_SCOPE)


def _stop_reported(session: ScopedSession, *, slot: int, report: SlotStopReport) -> bool:
    limit_hours = report.limit_hours if report.reason in ("idle", "max_time") else None
    record = _running_record(session, slot=slot)
    if record is not None and record.boot_id == report.boot_id:
        _stop(session, record, reason=report.reason, limit_hours=limit_hours, stopped_at=report.stopped_at)
        return True
    # reconcile_running may have stopped it first, from the slot's status,
    # which says why but carries neither the hours nor when: the report
    # fills them in.
    stopped = session.exec(
        session.select(LiveAppLaunchRecord).where(
            LiveAppLaunchRecord.slot == slot,
            LiveAppLaunchRecord.boot_id == report.boot_id,
            LiveAppLaunchRecord.state == "stopped",
            LiveAppLaunchRecord.stop_reason == report.reason,
        )
    ).first()
    if stopped is None:
        return False
    stopped.limit_hours = limit_hours
    stopped.stopped_at = _when_stopped(stopped, report.stopped_at)
    session.add(stopped)
    return True


def _limit_text(limit_hours: float) -> str:
    """An idle or maximum time as a stop's sentence shows it: whole minutes
    under an hour ("2 min"), else hours ("2 h", "1.5 h")."""
    if limit_hours < 1:
        return f"{max(1, round(limit_hours * 60))} min"
    return f"{limit_hours:g} h"


def stop_reason_text(reason: StopReason | None, *, limit_hours: float | None) -> str | None:
    """The sentence the card, the stopped page and the turn show for a stop."""
    limit = "" if limit_hours is None else _limit_text(limit_hours)
    texts: dict[str, str] = {
        "idle": f"no requests for {limit}" if limit else "no requests for a while",
        "max_time": f"it reached its {limit} limit" if limit else "it reached its time limit",
        "app_exited": "its backend stopped by itself",
        "terminated": "the operator or a stack restart stopped it",
        # True both for another app of theirs and for a relaunch of this one
        # that didn't start, since claim stops the record before either.
        "replaced": "a launch of yours took its slot",
        "relaunched": "you relaunched it",
        "launch_failed": "a launch of yours into its slot didn't start",
        "lost": "its app slot restarted",
        "moved": "its conversation moved to another project",
    }
    return None if reason is None else texts[reason]


def stop_time_text(at: datetime, *, now: datetime | None = None) -> str:
    """When a stop happened, as the stopped page and the turn show it: "12:04
    UTC" for a stop today (UTC), "19 Oct 12:04 UTC" for an older one."""
    when = ensure_utc(at).astimezone(timezone.utc)
    today = ensure_utc(now or _utc_now()).astimezone(timezone.utc).date()
    if when.date() == today:
        return when.strftime("%H:%M UTC")
    return f"{when.day} {when.strftime('%b %H:%M')} UTC"


# ── the person's apps ────────────────────────────────────────────────────────


class PersonApp(NamedTuple):
    """The latest launch of one of a person's apps."""

    record_id: UUID
    project_id: UUID
    slug: str
    slot: int
    address: str
    boot_id: str
    running: bool
    started_at: datetime
    stopped_at: datetime | None
    stop_reason: StopReason | None
    limit_hours: float | None
    stop_noticed: bool

    @property
    def stop_text(self) -> str | None:
        return stop_reason_text(self.stop_reason, limit_hours=self.limit_hours)


def _as_person_app(record: LiveAppLaunchRecord) -> PersonApp:
    return PersonApp(
        record_id=record.id,
        project_id=record.project_id,
        slug=record.slug,
        slot=record.slot,
        address=record.address,
        boot_id=record.boot_id,
        running=record.state == "running",
        started_at=record.started_at,
        stopped_at=record.stopped_at,
        stop_reason=record.stop_reason,
        limit_hours=record.limit_hours,
        stop_noticed=record.stop_noticed,
    )


def _latest_apps(session: ScopedSession, *, person: LiveAppPerson) -> list[PersonApp]:
    """The latest launch of each of ``person``'s apps, newest first."""
    records = session.exec(
        session.select(LiveAppLaunchRecord)
        .where(
            LiveAppLaunchRecord.launcher_partition_id == person.partition_id,
            LiveAppLaunchRecord.launcher_user_id == person.user_id,
        )
        .order_by(LiveAppLaunchRecord.started_at.desc())
    ).all()
    latest: dict[tuple[UUID, str], PersonApp] = {}
    for record in records:
        latest.setdefault((record.project_id, record.slug), _as_person_app(record))
    return list(latest.values())


class _FoundStop(NamedTuple):
    """A stop reconcile_running found: why, and when (None for now)."""

    reason: StopReason
    at: datetime | None


#: The stops a slot reports (SlotStopReport), which its status repeats while
#: it stays exited. A ``reset`` isn't one (record_slot_stop).
_SLOT_STOPS: frozenset[str] = frozenset({"idle", "max_time", "app_exited", "terminated"})


def _mark_stopped(session: ScopedSession, *, found: dict[UUID, _FoundStop]) -> dict[UUID, PersonApp]:
    """Stop each record in ``found`` still running, for the reason and at the
    time found for it, and return each record as it now stands, one a report
    stopped meanwhile included."""
    now_standing: dict[UUID, PersonApp] = {}
    for record_id, stop in found.items():
        record = session.get(LiveAppLaunchRecord, record_id)
        if record is None:
            continue
        if record.state == "running":
            _stop(session, record, reason=stop.reason, stopped_at=stop.at)
        now_standing[record_id] = _as_person_app(record)
    return now_standing


async def _slot_boot_time(slot: int, *, boot_id: str | None = None) -> datetime | None:
    """When slot ``slot``'s current boot began (only when it is ``boot_id``,
    if given): the latest an app the slot ran before can have ended. None
    when the slot can't say. Never raises, since anton fails a launch whose
    host call raises."""
    try:
        async with _http_client() as client:
            status = await _slot_status(client, slot)
    except Exception:
        return None
    if boot_id is not None and status.boot_id != boot_id:
        return None
    return _booted_at(status)


def _booted_at(status: SlotStatus) -> datetime | None:
    """When the slot's current boot began, the latest its earlier app can
    have ended, or None when the slot's time doesn't parse."""
    try:
        return datetime.fromisoformat(status.booted_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


async def reconcile_running(apps: list[PersonApp]) -> list[PersonApp]:
    """``apps`` with every running one its slot no longer runs marked
    stopped, in the database too: ``lost`` for a slot on a later boot, or a
    slot the slot list no longer has, and the slot's own reason for one that
    shows the record's boot exited and says why. A slot that still runs the
    record's boot keeps it running, and so does one exited without a reason
    yet. A slot that can't answer leaves its app as recorded."""
    running = [app for app in apps if app.running]
    if not running:
        return apps
    count = len(live_app_slot_addresses())
    # A slot past the list (the operator lowered the slot count) has no
    # address to ask and no route for its report.
    found: dict[UUID, _FoundStop] = {
        app.record_id: _FoundStop("lost", None) for app in running if not 1 <= app.slot <= count
    }
    asked = [app for app in running if 1 <= app.slot <= count]
    async with _http_client() as client:
        statuses = await asyncio.gather(
            *(_slot_status(client, app.slot) for app in asked), return_exceptions=True
        )
    for app, status in zip(asked, statuses):
        if isinstance(status, LiveAppAccessUnavailable):
            logger.warning(
                "Live app slot status unreadable",
                extra={"event": "live_app_status_unreadable", "slot": app.slot, "reason": status.reason},
            )
        elif isinstance(status, BaseException):
            raise status
        elif status.boot_id != app.boot_id:
            found[app.record_id] = _FoundStop("lost", _booted_at(status))
        elif status.state == "exited" and status.app is not None and status.app.stop_reason in _SLOT_STOPS:
            # The slot's report of why can come before launched wrote this
            # record, which leaves it nothing to stop (_stop_reported), so
            # the status says why instead.
            found[app.record_id] = _FoundStop(status.app.stop_reason, None)
    if not found:
        return apps
    now_standing = await run_db(partial(_mark_stopped, found=found), scope=SYSTEM_SCOPE)
    return [now_standing.get(app.record_id, app) for app in apps]


async def person_apps(person: LiveAppPerson) -> list[PersonApp]:
    """The latest launch of each of ``person``'s apps, newest first, with any
    their slot no longer runs marked lost."""
    apps = await run_db(partial(_latest_apps, person=person), scope=SYSTEM_SCOPE)
    return await reconcile_running(apps)


# ── the stopped page ─────────────────────────────────────────────────────────


class StoppedPage(NamedTuple):
    status_code: int
    html: str


_NO_ACCESS = "You don't have access to this app."


def _page(body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Live app</title></head>"
        "<body style=\"font-family: system-ui, sans-serif; max-width: 36rem; margin: 4rem auto; padding: 0 1rem;\">"
        f"{body}</body></html>"
    )


class _SlotHistory(NamedTuple):
    """The visitor's latest launch on a slot (``mine``), with the space it
    lives in; the same app running on another slot now (``elsewhere``); and
    whether someone else's app runs on the slot now."""

    mine: PersonApp | None
    partition_id: str | None
    elsewhere: PersonApp | None
    other_running: bool


def _slot_history(session: ScopedSession, *, slot: int, user_id: str) -> _SlotHistory:
    """The launches on slot ``slot`` of the person whose ``sub`` is
    ``user_id``, as _SlotHistory tells them."""
    mine = session.exec(
        session.select(LiveAppLaunchRecord)
        .where(LiveAppLaunchRecord.slot == slot, LiveAppLaunchRecord.launcher_user_id == user_id)
        .order_by(LiveAppLaunchRecord.started_at.desc())
    ).first()
    running = _running_record(session, slot=slot)
    other_running = running is not None and running.launcher_user_id != user_id
    if mine is None:
        return _SlotHistory(None, None, None, other_running)
    elsewhere = session.exec(
        session.select(LiveAppLaunchRecord).where(
            LiveAppLaunchRecord.launcher_partition_id == mine.launcher_partition_id,
            LiveAppLaunchRecord.launcher_user_id == user_id,
            LiveAppLaunchRecord.project_id == mine.project_id,
            LiveAppLaunchRecord.slug == mine.slug,
            LiveAppLaunchRecord.slot != slot,
            LiveAppLaunchRecord.state == "running",
        )
    ).first()
    return _SlotHistory(
        _as_person_app(mine),
        mine.launcher_partition_id,
        None if elsewhere is None else _as_person_app(elsewhere),
        other_running,
    )


async def _app_name(*, partition_id: str, user_id: str, project_id: UUID, slug: str) -> str:
    """The name the person's app ``slug`` has in its metadata.json, as their
    card shows it, or the slug when it can't be read."""
    from cowork.db.scoped import scope_for_identity
    from cowork.services.projects import ProjectNotFoundError, ProjectService

    def project_path(session: ScopedSession) -> Path | None:
        try:
            return Path(ProjectService(session).get_project(project_id).path)
        except ProjectNotFoundError:
            return None

    project_dir = await run_db(project_path, scope=scope_for_identity(org_id=partition_id, user_id=user_id))
    if project_dir is None:
        return slug
    try:
        metadata = await asyncio.to_thread(read_app_metadata, project_dir, slug)
    except (LiveAppGone, LiveAppCopyRefused):
        return slug
    return shared_text(metadata.name) or slug


async def stopped_page(slot: int, *, cookie: str, owner_id: str | None = None) -> StoppedPage:
    """The page the edge shows at slot ``slot``'s address instead of a bare
    403, 502 or 503: why the visitor's app there stopped, or where it runs
    now, for a visitor who launched one there, and a plain refusal for anyone
    else. ``owner_id`` names the visitor when the request carries a preview
    lease of theirs instead of the viewer cookie."""
    refusal = StoppedPage(403, _page(f"<p>{_NO_ACCESS}</p>"))
    if not cookie:
        return refusal
    async with _http_client() as client:
        visitor = await viewer_identity(client, cookie)
    user_id = visitor.user_id if visitor is not None and visitor.user_id else owner_id
    if not user_id:
        return refusal
    history = await run_db(partial(_slot_history, slot=slot, user_id=user_id), scope=SYSTEM_SCOPE)
    app = history.mine
    if app is None or history.partition_id is None:
        return refusal
    if app.running:
        (app,) = await reconcile_running([app])
    elsewhere = history.elsewhere
    if elsewhere is not None and not app.running:
        (elsewhere,) = await reconcile_running([elsewhere])
    name = html.escape(
        await _app_name(partition_id=history.partition_id, user_id=user_id, project_id=app.project_id, slug=app.slug)
    )
    if app.running:
        return StoppedPage(
            200,
            _page(
                f"<p>Your app <code>{name}</code> isn't answering right now.</p>"
                "<p>Ask your agent to relaunch it.</p>"
            ),
        )
    if elsewhere is not None and elsewhere.running:
        address = html.escape(elsewhere.address)
        return StoppedPage(
            200, _page(f"<p>Your app <code>{name}</code> now runs at <a href=\"{address}\">{address}</a>.</p>")
        )
    stopped_at = stop_time_text(app.stopped_at or _utc_now())
    reason = html.escape(app.stop_text or "")
    body = f"<p>Your app <code>{name}</code> stopped at {stopped_at}: {reason}.</p><p>Ask your agent to relaunch it.</p>"
    if history.other_running:
        body += (
            "<p>This address now serves another person's app, so a relaunch gets a new one, "
            "which its card in Cowork shows.</p>"
        )
    return StoppedPage(200, _page(body))


# ── the app's files ──────────────────────────────────────────────────────────


class LiveAppCopyRefused(ValueError):
    """The owner's folder holds something the copy rule refuses. The message
    names the path, never content."""


class NotALiveApp(LiveAppCopyRefused):
    """The artifact isn't a full-stack app, so it can't be shared or copied as one."""


class LiveAppGone(LookupError):
    """The owner's app folder, or its metadata, is no longer there."""


class AppFile(NamedTuple):
    path: PurePosixPath
    content: bytes


class AppMetadata(BaseModel):
    """The few metadata.json keys a share reads and a copy carries. Lenient
    beyond them, unlike anton's Artifact, so an app with an older identity
    still shares."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    description: str = ""
    type: str
    primary: str | None = None
    datasources: list[DatasourceRef] = []


class SharedApp(NamedTuple):
    """What a copy carries of the owner's app."""

    name: str
    description: str
    type: str
    primary: str | None
    datasources: tuple[DatasourceRef, ...]
    files: tuple[AppFile, ...]


def _read_regular(folder: PinnedDir, name: str, *, path: PurePosixPath, limit: int) -> bytes | None:
    """The bytes of regular file ``name`` in ``folder``, or None when there is
    none. Refuses a link, a hard-linked file, anything but a regular file, and
    a file over ``limit`` bytes."""
    try:
        # Non-blocking, so a FIFO planted at the name can't hold the open.
        fd = dir_open(folder, name, os.O_RDONLY | O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise LiveAppCopyRefused(f"{path} is a link") from None
        raise LiveAppCopyRefused(f"{path} can't be read as a regular file") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LiveAppCopyRefused(f"{path} isn't a regular file")
        if info.st_nlink > 1:
            raise LiveAppCopyRefused(f"{path} has a second name (a hard link)")
        if info.st_size > limit:
            raise LiveAppCopyRefused(f"{path} is larger than {limit} bytes")
        return b"".join(read_bounded(fd, info.st_size))
    finally:
        os.close(fd)


def _check_name(name: str, *, path: PurePosixPath) -> None:
    if name in {"", ".", ".."} or "/" in name or "\\" in name or "\0" in name:
        raise LiveAppCopyRefused(f"{path} has a name the copy refuses")


class _LaunchCaps(NamedTuple):
    files: int
    file_bytes: int
    total_bytes: int


def _add_launched(out: list[AppFile], file: AppFile, *, caps: _LaunchCaps) -> None:
    """Add ``file`` to the launch files, refusing as soon as they pass a cap,
    so a huge tree is never read whole."""
    out.append(file)
    if len(out) > caps.files:
        raise LiveAppCopyRefused(f"the app has more than {caps.files} files")
    if sum(len(launched.content) for launched in out) > caps.total_bytes:
        raise LiveAppCopyRefused(f"the app's files are larger than {caps.total_bytes} bytes")


def _read_tree(folder: PinnedDir, path: PurePosixPath, *, depth: int, caps: _LaunchCaps, out: list[AppFile]) -> None:
    """Every regular file below ``folder``, at ``path`` in the app, into ``out``."""
    if depth > COPY_MAX_DEPTH:
        raise LiveAppCopyRefused(f"{path} is more than {COPY_MAX_DEPTH} folders deep")
    for entry in sorted(dir_scandir(folder), key=lambda entry: entry.name):
        child = path / entry.name
        _check_name(entry.name, path=child)
        if entry.is_symlink():
            raise LiveAppCopyRefused(f"{child} is a link")
        if entry.is_dir(follow_symlinks=False):
            below = open_pinned_child(folder, entry.name)
            try:
                _read_tree(below, child, depth=depth + 1, caps=caps, out=out)
            finally:
                below.close()
            continue
        content = _read_regular(folder, entry.name, path=child, limit=caps.file_bytes)
        if content is not None:
            _add_launched(out, AppFile(child, content), caps=caps)


def read_shared_app(project_dir: Path, slug: str) -> SharedApp:
    """The owner's app ``slug`` in ``project_dir``, read for a copy with
    no-follow opens: anton's launch files (LAUNCH_FILES, every file under
    LAUNCH_DIRS) within its caps, the PRD and the spec, and what metadata.json
    says of the app.

    Raises LiveAppGone when the folder or its metadata is gone, and
    LiveAppCopyRefused for a link (at the owner's project folder too), a
    hard-linked file, a name the copy refuses, a file over its cap, a folder
    that can't be read, or an app that isn't full-stack.
    """
    from anton.core.artifacts.live_app_slots import (
        LAUNCH_DIRS,
        LAUNCH_FILES,
        MAX_LAUNCH_FILE_BYTES,
        MAX_LAUNCH_FILES,
        MAX_LAUNCH_TOTAL_BYTES,
    )

    root = PurePosixPath()
    caps = _LaunchCaps(
        files=MAX_LAUNCH_FILES, file_bytes=MAX_LAUNCH_FILE_BYTES, total_bytes=MAX_LAUNCH_TOTAL_BYTES
    )
    _check_name(slug, path=PurePosixPath(slug))
    try:
        # opened_subdir_nofollow trusts its base, so the owner's project
        # folder itself is checked from the server's roots first.
        refuse_a_linked_folder(project_dir)
        with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH, slug) as folder:
            metadata = _read_metadata(folder)
            launched: list[AppFile] = []
            for name in LAUNCH_FILES:
                content = _read_regular(folder, name, path=root / name, limit=caps.file_bytes)
                if content is not None:
                    _add_launched(launched, AppFile(root / name, content), caps=caps)
            for name in LAUNCH_DIRS:
                try:
                    below = open_pinned_child(folder, name)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    if error.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise LiveAppCopyRefused(f"{name} isn't a folder") from None
                    raise
                try:
                    _read_tree(below, root / name, depth=1, caps=caps, out=launched)
                finally:
                    below.close()
            docs = []
            for name in COPY_DOCS:
                content = _read_regular(folder, name, path=root / name, limit=COPY_DOC_MAX_BYTES)
                if content is not None:
                    docs.append(AppFile(root / name, content))
    except OSError as error:
        raise _folder_refusal(error, slug=slug) from None
    if not any(file.path == root / "backend.py" for file in launched):
        raise LiveAppCopyRefused("backend.py is missing")
    return SharedApp(
        name=metadata.name,
        description=metadata.description,
        type=metadata.type,
        primary=metadata.primary,
        datasources=tuple(metadata.datasources),
        files=(*launched, *docs),
    )


def _read_metadata(folder: PinnedDir) -> AppMetadata:
    content = _read_regular(
        folder, METADATA_FILENAME, path=PurePosixPath(METADATA_FILENAME), limit=COPY_DOC_MAX_BYTES
    )
    if content is None:
        raise LiveAppGone(METADATA_FILENAME)
    try:
        metadata = AppMetadata.model_validate_json(content)
    except ValidationError:
        raise LiveAppCopyRefused(f"{METADATA_FILENAME} doesn't parse") from None
    if metadata.type not in LIVE_APP_TYPES:
        raise NotALiveApp(f"the artifact is a {metadata.type}, not a full-stack app")
    return metadata


def read_app_metadata(project_dir: Path, slug: str) -> AppMetadata:
    """What ``slug``'s metadata.json says of a full-stack app in
    ``project_dir``, read with no-follow opens. Raises LiveAppGone and
    LiveAppCopyRefused as read_shared_app does."""
    _check_name(slug, path=PurePosixPath(slug))
    try:
        refuse_a_linked_folder(project_dir)
        with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH, slug) as folder:
            return _read_metadata(folder)
    except OSError as error:
        raise _folder_refusal(error, slug=slug) from None


def _folder_refusal(error: OSError, *, slug: str) -> LiveAppGone | LiveAppCopyRefused:
    """What reading the owner's app ``slug`` raises for ``error``: gone for a
    missing folder, a refusal naming the folder for a link on the way or one
    the server can't read (a permission, say), so one bad folder never fails
    a list of other people's apps."""
    if isinstance(error, FileNotFoundError):
        return LiveAppGone(slug)
    if error.errno in (errno.ELOOP, errno.ENOTDIR):
        return LiveAppCopyRefused(f"the folder of {slug} is reached through a link")
    return LiveAppCopyRefused(f"the folder of {slug} can't be read")


def write_app_copy(project_dir: Path, app: SharedApp) -> str:
    """A new artifact in ``project_dir`` holding ``app``, and its slug.

    anton's store creates the folder, its metadata.json (the app's name,
    description, type and entry file, under a new id) and its README, and
    records the declared connections; then each file is written with no-follow
    opens. A failed write removes the new folder.
    """
    from anton.core.artifacts.store import ArtifactStore

    refuse_a_linked_project(project_dir)
    artifacts_base = project_artifacts_base(str(project_dir))
    makedirs_below_nofollow(project_dir, target=artifacts_base)
    store = ArtifactStore(artifacts_base)
    created = store.create(
        name=app.name, description=app.description, type=app.type, primary=app.primary
    )
    try:
        store.update(created.slug, datasources=list(app.datasources))
        with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH, created.slug) as folder:
            for file in app.files:
                _write_file(folder, file)
    except BaseException:
        remove_app_copy(project_dir, created.slug)
        raise
    return created.slug


def remove_app_copy(project_dir: Path, slug: str) -> None:
    """Remove the copy ``slug`` write_app_copy made in ``project_dir``,
    without following links."""
    with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH) as artifacts:
        dir_rmtree(artifacts, slug)


def _write_file(folder: PinnedDir, file: AppFile) -> None:
    *parents, name = file.path.parts
    opened: list[PinnedDir] = []
    here = folder
    try:
        for part in parents:
            try:
                dir_mkdir(here, part)
            except FileExistsError:
                pass
            here = open_pinned_child(here, part)
            opened.append(here)
        fd = dir_open(here, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, 0o644)
        try:
            view = memoryview(file.content)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
    finally:
        for pinned in reversed(opened):
            pinned.close()


# ── shares and copies ────────────────────────────────────────────────────────


def _recipient_is(recipient_email: str | None):
    if recipient_email is None:
        return LiveAppShare.recipient_email.is_(None)
    return LiveAppShare.recipient_email == recipient_email


def find_share(
    session: ScopedSession,
    *,
    owner_partition_id: str,
    project_id: UUID,
    slug: str,
    recipient_email: str | None,
) -> LiveAppShare | None:
    return session.exec(
        session.select(LiveAppShare).where(
            LiveAppShare.owner_partition_id == owner_partition_id,
            LiveAppShare.project_id == project_id,
            LiveAppShare.slug == slug,
            _recipient_is(recipient_email),
        )
    ).first()


class ShareRecord(NamedTuple):
    """A share as the share route answers it."""

    share_id: UUID
    slug: str
    recipient_email: str | None
    shared_at: datetime


def _as_share_record(share: LiveAppShare) -> ShareRecord:
    return ShareRecord(
        share_id=share.id,
        slug=share.slug,
        recipient_email=share.recipient_email,
        shared_at=share.created_at or _utc_now(),
    )


def record_share(
    session: ScopedSession,
    *,
    owner: LiveAppPerson,
    project_id: UUID,
    slug: str,
    recipient_email: str | None,
) -> ShareRecord:
    """The share of ``owner``'s app ``slug`` with ``recipient_email``, or with
    everyone for None: the one already recorded, or a new one."""
    keys = dict(
        owner_partition_id=owner.partition_id,
        project_id=project_id,
        slug=slug,
        recipient_email=recipient_email,
    )
    existing = find_share(session, **keys)
    if existing is not None:
        return _as_share_record(existing)
    share = LiveAppShare(owner_user_id=owner.user_id, owner_email=owner.email, **keys)
    session.add(share)
    try:
        session.commit()
    except IntegrityError:
        # The same share, recorded at the same moment by another request.
        session.rollback()
        existing = find_share(session, **keys)
        if existing is None:
            raise
        return _as_share_record(existing)
    session.refresh(share)
    return _as_share_record(share)


def _owned_by(owner: LiveAppPerson):
    """Shares ``owner`` made: their space and their ``sub``."""
    return sa.and_(
        LiveAppShare.owner_partition_id == owner.partition_id, LiveAppShare.owner_user_id == owner.user_id
    )


def shares_of_app(session: ScopedSession, *, owner: LiveAppPerson, project_id: UUID, slug: str) -> list[ShareRecord]:
    """Every share ``owner`` made of their app ``slug`` in project
    ``project_id``, newest first."""
    rows = session.exec(
        session.select(LiveAppShare)
        .where(_owned_by(owner), LiveAppShare.project_id == project_id, LiveAppShare.slug == slug)
        .order_by(LiveAppShare.created_at.desc())
    ).all()
    return [_as_share_record(row) for row in rows]


def delete_share(session: ScopedSession, *, owner: LiveAppPerson, share_id: UUID) -> bool:
    """Withdraw share ``share_id`` when ``owner`` made it, and say whether it
    did. Copies already made keep their own rows, so their "received from"."""
    row = session.exec(session.select(LiveAppShare).where(LiveAppShare.id == share_id, _owned_by(owner))).first()
    if row is None:
        return False
    session.delete(row)
    return True


class ShareForRecipient(NamedTuple):
    """A share someone made with the recipient, and where its app lives."""

    share_id: UUID
    owner_partition_id: str
    owner_user_id: str
    owner_email: str
    project_id: UUID
    slug: str
    everyone: bool
    shared_at: datetime


def _as_share_for_recipient(share: LiveAppShare) -> ShareForRecipient:
    return ShareForRecipient(
        share_id=share.id,
        owner_partition_id=share.owner_partition_id,
        owner_user_id=share.owner_user_id,
        owner_email=share.owner_email,
        project_id=share.project_id,
        slug=share.slug,
        everyone=share.recipient_email is None,
        shared_at=share.created_at or _utc_now(),
    )


def _shared_with(recipient: LiveAppPerson):
    """Shares with ``recipient``'s email or with everyone, from someone else."""
    return sa.and_(
        sa.or_(LiveAppShare.recipient_email == recipient.email, LiveAppShare.recipient_email.is_(None)),
        LiveAppShare.owner_partition_id != recipient.partition_id,
    )


def shares_with(session: ScopedSession, *, recipient: LiveAppPerson) -> list[ShareForRecipient]:
    """Every app shared with ``recipient``, newest first, each once: an app
    shared both with their address and with everyone is listed as the share
    with their address."""
    rows = session.exec(
        session.select(LiveAppShare)
        .where(_shared_with(recipient))
        .order_by(LiveAppShare.created_at.desc())
    ).all()
    by_app: dict[tuple[str, UUID, str], ShareForRecipient] = {}
    for row in rows:
        share = _as_share_for_recipient(row)
        app = (share.owner_partition_id, share.project_id, share.slug)
        listed = by_app.get(app)
        if listed is None or (listed.everyone and not share.everyone):
            by_app[app] = share
    return sorted(by_app.values(), key=lambda share: share.shared_at, reverse=True)


def share_for_recipient(
    session: ScopedSession, *, share_id: UUID, recipient: LiveAppPerson
) -> ShareForRecipient | None:
    """Share ``share_id`` when it is with ``recipient`` (_shared_with)."""
    row = session.exec(
        session.select(LiveAppShare).where(LiveAppShare.id == share_id, _shared_with(recipient))
    ).first()
    return None if row is None else _as_share_for_recipient(row)


def record_copy(
    session: ScopedSession,
    *,
    recipient: LiveAppPerson,
    project_id: UUID,
    slug: str,
    share: ShareForRecipient,
) -> None:
    session.add(
        LiveAppCopy(
            partition_id=recipient.partition_id,
            project_id=project_id,
            slug=slug,
            owner_email=share.owner_email,
            share_id=share.share_id,
        )
    )


def forget_copy(session: ScopedSession, *, recipient: LiveAppPerson, project_id: UUID, slug: str) -> None:
    """Remove the received mark of ``recipient``'s copy ``slug``, for a copy
    whose folder the duplicate removed again."""
    row = session.exec(
        session.select(LiveAppCopy).where(
            LiveAppCopy.partition_id == recipient.partition_id,
            LiveAppCopy.project_id == project_id,
            LiveAppCopy.slug == slug,
        )
    ).first()
    if row is not None:
        session.delete(row)


def received_from(session: ScopedSession, *, partition_id: str) -> dict[tuple[UUID, str], str]:
    """Whom each copy in ``partition_id``'s projects came from: the owner's
    email by (project id, slug)."""
    rows = session.exec(
        session.select(LiveAppCopy).where(LiveAppCopy.partition_id == partition_id)
    ).all()
    return {(row.project_id, row.slug): row.owner_email for row in rows}


class CopyOfShare(NamedTuple):
    project_id: UUID
    slug: str


def copies_of_shares(session: ScopedSession, *, recipient: LiveAppPerson) -> dict[UUID, list[CopyOfShare]]:
    """``recipient``'s copies by the share each came from. A copy whose folder
    is gone keeps its row, so the caller checks the folder (kept_copy)."""
    rows = session.exec(
        session.select(LiveAppCopy).where(LiveAppCopy.partition_id == recipient.partition_id)
    ).all()
    by_share: dict[UUID, list[CopyOfShare]] = {}
    for row in rows:
        by_share.setdefault(row.share_id, []).append(CopyOfShare(row.project_id, row.slug))
    return by_share


def kept_copy(project_dirs: Mapping[UUID, Path | None], copies: Iterable[CopyOfShare]) -> CopyOfShare | None:
    """The first of ``copies`` whose folder is still there, in a project the
    person still has (``project_dirs``, by project id): one copy per person
    per share, until they delete it."""
    for copy in copies:
        project_dir = project_dirs.get(copy.project_id)
        if project_dir is not None and _still_there(Path(project_artifacts_base(str(project_dir))), copy.slug):
            return copy
    return None


class ReceivedApp(NamedTuple):
    slug: str
    owner_email: str


def received_in_project(session: ScopedSession, *, person: LiveAppPerson, project_id: UUID) -> list[ReceivedApp]:
    """The copies ``person`` duplicated into project ``project_id`` and never
    launched, by slug: the ones whose ``backend.py`` the agent reads first."""
    launched = (
        session.select(LiveAppLaunchRecord)
        .where(
            LiveAppLaunchRecord.launcher_partition_id == person.partition_id,
            LiveAppLaunchRecord.project_id == project_id,
        )
    )
    launched_slugs = {record.slug for record in session.exec(launched).all()}
    rows = session.exec(
        session.select(LiveAppCopy)
        .where(LiveAppCopy.partition_id == person.partition_id, LiveAppCopy.project_id == project_id)
        .order_by(LiveAppCopy.slug)
    ).all()
    return [ReceivedApp(row.slug, row.owner_email) for row in rows if row.slug not in launched_slugs]


def _still_there(artifacts_dir: Path, slug: str) -> bool:
    """Whether the copy ``slug`` still has its folder, which a delete from the
    card or from a cell removes without touching its row. A folder that can't
    be checked counts as there, so the agent still hears the copy is someone
    else's."""
    try:
        return stat.S_ISDIR(os.lstat(artifacts_dir / slug).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        logger.warning(
            "Live app copy folder unchecked",
            extra={
                "event": "live_app_copy_unchecked",
                "artifact_slug": slug,
                "reason": errno.errorcode.get(error.errno or 0, "os_error"),
            },
        )
        return True


def _claim_notices(session: ScopedSession, *, record_ids: Iterable[UUID]) -> list[UUID]:
    """Mark the stops ``record_ids`` heard, and return the ones this call
    marked: a stop another turn marked first is that turn's to tell."""
    ids = list(record_ids)
    # FOR UPDATE: on Postgres a second turn's claim waits for the first's
    # commit, then reads stop_noticed again and skips what it marked.
    rows = session.exec(
        session.select(LiveAppLaunchRecord)
        .where(LiveAppLaunchRecord.id.in_(ids), LiveAppLaunchRecord.stop_noticed.is_(False))
        .with_for_update()
    ).all()
    for record in rows:
        record.stop_noticed = True
        session.add(record)
    return [record.id for record in rows]


def _unmark_notices(session: ScopedSession, *, record_ids: Iterable[UUID]) -> None:
    for record_id in record_ids:
        record = session.get(LiveAppLaunchRecord, record_id)
        if record is not None:
            record.stop_noticed = False
            session.add(record)


async def release_stop_notices(record_ids: Iterable[UUID]) -> None:
    """Hand back the stops a turn claimed (turn_live_apps) and didn't get to
    tell, because it failed or was stopped, so the person's next turn does."""
    ids = list(record_ids)
    if ids:
        await run_db(partial(_unmark_notices, record_ids=ids), scope=SYSTEM_SCOPE)


class TurnLiveApps(NamedTuple):
    """What a turn's live-app rule names: the copies in its project not yet
    launched, the person's running apps, and the stops they haven't heard of."""

    received: tuple[ReceivedApp, ...]
    running: tuple[PersonApp, ...]
    stopped: tuple[PersonApp, ...]


async def turn_live_apps(
    person: LiveAppPerson, *, project_id: UUID, artifacts_dir: Path, claimed_into: list[UUID]
) -> TurnLiveApps:
    """The live apps a turn of ``person``'s in ``project_id`` hears of, with
    ``artifacts_dir`` the project's artifacts folder. The turn claims each
    stop it names, so two turns at once never both tell it, into
    ``claimed_into``, which a Stop can't leave empty once the claim commits;
    the turn hands back what it didn't get to tell (release_stop_notices)."""
    received = await run_db(
        partial(received_in_project, person=person, project_id=project_id), scope=SYSTEM_SCOPE
    )
    received = [copy for copy in received if await asyncio.to_thread(_still_there, artifacts_dir, copy.slug)]
    apps = await person_apps(person)
    unheard = [
        app for app in apps if not app.running and not app.stop_noticed and app.stop_reason != "relaunched"
    ]
    if unheard:

        async def claim() -> None:
            claimed_into.extend(
                await run_db(
                    partial(_claim_notices, record_ids=[app.record_id for app in unheard]), scope=SYSTEM_SCOPE
                )
            )

        await run_to_completion(claim())
    claimed = set(claimed_into)
    return TurnLiveApps(
        received=tuple(received),
        running=tuple(app for app in apps if app.running),
        stopped=tuple(app for app in unheard if app.record_id in claimed),
    )


def follow_moved_app(
    session: ScopedSession,
    *,
    partition_id: str,
    source_project_id: UUID,
    old_slug: str,
    dest_project_id: UUID,
    new_slug: str,
) -> None:
    """Retarget the live-app rows of app ``old_slug`` in ``partition_id``'s
    project ``source_project_id`` to its new home after a conversation move
    took its folder there: its shares, its received mark and its launches,
    so recipients still list it and its card keeps its state. Rows already
    at the new address belong to a folder that is gone there, so they go.

    A launch still running stops as ``moved``: anton finds a running app by
    its folder, which the move changed, so it would no longer relaunch it
    after an edit, and a relaunch would take a second slot. The stop closes
    the old address and tells the person on their next turn."""
    keyed = (
        (LiveAppShare, LiveAppShare.owner_partition_id),
        (LiveAppCopy, LiveAppCopy.partition_id),
        (LiveAppLaunchRecord, LiveAppLaunchRecord.launcher_partition_id),
    )
    for model, partition in keyed:
        for stale in session.exec(
            session.select(model).where(
                partition == partition_id, model.project_id == dest_project_id, model.slug == new_slug
            )
        ).all():
            session.delete(stale)
    session.flush()
    for model, partition in keyed:
        for row in session.exec(
            session.select(model).where(
                partition == partition_id, model.project_id == source_project_id, model.slug == old_slug
            )
        ).all():
            row.project_id = dest_project_id
            row.slug = new_slug
            session.add(row)
            if isinstance(row, LiveAppLaunchRecord) and row.state == "running":
                _stop(session, row, reason="moved")


def shared_text(value: str) -> str:
    """A name or description from another person's metadata, as "Shared with you" shows it."""
    return value.strip()[:SHARED_TEXT_MAX_LENGTH]
