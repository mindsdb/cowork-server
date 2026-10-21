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
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
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

from cowork.code_sandbox import live_apps_on, refuse_a_linked_project
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
from cowork.db.units import run_db
from cowork.models.live_app import LiveAppCopy, LiveAppLaunchRecord, LiveAppShare, StopReason
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

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def normalized_email(value: str) -> str:
    """``value`` as live apps compare addresses: trimmed and lowercased."""
    return value.strip().lower()


def is_email(value: str) -> bool:
    return _EMAIL.fullmatch(value) is not None


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
    of it."""
    if principal is None or not principal.user_id or not principal.org_id:
        return None
    email = normalized_email(principal.email)
    if not email:
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
    until a later launch succeeds. ``launched`` starts the new record once the
    slot has accepted, ending any record still running there (_start_record).
    """

    def __init__(self, launcher: LiveAppPerson, *, project_id: UUID | None, launch_token: str) -> None:
        from anton.core.artifacts.live_app_slots import LiveAppLauncher

        self.launcher = LiveAppLauncher(subject=launcher.user_id, email=launcher.email)
        self.launch_token = launch_token
        self._person = launcher
        self._project_id = project_id
        # The record each claim stopped, by slot, so launched can tell the
        # person's relaunch of the same app from a replacement.
        self._claimed: dict[int, UUID] = {}

    async def claim(self, slot: int) -> None:
        stopped = await run_db(partial(_stop_for_claim, slot=slot, claimer=self._person), scope=SYSTEM_SCOPE)
        if stopped is None:
            self._claimed.pop(slot, None)
        else:
            self._claimed[slot] = stopped

    async def launched(self, launch: LiveAppLaunch) -> None:
        await run_db(
            partial(
                _start_record,
                launch=launch,
                launcher=self._person,
                project_id=self._project_id,
                claimed=self._claimed.pop(launch.slot, None),
            ),
            scope=SYSTEM_SCOPE,
        )


def live_app_host_for_turn(*, project_id: UUID | None) -> CoworkLiveAppHost | None:
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
) -> None:
    record.state = "stopped"
    record.stopped_at = _utc_now()
    record.stop_reason = reason
    record.limit_hours = limit_hours
    record.stop_noticed = noticed
    session.add(record)
    # Before any insert of the slot's next running record, which the unique
    # running-slot index would refuse beside this one.
    session.flush()


def _same_person(record: LiveAppLaunchRecord, person: LiveAppPerson) -> bool:
    return record.launcher_partition_id == person.partition_id and record.launcher_user_id == person.user_id


def _stop_for_claim(session: ScopedSession, *, slot: int, claimer: LiveAppPerson) -> UUID | None:
    """Stop slot ``slot``'s running record before ``claimer``'s launch, and
    return its id."""
    record = _running_record(session, slot=slot)
    if record is None:
        return None
    if _same_person(record, claimer):
        # The person approved a card naming the app it replaces, so no turn
        # repeats it.
        _stop(session, record, reason="replaced", noticed=True)
    else:
        # anton claims a slot that ran another person's app only once that
        # slot is empty (its slot choice never replaces anyone else's app), so
        # their app had already gone without a report.
        _stop(session, record, reason="lost")
    return record.id


def _start_record(
    session: ScopedSession,
    *,
    launch: LiveAppLaunch,
    launcher: LiveAppPerson,
    project_id: UUID | None,
    claimed: UUID | None,
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
            _stop(session, stale, reason="lost")
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
_SLOT_PATHS = ("access", "stopped-page", "stopped")


def live_app_slot_paths() -> frozenset[str]:
    """The slot-facing path of each slot (_SLOT_PATHS); none with live apps off."""
    if not live_apps_on():
        return frozenset()
    count = len(live_app_slot_addresses())
    return frozenset(
        f"/api/v1/live-apps/{n}/{path}" for n in range(1, count + 1) for path in _SLOT_PATHS
    )


class ViewerSession(BaseModel):
    """viewer-login's /oauth2/userinfo answer (oauth2-proxy's UserInfo). It
    answers 200 with ``{}`` when it finds no session and no error, so an empty
    email is a visitor who isn't signed in."""

    model_config = ConfigDict(extra="ignore")

    email: str = ""


class LiveAppAccessUnavailable(RuntimeError):
    """viewer-login or the slot didn't answer, so nobody's access can be told."""


AccessOutcome = Literal["allowed", "signed_out", "refused"]


class _LaunchedBy(NamedTuple):
    email: str
    boot_id: str


def _launched_by(session: ScopedSession, *, slot: int) -> _LaunchedBy | None:
    record = _running_record(session, slot=slot)
    return None if record is None else _LaunchedBy(email=record.launcher_email, boot_id=record.boot_id)


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=ACCESS_TIMEOUT_SECONDS, follow_redirects=False)


async def check_access(slot: int, *, cookie: str) -> AccessOutcome:
    """Whether the visitor whose viewer sign-in cookie is ``cookie`` launched
    the app slot ``slot`` runs now.

    The launcher comes from the slot's running record and from the slot's own
    status, never from the request: both must name the visitor, and the slot
    must still run the launch the record names (its boot id). Raises
    LiveAppAccessUnavailable when viewer-login or the slot can't answer.
    """
    async with _http_client() as client:
        email = await viewer_email(client, cookie)
        if not email:
            return "signed_out"
        launched_by = await run_db(partial(_launched_by, slot=slot), scope=SYSTEM_SCOPE)
        if launched_by is None or launched_by.email != email:
            return "refused"
        slot_status = await _slot_status(client, slot)
    app = slot_status.app
    if (
        slot_status.state != "running"
        or app is None
        or slot_status.boot_id != launched_by.boot_id
        or normalized_email(app.launcher.email) != email
    ):
        return "refused"
    return "allowed"


async def viewer_email(client: httpx.AsyncClient, cookie: str) -> str:
    """The email of the visitor signed in with ``cookie``, normalized, or ""
    for nobody. Raises LiveAppAccessUnavailable when viewer-login can't say."""
    if not cookie:
        return ""
    url = get_app_settings().live_apps_viewer_login_url.rstrip("/") + "/oauth2/userinfo"
    try:
        response = await client.get(url, headers={"Cookie": cookie})
    except httpx.HTTPError as error:
        raise LiveAppAccessUnavailable(
            f"viewer-login didn't answer ({type(error).__name__})"
        ) from error
    if response.status_code == 401:
        return ""
    if response.status_code != 200:
        raise LiveAppAccessUnavailable(f"viewer-login answered {response.status_code}")
    try:
        session = ViewerSession.model_validate_json(response.content)
    except ValidationError as error:
        raise LiveAppAccessUnavailable("viewer-login's answer doesn't parse") from error
    return normalized_email(session.email)


async def _slot_status(client: httpx.AsyncClient, slot: int) -> SlotStatus:
    from anton.core.artifacts.live_app_slots import SlotStatus

    address = live_app_slot_addresses()[slot - 1]
    token = get_app_settings().live_apps_launch_token.get_secret_value()
    try:
        response = await client.get(f"{address}/status", headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as error:
        raise LiveAppAccessUnavailable(
            f"live-app slot {slot} didn't answer ({type(error).__name__})"
        ) from error
    if response.status_code != 200:
        raise LiveAppAccessUnavailable(f"live-app slot {slot} answered {response.status_code}")
    try:
        return SlotStatus.model_validate_json(response.content)
    except ValidationError as error:
        raise LiveAppAccessUnavailable(f"live-app slot {slot}'s status doesn't parse") from error


# ── stops ────────────────────────────────────────────────────────────────────


class SlotStopReport(BaseModel):
    """A slot program's report that it ended its app (POST
    /api/v1/live-apps/<n>/stopped). ``limit_hours`` is the idle or maximum time
    for an ``idle`` or ``max_time`` stop."""

    model_config = ConfigDict(extra="ignore")

    boot_id: str = Field(min_length=1, max_length=64)
    artifact: str = Field(default="", max_length=255)
    reason: Literal["idle", "max_time", "app_exited", "reset", "terminated"]
    limit_hours: float | None = Field(default=None, ge=0, le=100_000)


class SlotReportRefused(PermissionError):
    """A stop report from an address other than its slot's own."""


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
        raise LiveAppAccessUnavailable(f"live-app slot {slot}'s host doesn't resolve ({error})") from error
    return frozenset(info[4][0] for info in infos)


async def record_slot_stop(
    slot: int, report: SlotStopReport, *, peer: str | None, headers: Iterable[str]
) -> bool:
    """Stop slot ``slot``'s running record for ``report``, and say whether it did.

    Believed only from the slot's own address, with no forwarding header, and
    only for the boot id of the slot's running record, which only anton and
    the slot know. A ``reset`` changes nothing: anton resets a slot only
    after its claim stopped the record. Raises SlotReportRefused for any other
    address.
    """
    if peer is None or any(name.lower() in _FORWARDING_HEADERS for name in headers):
        raise SlotReportRefused(f"a stop report for live-app slot {slot} came through a proxy")
    if peer not in await _slot_ips(slot):
        raise SlotReportRefused(f"a stop report for live-app slot {slot} came from another address")
    if report.reason == "reset":
        return False
    return await run_db(partial(_stop_reported, slot=slot, report=report), scope=SYSTEM_SCOPE)


def _stop_reported(session: ScopedSession, *, slot: int, report: SlotStopReport) -> bool:
    record = _running_record(session, slot=slot)
    if record is None or record.boot_id != report.boot_id:
        return False
    limit_hours = report.limit_hours if report.reason in ("idle", "max_time") else None
    _stop(session, record, reason=report.reason, limit_hours=limit_hours)
    return True


def stop_reason_text(reason: StopReason | None, *, limit_hours: float | None) -> str | None:
    """The sentence the card, the stopped page and the turn show for a stop."""
    hours = "" if limit_hours is None else f"{limit_hours:g}"
    texts: dict[str, str] = {
        "idle": f"no requests for {hours} h" if hours else "no requests for a while",
        "max_time": f"it reached its {hours} h limit" if hours else "it reached its time limit",
        "app_exited": "its backend stopped by itself",
        "terminated": "the operator or a stack restart stopped it",
        "replaced": "you launched another app in its slot",
        "relaunched": "you relaunched it",
        "lost": "its app slot restarted",
    }
    return None if reason is None else texts[reason]


# ── the person's apps ────────────────────────────────────────────────────────


class PersonApp(NamedTuple):
    """The latest launch of one of a person's apps."""

    record_id: UUID
    project_id: UUID | None
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
    latest: dict[tuple[UUID | None, str], PersonApp] = {}
    for record in records:
        latest.setdefault((record.project_id, record.slug), _as_person_app(record))
    return list(latest.values())


def _mark_lost(session: ScopedSession, *, record_ids: Iterable[UUID]) -> None:
    for record_id in record_ids:
        record = session.get(LiveAppLaunchRecord, record_id)
        if record is not None and record.state == "running":
            _stop(session, record, reason="lost")


async def reconcile_running(apps: list[PersonApp]) -> list[PersonApp]:
    """``apps`` with every running one its slot no longer runs marked
    stopped, ``lost``, in the database too. A slot that can't answer leaves
    its app as recorded."""
    running = [app for app in apps if app.running]
    if not running:
        return apps
    async with _http_client() as client:
        statuses = await asyncio.gather(
            *(_slot_status(client, app.slot) for app in running), return_exceptions=True
        )
    lost = {
        app.record_id
        for app, status in zip(running, statuses)
        if not isinstance(status, BaseException)
        and (status.state != "running" or status.boot_id != app.boot_id)
    }
    for app, status in zip(running, statuses):
        if isinstance(status, LiveAppAccessUnavailable):
            logger.warning("live app %d: its status can't be read: %s", app.slot, status)
        elif isinstance(status, BaseException):
            raise status
    if not lost:
        return apps
    await run_db(partial(_mark_lost, record_ids=lost), scope=SYSTEM_SCOPE)
    now = _utc_now()
    return [
        app._replace(running=False, stopped_at=now, stop_reason="lost", stop_noticed=False)
        if app.record_id in lost
        else app
        for app in apps
    ]


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


def _slot_history(session: ScopedSession, *, slot: int, email: str) -> tuple[PersonApp | None, bool]:
    """``email``'s latest launch on slot ``slot``, and whether someone else's
    app runs there now."""
    mine = session.exec(
        session.select(LiveAppLaunchRecord)
        .where(LiveAppLaunchRecord.slot == slot, LiveAppLaunchRecord.launcher_email == email)
        .order_by(LiveAppLaunchRecord.started_at.desc())
    ).first()
    running = _running_record(session, slot=slot)
    other_running = running is not None and running.launcher_email != email
    return (None if mine is None else _as_person_app(mine)), other_running


async def stopped_page(slot: int, *, cookie: str) -> StoppedPage:
    """The page the edge shows at slot ``slot``'s address instead of a bare
    403, 502 or 503: why the visitor's app there stopped, for a visitor who
    launched one there, and a plain refusal for anyone else."""
    async with _http_client() as client:
        email = await viewer_email(client, cookie)
    refusal = StoppedPage(403, _page(f"<p>{_NO_ACCESS}</p>"))
    if not email:
        return refusal
    app, other_running = await run_db(partial(_slot_history, slot=slot, email=email), scope=SYSTEM_SCOPE)
    if app is None:
        return refusal
    if app.running:
        (app,) = await reconcile_running([app])
    name = html.escape(app.slug)
    if app.running:
        return StoppedPage(
            200,
            _page(
                f"<p>Your app <code>{name}</code> isn't answering right now.</p>"
                "<p>Ask your agent to relaunch it.</p>"
            ),
        )
    stopped_at = (app.stopped_at or _utc_now()).astimezone(timezone.utc).strftime("%H:%M")
    reason = html.escape(app.stop_text or "")
    body = f"<p>Your app <code>{name}</code> stopped at {stopped_at} UTC: {reason}.</p><p>Ask your agent to relaunch it.</p>"
    if other_running:
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
    LiveAppCopyRefused for a link, a hard-linked file, a name the copy
    refuses, a file over its cap, or an app that isn't full-stack.
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
    except FileNotFoundError:
        raise LiveAppGone(slug) from None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise LiveAppCopyRefused(f"the folder of {slug} is reached through a link") from None
        raise
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
        with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH, slug) as folder:
            return _read_metadata(folder)
    except FileNotFoundError:
        raise LiveAppGone(slug) from None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise LiveAppCopyRefused(f"the folder of {slug} is reached through a link") from None
        raise


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
        with opened_subdir_nofollow(project_dir, *_ARTIFACTS_SUBPATH) as artifacts:
            dir_rmtree(artifacts, created.slug)
        raise
    return created.slug


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


def received_from(session: ScopedSession, *, partition_id: str) -> dict[tuple[UUID, str], str]:
    """Whom each copy in ``partition_id``'s projects came from: the owner's
    email by (project id, slug)."""
    rows = session.exec(
        session.select(LiveAppCopy).where(LiveAppCopy.partition_id == partition_id)
    ).all()
    return {(row.project_id, row.slug): row.owner_email for row in rows}


class ReceivedApp(NamedTuple):
    slug: str
    owner_email: str


def received_in_project(session: ScopedSession, *, person: LiveAppPerson, project_id: UUID) -> list[ReceivedApp]:
    """The copies ``person`` duplicated into project ``project_id``, by slug."""
    rows = session.exec(
        session.select(LiveAppCopy)
        .where(LiveAppCopy.partition_id == person.partition_id, LiveAppCopy.project_id == project_id)
        .order_by(LiveAppCopy.slug)
    ).all()
    return [ReceivedApp(row.slug, row.owner_email) for row in rows]


def _mark_noticed(session: ScopedSession, *, record_ids: Iterable[UUID]) -> None:
    for record_id in record_ids:
        record = session.get(LiveAppLaunchRecord, record_id)
        if record is not None:
            record.stop_noticed = True
            session.add(record)


class TurnLiveApps(NamedTuple):
    """What a turn's live-app rule names: the copies in its project, the
    person's running apps, and the stops they haven't heard of yet."""

    received: tuple[ReceivedApp, ...]
    running: tuple[PersonApp, ...]
    stopped: tuple[PersonApp, ...]


async def turn_live_apps(person: LiveAppPerson, *, project_id: UUID) -> TurnLiveApps:
    """The live apps a turn of ``person``'s in ``project_id`` hears of. A stop
    it names counts as heard, so the next turn doesn't repeat it."""
    received = await run_db(
        partial(received_in_project, person=person, project_id=project_id), scope=SYSTEM_SCOPE
    )
    apps = await person_apps(person)
    stopped = tuple(
        app for app in apps if not app.running and not app.stop_noticed and app.stop_reason != "relaunched"
    )
    if stopped:
        await run_db(
            partial(_mark_noticed, record_ids=[app.record_id for app in stopped]), scope=SYSTEM_SCOPE
        )
    return TurnLiveApps(
        received=tuple(received),
        running=tuple(app for app in apps if app.running),
        stopped=stopped,
    )


def shared_text(value: str) -> str:
    """A name or description from another person's metadata, as "Shared with you" shows it."""
    return value.strip()[:SHARED_TEXT_MAX_LENGTH]
