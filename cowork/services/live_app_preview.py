"""Owner-selected previews with one-use grants and revocable browser leases.

The app never receives these credentials: the edge strips Cookie before
proxying app requests. Only the trusted bootstrap exchanges a grant. Every
request rechecks the immutable viewer proof and the slot's current launch.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import partial
from http.cookies import SimpleCookie, CookieError
from typing import TypeVar
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from cryptography.fernet import InvalidToken
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import StaleDataError

from cowork.common.datetime_utils import ensure_utc
from cowork.common.encryption import decrypt, encrypt
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.units import run_db
from cowork.models.live_app import LiveAppLaunchRecord
from cowork.models.live_app_preview import LiveAppPreview, LiveAppPreviewOwner
from cowork.services import live_apps

COWORK_ORIGIN = "http://cowork.localhost:8080"
PREVIEW_COOKIE = "__Host-live-app-preview"
INTENT_SECONDS = 120
GRANT_SECONDS = 30
LEASE_SECONDS = 60
MAX_PREVIEWS = 8
T = TypeVar("T")
_TOKEN = re.compile(r"[0-9a-f]{64}")
_VIEWER_COOKIE = re.compile(r"_oauth2_proxy(?:_\d+)?")


class PreviewRefused(Exception):
    def __init__(self, message: str = "This preview is no longer available. Reload to try again.", status: int = 403):
        self.message, self.status = message, status


def now() -> datetime:
    return datetime.now(timezone.utc)


def app_origin(slot: int) -> str:
    return f"http://app{slot}.view.localhost:8080"


def digest(token: str) -> str:
    if not _TOKEN.fullmatch(token):
        raise PreviewRefused()
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def cookie_value(cookie: str, name: str) -> str:
    try:
        parsed = SimpleCookie(cookie)
        return parsed[name].value if name in parsed else ""
    except CookieError:
        return ""


def viewer_cookie_only(cookie: str) -> str:
    if len(cookie) > 32768:
        raise PreviewRefused()
    try:
        parsed = SimpleCookie(cookie)
    except CookieError:
        return ""
    return "; ".join(m.OutputString() for k, m in parsed.items() if _VIEWER_COOKIE.fullmatch(k))


def _launch(session: ScopedSession, launch_id: UUID) -> LiveAppLaunchRecord:
    launch = session.get(LiveAppLaunchRecord, launch_id)
    if launch is None or launch.state != "running" or launch.address != app_origin(launch.slot) + "/":
        raise PreviewRefused()
    return launch


def _owner(row: LiveAppPreview | LiveAppLaunchRecord, person: live_apps.LiveAppPerson) -> bool:
    if isinstance(row, LiveAppLaunchRecord):
        return (row.launcher_partition_id, row.launcher_user_id, row.launcher_email) == (person.partition_id, person.user_id, person.email)
    return (row.owner_partition_id, row.owner_user_id, row.owner_email) == (person.partition_id, person.user_id, person.email)


def _valid(row: LiveAppPreview | None, *, state: str | None = None) -> LiveAppPreview:
    if row is None or row.state == "revoked" or ensure_utc(row.expires_at) <= now() or (state and row.state != state):
        raise PreviewRefused()
    return row


def _by_token(session: ScopedSession, token: str, field: str) -> LiveAppPreview:
    row = session.exec(session.select(LiveAppPreview).where(getattr(LiveAppPreview, field) == digest(token))).first()
    return _valid(row)


async def _unit(fn: Callable[[ScopedSession], T]) -> T:
    try:
        return await run_db(fn, scope=SYSTEM_SCOPE)
    except (IntegrityError, StaleDataError):
        # A concurrent consume/revoke won. No secret leaves a failed commit.
        raise PreviewRefused() from None


def _touch_owner(session: ScopedSession, user_id: str, email: str) -> None:
    # The same stable owner key is available at both authenticated Cowork
    # logout and viewer logout, independent of the browser cookie's rotation.
    owner_id = uuid5(NAMESPACE_URL, f"live-app-preview:{user_id}:{email}")
    owner = session.get(LiveAppPreviewOwner, owner_id)
    if owner is None:
        owner = LiveAppPreviewOwner(id=owner_id)
    else:
        owner.marker += 1
    session.add(owner)
    session.flush()


async def _revoke_unit(fn: Callable[[ScopedSession], None]) -> None:
    for _ in range(3):
        try:
            await run_db(fn, scope=SYSTEM_SCOPE)
            return
        except (IntegrityError, StaleDataError):
            # A heartbeat or new lease won. Re-read all leases in a new
            # transaction; never return success from a rolled-back revoke.
            continue
    raise PreviewRefused("Couldn't close app previews. Please try again.", 503)


def _create(session: ScopedSession, person: live_apps.LiveAppPerson, launch_id: UUID, token: str) -> UUID:
    _touch_owner(session, person.user_id, person.email)
    launch = _launch(session, launch_id)
    if not _owner(launch, person):
        raise PreviewRefused()
    rows = session.exec(session.select(LiveAppPreview).where(
        LiveAppPreview.owner_partition_id == person.partition_id,
        LiveAppPreview.owner_user_id == person.user_id,
    ).order_by(LiveAppPreview.position)).all()
    occupied = set()
    reusable = None
    for row in rows:
        occupied.add(row.position)
        if reusable is None and (row.state == "revoked" or ensure_utc(row.expires_at) <= now()):
            reusable = row
    if reusable is None:
        position = next((n for n in range(MAX_PREVIEWS) if n not in occupied), None)
        if position is None:
            raise PreviewRefused("Close another app preview and try again.", 429)
        reusable = LiveAppPreview(owner_partition_id=person.partition_id, owner_user_id=person.user_id,
            owner_email=person.email, position=position, launch_id=launch_id, origin=COWORK_ORIGIN,
            intent_hash=digest(token), state="pending", expires_at=now() + timedelta(seconds=INTENT_SECONDS))
    else:
        reusable.id = uuid4()
        reusable.owner_email = person.email
        reusable.launch_id = launch_id
        reusable.origin = COWORK_ORIGIN
        reusable.intent_hash = digest(token)
        reusable.grant_hash = reusable.session_hash = reusable.viewer_cookie = None
        reusable.state = "pending"
        reusable.expires_at = now() + timedelta(seconds=INTENT_SECONDS)
    session.add(reusable)
    session.flush()
    return reusable.id


async def create(person: live_apps.LiveAppPerson, launch_id: UUID, origin: str) -> tuple[UUID, str]:
    if origin != COWORK_ORIGIN:
        raise PreviewRefused()
    token = secrets.token_hex(32)
    record_id = await _unit(partial(_create, person=person, launch_id=launch_id, token=token))
    return record_id, token


def _snapshot(session: ScopedSession, token: str, field: str, slot: int) -> tuple[LiveAppPreview, LiveAppLaunchRecord]:
    row = _by_token(session, token, field)
    launch = _launch(session, row.launch_id)
    if launch.slot != slot or row.origin != COWORK_ORIGIN or (row.owner_user_id, row.owner_email, row.owner_partition_id) != (
        launch.launcher_user_id, launch.launcher_email, launch.launcher_partition_id
    ):
        raise PreviewRefused()
    return row, launch


async def _verify_viewer(slot: int, cookie: str, launch: LiveAppLaunchRecord) -> None:
    async with live_apps._http_client() as client:
        visitor = await live_apps.viewer_identity(client, cookie)
        if visitor is None:
            raise PreviewRefused("Sign in to the app viewer to continue.", 401)
        if (visitor.user_id, visitor.email) != (launch.launcher_user_id, launch.launcher_email):
            raise PreviewRefused("Sign in to the app viewer with the account that launched this app.")
        running = await live_apps._slot_status(client, slot)
    app = running.app
    if running.state != "running" or running.boot_id != launch.boot_id or app is None or (
        app.launcher.subject, live_apps.normalized_email(app.launcher.email)
    ) != (launch.launcher_user_id, launch.launcher_email):
        raise PreviewRefused()


def _grant(session: ScopedSession, token: str, slot: int, launch_id: UUID, proof: str, grant: str) -> None:
    row, launch = _snapshot(session, token, "intent_hash", slot)
    _valid(row, state="pending")
    if launch.id != launch_id:
        raise PreviewRefused()
    row.state = "granted"
    row.viewer_cookie = proof
    row.grant_hash = digest(grant)
    row.expires_at = now() + timedelta(seconds=GRANT_SECONDS)
    session.add(row)


async def authorize(slot: int, intent: str, cookie: str) -> str:
    row, launch = await _unit(partial(_snapshot, token=intent, field="intent_hash", slot=slot))
    _valid(row, state="pending")
    cookie = viewer_cookie_only(cookie)
    await _verify_viewer(slot, cookie, launch)
    grant = secrets.token_hex(32)
    await _unit(partial(_grant, token=intent, slot=slot, launch_id=launch.id, proof=encrypt(cookie), grant=grant))
    return grant


def _exchange(session: ScopedSession, grant: str, slot: int, token: str) -> None:
    row, _ = _snapshot(session, grant, "grant_hash", slot)
    _valid(row, state="granted")
    row.state = "active"
    row.grant_hash = None
    row.session_hash = digest(token)
    row.expires_at = now() + timedelta(seconds=LEASE_SECONDS)
    session.add(row)


async def exchange(slot: int, grant: str) -> str:
    token = secrets.token_hex(32)
    await _unit(partial(_exchange, grant=grant, slot=slot, token=token))
    return token


async def check(slot: int, cookie: str, *, intent: str | None = None) -> LiveAppPreview:
    if intent is not None:
        target, _ = await _unit(partial(_snapshot, token=intent, field="intent_hash", slot=slot))
        if target.state != "active":
            # A newly opened pane must establish its own lease even when the
            # browser still holds a cookie from a pane that was just closed.
            raise PreviewRefused("Sign in to the app viewer to continue.", 401)
    token = cookie_value(cookie, PREVIEW_COOKIE)
    if not token:
        raise PreviewRefused("Sign in to the app viewer to continue.", 401)
    row, launch = await _unit(partial(_snapshot, token=token, field="session_hash", slot=slot))
    _valid(row, state="active")
    if intent is not None and row.intent_hash != digest(intent):
        raise PreviewRefused("Sign in to the app viewer to continue.", 401)
    if not row.viewer_cookie:
        raise PreviewRefused()
    try:
        proof = decrypt(row.viewer_cookie)
    except InvalidToken:
        raise PreviewRefused() from None
    await _verify_viewer(slot, proof, launch)
    # A revoke or replacement during external checks cannot turn into success.
    def unchanged(session: ScopedSession) -> None:
        current = _valid(session.get(LiveAppPreview, row.id), state="active")
        _launch(session, launch.id)
        if current.session_hash != row.session_hash:
            raise PreviewRefused()
    await _unit(unchanged)
    return row


async def keepalive(person: live_apps.LiveAppPerson, record_id: UUID) -> None:
    def read(session: ScopedSession) -> tuple[LiveAppPreview, LiveAppLaunchRecord]:
        row = _valid(session.get(LiveAppPreview, record_id))
        if not _owner(row, person):
            raise PreviewRefused()
        return row, _launch(session, row.launch_id)
    row, launch = await _unit(read)
    if row.state != "active":
        return
    if not row.viewer_cookie:
        raise PreviewRefused()
    try:
        proof = decrypt(row.viewer_cookie)
    except InvalidToken:
        raise PreviewRefused() from None
    await _verify_viewer(launch.slot, proof, launch)
    def renew(session: ScopedSession) -> None:
        current = _valid(session.get(LiveAppPreview, record_id), state="active")
        _launch(session, current.launch_id)
        if current.session_hash != row.session_hash or not _owner(current, person):
            raise PreviewRefused()
        current.expires_at = now() + timedelta(seconds=LEASE_SECONDS)
        session.add(current)
    await _unit(renew)


def _revoke_rows(session: ScopedSession, *, user_id: str, email: str, partition: str | None, record_id: UUID | None) -> None:
    if record_id is None:
        _touch_owner(session, user_id, email)
    query = session.select(LiveAppPreview).where(LiveAppPreview.owner_user_id == user_id, LiveAppPreview.owner_email == email)
    if partition is not None:
        query = query.where(LiveAppPreview.owner_partition_id == partition)
    if record_id is not None:
        query = query.where(LiveAppPreview.id == record_id)
    for row in session.exec(query).all():
        row.state = "revoked"
        row.viewer_cookie = row.grant_hash = row.session_hash = None
        session.add(row)


async def revoke(person: live_apps.LiveAppPerson, record_id: UUID | None = None) -> None:
    await _revoke_unit(partial(_revoke_rows, user_id=person.user_id, email=person.email, partition=person.partition_id, record_id=record_id))


async def revoke_viewer(cookie: str) -> None:
    cookie = viewer_cookie_only(cookie)
    if not cookie:
        return
    async with live_apps._http_client() as client:
        visitor = await live_apps.viewer_identity(client, cookie)
    if visitor is not None:
        await _revoke_unit(partial(_revoke_rows, user_id=visitor.user_id, email=visitor.email, partition=None, record_id=None))
