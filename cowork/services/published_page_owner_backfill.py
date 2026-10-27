"""Import only unambiguous, publisher-confirmed owners from personal publish history."""
from __future__ import annotations

import json
import os
import stat
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from anton.publisher import list_published
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from cowork.common.paths import O_NOFOLLOW, O_NONBLOCK, cowork_home, dir_open, opened_subdir_nofollow
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.published_page import PublishedPageOwner
from cowork.models.setting import Setting
from cowork.services.artifacts import is_publish_record_name
from cowork.services.publish import _resolve_publish_endpoint
from cowork.services.published_page_ownership import (
    BACKFILL_SENTINEL_KEY,
    find_owner,
    protects_publisher,
    publisher_key,
)
from cowork.services.shared_resources import RESOURCE_LOCK_TIMEOUT_SECONDS, _DatabaseResourceLocks, _process_lock

SENTINEL_KEY = BACKFILL_SENTINEL_KEY
_LOCK_ID = ("global", "published_page", SENTINEL_KEY)
_MAX_RECORD_BYTES = 1024 * 1024


class PublishedPageBackfillUnavailable(RuntimeError):
    """The pass cannot rule out conflicting history or verify the publisher."""


class _HistoryRow(BaseModel):
    # cowork.services.publish.publish_artifact writes exactly these history fields.
    model_config = ConfigDict(strict=True)

    artifact: str = Field(min_length=1)
    artifactName: str = Field(min_length=1)
    url: str = Field(min_length=1)
    reportId: str = Field(min_length=1, max_length=255)
    publishedAt: str = Field(min_length=1)


class _PublishedPage(BaseModel):
    model_config = ConfigDict(strict=True)

    report_id: str = Field(min_length=1, max_length=255)
    view_url: str = Field(min_length=1)


@dataclass(frozen=True)
class _Claim:
    user_id: str
    artifact_path: str
    view_url: str


@dataclass(frozen=True)
class BackfillSummary:
    recorded: int
    unknown: int


def _uuid(value: object) -> str | None:
    try:
        return str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        return None


def _read_history(root: Path, *parts: str) -> list[object]:
    """Read the real file without following a planted link in its writable path."""
    try:
        with opened_subdir_nofollow(root, *parts[:-1]) as folder:
            fd = dir_open(folder, parts[-1], os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
            with os.fdopen(fd) as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError
                state = json.load(source)
        if not isinstance(state, dict) or not isinstance(state.get("publish_history", []), list):
            raise ValueError
        return state.get("publish_history", [])
    except FileNotFoundError:
        return []
    except (OSError, ValueError, UnicodeError):
        # Do not expose decoder text: it can include the history's contents.
        raise PublishedPageBackfillUnavailable("Published page ownership backfill could not read publish history.") from None


def _sources(session: Session) -> tuple[dict[str, set[str]], dict[str, list[Path]]]:
    users: dict[str, set[str]] = defaultdict(set)
    projects: dict[str, list[Path]] = defaultdict(list)
    pairs = list(session.exec(select(Setting.org_id, Setting.user_id).where(Setting.scope == "user")).all())
    pairs.extend(session.exec(select(Conversation.org_id, Conversation.created_by)).all())
    for project in session.exec(select(Project)).all():
        partition = _uuid(project.org_id)
        if partition is not None:
            projects[partition].append(Path(project.path))
        pairs.append((project.org_id, project.created_by))
    for partition, user in pairs:
        partition, user = _uuid(partition), _uuid(user)
        if partition is not None and user is not None:
            users[partition].add(user)
    return users, projects


def _artifact_path(row: _HistoryRow, projects: list[Path]) -> str | None:
    target = Path(row.artifact)
    if not target.is_absolute() or ".." in target.parts or target.name != row.artifactName:
        return None
    try:
        published = datetime.fromisoformat(row.publishedAt)
        if published.utcoffset() is None:
            return None
    except ValueError:
        return None
    for project in projects:
        try:
            relative = target.relative_to(project / ".anton" / "artifacts")
            # A loose page or one directly in an artifact folder: _published_pages
            # reads the records that could refuse it, and no deeper ones.
            if not relative.parts or len(relative.parts) > 2:
                continue
            with opened_subdir_nofollow(project.parent, project.name, ".anton", "artifacts", *relative.parts[:-1]) as folder:
                fd = dir_open(folder, relative.name, os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
                try:
                    info = os.fstat(fd)
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        return str(target.resolve(strict=True))
                finally:
                    os.close(fd)
        except (OSError, ValueError):
            continue
    return None


def _candidates(session: Session) -> tuple[dict[str, _Claim], set[str]]:
    users, projects = _sources(session)
    shared = Path(get_app_settings().storage.shared_root)
    # The unpartitioned path is cowork.services.publish._cowork_state_dir's
    # local path. Global/shared history names no tester, even if settings do.
    global_root = Path(os.environ.get("ANTON_COWORK_STATE_DIR") or cowork_home()).expanduser()
    histories: list[tuple[str | None, list[object]]] = [(None, _read_history(global_root, "state.json"))]
    try:
        partitions = list(shared.iterdir()) if shared.exists() else []
        for path in partitions:
            partition = _uuid(path.name)
            if partition is not None:
                histories.append((partition, _read_history(shared, path.name, "publish", "state.json")))
    except OSError:
        raise PublishedPageBackfillUnavailable("Published page ownership backfill could not read publish history.") from None

    claims: dict[str, set[_Claim]] = defaultdict(set)
    blocked: set[str] = set()
    for partition, rows in histories:
        for value in rows:
            report_id = value.get("reportId") if isinstance(value, dict) else None
            if not isinstance(report_id, str) or not report_id or len(report_id) > 255:
                continue
            try:
                row = _HistoryRow.model_validate(value)
            except ValidationError:
                blocked.add(report_id)
                continue
            owners = users.get(partition, set())
            artifact = _artifact_path(row, projects.get(partition, [])) if len(owners) == 1 else None
            if partition is None or artifact is None:
                blocked.add(report_id)
            else:
                claims[report_id].add(_Claim(next(iter(owners)), artifact, row.url))
    blocked.update(report_id for report_id, matches in claims.items() if len(matches) != 1)
    chosen = {report_id: next(iter(matches)) for report_id, matches in claims.items() if report_id not in blocked}
    if chosen:
        # Another folder naming the page means two artifacts once pointed at
        # it, and history that rotated out may have named the first of them.
        for project_path in session.exec(select(Project.path)).all():
            for report_id, source in _published_pages(Path(project_path)):
                claim = chosen.get(report_id)
                if claim is not None and source != claim.artifact_path:
                    blocked.add(report_id)
    return {report_id: claim for report_id, claim in chosen.items() if report_id not in blocked}, blocked


def _published_pages(project: Path) -> list[tuple[str, str]]:
    """List (page id, resolved source) from a project's artifact publish records.

    Evidence that can only refuse a claim: a planted entry costs that page its
    import, never grants one. Reads the artifacts root's record (loose pages)
    and each direct artifact folder's without following a link. Agents write
    these files, so a record or entry that cannot be read is skipped.
    """
    parts = (project.name, ".anton", "artifacts")
    try:
        with opened_subdir_nofollow(project.parent, *parts) as root:
            names = os.listdir(root.fd if root.fd is not None else root.path)
    except OSError:
        return []
    pages: list[tuple[str, str]] = []
    for folder in [(), *((name,) for name in names)]:
        try:
            pages.extend(_published_record(project.parent, *parts, *folder))
        except Exception:
            continue
    return pages


def _published_record(base: Path, *names: str) -> list[tuple[str, str]]:
    with opened_subdir_nofollow(base, *names) as pinned:
        fd = dir_open(pinned, ".published.json", os.O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                return []
            raw = source.read(_MAX_RECORD_BYTES + 1)
        folder = pinned.path
    if len(raw) > _MAX_RECORD_BYTES:
        return []
    record = json.loads(raw)
    if not isinstance(record, dict):
        return []
    pages: list[tuple[str, str]] = []
    for name, entry in record.items():
        report_id = entry.get("report_id") if isinstance(entry, dict) else None
        if not isinstance(report_id, str) or not report_id or not is_publish_record_name(name):
            continue
        try:
            pages.append((report_id, str((folder / name).resolve())))
        except (OSError, RuntimeError):
            continue
    return pages


def _inventory(api_key: str, publish_url: str) -> dict[str, set[str]]:
    try:
        values = list_published(
            api_key=api_key, publish_url=publish_url,
            ssl_verify=os.environ.get("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true",
        )
        if not isinstance(values, list):
            raise ValueError
        pages: dict[str, set[str]] = defaultdict(set)
        for value in values:
            page = _PublishedPage.model_validate(value)
            pages[page.report_id].add(page.view_url)
        return pages
    except Exception:
        # An HTTP error may hold a credential or response body. Startup needs
        # only the retryable failure, never that exception's text or chain.
        raise PublishedPageBackfillUnavailable("Published page ownership backfill could not verify the publisher's page list.") from None


def run_published_page_owner_backfill(engine=None) -> BackfillSummary | None:
    """Run before readiness; any failed pass rolls back owners and its sentinel together."""
    settings = get_app_settings()
    if settings.tenancy_mode == "org" or not settings.trust_identity_headers:
        return None
    engine = engine or get_engine(settings.database.uri)
    with _process_lock(*_LOCK_ID, timeout=RESOURCE_LOCK_TIMEOUT_SECONDS):
        with Session(engine) as session:
            locks = _DatabaseResourceLocks(ScopedSession(session, LOCAL_SCOPE))
            with locks.lock(*_LOCK_ID):
                if session.exec(select(Setting.id).where(Setting.key == SENTINEL_KEY, Setting.scope.is_(None))).first() is not None:
                    return None
                try:
                    publish_url, api_key = _resolve_publish_endpoint(get_user_settings())
                except Exception:
                    raise PublishedPageBackfillUnavailable("Published page ownership backfill could not resolve the publisher.") from None
                if not protects_publisher(publish_url):
                    return None
                publisher = publisher_key(publish_url)
                try:
                    candidates, unknown = _candidates(session)
                except (PublishedPageBackfillUnavailable, SQLAlchemyError):
                    raise
                except Exception:
                    # History and publish records are files agents can write.
                    raise PublishedPageBackfillUnavailable(
                        "Published page ownership backfill could not read publish history."
                    ) from None
                candidates = {report_id: claim for report_id, claim in candidates.items()
                              if find_owner(session, publisher, report_id) is None}
                if candidates and not api_key:
                    raise PublishedPageBackfillUnavailable("Published page ownership backfill requires a publisher credential.")
                inventory = _inventory(api_key, publish_url) if candidates else {}
                recorded = 0
                for report_id, claim in candidates.items():
                    # Both must match exactly. History carries no endpoint, so
                    # an id alone cannot prove this publisher served that URL.
                    if inventory.get(report_id) != {claim.view_url}:
                        unknown.add(report_id)
                        continue
                    session.add(PublishedPageOwner(
                        publisher_key=publisher, report_id=report_id,
                        artifact_path=claim.artifact_path, owner_user_id=claim.user_id,
                    ))
                    recorded += 1
                session.add(Setting(key=SENTINEL_KEY, value="1"))
                session.commit()
                return BackfillSummary(recorded, len(unknown))
