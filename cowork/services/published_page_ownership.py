"""Authorize shared-publisher mutations using server-owned page records."""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.published_page import PublishedPageOwner
from cowork.models.setting import Setting
from cowork.services.providers import is_mindshub_publish_url, normalize_publish_url
from cowork.services.shared_resources import (
    RESOURCE_LOCK_TIMEOUT_SECONDS,
    _DatabaseResourceLocks,
    _process_lock,
)

logger = logging.getLogger(__name__)

# The setting row published_page_owner_backfill writes once its pass completes.
BACKFILL_SENTINEL_KEY = "_published_page_owner_backfill_v1"
OWNERSHIP_UNAVAILABLE = "Published page ownership is unavailable. Try again later."


def protects_publisher(publish_url: str) -> bool:
    settings = get_app_settings()
    return (
        settings.tenancy_mode != "org"
        and settings.trust_identity_headers
        and not is_mindshub_publish_url(publish_url)
    )


def publisher_key(publish_url: str) -> str:
    normalized = normalize_publish_url(publish_url)
    if not normalized:
        raise ValueError("Publishing requires a publisher URL")
    return hashlib.sha256(normalized.rstrip("/").encode()).hexdigest()


def find_owner(session: Session, publisher: str, report_id: str) -> PublishedPageOwner | None:
    return session.exec(select(PublishedPageOwner).where(
        PublishedPageOwner.publisher_key == publisher,
        PublishedPageOwner.report_id == report_id,
    )).first()


@dataclass
class PublicationOwner:
    session: Session
    publisher: str
    artifact_path: str
    user_id: str

    def checked_report_id(self, report_id: object, *, url: object = "") -> str | None:
        """Only the recorded tester may send this artifact's recorded page id."""
        try:
            if not isinstance(report_id, str) or not report_id:
                return None
            row = find_owner(self.session, self.publisher, report_id)
            if row is None:
                # The caller drops this folder's pointer, and the page stays
                # live with no owner on record. This line is how an operator
                # finds it; keep the orphaned_publish prefix.
                logger.warning(
                    "orphaned_publish identifier=%s url=%s reason=unrecorded_owner",
                    report_id, url if isinstance(url, str) else "",
                )
                return None
            if row.artifact_path != self.artifact_path:
                return None
            if row.owner_user_id != self.user_id:
                from cowork.services.publish import PublishOwnedByOtherUser

                raise PublishOwnedByOtherUser()
            return report_id
        finally:
            # End the read transaction before the caller's upload or delete.
            # The advisory lock is on its own connection and stays held.
            self.session.rollback()

    def record(self, report_id: str) -> None:
        """Commit before exposing the successful upload in editable local state."""
        if not isinstance(report_id, str) or not report_id or len(report_id) > 255:
            raise RuntimeError("The publisher returned no valid page id")
        row = find_owner(self.session, self.publisher, report_id)
        if row is not None:
            if row.artifact_path != self.artifact_path or row.owner_user_id != self.user_id:
                raise RuntimeError("The publisher returned a page id already owned elsewhere")
            return
        self.session.add(PublishedPageOwner(
            publisher_key=self.publisher, report_id=report_id,
            artifact_path=self.artifact_path, owner_user_id=self.user_id,
        ))
        # The unique constraint also refuses a competing process's claim. No
        # failure path changes an existing ownership record; only
        # rekey_published_page_paths moves one, along with its folder.
        self.session.commit()


def _require_backfill(session: Session) -> None:
    """Retry the one-time legacy-owner import until it has run.

    Startup runs it first. While it cannot finish, a protected publish or
    delete answers 503 rather than minting a new link for a page the import
    would give back to its owner. A pass that skips this publisher writes no
    sentinel and refuses nothing.
    """
    done = session.exec(select(Setting.id).where(
        Setting.key == BACKFILL_SENTINEL_KEY, Setting.scope.is_(None),
    )).first()
    # The pass writes from its own session, which SQLite would block behind
    # this read.
    session.rollback()
    if done is not None:
        return
    from cowork.services.published_page_owner_backfill import (
        PublishedPageBackfillUnavailable,
        run_published_page_owner_backfill,
    )

    try:
        run_published_page_owner_backfill()
    except PublishedPageBackfillUnavailable:
        raise HTTPException(503, OWNERSHIP_UNAVAILABLE) from None


@contextmanager
def publication_owner(
    artifact: Path, *, published_dir: Path, publish_url: str, scope: TenantScope | None,
) -> Iterator[PublicationOwner | None]:
    if not protects_publisher(publish_url):
        yield None
        return
    try:
        if scope is None or not scope.personal or not scope.org_id or not scope.user_id:
            raise ValueError
        user_id = str(UUID(scope.user_id))
    except ValueError:
        raise HTTPException(401, "Publishing requires an authenticated user.") from None

    publisher = publisher_key(publish_url)
    path = str(artifact.resolve())
    # A folder can hold several loose pages in the same .published.json map.
    # Lock the map while reading it, uploading, recording ownership and saving
    # it. Use the existing PostgreSQL cross-process lock where available.
    lock_key = hashlib.sha256(str(published_dir.resolve()).encode()).hexdigest()
    try:
        engine = get_engine(get_app_settings().database.uri)
        with _process_lock(publisher, "published_page", lock_key, timeout=RESOURCE_LOCK_TIMEOUT_SECONDS):
            with Session(engine) as session:
                locks = _DatabaseResourceLocks(ScopedSession(session, LOCAL_SCOPE))
                with locks.lock(publisher, "published_page", lock_key):
                    # Force a database/schema check even on a first publish.
                    session.exec(select(PublishedPageOwner.id).limit(0)).all()
                    _require_backfill(session)
                    yield PublicationOwner(session, publisher, path, user_id)
    except SQLAlchemyError as exc:
        raise HTTPException(503, OWNERSHIP_UNAVAILABLE) from exc


def rekey_published_page_paths(old_dir: Path, new_dir: Path) -> None:
    """Carry page owners along when the server moves a folder of artifacts.

    A project rename and a task move take each artifact's .published.json
    with its folder, so the owner keeps the link. A copy, or a move the
    server did not make, still publishes as a new page.
    """
    settings = get_app_settings()
    if settings.tenancy_mode == "org" or not settings.trust_identity_headers:
        return
    # The old folder is gone; its parent still resolves as it did at publish.
    old_prefix = str(old_dir.parent.resolve() / old_dir.name)
    new_prefix = str(new_dir.resolve())
    try:
        with Session(get_engine(settings.database.uri)) as session:
            rows = session.exec(select(PublishedPageOwner).where(or_(
                PublishedPageOwner.artifact_path == old_prefix,
                PublishedPageOwner.artifact_path.startswith(old_prefix + os.sep, autoescape=True),
            ))).all()
            for row in rows:
                # SQLite's LIKE ignores case: recheck the prefix exactly.
                if row.artifact_path != old_prefix and not row.artifact_path.startswith(old_prefix + os.sep):
                    continue
                row.artifact_path = new_prefix + row.artifact_path[len(old_prefix):]
                session.add(row)
            session.commit()
    except SQLAlchemyError:
        # The folder has already moved. The owner's next publish of it gets a
        # new link, as a copy would, and the old page keeps its record.
        logger.warning(
            "Could not move published page owners from %s to %s", old_dir, new_dir, exc_info=True,
        )
