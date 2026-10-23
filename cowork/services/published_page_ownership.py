"""Authorize shared-publisher mutations using server-owned page records."""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.published_page import PublishedPageOwner
from cowork.services.providers import is_mindshub_publish_url, normalize_publish_url
from cowork.services.shared_resources import (
    RESOURCE_LOCK_TIMEOUT_SECONDS,
    _DatabaseResourceLocks,
    _process_lock,
)


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

    def checked_report_id(self, report_id: object) -> str | None:
        """Only the recorded tester may send this artifact's recorded page id."""
        if not isinstance(report_id, str) or not report_id:
            return None
        row = find_owner(self.session, self.publisher, report_id)
        if row is None or row.artifact_path != self.artifact_path:
            return None
        if row.owner_user_id != self.user_id:
            from cowork.services.publish import PublishOwnedByOtherUser

            raise PublishOwnedByOtherUser()
        return report_id

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
        # failure path updates or removes an existing ownership record.
        self.session.commit()


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
                    yield PublicationOwner(session, publisher, path, user_id)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Published page ownership is unavailable. Try again later.") from exc
