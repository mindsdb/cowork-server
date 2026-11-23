"""A share URL identifies a candidate. The draft review and live publish record authorize the read."""
from hashlib import sha256
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cowork.db.scoped import ScopedSession
from cowork.models.artifact_share_link import ArtifactShareLink


def legacy_share_ref(session: ScopedSession, url: str) -> tuple[str, str] | None:
    digest = sha256(url.encode()).hexdigest()
    rows = session.exec(session.select(ArtifactShareLink).where(
        ArtifactShareLink.url_digest == digest,
    ).limit(2)).all()
    if len(rows) != 1:
        return None
    return str(rows[0].project_id), rows[0].artifact_id


def record_share_link(session: ScopedSession, project_id: UUID, artifact_id: str, url: str) -> None:
    digest = sha256(url.encode()).hexdigest()
    query = session.select(ArtifactShareLink).where(
        ArtifactShareLink.project_id == project_id,
        ArtifactShareLink.artifact_id == artifact_id,
        ArtifactShareLink.url_digest == digest,
    )
    if session.exec(query).first() is not None:
        return
    session.add(ArtifactShareLink(
        org_id=session.scope.org_id, project_id=project_id, artifact_id=artifact_id, url_digest=digest,
    ))
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        if session.exec(query).first() is None:
            raise
