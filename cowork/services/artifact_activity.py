from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.artifact import Artifact, ArtifactActivityEvent
from cowork.models.project import Project
from cowork.services.artifact_versions import _external_artifact_id
from cowork.services.project_permissions import has_project_permission, project_has_owner


def activity_event_payload(
    session: Session,
    event: ArtifactActivityEvent,
    *,
    artifact: Artifact | None = None,
) -> dict:
    artifact = artifact or session.get(Artifact, event.artifact_id)
    project = session.get(Project, artifact.project_id) if artifact is not None and artifact.project_id else None
    return {
        "id": str(event.id),
        "artifactId": str(event.artifact_id),
        "externalArtifactId": _external_artifact_id(artifact) if artifact is not None else None,
        "artifactPath": artifact.path if artifact is not None else None,
        "artifactTitle": artifact.title if artifact is not None else None,
        "artifactSlug": artifact.slug if artifact is not None else None,
        "projectId": str(project.id) if project is not None else None,
        "projectName": project.name if project is not None else None,
        "versionId": str(event.version_id) if event.version_id else None,
        "eventType": event.event_type,
        "actorName": event.actor_name,
        "details": event.details or {},
        "createdAt": event.created_at.isoformat() if event.created_at else None,
    }


def list_project_activity(session: Session, project_id: UUID, *, limit: int = 50) -> dict:
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    rows = session.exec(
        select(ArtifactActivityEvent)
        .join(Artifact, Artifact.id == ArtifactActivityEvent.artifact_id)
        .where(Artifact.project_id == project.id)
        .order_by(ArtifactActivityEvent.created_at.desc(), ArtifactActivityEvent.id.desc())
        .limit(limit)
    ).all()
    return {
        "projectId": str(project.id),
        "projectName": project.name,
        "activity": [activity_event_payload(session, row) for row in rows],
    }


def list_global_activity(
    session: Session,
    *,
    actor_email: str | None,
    limit: int = 50,
) -> dict:
    events = session.exec(
        select(ArtifactActivityEvent)
        .order_by(ArtifactActivityEvent.created_at.desc(), ArtifactActivityEvent.id.desc())
        .limit(min(max(limit * 4, limit), 500))
    ).all()
    activity = []
    for event in events:
        artifact = session.get(Artifact, event.artifact_id)
        if artifact is None:
            continue
        if artifact.project_id is not None and project_has_owner(session, artifact.project_id):
            if not has_project_permission(session, artifact.project_id, actor_email, "view"):
                continue
        activity.append(activity_event_payload(session, event, artifact=artifact))
        if len(activity) >= limit:
            break
    return {"activity": activity}
