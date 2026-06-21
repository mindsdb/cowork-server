from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from sqlmodel import Session

from cowork.handlers.responses import ResponsesHandler
from cowork.models.artifact import Artifact, ArtifactActivityEvent, ArtifactComment, ArtifactVersion
from cowork.models.project import Project
from cowork.schemas.responses import ResponsesRequest
from cowork.services.artifact_versions import (
    ArtifactVersionService,
    _external_artifact_id,
    _artifact_from_identifier_or_path,
    comment_to_dict,
    version_to_dict,
)
from cowork.services.conversations import ConversationService


async def handoff_artifact_to_conversation(
    session: Session,
    *,
    path: str | None = None,
    artifact_id: str | UUID | None = None,
    version_id: str | UUID | None = None,
    comment_id: str | UUID | None = None,
    prompt: str | None = None,
    title: str | None = None,
    project_id: UUID | None = None,
    project: str | None = None,
    model: str | None = None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    artifact = _artifact_from_identifier_or_path(session, artifact_id=artifact_id, path=path)
    version = _version_for_handoff(session, artifact, version_id)
    comment = _comment_for_handoff(session, artifact, comment_id)
    resolved_project_id = project_id or artifact.project_id
    if resolved_project_id is None and project:
        project_row = ConversationService(session).project_by_name(project)
        resolved_project_id = project_row.id if project_row is not None else None

    conversation = ConversationService(session).create_conversation(
        topic=title or _default_title(artifact, comment=comment),
        project_id=resolved_project_id,
    )
    handoff_path, pre_turn_checkpoint = _handoff_source_path(
        session,
        artifact,
        version=version,
        conversation_id=conversation.id,
        project_id=resolved_project_id,
        prompt=prompt,
    )
    initial_prompt = _build_handoff_prompt(
        artifact,
        version=version,
        comment=comment,
        user_prompt=prompt,
        source_path=handoff_path,
        source_is_materialized=version is not None,
    )
    await ResponsesHandler(session).handle(
        ResponsesRequest(
            input=initial_prompt,
            conversation=str(conversation.id),
            project_id=resolved_project_id,
            project=project,
            model=model,
            stream=True,
        )
    )
    session.add(
        ArtifactActivityEvent(
            artifact_id=artifact.id,
            version_id=version.id if version is not None else artifact.current_version_id,
            event_type="handoff",
            actor_name=actor_name,
            details={
                "conversationId": str(conversation.id),
                "conversationTitle": conversation.topic,
                "targetProjectId": str(resolved_project_id) if resolved_project_id else None,
                "versionId": str(version.id) if version is not None else None,
                "commentId": str(comment.id) if comment is not None else None,
                "handoffPath": str(handoff_path),
                "materializedVersion": version is not None,
                "preTurnCheckpointId": str(pre_turn_checkpoint.id) if pre_turn_checkpoint is not None else None,
                "actorName": actor_name or "",
                **({"actorEmail": actor_email} if actor_email else {}),
                **({"actorSubject": actor_subject} if actor_subject else {}),
            },
        )
    )
    session.commit()
    session.refresh(conversation)
    pre_turn_checkpoint_payload = None
    if pre_turn_checkpoint is not None:
        checkpoint_artifact = session.get(Artifact, pre_turn_checkpoint.artifact_id)
        checkpoint_external_id = _external_artifact_id(checkpoint_artifact) if checkpoint_artifact is not None else None
        pre_turn_checkpoint_payload = version_to_dict(
            pre_turn_checkpoint,
            session=session,
            artifact_external_id=checkpoint_external_id,
        )
    return {
        "conversationId": str(conversation.id),
        "conversation": {
            "id": str(conversation.id),
            "title": conversation.topic,
            "projectId": str(conversation.project_id) if conversation.project_id else None,
        },
        "artifact": _artifact_payload(artifact),
        "version": version_to_dict(version, session=session, artifact_external_id=_external_artifact_id(artifact)) if version else None,
        "preTurnCheckpoint": pre_turn_checkpoint_payload,
        "comment": comment_to_dict(comment) if comment else None,
        "handoffPath": str(handoff_path),
        "materializedVersion": version is not None,
        "initialPrompt": initial_prompt,
        "started": True,
    }


def _version_for_handoff(
    session: Session,
    artifact: Artifact,
    version_id: str | UUID | None,
) -> ArtifactVersion | None:
    if version_id is None:
        return None
    try:
        internal_id = version_id if isinstance(version_id, UUID) else UUID(str(version_id))
    except ValueError as exc:
        raise ValueError("Invalid version id") from exc
    version = session.get(ArtifactVersion, internal_id)
    if version is None or version.artifact_id != artifact.id:
        raise ValueError("Artifact version not found")
    return version


def _comment_for_handoff(
    session: Session,
    artifact: Artifact,
    comment_id: str | UUID | None,
) -> ArtifactComment | None:
    if comment_id is None:
        return None
    try:
        internal_id = comment_id if isinstance(comment_id, UUID) else UUID(str(comment_id))
    except ValueError as exc:
        raise ValueError("Invalid comment id") from exc
    comment = session.get(ArtifactComment, internal_id)
    if comment is None or comment.artifact_id != artifact.id:
        raise ValueError("Artifact comment not found")
    return comment


def _build_handoff_prompt(
    artifact: Artifact,
    *,
    version: ArtifactVersion | None,
    comment: ArtifactComment | None,
    user_prompt: str | None,
    source_path: Path,
    source_is_materialized: bool,
) -> str:
    lines = [
        user_prompt.strip() if user_prompt and user_prompt.strip() else "Continue work on this artifact.",
        "",
        "Artifact context:",
        f"- Title: {artifact.title}",
        f"- Artifact id: {_external_artifact_id(artifact) or artifact.id}",
        f"- Slug: {artifact.slug}",
        f"- Folder: {source_path}",
    ]
    if artifact.description:
        lines.append(f"- Description: {artifact.description}")
    if artifact.artifact_type:
        lines.append(f"- Type: {artifact.artifact_type}")
    if version is not None:
        lines.extend(
            [
                "",
                "Selected version:",
                f"- Version id: {version.id}",
                f"- Version number: {version.version_number}",
                f"- Label: {version.label or ''}",
                f"- Operation: {version.operation_type}",
                f"- Files hash: {version.files_hash}",
            ]
        )
        if source_is_materialized:
            lines.append(f"- Exact version folder: {source_path}")
    if comment is not None:
        lines.extend(
            [
                "",
                "Selected review note:",
                f"- Comment id: {comment.id}",
                f"- Kind: {comment.kind}",
                f"- Status: {comment.status}",
                f"- Body: {comment.body}",
            ]
        )
        if comment.anchor:
            lines.append(f"- Anchor: {comment.anchor}")
        if comment.proposed_patch:
            lines.append(f"- Proposed patch: {comment.proposed_patch}")
    lines.extend(
        [
            "",
            "Use the folder above as the source of truth. If you change the artifact, update files in that folder.",
        ]
    )
    return "\n".join(lines)


def _handoff_source_path(
    session: Session,
    artifact: Artifact,
    *,
    version: ArtifactVersion | None,
    conversation_id: UUID,
    project_id: UUID | None,
    prompt: str | None,
) -> tuple[Path, ArtifactVersion | None]:
    if version is None:
        checkpoint = ArtifactVersionService(session).snapshot_artifact(
            Path(artifact.path),
            artifact_id=artifact.id,
            source_conversation_id=conversation_id,
            prompt=prompt or "Before artifact handoff",
            label="Before follow-up task",
            operation_type="handoff_safety",
            snapshot_role="pre",
        )
        return Path(artifact.path), checkpoint
    service = ArtifactVersionService(session)
    target = _handoff_target_root(session, artifact, version=version, conversation_id=conversation_id, project_id=project_id)
    external_id = _external_artifact_id(artifact) or artifact.slug or str(artifact.id)
    metadata_overrides = {
        "id": f"{external_id}-handoff-{str(conversation_id)[:8]}",
        "slug": target.name,
        "name": artifact.title or target.name,
    }
    service.replace_with_version(
        version.id,
        target,
        metadata_overrides=metadata_overrides,
        clear_published=True,
    )
    checkpoint = None
    if project_id is not None:
        checkpoint = service.snapshot_artifact(
            target,
            project_id=project_id,
            slug=target.name,
            title=artifact.title or target.name,
            description=artifact.description,
            artifact_type=artifact.artifact_type,
            source_conversation_id=conversation_id,
            prompt=_handoff_snapshot_prompt(version, prompt),
            label=f"Handoff copy from v{version.version_number}",
            operation_type="fork",
            preview_status=version.preview_status,
            publish_status="unpublished",
            branch_name=f"handoff/{str(conversation_id)[:8]}",
            forked_from_version_id=version.id,
            snapshot_role="pre",
        )
    return target, checkpoint


def _handoff_target_root(
    session: Session,
    artifact: Artifact,
    *,
    version: ArtifactVersion,
    conversation_id: UUID,
    project_id: UUID | None,
) -> Path:
    project = session.get(Project, project_id) if project_id is not None else None
    if project is not None:
        base = Path(project.path).expanduser().resolve(strict=False) / ".anton" / "artifacts"
    else:
        base = Path(artifact.path).expanduser().resolve(strict=False).parent / ".handoffs"
    base.mkdir(parents=True, exist_ok=True)
    return base / _unique_handoff_slug(base, artifact.slug, version=version, conversation_id=conversation_id)


def _unique_handoff_slug(base: Path, source_slug: str, *, version: ArtifactVersion, conversation_id: UUID) -> str:
    stem = f"{source_slug}-v{version.version_number}-handoff-{str(conversation_id)[:8]}"
    candidate = stem
    while (base / candidate).exists():
        candidate = f"{stem}-{uuid4().hex[:6]}"
    return candidate


def _handoff_snapshot_prompt(version: ArtifactVersion, prompt: str | None) -> str:
    prefix = f"Handoff from artifact version {version.version_number}"
    clean_prompt = (prompt or "").strip()
    return f"{prefix}: {clean_prompt}" if clean_prompt else prefix


def _default_title(artifact: Artifact, *, comment: ArtifactComment | None) -> str:
    prefix = "Review" if comment is not None else "Work on"
    return f"{prefix} {artifact.title or artifact.slug}"[:120]


def _artifact_payload(artifact: Artifact) -> dict:
    return {
        "id": _external_artifact_id(artifact) or str(artifact.id),
        "internalId": str(artifact.id),
        "slug": artifact.slug,
        "title": artifact.title,
        "description": artifact.description,
        "type": artifact.artifact_type,
        "path": artifact.path,
        "projectId": str(artifact.project_id) if artifact.project_id else None,
    }
