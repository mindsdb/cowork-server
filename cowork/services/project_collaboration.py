from __future__ import annotations

import re
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlmodel import Session, select

from cowork.common.encryption import decrypt, encrypt
from cowork.models.artifact import Artifact
from cowork.models.project import Project
from cowork.models.project_collaboration import (
    NotificationDelivery,
    ProjectCollaborator,
    ProjectInvitation,
    ProjectNotificationHook,
)


COLLABORATOR_ROLES = {"owner", "editor", "reviewer", "commenter", "viewer"}
HOOK_KINDS = {"email", "webhook"}
SECRET_CONFIG_KEY_PARTS = ("password", "passwd", "pwd", "secret", "token", "api_key", "apikey", "authorization")
DEFAULT_NOTIFICATION_EVENTS = [
    "project.invited",
    "artifact.commented",
    "artifact.suggested",
    "artifact.review_requested",
    "artifact.resolved",
    "artifact.reopened",
    "artifact.accepted",
    "artifact.rejected",
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ProjectCollaborationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_collaborators(self, project_id: UUID) -> dict:
        project = self._project(project_id)
        rows = self.session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == project.id)
            .order_by(ProjectCollaborator.email)
        ).all()
        return {
            "projectId": str(project.id),
            "collaborators": [collaborator_to_dict(row) for row in rows],
        }

    def list_invitations(self, project_id: UUID, *, include_closed: bool = False) -> dict:
        project = self._project(project_id)
        query = select(ProjectInvitation).where(ProjectInvitation.project_id == project.id)
        if not include_closed:
            query = query.where(ProjectInvitation.status == "pending")
        rows = self.session.exec(query.order_by(ProjectInvitation.created_at.desc())).all()
        return {
            "projectId": str(project.id),
            "invitations": [invitation_to_dict(row) for row in rows],
        }

    def invite_collaborator(
        self,
        project_id: UUID,
        *,
        email: str,
        display_name: str | None = None,
        role: str = "viewer",
        invited_by_subject: str | None = None,
        invited_by_email: str | None = None,
        invited_by_name: str | None = None,
        ttl_days: int = 7,
    ) -> dict:
        project = self._project(project_id)
        normalized_email = normalize_email(email)
        clean_role = validate_role(role)
        token = new_invitation_token()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=max(1, min(int(ttl_days or 7), 30)))
        row = self.session.exec(
            select(ProjectInvitation)
            .where(ProjectInvitation.project_id == project.id)
            .where(ProjectInvitation.email == normalized_email)
            .where(ProjectInvitation.status == "pending")
        ).first()
        created = row is None
        if row is None:
            row = ProjectInvitation(project_id=project.id, email=normalized_email)
        row.display_name = display_name or row.display_name
        row.role = clean_role
        row.status = "pending"
        row.token_hash = invitation_token_hash(token)
        row.expires_at = expires_at
        row.accepted_at = None
        row.revoked_at = None
        row.invited_by_subject = invited_by_subject or row.invited_by_subject
        row.invited_by_email = normalize_optional_email(invited_by_email)
        row.invited_by_name = invited_by_name or row.invited_by_name
        self.session.add(row)
        self.session.flush()
        deliveries = dispatch_project_email(
            self.session,
            project,
            "project.invited",
            recipient_email=normalized_email,
            details={
                "projectId": str(project.id),
                "projectName": project.name,
                "invitationId": str(row.id),
                "invitedEmail": normalized_email,
                "invitedRole": clean_role,
                "inviterSubject": row.invited_by_subject,
                "inviterEmail": row.invited_by_email,
                "inviterName": row.invited_by_name,
                "inviteToken": token,
                "invitationSendId": str(uuid4()),
                "expiresAt": expires_at.isoformat(),
            },
        )
        state = dict(row.notification_state or {})
        state["lastSentAt"] = now.isoformat()
        state["deliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
        row.notification_state = state
        row.last_sent_at = now
        row.send_count = (row.send_count or 0) + len(deliveries)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {
            "created": created,
            "invitation": invitation_to_dict(row, token=token),
            "deliveries": [delivery_to_dict(delivery) for delivery in deliveries],
        }

    def resend_invitation(
        self,
        project_id: UUID,
        invitation_id: UUID,
        *,
        invited_by_subject: str | None = None,
        invited_by_email: str | None = None,
        invited_by_name: str | None = None,
    ) -> dict:
        project = self._project(project_id)
        row = self._invitation(project.id, invitation_id)
        if row.status != "pending":
            raise ValueError("Only pending invitations can be resent")
        if invitation_expired(row):
            row.status = "expired"
            self.session.add(row)
            self.session.commit()
            raise ValueError("Invitation has expired")
        token = new_invitation_token()
        now = datetime.now(timezone.utc)
        row.token_hash = invitation_token_hash(token)
        row.expires_at = now + timedelta(days=7)
        row.invited_by_subject = invited_by_subject or row.invited_by_subject
        row.invited_by_email = normalize_optional_email(invited_by_email) or row.invited_by_email
        row.invited_by_name = invited_by_name or row.invited_by_name
        self.session.add(row)
        self.session.flush()
        deliveries = dispatch_project_email(
            self.session,
            project,
            "project.invited",
            recipient_email=row.email,
            details={
                "projectId": str(project.id),
                "projectName": project.name,
                "invitationId": str(row.id),
                "invitedEmail": row.email,
                "invitedRole": row.role,
                "inviterSubject": row.invited_by_subject,
                "inviterEmail": row.invited_by_email,
                "inviterName": row.invited_by_name,
                "inviteToken": token,
                "invitationSendId": str(uuid4()),
                "expiresAt": row.expires_at.isoformat() if row.expires_at else None,
                "resent": True,
            },
        )
        state = dict(row.notification_state or {})
        state["lastSentAt"] = now.isoformat()
        state["deliveries"] = [delivery_to_dict(delivery) for delivery in deliveries]
        row.notification_state = state
        row.last_sent_at = now
        row.send_count = (row.send_count or 0) + len(deliveries)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {
            "invitation": invitation_to_dict(row, token=token),
            "deliveries": [delivery_to_dict(delivery) for delivery in deliveries],
        }

    def revoke_invitation(self, project_id: UUID, invitation_id: UUID) -> dict:
        row = self._invitation(project_id, invitation_id)
        if row.status != "pending":
            raise ValueError("Only pending invitations can be revoked")
        row.status = "revoked"
        row.revoked_at = datetime.now(timezone.utc)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"invitation": invitation_to_dict(row)}

    def accept_invitation(
        self,
        project_id: UUID,
        *,
        token: str,
        principal_email: str | None,
        principal_subject: str | None = None,
        principal_name: str | None = None,
        display_name: str | None = None,
    ) -> dict:
        project = self._project(project_id)
        if not principal_email:
            raise ValueError("Authentication is required to accept an invitation")
        email = normalize_email(principal_email)
        row = self.session.exec(
            select(ProjectInvitation)
            .where(ProjectInvitation.project_id == project.id)
            .where(ProjectInvitation.token_hash == invitation_token_hash(token))
            .where(ProjectInvitation.status == "pending")
        ).first()
        if row is None:
            raise ValueError("Invitation not found")
        if row.email != email:
            raise ValueError("Invitation can only be accepted by the invited email")
        if invitation_expired(row):
            row.status = "expired"
            self.session.add(row)
            self.session.commit()
            raise ValueError("Invitation has expired")
        collaborator = self.session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == project.id)
            .where(ProjectCollaborator.email == row.email)
        ).first()
        created = collaborator is None
        if collaborator is None:
            collaborator = ProjectCollaborator(project_id=project.id, email=row.email)
        collaborator.display_name = display_name or row.display_name or collaborator.display_name
        collaborator.role = validate_role(row.role)
        row.status = "accepted"
        row.accepted_at = datetime.now(timezone.utc)
        row.accepted_by_subject = principal_subject
        row.accepted_by_email = email
        row.accepted_by_name = principal_name or display_name
        self.session.add(collaborator)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        self.session.refresh(collaborator)
        return {
            "created": created,
            "invitation": invitation_to_dict(row),
            "collaborator": collaborator_to_dict(collaborator),
        }

    def upsert_collaborator(
        self,
        project_id: UUID,
        *,
        email: str,
        display_name: str | None = None,
        role: str = "viewer",
    ) -> dict:
        project = self._project(project_id)
        normalized_email = normalize_email(email)
        clean_role = validate_role(role)
        row = self.session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == project.id)
            .where(ProjectCollaborator.email == normalized_email)
        ).first()
        created = row is None
        if row is None:
            row = ProjectCollaborator(project_id=project.id, email=normalized_email)
        else:
            self._ensure_project_keeps_owner(row, clean_role)
        row.display_name = display_name or row.display_name
        row.role = clean_role
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"created": created, "collaborator": collaborator_to_dict(row)}

    def update_collaborator(
        self,
        project_id: UUID,
        collaborator_id: UUID,
        *,
        display_name: str | None = None,
        role: str | None = None,
    ) -> dict:
        row = self._collaborator(project_id, collaborator_id)
        if display_name is not None:
            row.display_name = display_name
        if role is not None:
            next_role = validate_role(role)
            self._ensure_project_keeps_owner(row, next_role)
            row.role = next_role
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"collaborator": collaborator_to_dict(row)}

    def delete_collaborator(self, project_id: UUID, collaborator_id: UUID) -> bool:
        row = self._collaborator(project_id, collaborator_id)
        self._ensure_project_keeps_owner(row, None)
        self.session.delete(row)
        self.session.commit()
        return True

    def list_hooks(self, project_id: UUID) -> dict:
        project = self._project(project_id)
        rows = self.session.exec(
            select(ProjectNotificationHook)
            .where(ProjectNotificationHook.project_id == project.id)
            .order_by(ProjectNotificationHook.created_at.desc())
        ).all()
        return {
            "projectId": str(project.id),
            "hooks": [hook_to_dict(row) for row in rows],
        }

    def create_hook(
        self,
        project_id: UUID,
        *,
        kind: str,
        target: str,
        enabled: bool = True,
        events: list[str] | None = None,
        secret: str | None = None,
        config: dict | None = None,
    ) -> dict:
        project = self._project(project_id)
        clean_kind = validate_hook_kind(kind)
        row = ProjectNotificationHook(
            project_id=project.id,
            kind=clean_kind,
            target=clean_target(target, kind=clean_kind, secret=secret),
            enabled=enabled,
            events=normalize_events(events),
            secret_ciphertext=encrypt(secret) if secret else None,
            config=normalize_hook_config(clean_kind, config),
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"hook": hook_to_dict(row)}

    def update_hook(
        self,
        project_id: UUID,
        hook_id: UUID,
        *,
        kind: str | None = None,
        target: str | None = None,
        enabled: bool | None = None,
        events: list[str] | None = None,
        secret: str | None = None,
        config: dict | None = None,
    ) -> dict:
        row = self._hook(project_id, hook_id)
        clean_kind = validate_hook_kind(kind) if kind is not None else row.kind
        clean_config = normalize_hook_config(clean_kind, config if config is not None else row.config)
        clean_target = clean_target_for_update(row, clean_kind, target=target, secret=secret)
        row.kind = clean_kind
        row.target = clean_target
        if target is not None:
            row.target = clean_target
        elif secret and clean_kind == "webhook" and (not row.target or row.target.startswith("secret:")):
            row.target = clean_target
        if enabled is not None:
            row.enabled = enabled
        if events is not None:
            row.events = normalize_events(events)
        if secret is not None:
            row.secret_ciphertext = encrypt(secret) if secret else None
        row.config = clean_config
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"hook": hook_to_dict(row)}

    def delete_hook(self, project_id: UUID, hook_id: UUID) -> bool:
        row = self._hook(project_id, hook_id)
        self.session.delete(row)
        self.session.commit()
        return True

    async def test_hook(self, project_id: UUID, hook_id: UUID) -> dict:
        row = self._hook(project_id, hook_id)
        delivery = _record_delivery(
            self.session,
            hook=row,
            event_key="test",
            dedupe_key=f"test:{row.id}:{uuid4()}",
            details={"message": "Test notification"},
        )
        self.session.commit()
        self.session.refresh(delivery)
        from cowork.services.notifications import send_notification_delivery

        return await send_notification_delivery(self.session, delivery.id)

    def list_deliveries(self, project_id: UUID, *, limit: int = 50) -> dict:
        project = self._project(project_id)
        rows = self.session.exec(
            select(NotificationDelivery)
            .where(NotificationDelivery.project_id == project.id)
            .order_by(NotificationDelivery.created_at.desc())
            .limit(max(1, min(limit, 200)))
        ).all()
        return {
            "projectId": str(project.id),
            "deliveries": [delivery_to_dict(row) for row in rows],
        }

    def retry_delivery(self, project_id: UUID, delivery_id: UUID) -> dict:
        row = self._delivery(project_id, delivery_id)
        if row.status == "sent":
            raise ValueError("Sent notifications cannot be retried")
        details = dict(row.details or {})
        details["retryRequestedAt"] = datetime.now(timezone.utc).isoformat()
        details["previousAttempts"] = row.attempts
        details["retryable"] = True
        details.pop("nextAttemptAt", None)
        details.pop("exhaustedAt", None)
        row.status = "queued"
        row.attempts = 0
        row.error = None
        row.details = details
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return {"delivery": delivery_to_dict(row)}

    def _project(self, project_id: UUID) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")
        return project

    def _collaborator(self, project_id: UUID, collaborator_id: UUID) -> ProjectCollaborator:
        self._project(project_id)
        row = self.session.get(ProjectCollaborator, collaborator_id)
        if row is None or row.project_id != project_id:
            raise ValueError("Collaborator not found")
        return row

    def _invitation(self, project_id: UUID, invitation_id: UUID) -> ProjectInvitation:
        self._project(project_id)
        row = self.session.get(ProjectInvitation, invitation_id)
        if row is None or row.project_id != project_id:
            raise ValueError("Invitation not found")
        return row

    def _hook(self, project_id: UUID, hook_id: UUID) -> ProjectNotificationHook:
        self._project(project_id)
        row = self.session.get(ProjectNotificationHook, hook_id)
        if row is None or row.project_id != project_id:
            raise ValueError("Notification hook not found")
        return row

    def _delivery(self, project_id: UUID, delivery_id: UUID) -> NotificationDelivery:
        self._project(project_id)
        row = self.session.get(NotificationDelivery, delivery_id)
        if row is None or row.project_id != project_id:
            raise ValueError("Notification delivery not found")
        return row

    def _ensure_project_keeps_owner(self, row: ProjectCollaborator, next_role: str | None) -> None:
        current_is_owner = validate_role(row.role) == "owner"
        next_is_owner = next_role == "owner"
        if not current_is_owner or next_is_owner:
            return
        owner_count = self.session.exec(
            select(ProjectCollaborator)
            .where(ProjectCollaborator.project_id == row.project_id)
            .where(ProjectCollaborator.role == "owner")
        ).all()
        if len(owner_count) <= 1:
            raise ValueError("A shared project must keep at least one owner")


def dispatch_project_notification(
    session: Session,
    artifact: Artifact,
    event_type: str,
    *,
    details: dict | None = None,
) -> list[NotificationDelivery]:
    if artifact.project_id is None:
        return []
    event_key = notification_event_key(event_type)
    hooks = session.exec(
        select(ProjectNotificationHook)
        .where(ProjectNotificationHook.project_id == artifact.project_id)
        .where(ProjectNotificationHook.enabled == True)  # noqa: E712
    ).all()
    deliveries: list[NotificationDelivery] = []
    payload = {
        "artifactId": str(artifact.id),
        "artifactTitle": artifact.title,
        "artifactPath": artifact.path,
        **(details or {}),
    }
    for hook in hooks:
        if not hook_accepts_event(hook, event_key):
            continue
        dedupe_key = notification_dedupe_key(hook, event_key, payload)
        existing = session.exec(
            select(NotificationDelivery)
            .where(NotificationDelivery.hook_id == hook.id)
            .where(NotificationDelivery.dedupe_key == dedupe_key)
        ).first()
        if existing is not None:
            deliveries.append(existing)
            continue
        deliveries.append(
            _record_delivery(
                session,
                hook=hook,
                event_key=event_key,
                dedupe_key=dedupe_key,
                details=payload,
            )
        )
    return deliveries


def dispatch_project_email(
    session: Session,
    project: Project,
    event_type: str,
    *,
    recipient_email: str,
    details: dict | None = None,
) -> list[NotificationDelivery]:
    recipient = normalize_email(recipient_email)
    event_key = notification_event_key(event_type)
    hooks = session.exec(
        select(ProjectNotificationHook)
        .where(ProjectNotificationHook.project_id == project.id)
        .where(ProjectNotificationHook.kind == "email")
        .where(ProjectNotificationHook.enabled == True)  # noqa: E712
    ).all()
    payload = {
        "projectId": str(project.id),
        "projectName": project.name,
        "recipientEmail": recipient,
        **(details or {}),
    }
    deliveries: list[NotificationDelivery] = []
    for hook in hooks:
        if not hook_accepts_event(hook, event_key):
            continue
        dedupe_key = notification_dedupe_key(hook, event_key, payload)
        existing = session.exec(
            select(NotificationDelivery)
            .where(NotificationDelivery.hook_id == hook.id)
            .where(NotificationDelivery.dedupe_key == dedupe_key)
        ).first()
        if existing is not None:
            deliveries.append(existing)
            continue
        deliveries.append(
            _record_delivery(
                session,
                hook=hook,
                event_key=event_key,
                dedupe_key=dedupe_key,
                details=payload,
            )
        )
    return deliveries


def _record_delivery(
    session: Session,
    *,
    hook: ProjectNotificationHook,
    event_key: str,
    dedupe_key: str,
    details: dict,
) -> NotificationDelivery:
    delivery = NotificationDelivery(
        project_id=hook.project_id,
        hook_id=hook.id,
        event_key=event_key,
        dedupe_key=dedupe_key,
        status="queued",
        attempts=0,
        details={
            **details,
            "hookKind": hook.kind,
            "hookTarget": hook.target,
        },
    )
    session.add(delivery)
    session.flush()
    return delivery


def notification_event_key(event_type: str) -> str:
    clean = (event_type or "").strip().lower()
    mapping = {
        "invited": "project.invited",
        "project_invited": "project.invited",
        "collaborator_invited": "project.invited",
        "commented": "artifact.commented",
        "suggested": "artifact.suggested",
        "review_requested": "artifact.review_requested",
        "resolved": "artifact.resolved",
        "reopened": "artifact.reopened",
        "accepted": "artifact.accepted",
        "rejected": "artifact.rejected",
        "accepted_patch": "artifact.accepted",
        "published": "artifact.published",
        "publish_failed": "artifact.publish_failed",
        "preview_failed": "artifact.preview_failed",
        "restored": "artifact.restored",
        "restore": "artifact.restored",
        "restore_deleted": "artifact.restored",
        "forked": "artifact.forked",
        "fork": "artifact.forked",
        "generated_updated": "artifact.generated_updated",
        "generated_update": "artifact.generated_updated",
        "deleted": "artifact.deleted",
    }
    return mapping.get(clean, clean if "." in clean else f"artifact.{clean or 'event'}")


def notification_dedupe_key(hook: ProjectNotificationHook, event_key: str, payload: dict) -> str:
    parts = [
        event_key,
        str(payload.get("artifactId") or ""),
        str(payload.get("commentId") or payload.get("versionId") or payload.get("invitationId") or ""),
        str(payload.get("recipientEmail") or payload.get("invitedEmail") or ""),
        str(payload.get("invitationSendId") or ""),
        str(payload.get("status") or ""),
        str(hook.id),
    ]
    return ":".join(parts)[:255]


def hook_accepts_event(hook: ProjectNotificationHook, event_key: str) -> bool:
    events = normalize_events(hook.events)
    return "*" in events or event_key in events


def normalize_email(email: str) -> str:
    clean = (email or "").strip().lower()
    if not clean or not _EMAIL_RE.match(clean):
        raise ValueError("A valid collaborator email is required")
    return clean


def normalize_optional_email(email: str | None) -> str | None:
    if not email:
        return None
    return normalize_email(email)


def validate_role(role: str) -> str:
    clean = (role or "viewer").strip().lower()
    if clean not in COLLABORATOR_ROLES:
        raise ValueError(f"Collaborator role must be one of: {', '.join(sorted(COLLABORATOR_ROLES))}")
    return clean


def validate_hook_kind(kind: str) -> str:
    clean = (kind or "").strip().lower()
    if clean not in HOOK_KINDS:
        raise ValueError(f"Notification hook kind must be one of: {', '.join(sorted(HOOK_KINDS))}")
    return clean


def normalize_events(events: list[str] | None) -> list[str]:
    if not events:
        return list(DEFAULT_NOTIFICATION_EVENTS)
    clean = []
    for event in events:
        value = str(event or "").strip().lower()
        if not value:
            continue
        clean.append(value if value == "*" or "." in value else f"artifact.{value}")
    return sorted(set(clean)) or list(DEFAULT_NOTIFICATION_EVENTS)


def clean_target(target: str, *, kind: str, secret: str | None = None) -> str:
    clean = (target or "").strip()
    if kind == "email":
        if not clean or not _EMAIL_RE.match(clean):
            raise ValueError("A valid notification recipient email is required")
        return clean
    if clean:
        return clean
    if secret:
        return f"secret:{mask_secret(secret)}"
    raise ValueError("Notification target is required")


def clean_target_for_update(
    row: ProjectNotificationHook,
    kind: str,
    *,
    target: str | None,
    secret: str | None,
) -> str:
    if target is not None:
        return clean_target(target, kind=kind, secret=secret)
    if kind == "email":
        return clean_target(row.target, kind=kind)
    if secret and (not row.target or row.target.startswith("secret:")):
        return clean_target("", kind=kind, secret=secret)
    return clean_target(row.target, kind=kind, secret=secret)


def normalize_hook_config(kind: str, config: dict | None) -> dict:
    raw = config or {}
    if not isinstance(raw, dict):
        raise ValueError("Notification hook config must be an object")
    secret_key = first_secret_config_key(raw)
    if secret_key:
        raise ValueError(f"Notification hook config must not include secret value '{secret_key}'; pass credentials as the hook secret")
    if kind == "email":
        return normalize_email_hook_config(raw)
    return dict(raw)


def normalize_email_hook_config(config: dict) -> dict:
    host = str(config.get("smtpHost") or config.get("host") or "").strip()
    sender = str(config.get("from") or config.get("sender") or "").strip()
    if not host or any(ch.isspace() for ch in host):
        raise ValueError("Email notification hooks require a valid SMTP host")
    if not sender or not _EMAIL_RE.match(sender):
        raise ValueError("Email notification hooks require a valid from address")
    try:
        port = int(config.get("smtpPort") or config.get("port") or 587)
    except (TypeError, ValueError) as exc:
        raise ValueError("Email notification SMTP port must be a number") from exc
    if port < 1 or port > 65535:
        raise ValueError("Email notification SMTP port must be between 1 and 65535")

    normalized = {
        "smtpHost": host,
        "smtpPort": port,
        "from": sender,
        "smtpStartTls": coerce_bool(config.get("smtpStartTls", config.get("startTls", True))),
    }
    username = str(config.get("smtpUsername") or config.get("username") or "").strip()
    if username:
        normalized["smtpUsername"] = username
    subject = str(config.get("subject") or "").strip()
    if subject:
        normalized["subject"] = subject[:200]
    return normalized


def first_secret_config_key(value, *, prefix: str = "") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(part in lowered for part in SECRET_CONFIG_KEY_PARTS):
                return path
            found = first_secret_config_key(child, prefix=path)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = first_secret_config_key(child, prefix=f"{prefix}[{index}]")
            if found:
                return found
    return None


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def mask_secret(secret: str) -> str:
    clean = (secret or "").strip()
    if len(clean) <= 8:
        return "********"
    return f"********{clean[-4:]}"


def collaborator_to_dict(row: ProjectCollaborator) -> dict:
    return {
        "id": str(row.id),
        "projectId": str(row.project_id),
        "email": row.email,
        "displayName": row.display_name,
        "role": row.role,
        "notificationState": row.notification_state or {},
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }


def invitation_to_dict(row: ProjectInvitation, *, token: str | None = None) -> dict:
    payload = {
        "id": str(row.id),
        "projectId": str(row.project_id),
        "email": row.email,
        "displayName": row.display_name,
        "role": row.role,
        "status": row.status,
        "expiresAt": row.expires_at.isoformat() if row.expires_at else None,
        "acceptedAt": row.accepted_at.isoformat() if row.accepted_at else None,
        "revokedAt": row.revoked_at.isoformat() if row.revoked_at else None,
        "lastSentAt": row.last_sent_at.isoformat() if row.last_sent_at else None,
        "sendCount": row.send_count or 0,
        "invitedBySubject": row.invited_by_subject,
        "invitedByEmail": row.invited_by_email,
        "invitedByName": row.invited_by_name,
        "acceptedBySubject": row.accepted_by_subject,
        "acceptedByEmail": row.accepted_by_email,
        "acceptedByName": row.accepted_by_name,
        "notificationState": row.notification_state or {},
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }
    if token:
        payload["acceptToken"] = token
    return payload


def new_invitation_token() -> str:
    return secrets.token_urlsafe(32)


def invitation_token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def invitation_expired(row: ProjectInvitation) -> bool:
    if row.expires_at is None:
        return False
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires <= datetime.now(timezone.utc)


def hook_to_dict(row: ProjectNotificationHook) -> dict:
    return {
        "id": str(row.id),
        "projectId": str(row.project_id),
        "kind": row.kind,
        "target": row.target,
        "enabled": row.enabled,
        "events": normalize_events(row.events),
        "secretSet": bool(row.secret_ciphertext),
        "config": public_hook_config(row.config or {}),
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }


def delivery_to_dict(row: NotificationDelivery) -> dict:
    return {
        "id": str(row.id),
        "projectId": str(row.project_id),
        "hookId": str(row.hook_id) if row.hook_id else None,
        "eventKey": row.event_key,
        "dedupeKey": row.dedupe_key,
        "status": row.status,
        "attempts": row.attempts,
        "error": row.error,
        "details": public_delivery_details(row.details or {}),
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }


def decrypt_hook_secret(row: ProjectNotificationHook) -> str | None:
    return decrypt(row.secret_ciphertext) if row.secret_ciphertext else None


def public_hook_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    safe = {}
    for key, value in config.items():
        lowered = str(key).lower()
        if any(part in lowered for part in SECRET_CONFIG_KEY_PARTS):
            continue
        safe[key] = value
    return safe


def public_delivery_details(details) -> dict:
    if not isinstance(details, dict):
        return {}
    safe = {}
    for key, value in details.items():
        lowered = str(key).lower()
        if any(part in lowered for part in SECRET_CONFIG_KEY_PARTS):
            continue
        if isinstance(value, dict):
            safe[key] = public_delivery_details(value)
        elif isinstance(value, list):
            safe[key] = [
                public_delivery_details(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            safe[key] = value
    return safe
