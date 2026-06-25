from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Protocol
from uuid import UUID

import httpx
from sqlalchemy import update
from sqlmodel import Session, select

from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_open_session
from cowork.models.project_collaboration import NotificationDelivery, ProjectNotificationHook
from cowork.services.project_collaboration import decrypt_hook_secret, delivery_to_dict


logger = setup_logging()
MAX_DELIVERY_ATTEMPTS = 5
RETRY_BASE_DELAY_SECONDS = 30
RETRY_MAX_DELAY_SECONDS = 15 * 60
DEFAULT_DISPATCH_INTERVAL_SECONDS = 5.0
_dispatcher_task: asyncio.Task | None = None


@dataclass(frozen=True)
class SendResult:
    status: str
    retryable: bool = False
    external_id: str | None = None
    error: str | None = None
    details: dict | None = None


class NotificationSender(Protocol):
    kind: str

    async def send(self, hook: ProjectNotificationHook, delivery: NotificationDelivery) -> SendResult:
        ...


class WebhookSender:
    kind = "webhook"

    async def send(self, hook: ProjectNotificationHook, delivery: NotificationDelivery) -> SendResult:
        url = decrypt_hook_secret(hook)
        if not url:
            return SendResult(status="failed", error="webhook_url_missing", retryable=False)
        return await _post_json(url, _public_payload(hook, delivery), provider="webhook")


class EmailSmtpSender:
    kind = "email"

    async def send(self, hook: ProjectNotificationHook, delivery: NotificationDelivery) -> SendResult:
        config = hook.config or {}
        host = str(config.get("smtpHost") or config.get("host") or "").strip()
        sender = str(config.get("from") or config.get("sender") or "").strip()
        if not host or not sender:
            return SendResult(status="failed", error="smtp_not_configured", retryable=False)

        try:
            port = int(config.get("smtpPort") or config.get("port") or 587)
        except (TypeError, ValueError):
            return SendResult(status="failed", error="smtp_port_invalid", retryable=False)

        username = str(config.get("smtpUsername") or config.get("username") or "").strip()
        password = decrypt_hook_secret(hook)
        use_tls = bool(config.get("smtpStartTls", config.get("startTls", True)))
        subject = str(config.get("subject") or "Cowork artifact review update").strip()
        details = delivery.details if isinstance(delivery.details, dict) else {}
        recipient = str(details.get("recipientEmail") or details.get("recipient") or hook.target).strip()

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = recipient
        message.set_content(_message_text(hook, delivery))

        try:
            await asyncio.to_thread(
                _send_email_sync,
                host,
                port,
                message,
                username=username,
                password=password,
                use_tls=use_tls,
            )
        except smtplib.SMTPAuthenticationError:
            return SendResult(status="failed", error="smtp_auth_failed", retryable=False)
        except smtplib.SMTPRecipientsRefused:
            return SendResult(status="failed", error="smtp_recipient_refused", retryable=False)
        except smtplib.SMTPResponseException as exc:
            code = int(exc.smtp_code or 0)
            retryable = 400 <= code < 500
            return SendResult(
                status="failed",
                error=f"smtp_{code}" if code else "smtp_response_error",
                retryable=retryable,
                details={"smtpCode": code} if code else None,
            )
        except (TimeoutError, OSError, smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError):
            return SendResult(status="failed", error="smtp_transient_error", retryable=True)
        except smtplib.SMTPException:
            return SendResult(status="failed", error="smtp_error", retryable=False)
        return SendResult(status="sent", external_id=f"email:{recipient}")


def default_senders() -> dict[str, NotificationSender]:
    return {
        "webhook": WebhookSender(),
        "email": EmailSmtpSender(),
    }


async def dispatch_pending_notifications(
    session: Session,
    *,
    limit: int = 25,
    senders: dict[str, NotificationSender] | None = None,
) -> list[dict]:
    rows = session.exec(
        select(NotificationDelivery)
        .where(NotificationDelivery.status.in_(["queued", "failed"]))  # type: ignore[attr-defined]
        .order_by(NotificationDelivery.created_at)
    ).all()
    results = []
    for row in rows:
        if len(results) >= limit:
            break
        if row.status == "failed" and not _retry_ready(row):
            continue
        results.append(await send_notification_delivery(session, row.id, senders=senders))
    return results


async def send_notification_delivery(
    session: Session,
    delivery_id: UUID,
    *,
    senders: dict[str, NotificationSender] | None = None,
) -> dict:
    delivery = _claim_delivery(session, delivery_id)
    if delivery is None:
        existing = session.get(NotificationDelivery, delivery_id)
        if existing is None:
            raise ValueError("Notification delivery not found")
        return {"delivery": delivery_to_dict(existing), "skipped": True}
    hook = session.get(ProjectNotificationHook, delivery.hook_id) if delivery.hook_id else None
    if hook is None:
        _mark_delivery(delivery, SendResult(status="skipped", error="hook_missing"))
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        return {"delivery": delivery_to_dict(delivery)}
    if not hook.enabled:
        _mark_delivery(delivery, SendResult(status="skipped", error="hook_disabled"))
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        return {"delivery": delivery_to_dict(delivery)}

    sender_map = senders or default_senders()
    sender = sender_map.get(hook.kind)
    if sender is None:
        _mark_delivery(delivery, SendResult(status="skipped", error="sender_unavailable"))
        session.add(delivery)
        session.commit()
        session.refresh(delivery)
        return {"delivery": delivery_to_dict(delivery)}

    try:
        result = await sender.send(hook, delivery)
    except Exception:
        logger.exception("Notification sender failed")
        result = SendResult(status="failed", error="sender_exception", retryable=True)
    _mark_delivery(delivery, result)
    session.add(delivery)
    session.commit()
    session.refresh(delivery)
    return {"delivery": delivery_to_dict(delivery)}


def start_notification_dispatcher(interval_seconds: float = DEFAULT_DISPATCH_INTERVAL_SECONDS) -> asyncio.Task | None:
    global _dispatcher_task
    if _dispatcher_task is not None and not _dispatcher_task.done():
        return _dispatcher_task
    try:
        _dispatcher_task = asyncio.create_task(_notification_dispatch_loop(interval_seconds))
        return _dispatcher_task
    except RuntimeError:
        logger.exception("Notification dispatcher could not start")
        return None


async def stop_notification_dispatcher() -> None:
    global _dispatcher_task
    task = _dispatcher_task
    _dispatcher_task = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _notification_dispatch_loop(interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            with get_open_session(get_app_settings().database.uri) as session:
                await dispatch_pending_notifications(session)
        except Exception:
            logger.exception("Notification dispatch loop error")


def _mark_delivery(delivery: NotificationDelivery, result: SendResult) -> None:
    status = result.status if result.status in {"queued", "sent", "skipped", "failed", "exhausted"} else "failed"
    now = datetime.now(timezone.utc)
    details = dict(delivery.details or {})
    details["retryable"] = bool(result.retryable)
    details["lastAttemptAt"] = now.isoformat()
    details.pop("nextAttemptAt", None)
    if status == "failed" and result.retryable:
        if delivery.attempts >= MAX_DELIVERY_ATTEMPTS:
            status = "exhausted"
            details["retryable"] = False
            details["exhaustedAt"] = now.isoformat()
        else:
            details["nextAttemptAt"] = (now + retry_delay(delivery.attempts)).isoformat()
    if result.external_id:
        details["externalId"] = result.external_id
    if result.details:
        details.update(_safe_details(result.details))
    delivery.status = status
    delivery.error = _safe_error(result.error)
    delivery.details = details


def _retry_ready(delivery: NotificationDelivery) -> bool:
    details = delivery.details or {}
    if not bool(details.get("retryable")) or delivery.attempts >= MAX_DELIVERY_ATTEMPTS:
        return False
    next_attempt = details.get("nextAttemptAt")
    if not next_attempt:
        return True
    try:
        ready_at = datetime.fromisoformat(str(next_attempt))
    except ValueError:
        return True
    if ready_at.tzinfo is None:
        ready_at = ready_at.replace(tzinfo=timezone.utc)
    return ready_at <= datetime.now(timezone.utc)


def retry_delay(attempts: int) -> timedelta:
    exponent = max(0, attempts - 1)
    seconds = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** exponent))
    return timedelta(seconds=seconds)


def _claim_delivery(session: Session, delivery_id: UUID) -> NotificationDelivery | None:
    result = session.exec(
        update(NotificationDelivery)
        .where(NotificationDelivery.id == delivery_id)
        .where(NotificationDelivery.status.in_(["queued", "failed"]))  # type: ignore[attr-defined]
        .values(
            status="sending",
            attempts=NotificationDelivery.attempts + 1,
            error=None,
        )
    )
    if getattr(result, "rowcount", 0) != 1:
        session.rollback()
        return None
    session.commit()
    delivery = session.get(NotificationDelivery, delivery_id)
    if delivery is not None:
        session.refresh(delivery)
    return delivery


async def _post_json(url: str, payload: dict, *, provider: str) -> SendResult:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.post(url, json=payload)
    except (httpx.TimeoutException, httpx.TransportError):
        return SendResult(status="failed", error=f"{provider}_transient_error", retryable=True)
    if response.status_code in {408, 425, 429} or response.status_code >= 500:
        return SendResult(status="failed", error=f"{provider}_{response.status_code}", retryable=True)
    if response.status_code >= 400:
        return SendResult(status="failed", error=f"{provider}_{response.status_code}", retryable=False)
    external_id = response.headers.get("x-request-id")
    return SendResult(status="sent", external_id=external_id)


def _send_email_sync(
    host: str,
    port: int,
    message: EmailMessage,
    *,
    username: str,
    password: str | None,
    use_tls: bool,
) -> None:
    with smtplib.SMTP(host, port, timeout=15) as smtp:
        if use_tls:
            smtp.starttls()
        if username or password:
            smtp.login(username, password or "")
        smtp.send_message(message)


def _message_text(hook: ProjectNotificationHook, delivery: NotificationDelivery) -> str:
    details = delivery.details or {}
    if delivery.event_key == "project.invited":
        inviter = details.get("inviterName") or details.get("inviterEmail") or "Someone"
        project = details.get("projectName") or "a Cowork project"
        role = details.get("invitedRole") or "collaborator"
        token = details.get("inviteToken") or ""
        expiry = details.get("expiresAt") or ""
        return "\n".join(
            [
                f"{inviter} invited you to collaborate on {project} as {role}.",
                f"Accept token: {token}" if token else "",
                f"Expires: {expiry}" if expiry else "",
            ]
        ).strip()
    artifact = details.get("artifactTitle") or details.get("artifactPath") or "Artifact"
    actor = details.get("actorName") or "Someone"
    event = delivery.event_key.replace("artifact.", "").replace("_", " ")
    return f"{actor} {event} on {artifact}."


def _public_payload(hook: ProjectNotificationHook, delivery: NotificationDelivery) -> dict:
    return {
        "eventKey": delivery.event_key,
        "projectId": str(delivery.project_id),
        "deliveryId": str(delivery.id),
        "target": hook.target,
        "details": _safe_details(delivery.details or {}),
    }


def _safe_details(details: dict) -> dict:
    safe = {}
    for key, value in details.items():
        if "secret" in str(key).lower() or "password" in str(key).lower() or "token" in str(key).lower():
            continue
        safe[key] = value
    return safe


def _safe_error(error: str | None) -> str | None:
    if not error:
        return None
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(error).lower())
    return clean[:128] or "notification_error"
