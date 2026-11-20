"""Fail-closed adapter for the deployment's approved moderation service.

Without a dedicated endpoint, checks use the existing MindsHub gateway with
short-lived caller-scoped keys. Enforcement stays off until classifier and
coverage evaluation passes. Never log content, keys or provider response bodies.
"""
from __future__ import annotations
import asyncio
import json
from dataclasses import asdict, is_dataclass
from typing import Literal
from urllib.parse import urlsplit
import httpx
from fastapi import HTTPException
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict
from cowork.common.settings.app_settings import Settings, MINDS_FREE_MODEL
from cowork.db.scoped import TenantScope

class ContentSafetySettings(Settings):
    model_config = SettingsConfigDict(env_prefix="COWORK_CONTENT_SAFETY_")
    required: bool = False
    model: str = MINDS_FREE_MODEL
    endpoint: str = ""
    api_key: SecretStr | None = None

MAX_CONTENT_CHARS = 100_000
MAX_FILE_BYTES = 100_000

async def check_file_bytes(data: bytes, *, content_type: str, purpose: Literal['input', 'output'], scope: TenantScope) -> None:
    """Check an immutable plain-text file; unsupported formats fail closed.

    MIME labels cannot approve embedded media or executable HTML. Callers must
    serve/write these exact bytes after approval, never reopen a mutable path.
    """
    settings = ContentSafetySettings()
    if not settings.required:
        return
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, 'This file is too large for a safety check.')
    if content_type.split(';', 1)[0].strip().lower() != 'text/plain':
        raise HTTPException(503, 'Safety checks for this file type are not available yet.')
    try:
        text = data.decode('utf-8', errors='strict')
    except UnicodeError:
        raise HTTPException(503, 'Safety checks for this file encoding are not available yet.') from None
    if any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        raise HTTPException(503, 'Safety checks for this file encoding are not available yet.')
    await check_content(text, purpose=purpose, settings=settings, scope=scope)

async def check_content(content: str, *, purpose: Literal['input', 'output'], settings=None, transport=None, scope: TenantScope | None = None) -> None:
    settings = settings or ContentSafetySettings()
    if not settings.required:
        return
    url = urlsplit(settings.endpoint)
    if settings.endpoint and (url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment):
        raise HTTPException(503, 'Content safety checks are unavailable. Please try again later.')
    if len(content) > MAX_CONTENT_CHARS:
        raise HTTPException(413, 'This content is too large for a safety check.')
    if not content.strip():
        return
    # Match the main model's redaction boundary before either transport. This
    # covers known key shapes and exact secrets registered in the current turn.
    from anton.utils.datasources import scrub_credentials
    content = scrub_credentials(content)
    headers = {}
    if settings.api_key:
        headers['Authorization'] = f'Bearer {settings.api_key.get_secret_value()}'
    try:
        async with asyncio.timeout(10):
            if settings.endpoint:
                async with httpx.AsyncClient(timeout=8, transport=transport, follow_redirects=False) as client:
                    response = await client.post(settings.endpoint, headers=headers,
                        json={'policy': 'cowork-store-safety-v1', 'purpose': purpose, 'content': content})
                    response.raise_for_status()
                    from cowork.services.content_safety_gateway import _unique_object
                    result = response.json(object_pairs_hook=_unique_object)
            else:
                from cowork.common.settings.user_settings import current_settings_scope
                from cowork.services.content_safety_gateway import gateway_verdict
                caller = scope or current_settings_scope()
                if caller is None or not caller.org_mode or not caller.org_id or not caller.user_id:
                    raise ValueError('Missing moderation caller')
                result = await gateway_verdict(content, purpose=purpose, scope=caller, model=settings.model)
            if not isinstance(result, dict) or set(result) != {'decision'} or result.get('decision') not in ('allow', 'block'):
                raise ValueError('invalid verdict')
            if result['decision'] == 'block':
                raise HTTPException(422, 'This content cannot be processed under our safety policy.')
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Content safety checks are unavailable. Please try again later.') from None

async def check_request(request, *, scope: TenantScope | None = None) -> None:
    settings = ContentSafetySettings()
    if not settings.required:
        return
    # Attachment IDs are not their contents. Never silently approve an upload
    # on the strength of its filename while the attachment scanner is pending.
    payload = request.model_dump(mode='json').get('input')
    non_text = False
    if isinstance(payload, list):
        for message in payload:
            content = message.get('content')
            if isinstance(content, list):
                non_text |= any(not isinstance(part, dict) or part.get('type') != 'input_text' for part in content)
            elif content is not None and not isinstance(content, str):
                non_text = True
    if request.attachment_ids or non_text:
        raise HTTPException(503, 'Attachment safety checks are not available yet.')
    await check_content(json.dumps(payload, ensure_ascii=False),
                        purpose='input', settings=settings, scope=scope)

async def guarded_events(events):
    """Check visible semantic blocks, not individual wire fragments/history.

    Retain a bounded tail across blocks to catch content split at an event
    boundary. This is a text guard, not approval of referenced files or images.
    """
    settings = ContentSafetySettings()
    if not settings.required:
        async for event in events:
            yield event
        return
    from anton.core.llm import provider
    from anton.core.llm.provider import (
        StreamTextDelta, StreamToolUseDelta,
        StreamToolUseEnd, StreamComplete, StreamBrowserSession,
    )
    text_types = (StreamTextDelta,)
    reasoning_type = getattr(provider, 'StreamReasoningDelta', None)
    if reasoning_type is not None:
        text_types += (reasoning_type,)
    from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory
    pending = []
    pending_type = None
    size = 0
    context = ''
    tool_parts = {}
    tool_size = 0

    async def check_next(content):
        nonlocal context
        # Limit the new block independently; history cannot consume its budget.
        if len(content) > MAX_CONTENT_CHARS:
            raise HTTPException(413, 'This content is too large for a safety check.')
        await check_content(context + content, purpose='output', settings=settings)
        context = (context + content)[-8000:]

    async def flush():
        nonlocal size, pending_type
        text = ''.join(pending)
        await check_next(text)
        event = pending_type(text=text)
        pending.clear()
        pending_type = None
        size = 0
        return event

    async for event in events:
        if isinstance(event, text_types):
            if pending and type(event) is not pending_type:
                yield await flush()
            pending_type = type(event)
            pending.append(event.text)
            size += len(event.text)
            if size > MAX_CONTENT_CHARS - 8000:
                raise HTTPException(413, 'This content is too large for a safety check.')
            continue
        if pending:
            yield await flush()
        if isinstance(event, StreamToolUseDelta):
            tool_parts.setdefault(event.id, []).append(event.json_delta)
            tool_size += len(event.json_delta)
            if tool_size > MAX_CONTENT_CHARS - 8000:
                raise HTTPException(413, 'This content is too large for a safety check.')
            continue
        if isinstance(event, StreamToolUseEnd):
            text = ''.join(tool_parts.pop(event.id, []))
            tool_size -= len(text)
            if text:
                await check_next(text)
                yield StreamToolUseDelta(id=event.id, json_delta=text)
            yield event
            continue
        if isinstance(event, (TurnContext, TurnHistory, StreamComplete, StreamBrowserSession)):
            # Internal replay/history, duplicate completion, or an opaque ID.
            # Visible answer and tool content have their own guarded events.
            yield event
            continue
        payload = event.model_dump(mode='json') if hasattr(event, 'model_dump') else asdict(event) if is_dataclass(event) else event
        await check_next(json.dumps(payload, default=str, ensure_ascii=False))
        yield event
    if pending:
        yield await flush()
    if tool_parts:
        # A truncated tool call must never be released without its check.
        raise HTTPException(503, 'The content safety check could not finish.')
