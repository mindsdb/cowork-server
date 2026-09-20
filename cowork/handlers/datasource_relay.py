"""Cloud connector submissions, handed to auth's encrypted store.

In org mode a database form never reaches the staging store, the probe or a
local vault. The submitted values are checked against the capability policy,
normalized into auth's payload and relayed under the caller's own bearer;
auth holds the credential encrypted and validates it out of band. What comes
back is masked metadata, which is all this turn ever says out loud.

A refusal raises before anything streams, so a client sees an ordinary status
code rather than a failure buried in a 200 event. The SSE framing is written
a second time here rather than shared with `probe.py`: there is one round trip
and nothing to stream incrementally, so every event exists before the response
starts, which also keeps the database write out of a generator that runs after
the request's session dependency has exited.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any
from uuid import UUID

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from cowork.common.settings.app_settings import OAuthSettings
from cowork.db.scoped import ScopedSession
from cowork.schemas.connectors import DatasourceCreateRequest, SubmitFormRequest
from cowork.services.connectors.datasource_capabilities import load_datasource_capabilities
from cowork.services.connectors.datasources import (
    InvalidDatasourceInput,
    UnsupportedDatasourceCapability,
    normalize_datasource_input,
)
from cowork.services.connectors.oauth import auth_proxy
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import missing_required_fields
from cowork.services.conversations import ConversationService

logger = logging.getLogger(__name__)


def _rendered(name: str) -> str:
    """The connection name as it may appear inside markdown.

    The name is the submitter's own text and this turn is rendered as
    markdown, so a newline followed by a fenced block would put a form of
    their choosing into their conversation. Whitespace collapses and the fence
    character goes.
    """
    return re.sub(r"\s+", " ", name).replace("`", "").strip()


def _sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _to_payload(req: SubmitFormRequest, connector_id: str, method: str, values: dict[str, Any]) -> dict[str, Any]:
    """Build auth's create payload out of the submitted values.

    The model is constructed inside the guard because `values` is whatever the
    wire carried: a number where a password belongs raises a pydantic error
    whose text quotes the input, and that must not become a 500 with the value
    in the log.
    """
    try:
        model = DatasourceCreateRequest(
            connector_id=connector_id,
            method=method,
            name=req.name,
            host=values.get("host"),
            port=values.get("port"),
            database=values.get("database"),
            username=values.get("username"),
            password=values.get("password"),
            # The cloud form asks nothing about certificates, so this is absent
            # for nearly every submission and the server applies its own
            # default. A spec that does ask still travels as it was answered.
            tls=(
                {"mode": values["tls_mode"], "ca_pem": values.get("ca_pem") or None}
                if values.get("tls_mode")
                else None
            ),
        )
    except ValidationError as exc:
        raise InvalidDatasourceInput("the submitted connection fields are invalid") from exc
    return normalize_datasource_input(model)


def _save_turn(scoped: ScopedSession, conversation_id: str | None, text: str, events: list[dict]) -> None:
    """Record the turn in the conversation, if this submission came from one.

    Best effort on purpose: the credential is already stored in auth by the
    time this runs, so a conversation that has gone away must not turn a
    successful save into an error the user can act on.
    """
    if not conversation_id:
        return
    try:
        service = ConversationService(scoped)
        conversation = service.get_conversation(UUID(conversation_id))
        service.save_assistant_turn(conversation.id, text, events)
    except Exception:
        # With the cause: the text and events hold the connection name and the
        # connector id, nothing secret, and a silent loss here would otherwise
        # look like a conversation that never had the turn.
        logger.warning(
            "[datasources] could not record the submission turn for conversation %s",
            conversation_id,
            exc_info=True,
        )


async def relay_cloud_submission(
    req: SubmitFormRequest,
    connector_id: str,
    method: str | None,
    scoped: ScopedSession,
    request: Request,
) -> StreamingResponse:
    """Relay one enabled cloud connector submission and answer with its turn."""
    spec = registry.get_connector(connector_id)
    capabilities = load_datasource_capabilities()
    if spec is None or not method or not capabilities.is_available(connector_id, method):
        raise UnsupportedDatasourceCapability()

    fields = capabilities.cloud_fields(connector_id, method)
    missing = missing_required_fields(fields, req.values, req.skipped)
    if missing:
        raise InvalidDatasourceInput(f"missing required fields: {', '.join(missing)}")
    # Only the cloud form's own fields may travel. A driver option smuggled in
    # beside them would otherwise be stored by auth and used when dialing.
    if set(req.values) - {f.name for f in fields}:
        raise InvalidDatasourceInput("this connector accepts only its cloud connection fields")

    payload = _to_payload(req, connector_id, method, req.values)
    saved = await auth_proxy.proxy_datasource_create(request, OAuthSettings(), payload)

    name = str(saved.get("name") or payload["name"])
    shown = _rendered(name)
    response_id = "resp-" + uuid.uuid4().hex[:12]
    message_id = "msg-" + uuid.uuid4().hex[:12]
    text = f"Saved encrypted draft **{shown}** for {connector_id}. Validation has not run yet.\n\n"
    patch = {
        "form_id": spec.form.form_id,
        "title": f"Saved — {shown}",
        "subtitle": "Stored encrypted. Validation runs separately; the connection shows its result when it finishes.",
        "status_text": None,
        "_is_probing": False,
        "_is_success": True,
        "actions": [{"id": "dismiss", "label": "Close", "kind": "cancel"}],
    }
    patch_block = f"\n\n```data-vault-form-patch\n{json.dumps(patch, indent=2)}\n```\n\n"

    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {
            "type": "response.created",
            "response": {"id": response_id, "model": "datavault-agent", "status": "created"},
            "conversation_id": req.conversation_id,
        }),
        ("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": message_id, "delta": text,
        }),
        ("response.output_text.delta", {
            "type": "response.output_text.delta", "item_id": message_id, "delta": patch_block,
        }),
        ("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "success", "user_label": shown},
        }),
    ]
    recorded = [{**payload_, "sequence_number": index} for index, (_, payload_) in enumerate(events, start=1)]
    frames = [_sse(kind, recorded[index]) for index, (kind, _) in enumerate(events)]

    _save_turn(scoped, req.conversation_id, text + patch_block, recorded)

    async def stream():
        for frame in frames:
            yield frame

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )
