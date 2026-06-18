from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.schemas.responses import Role
from cowork.services.connectors.probe import CredentialProbe, ProbeOutcome
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import store
from cowork.services.conversations import ConversationService

logger = logging.getLogger(__name__)


class ProbeHandler:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def run(
        self,
        submission_id: str,
        connector_id: str,
        method: str | None,
        name: str,
        conversation_id: str | None,
    ) -> AsyncGenerator[str, None]:
        response_id = "resp-" + uuid.uuid4().hex[:12]
        message_id = "msg-" + uuid.uuid4().hex[:12]
        seq = 0
        body_parts: list[str] = []
        recorded_events: list[dict] = []

        def _sse(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        def _push(event_type: str, data: dict) -> str:
            nonlocal seq
            seq += 1
            payload = {**data, "sequence_number": seq}
            recorded_events.append(payload)
            return _sse(event_type, payload)

        def _delta(text: str) -> str:
            body_parts.append(text)
            return _push("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": message_id,
                "delta": text,
            })

        def _format_patch_block(patch: dict) -> str:
            body = json.dumps(patch, indent=2)
            return f"\n\n```data-vault-form-patch\n{body}\n```\n\n"

        def _field_patch(form_id: str, method_id: str | None, field_name: str, value: Any) -> dict:
            if method_id:
                return {"form_id": form_id, "methods": {method_id: {"fields": {field_name: value}}}}
            return {"form_id": form_id, "fields": {field_name: value}}

        def _patch_delta(patch: dict) -> str:
            return _delta(_format_patch_block(patch))

        _temp_workspace_dir: str | None = None

        try:
            # Resolve conversation + workspace
            db_conversation_id: UUID | None = None
            workspace = None
            workspace_path: Path | None = None

            if conversation_id:
                try:
                    conversation = ConversationService(self.session).get_conversation(UUID(conversation_id))
                    db_conversation_id = conversation.id
                    workspace_path = Path(conversation.project.path)
                except Exception:
                    logger.warning("Could not resolve conversation %s", conversation_id)

            # response.created — opens the SSE turn
            yield _push("response.created", {
                "type": "response.created",
                "response": {"id": response_id, "model": "datavault-agent", "status": "created"},
                "conversation_id": conversation_id,
            })

            # Stage check
            submission = store.get(submission_id)
            if not submission:
                yield _delta(
                    "The form submission expired before I could process it. "
                    "Please re-submit the form."
                )
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "failed"},
                })
                self._save_assistant_turn(db_conversation_id, "".join(body_parts), recorded_events)
                return

            values = submission.get("values", {}) or {}
            skipped = submission.get("skipped", []) or []
            credentials = {
                k: v for k, v in values.items()
                if k not in skipped and v is not None and v != ""
            }

            # Connector spec — absent for agent-handcrafted (non-registry)
            # connectors; those fall back to the form_spec staged with the
            # submission and skip the probe below.
            spec = registry.get_connector(connector_id)
            if spec is not None:
                form_id = spec.form.form_id
                form_spec = spec.form.model_dump()
            else:
                form_spec = dict(submission.get("form_spec") or {})
                form_id = form_spec.get("form_id") or f"{connector_id}-connector"
            if method:
                form_spec["selected_method"] = method

            # Save without probe only when there is no registry spec —
            # there is no engine to verify a handcrafted connector against.
            # Missing conversation context is fine: a temp workspace is created below.
            if spec is None:
                try:
                    from anton.core.datasources.data_vault import LocalDataVault
                    vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
                    slug = (name or "").strip() or f"{connector_id}-{uuid.uuid4().hex[:6]}"
                    payload_to_save = {**credentials, "_connector_id": connector_id}
                    if method:
                        payload_to_save["_method"] = method
                    vault.save(connector_id, slug, payload_to_save)
                except Exception as exc:
                    yield _delta(f"Could not save: `{exc}`.")
                    yield _push("response.completed", {
                        "type": "response.completed",
                        "response": {"id": response_id, "status": "failed"},
                    })
                    self._save_assistant_turn(db_conversation_id, "".join(body_parts), recorded_events)
                    return
                reason = "connector is not in the registry"
                yield _delta(f"Saved as `{slug}` (no live probe — {reason}).\n\n")
                yield _patch_delta({
                    "form_id": form_id,
                    "title": f"Saved — {slug}",
                    "subtitle": "Stored in the vault. No live verification was performed.",
                    "status_text": None,
                    "_is_probing": False,
                    "_is_success": True,
                    "actions": [{"id": "dismiss", "label": "Close", "kind": "cancel"}],
                })
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "success"},
                })
                self._save_assistant_turn(db_conversation_id, "".join(body_parts), recorded_events)
                return

            # Probe path: build workspace + LLM client
            if workspace_path is None:
                _temp_workspace_dir = tempfile.mkdtemp(prefix="cowork-probe-")
                workspace_path = Path(_temp_workspace_dir)

            try:
                from anton.workspace import Workspace
                workspace = Workspace(workspace_path)
            except Exception:
                logger.exception("Could not build workspace for probe")

            llm_client = None
            try:
                llm_client = self._build_llm_client()
            except Exception:
                logger.exception("Could not build LLM client for probe")

            # Workspace / LLM client availability
            if workspace is None or llm_client is None:
                err = "Could not initialize the probe (workspace or LLM client unavailable)."
                yield _delta(err)
                yield _patch_delta({"form_id": form_id, "form_error": err})
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "failed"},
                })
                self._save_assistant_turn(db_conversation_id, "".join(body_parts), recorded_events)
                return

            # Intro + initial probing patch
            yield _delta(f"Trying to connect to **{connector_id}**…\n\n")
            yield _patch_delta({
                "form_id": form_id,
                "_is_probing": True,
                "status_text": "Starting probe…",
                "form_error": None,
            })

            # Run probe
            final_outcome: ProbeOutcome | None = None
            pending_cell: dict = {}

            probe = CredentialProbe(
                engine=connector_id,
                credentials=credentials,
                llm_client=llm_client,
                workspace=workspace,
                form_spec=form_spec,
                skipped=skipped,
            )
            try:
                async for kind, payload in probe.run():
                    if kind == "text":
                        yield _delta(payload)
                    elif kind == "status":
                        yield _patch_delta({
                            "form_id": form_id,
                            "_is_probing": True,
                            "status_text": payload,
                        })
                    elif kind == "field_status":
                        field_name = (payload or {}).get("name")
                        if field_name:
                            status_val = (payload or {}).get("status")
                            mid = (payload or {}).get("method_id")
                            yield _patch_delta(_field_patch(form_id, mid, field_name, {"status": status_val}))
                    elif kind == "remove_field":
                        field_name = (payload or {}).get("name")
                        if field_name:
                            mid = (payload or {}).get("method_id")
                            yield _patch_delta(_field_patch(form_id, mid, field_name, None))
                    elif kind == "switch_method":
                        mid = (payload or {}).get("method_id")
                        reason = (payload or {}).get("reason") or ""
                        if mid:
                            patch: dict = {"form_id": form_id, "selected_method": mid}
                            if reason:
                                patch["status_text"] = reason
                            yield _patch_delta(patch)
                    elif kind == "scratchpad":
                        action = payload.get("action")
                        if action == "start":
                            yield _push("response.in_progress", {
                                "type": "response.in_progress",
                                "thought_role": "thought.scratchpad.start",
                                "tool_name": "scratchpad",
                            })
                        elif action == "end":
                            pending_cell.update(payload)
                            yield _push("response.in_progress", {
                                "type": "response.in_progress",
                                "thought_role": "thought.scratchpad.end",
                                "content": json.dumps({
                                    "name": payload.get("name", ""),
                                    "one_line_description": payload.get("one_line_description", ""),
                                    "code": payload.get("code", ""),
                                }),
                            })
                        elif action == "result":
                            yield _push("response.in_progress", {
                                "type": "response.in_progress",
                                "thought_role": "thought.scratchpad.result",
                                "content": json.dumps({
                                    "code": pending_cell.get("code", ""),
                                    "stdout": payload.get("content", ""),
                                    "stderr": "",
                                }),
                            })
                            pending_cell = {}
                    elif kind == "verdict":
                        final_outcome = payload
                        break
            except Exception as exc:
                logger.exception("Probe iteration failed")
                final_outcome = ProbeOutcome(
                    status="failure",
                    error=f"Probe runner crashed: {exc}",
                    follow_up="Try resubmitting; if it persists, restart the app.",
                )

            if final_outcome is None:
                final_outcome = ProbeOutcome(status="failure", error="Probe ended without a verdict.")

            # Apply verdict
            saved_slug: str | None = None
            if final_outcome.status == "success":
                try:
                    from anton.core.datasources.data_vault import LocalDataVault
                    vault = LocalDataVault(Path(get_app_settings().connector.vault_dir))
                    slug = (name or "").strip() or f"{connector_id}-{uuid.uuid4().hex[:6]}"
                    payload_to_save = {**credentials, "_connector_id": connector_id}
                    if method:
                        payload_to_save["_method"] = method
                    vault.save(connector_id, slug, payload_to_save)
                    saved_slug = slug
                except Exception as exc:
                    logger.exception("Vault save failed despite probe success")
                    final_outcome.status = "failure"
                    final_outcome.error = f"Probe succeeded but save failed: {exc}"

            if final_outcome.status == "success":
                summary = final_outcome.summary or "Connection works."
                yield _delta(f"\n\n{summary}\n")
                yield _patch_delta({
                    "form_id": form_id,
                    "title": f"Connected — {saved_slug}",
                    "subtitle": summary,
                    "status_text": None,
                    "form_error": None,
                    "_is_probing": False,
                    "_is_success": True,
                })
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "success"},
                })
            elif final_outcome.status == "needs_input":
                reason = final_outcome.follow_up or "We need a few more details before we can connect."
                yield _delta(f"\n\nI need a bit more info before I can finish: {reason}\n")
                extra: dict = {}
                for f in final_outcome.extra_fields:
                    fname = f.get("name")
                    if fname:
                        extra[fname] = {
                            "label": f.get("label") or fname,
                            "type": f.get("type") or "text",
                            "help": f.get("help") or "",
                            "placeholder": f.get("placeholder") or "",
                            "required": bool(f.get("required", True)),
                        }
                target_method = final_outcome.method_id
                if target_method:
                    yield _patch_delta({
                        "form_id": form_id,
                        "subtitle": reason,
                        "status_text": None,
                        "_is_probing": False,
                        "form_error": None,
                        "methods": {target_method: {"fields": extra}},
                    })
                else:
                    yield _patch_delta({
                        "form_id": form_id,
                        "subtitle": reason,
                        "status_text": None,
                        "_is_probing": False,
                        "form_error": None,
                        "fields": extra,
                    })
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "needs_input"},
                })
            else:
                err = final_outcome.error or "Connection failed."
                hint = final_outcome.follow_up or "Update the form and try again."
                yield _delta(f"\n\n{err} {hint}\n")
                yield _patch_delta({
                    "form_id": form_id,
                    "subtitle": hint,
                    "status_text": None,
                    "_is_probing": False,
                    "form_error": err,
                    "_is_success": False,
                })
                yield _push("response.completed", {
                    "type": "response.completed",
                    "response": {"id": response_id, "status": "retry"},
                })

            self._save_assistant_turn(db_conversation_id, "".join(body_parts), recorded_events)

        finally:
            if _temp_workspace_dir:
                shutil.rmtree(_temp_workspace_dir, ignore_errors=True)

    @staticmethod
    def _build_llm_client():
        from cowork.services.providers import build_llm_client
        return build_llm_client()

    def _save_assistant_turn(
        self,
        conversation_id: UUID | None,
        text: str,
        events: list[dict],
    ) -> None:
        if not conversation_id or not text:
            return
        ConversationService(self.session).save_assistant_turn(conversation_id, text, events)
