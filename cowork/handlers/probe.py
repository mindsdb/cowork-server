from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.code_sandbox import build_cell_sandbox, code_sandbox_on, probe_workspaces_dir
from cowork.common.logger import log_context
from cowork.common.settings.user_settings import UserSettings
from cowork.handlers.turn_errors import server_busy_message
from cowork.schemas.connectors import ConnectorSpecResponse
from cowork.schemas.responses import Role
from cowork.services.connectors.catalog import ConnectorCatalog
from cowork.services.connectors.connector_test_flow import (
    WAITING_FOR_APPROVAL,
    WAITING_FOR_CONFIRMATION,
    advance,
    save_definition,
)
from cowork.services.connectors.connector_test_run import ConnectorTestError, placeholder_names, request_endpoints
from cowork.services.connectors.custom_connectors import (
    NEEDS_REBUILD_FOR_MEMBER,
    USAGE_NOTES_MAX,
    test_request_for,
)
from cowork.services.connectors.identity import spec_secret_fields
from cowork.services.connectors.pending_tests import PendingTest
from cowork.services.connectors.persist import persist_connection, vault_for_scope
from cowork.services.connectors.probe import CredentialProbe, ProbeOutcome
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import store
from cowork.db.scoped import (
    LOCAL_SCOPE,
    ScopedSession,
    TenantScope,
    tenant_scope_required,
    unsafe_unscoped_session,
)
from cowork.db.units import busy_retry_seconds, conversation_writes, run_db
from cowork.services.conversations import ConversationService
from cowork.services.settings import SettingService

logger = logging.getLogger(__name__)


def _extract_connection_label_fields(credentials: dict) -> tuple[str, str]:
    """Pop `label`/`_label` and `user_label`/`_user_label` out of `credentials`
    in place, returning (label, user_label). Both are non-credential display
    fields that must never reach the probe or the saved credential set.

    Both pops on each line are unconditional (not `a or b`, which would
    short-circuit and skip the second `.pop()` whenever the first key is
    present and truthy — leaving the other key sitting in `credentials`,
    where it would be saved as an ordinary field instead of being merged into
    the single display value it actually represents).
    """
    raw_label = credentials.pop("label", "")
    raw_label_underscore = credentials.pop("_label", "")
    label = str(raw_label or raw_label_underscore).strip()

    raw_user_label = credentials.pop("user_label", "")
    raw_user_label_underscore = credentials.pop("_user_label", "")
    user_label = str(raw_user_label or raw_user_label_underscore).strip()

    return label, user_label


@dataclass(frozen=True)
class _ProbeConversation:
    """The conversation a probe's turn is saved into, and the project
    directory its probe runs in."""

    id: UUID
    workspace_path: Path


def _read_conversation(session: ScopedSession, *, conversation_id: UUID) -> _ProbeConversation:
    conversation = ConversationService(session).get_conversation(conversation_id)
    return _ProbeConversation(id=conversation.id, workspace_path=Path(conversation.project.path))


def _read_probe_settings(session: ScopedSession) -> UserSettings:
    """The settings the probe's model client is built from. They are read for
    LOCAL_SCOPE, the install's global rows, which is what get_user_settings()
    resolves when no tenant is bound for settings, as none is here.
    SettingService routes each key to its row itself, so it reads through the
    raw session."""
    return SettingService(unsafe_unscoped_session(session), LOCAL_SCOPE).load()


def _save_probe_turn(
    session: ScopedSession, *, conversation_id: UUID, text: str, events: tuple[dict, ...],
) -> UUID | None:
    """The probe's narrative as an assistant turn: the saved row's id, or
    None when nothing was saved."""
    message = ConversationService(session).save_assistant_turn(conversation_id, text, list(events))
    return message.id if message is not None else None


def _is_oauth_grant(form_spec: dict, method: str | None, credentials: dict) -> bool:
    """True when this submission is the result of an OAuth browser grant.

    The form is model-written, so a flag alone is not enough: the selected
    method must be an OAuth launch with an ``oauth`` block (the only shape the
    renderer runs a grant for), and the values must carry a token from it.
    """
    if not (credentials.get("access_token") or credentials.get("refresh_token")):
        return False
    for m in form_spec.get("methods") or []:
        if (
            isinstance(m, dict) and m.get("id") == method
            and m.get("submit_action") == "oauth_launch" and isinstance(m.get("oauth"), dict)
        ):
            return True
    return False


_NO_TEST_REQUEST = "This form has no test request, so its connection cannot be tested."


def _method_fields(form_spec: dict, method: str | None) -> list:
    """The fields a submission of ``method`` fills: the form's own, or that method's."""
    methods = [m for m in form_spec.get("methods") or [] if isinstance(m, dict)]
    if not methods:
        return list(form_spec.get("fields") or [])
    chosen = next((m for m in methods if m.get("id") == method), None)
    return list((chosen or {}).get("fields") or [])


def _form_usage_notes(form_spec: dict) -> str | None:
    """A handcrafted form's own ``connector.usage_notes``, for its probe.

    Returns the stripped text, or None when the form has none or they exceed
    USAGE_NOTES_MAX. The cap is checked again here rather than trusting that
    every route staging a form ran check_handcrafted_form.
    """
    meta = form_spec.get("connector")
    notes = meta.get("usage_notes") if isinstance(meta, dict) else None
    if not isinstance(notes, str) or len(notes) > USAGE_NOTES_MAX:
        return None
    return notes.strip() or None


_GRANT_UNLISTED = (
    "\n\nThe connection is saved, but it was not added to your connectors list "
    "because its connection test did not pass: {reason}\n"
)


def _grant_saved_patch(form_id: str, slug: str) -> dict:
    """The form patch for a connection saved from a provider's grant."""
    return {
        "form_id": form_id,
        "title": f"Saved — {slug}",
        "subtitle": "Stored in the vault after the provider's authorization.",
        "status_text": None,
        "_is_probing": False,
        "_is_success": True,
        "actions": [{"id": "dismiss", "label": "Close", "kind": "cancel"}],
    }


class ProbeHandler:
    """The connector form's stream: an LLM-driven credential probe, narrated
    as an assistant turn.

    Its database work runs as units (cowork.db.units): the conversation read
    and the settings read before the probe, and the assistant-turn write after
    it. No connection is held while the probe waits on the model, and none is
    checked out on the event loop.
    """

    def __init__(self, *, scope: TenantScope) -> None:
        self.scope = scope

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
        persisted = False

        def _sse(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        def _push(event_type: str, data: dict) -> str:
            nonlocal seq
            seq += 1
            payload = {**data, "sequence_number": seq}
            recorded_events.append(payload)
            return _sse(event_type, payload)

        async def _persist_once(conversation_id: UUID | None) -> UUID | None:
            # Idempotent — every response.completed below calls this so the
            # turn's real assistant message id can ride the SAME
            # frame the client's SSE reader stops at; a probe turn legitimately
            # persists nothing when there's no conversation or no body text.
            nonlocal persisted
            if persisted:
                return None
            persisted = True
            if not conversation_id or not "".join(body_parts):
                return None
            save = partial(
                _save_probe_turn,
                conversation_id=conversation_id,
                text="".join(body_parts),
                events=tuple(recorded_events),
            )
            try:
                async with conversation_writes(conversation_id):
                    return await run_db(save, scope=self.scope)
            except PoolTimeoutError:
                # The probe's outcome (a saved connection, or the reason it
                # failed) still reaches the form in the frame below; only its
                # narrative is missing from the conversation.
                logger.warning(
                    "No database connection freed in time to save the probe turn for conversation %s",
                    conversation_id,
                )
                return None
            except Exception:
                # Same outcome as a refused save: the form keeps the probe's
                # result, and only its narrative is missing from the
                # conversation.
                logger.exception("Could not save the probe turn", extra=log_context(conversation_id=conversation_id))
                return None

        async def _completed(conversation_id: UUID | None, response_fields: dict) -> str:
            # _push records the (id-less) event and bumps seq first, exactly
            # like every other event — persistence must see this completion
            # event in recorded_events, matching the ordering this replaced,
            # where _save_assistant_turn always ran after this push.
            wire = _push(
                "response.completed",
                {"type": "response.completed", "response": {"id": response_id, **response_fields}},
            )
            message_id = await _persist_once(conversation_id)
            if message_id is None:
                return wire
            # Same recorded payload, republished with the id added — the
            # persisted copy in recorded_events is deliberately left without
            # it, matching how response_failed_payload's persisted copy
            # doesn't carry it either.
            wire_payload = {**recorded_events[-1], "assistant_message_id": str(message_id)}
            return _sse("response.completed", wire_payload)

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
            conversation_busy = False

            if conversation_id:
                try:
                    conversation = await run_db(
                        partial(_read_conversation, conversation_id=UUID(conversation_id)),
                        scope=self.scope,
                    )
                    db_conversation_id = conversation.id
                    workspace_path = conversation.workspace_path
                except PoolTimeoutError:
                    conversation_busy = True
                    logger.warning(
                        "No database connection freed in time to read conversation %s; "
                        "the form is told Cowork is busy", conversation_id,
                    )
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
                yield await _completed(db_conversation_id, {"status": "failed"})
                return

            values = submission.get("values", {}) or {}
            skipped = submission.get("skipped", []) or []
            credentials = {
                k: v for k, v in values.items()
                if k not in skipped and v is not None and v != ""
            }
            # `label`/`user_label` are human display fields, not credentials —
            # pull them out so they never reach the probe, and hand them to
            # persist as the label/user_label.
            connection_label, connection_user_label = _extract_connection_label_fields(credentials)

            # Mirrors submit_form: a handcrafted form keeps its own spec, and a
            # custom connector's stored spec arrives staged with the submission.
            # Only an explicit False from submit_form skips the stored spec, so
            # a submission can never be validated as one connector and saved untested.
            stamped = submission.get("checked_against_stored_spec") is not False
            spec = registry.get_connector(connector_id) if stamped else None
            if spec is None and submission.get("custom_spec"):
                spec = ConnectorSpecResponse.model_validate(submission["custom_spec"])
            if spec is not None:
                form_id = spec.form.form_id
                form_spec = spec.form.model_dump()
            else:
                form_spec = dict(submission.get("form_spec") or {})
                form_id = form_spec.get("form_id") or f"{connector_id}-connector"
            if method:
                form_spec["selected_method"] = method

            if conversation_busy:
                # Without its conversation the probe would run in a temp
                # workspace and its turn would never be saved, so the form
                # says Cowork is busy, with the wait, as a refused settings
                # read does.
                busy = server_busy_message(busy_retry_seconds())
                yield _delta(busy)
                yield _patch_delta({"form_id": form_id, "form_error": busy})
                yield await _completed(None, {"status": "failed"})
                return

            extends_name = form_spec.get("_extends_connection") if spec is None else None
            # A handcrafted OAuth method arrives after the browser grant. Its
            # connection is saved first, so a probe failure never throws a
            # working grant away; the token is only the client's word, though,
            # so the shared definition waits for the probe below to pass.
            grant_slug: str | None = None
            grant_user_label: str | None = None
            if spec is None and _is_oauth_grant(form_spec, method, credentials):
                try:
                    vault = vault_for_scope(self.scope)
                    slug = persist_connection(
                        connector_id, method, extends_name or name, credentials,
                        label=connection_label, user_label=connection_user_label,
                        extends=bool(extends_name), spec_form=form_spec, vault=vault,
                        scope=self.scope,
                    )
                    saved_record = vault.read_record(connector_id, slug) or {}
                    saved_user_label = str(saved_record.get("fields", {}).get("_user_label", "")).strip() or None
                except Exception as exc:
                    yield _delta(f"Could not save: `{exc}`.")
                    yield await _completed(db_conversation_id, {"status": "failed"})
                    return
                yield _delta(f"Saved as `{slug}` (authorized with the provider).\n\n")
                if extends_name:
                    yield _patch_delta(_grant_saved_patch(form_id, slug))
                    yield await _completed(db_conversation_id, {"status": "success", "user_label": saved_user_label})
                    return
                grant_slug, grant_user_label = slug, saved_user_label

            # A follow-up step is probed with the fields of the connection it
            # extends too; the save below still merges them once, under the vault lock.
            probe_credentials = credentials
            saved_only_fields: set[str] = set()
            unpatchable_names: set[str] = set()
            if extends_name:
                try:
                    target = vault_for_scope(self.scope).read_record(connector_id, extends_name)
                except Exception as exc:
                    logger.warning("Could not read connection %s to extend: %s", extends_name, exc)
                    unreadable = f"Could not read `{extends_name}`: `{exc}`."
                    yield _delta(unreadable)
                    yield _patch_delta({"form_id": form_id, "form_error": unreadable})
                    yield await _completed(db_conversation_id, {"status": "failed"})
                    return
                if target is None:
                    missing = f"No {connector_id} connection named {extends_name!r} to add these fields to."
                    yield _delta(missing)
                    yield _patch_delta({"form_id": form_id, "form_error": missing})
                    yield await _completed(db_conversation_id, {"status": "failed"})
                    return
                stored = {k: v for k, v in target.get("fields", {}).items() if not k.startswith("_")}
                probe_credentials = {**stored, **credentials}
                step_field_names: set[str] = set()
                field_lists = [form_spec.get("fields") or []]
                field_lists += [m.get("fields") or [] for m in form_spec.get("methods") or [] if isinstance(m, dict)]
                for fields in field_lists:
                    step_field_names.update(f.get("name") for f in fields if isinstance(f, dict))
                saved_only_fields = set(stored) - step_field_names
                # The probe reads these as DS_<NAME> in its .env, so it may echo either form.
                for saved_name in saved_only_fields:
                    unpatchable_names.update({saved_name.lower(), f"ds_{saved_name.lower()}"})

            # A custom connector is tested by its own test request, which the
            # member confirms, never by the model probe. Self-hosted and org
            # mode keep the model probe until their own milestone.
            if registry.get_connector(connector_id) is None and not tenant_scope_required():
                # The staged copy goes now; the values wait only in the pending test.
                store.consume(submission_id)
                test_request = test_request_for(form_spec, method)
                # Every value the test sends from the saved record, declared in
                # this step's form or not: a blank field of the same name still
                # sends the saved value.
                from_saved = set(probe_credentials) - set(credentials)
                refusal = (
                    (NEEDS_REBUILD_FOR_MEMBER if spec is not None else _NO_TEST_REQUEST)
                    if test_request is None
                    else await self._follow_up_refusal(connector_id, test_request, form_spec, method, from_saved)
                )
                if refusal is not None:
                    yield _delta(refusal)
                    yield _patch_delta({"form_id": form_id, "_is_probing": False, "form_error": refusal})
                    yield await _completed(db_conversation_id, {"status": "failed"})
                    return
                entry = PendingTest(
                    kind="confirm", scope=self.scope, connector_id=connector_id, method=method, name=name,
                    conversation_id=conversation_id, form_id=form_id, form_spec=form_spec,
                    test_request=test_request, test_values=probe_credentials, credentials=credentials,
                    secret_names=sorted(set(spec_secret_fields(connector_id, method, form_spec)) | from_saved),
                    label=connection_label, user_label=connection_user_label, extends_name=extends_name,
                    grant_slug=grant_slug, handcrafted=spec is None,
                )
                yield _delta(f"Testing the connection to **{connector_id}**…\n\n")
                yield _patch_delta({
                    "form_id": form_id, "_is_probing": True, "status_text": "Testing the connection…", "form_error": None,
                })
                try:
                    result = await advance(entry)
                except ConnectorTestError as exc:
                    reason = exc.reason
                    if grant_slug:
                        yield _delta(_GRANT_UNLISTED.format(reason=reason))
                        yield _patch_delta(_grant_saved_patch(form_id, grant_slug))
                        yield await _completed(db_conversation_id, {"status": "success", "user_label": grant_user_label})
                        return
                    yield _delta(f"\n\n{reason} Update the form and try again.\n")
                    yield _patch_delta({
                        "form_id": form_id, "subtitle": "Update the form and try again.", "status_text": None,
                        "_is_probing": False, "form_error": reason, "_is_success": False,
                    })
                    yield await _completed(db_conversation_id, {"status": "retry"})
                    return
                # The id and the preview go only to this stream, never into the
                # saved turn, so no later turn of the agent can answer for the member.
                if "private_host_request" in result:
                    yield _sse("connector.private_host_request", {
                        "type": "connector.private_host_request", "form_id": form_id, **result["private_host_request"],
                    })
                    yield _delta(WAITING_FOR_APPROVAL)
                    yield _patch_delta({
                        "form_id": form_id, "_awaiting_private_approval": True, "_is_probing": False,
                        "status_text": WAITING_FOR_APPROVAL,
                    })
                    yield await _completed(db_conversation_id, {"status": "awaiting_private_approval"})
                    return
                yield _sse("connector.test_preview", {
                    "type": "connector.test_preview", "form_id": form_id,
                    "confirm_id": result["confirm_id"], "preview": result["preview"],
                })
                yield _delta(WAITING_FOR_CONFIRMATION)
                yield _patch_delta({
                    "form_id": form_id, "_awaiting_confirmation": True, "_is_probing": False,
                    "status_text": WAITING_FOR_CONFIRMATION,
                })
                yield await _completed(db_conversation_id, {"status": "awaiting_confirmation"})
                return

            # Probe path: build workspace + LLM client
            cell_sandbox = None
            try:
                if workspace_path is None:
                    # Under the code sandbox, in the asker's own sandbox store:
                    # the shared /tmp is denied to the cell, and every other
                    # person's cell could read a folder there.
                    _temp_workspace_dir = tempfile.mkdtemp(
                        prefix="cowork-probe-",
                        dir=probe_workspaces_dir(self.scope) if code_sandbox_on() else None,
                    )
                    workspace_path = Path(_temp_workspace_dir)
                from anton.workspace import Workspace
                workspace = Workspace(workspace_path)
                # None with the sandbox off.
                cell_sandbox = build_cell_sandbox(scope=self.scope, project_dir=workspace_path)
            except Exception:
                workspace = None
                logger.exception("Could not build workspace for probe")

            llm_client = None
            init_error = "Could not initialize the probe (workspace or LLM client unavailable)."
            settings_busy = False
            try:
                from cowork.services.providers import web_tool_kwargs_for

                # Capture the client and its web policy together, before the
                # yields below let a settings update run ahead of the probe.
                settings = await run_db(_read_probe_settings, scope=self.scope)
                web_tools = web_tool_kwargs_for(settings.resolved_planning_provider)
                llm_client = self._build_llm_client(settings=settings)
            except PoolTimeoutError:
                # The form says so, with the wait, so the user submits it
                # again rather than checking a model key that is fine.
                init_error = server_busy_message(busy_retry_seconds())
                settings_busy = True
                logger.warning("No database connection freed in time to read the probe's settings")
            except Exception:
                logger.exception("Could not build LLM client for probe")

            # Workspace / LLM client availability
            if (workspace is None or llm_client is None) and grant_slug:
                yield _delta(_GRANT_UNLISTED.format(reason=init_error))
                yield _patch_delta(_grant_saved_patch(form_id, grant_slug))
                yield await _completed(
                    None if settings_busy else db_conversation_id,
                    {"status": "success", "user_label": grant_user_label},
                )
                return
            if workspace is None or llm_client is None:
                yield _delta(init_error)
                yield _patch_delta({"form_id": form_id, "form_error": init_error})
                # A refused settings read ends here without saving, as a
                # refused conversation read does: the save would wait for
                # another unit, and the busy text is not part of the turn.
                yield await _completed(None if settings_busy else db_conversation_id, {"status": "failed"})
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

            usage_notes, notes_define_pass = await self._usage_notes(connector_id)
            if spec is None and not usage_notes:
                # The submitter wrote this form and its probe uses only their own
                # credentials, so its notes are hints in every scope, never the rule.
                usage_notes, notes_define_pass = _form_usage_notes(form_spec), False
            probe = CredentialProbe(
                engine=connector_id,
                credentials=probe_credentials,
                llm_client=llm_client,
                workspace=workspace,
                form_spec=form_spec,
                skipped=skipped,
                usage_notes=usage_notes,
                notes_define_pass=notes_define_pass,
                web_tool_kwargs=web_tools,
                saved_fields=sorted(saved_only_fields),
                cell_sandbox=cell_sandbox,
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
                        # A patch for a field the form doesn't hold would add it,
                        # untyped, so a saved secret typed there would show.
                        if field_name and field_name.lower() not in unpatchable_names:
                            status_val = (payload or {}).get("status")
                            mid = (payload or {}).get("method_id")
                            yield _patch_delta(_field_patch(form_id, mid, field_name, {"status": status_val}))
                    elif kind == "remove_field":
                        field_name = (payload or {}).get("name")
                        if field_name and field_name.lower() not in unpatchable_names:
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
            saved_user_label: str | None = None
            if grant_slug and final_outcome.status != "success":
                reason = final_outcome.error or final_outcome.follow_up or "its test did not pass"
                yield _delta(_GRANT_UNLISTED.format(reason=reason))
                yield _patch_delta(_grant_saved_patch(form_id, grant_slug))
                yield await _completed(db_conversation_id, {"status": "success", "user_label": grant_user_label})
                return
            if grant_slug:
                saved_slug, saved_user_label = grant_slug, grant_user_label
            elif final_outcome.status == "success":
                # A connector the static registry lacks has only this form to
                # say which fields are secret.
                builtin = registry.get_connector(connector_id) is not None
                try:
                    vault = vault_for_scope(self.scope)
                    slug = persist_connection(
                        connector_id, method, extends_name or name, credentials,
                        label=connection_label, user_label=connection_user_label,
                        extends=bool(extends_name), spec_form=None if builtin else form_spec,
                        vault=vault, scope=self.scope,
                    )
                    saved_slug = slug
                    saved_record = vault.read_record(connector_id, slug) or {}
                    saved_user_label = str(saved_record.get("fields", {}).get("_user_label", "")).strip() or None
                except Exception as exc:
                    logger.exception("Vault save failed despite probe success")
                    final_outcome.status = "failure"
                    final_outcome.error = f"Probe succeeded but save failed: {exc}"

            if final_outcome.status == "success":
                summary = final_outcome.summary or "Connection works."
                yield _delta(f"\n\n{summary}\n\nSaved as `{saved_slug}`.\n")
                if spec is None and not extends_name:
                    note = await self._save_definition(connector_id, form_spec)
                    if note:
                        yield _delta(note)
                yield _patch_delta({
                    "form_id": form_id,
                    "title": f"Connected — {saved_slug}",
                    "subtitle": summary,
                    "status_text": None,
                    "form_error": None,
                    "_is_probing": False,
                    "_is_success": True,
                })
                yield await _completed(db_conversation_id, {"status": "success", "user_label": saved_user_label})
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
                yield await _completed(db_conversation_id, {"status": "needs_input"})
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
                yield await _completed(db_conversation_id, {"status": "retry"})

        finally:
            if _temp_workspace_dir:
                shutil.rmtree(_temp_workspace_dir, ignore_errors=True)

    async def _save_definition(self, connector_id: str, form_spec: dict) -> str | None:
        """Store a handcrafted form whose model probe passed as the scope's
        custom connector (``connector_test_flow.save_definition``).

        Returns:
            None when saved or skipped, else a line telling the user the
            connection is saved but the connector could not be.
        """
        return await save_definition(self.scope, connector_id, form_spec)

    async def _follow_up_refusal(
        self, connector_id: str, test_request: dict, form_spec: dict, method: str | None, saved_fields: set[str],
    ) -> str | None:
        """Why a follow-up step's test request may not run, or None when it may.

        It may name only its own fields, the saved connection's and the token
        names, and it may send a saved field only to the scheme, host and port the
        stored connector's own test requests use, so a new step cannot carry a saved
        credential somewhere else.
        """
        used = {n for step in (test_request.get("token"), test_request.get("data")) if step for n in placeholder_names(step)}
        own = {f.get("name") for f in _method_fields(form_spec, method) if isinstance(f, dict)}
        unknown = used - own - saved_fields - {"token", "access_token", "refresh_token"}
        if unknown:
            return f"The test request names {', '.join(sorted(unknown))}, which this connection does not have."
        if not used & saved_fields:
            return None
        stored = await run_db(lambda session: ConnectorCatalog(session).get_connector(connector_id), scope=self.scope)
        allowed: set[tuple[str, str, int]] = set()
        if stored is not None and stored.custom:
            form = stored.form.model_dump()
            for holder in [form, *(form.get("methods") or [])]:
                if isinstance(holder.get("test_request"), dict):
                    allowed |= request_endpoints(holder["test_request"])
        if not request_endpoints(test_request) <= allowed:
            return "A follow-up step may send saved fields only to the hosts, ports and schemes this connector was tested with."
        return None

    async def _usage_notes(self, connector_id: str) -> tuple[str | None, bool]:
        """The connector's agent-facing notes from the catalog, or None when it
        has none or they could not be read; a handcrafted form's own notes may
        then stand in (``_form_usage_notes``). The flag is True
        only for a built-in spec's notes, the ones reviewed like code, which
        may tell the probe what counts as a pass."""
        try:
            notes = await run_db(
                lambda session: ConnectorCatalog(session).usage_notes_for([connector_id]), scope=self.scope,
            )
        except Exception:
            logger.exception("Could not read the catalog usage notes for %s", connector_id)
            return None, False
        text = notes.get(connector_id)
        return text, text is not None and text == registry.usage_notes_for([connector_id]).get(connector_id)

    @staticmethod
    def _build_llm_client(settings=None):
        from cowork.services.providers import build_llm_client
        return build_llm_client(settings=settings)
