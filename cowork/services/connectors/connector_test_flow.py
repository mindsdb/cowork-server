"""The steps of a custom connector's connection test that wait on the member.

The probe stream starts a test; the allow-private and confirm routes continue
it. All three go through here, so a test continues and saves the same way
whichever one runs it.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

from cowork.db.scoped import TenantScope
from cowork.db.units import run_db
from cowork.services.connectors.connector_test_run import run_test_request
from cowork.services.connectors.custom_connectors import CustomConnectorService, UnstorableFormError
from cowork.services.connectors.pending_tests import PendingTest, pending_tests
from cowork.services.connectors.persist import persist_connection, vault_for_scope

logger = logging.getLogger(__name__)

WAITING_FOR_APPROVAL = "Waiting for you to allow a private address."
WAITING_FOR_CONFIRMATION = "Waiting for you to confirm the test result."


async def advance(entry: PendingTest) -> dict[str, Any]:
    """Run ``entry``'s test with the approvals it has, and park it for the member.

    Returns:
        ``{"private_host_request": {...}}`` with a new ``approval_id`` when a
        private host must be allowed first, else ``{"preview": {...},
        "confirm_id": ...}``.

    Raises:
        ConnectorTestError: when the test fails or its answer is not data.
    """
    outcome = await run_test_request(
        entry.test_request, entry.test_values, secret_names=entry.secret_names, approved=entry.approved,
    )
    if outcome.private_host is not None:
        host = outcome.private_host
        parked = dataclasses.replace(
            entry, kind="approval", asking=(host.key, host.addresses), approved=dict(entry.approved),
            created_at=time.monotonic(),
        )
        approval_id = pending_tests.put(parked)
        return {"private_host_request": {"approval_id": approval_id, **host.to_dict()}}
    parked = dataclasses.replace(
        entry, kind="confirm", asking=None, approved=dict(entry.approved), created_at=time.monotonic(),
    )
    confirm_id = pending_tests.put(parked)
    return {"preview": outcome.preview.to_dict(), "confirm_id": confirm_id}


async def save_definition(scope: TenantScope, connector_id: str, form_spec: dict) -> str | None:
    """Store a tested handcrafted form as the scope's custom connector.

    Local installs only for now: who may publish a connector to a whole
    organization is not decided yet.

    Returns:
        None when saved or skipped, else a line telling the user the
        connection is saved but the connector could not be.
    """
    if scope.org_mode:
        return None
    try:
        await run_db(
            lambda session: CustomConnectorService(session).upsert_from_form(connector_id, form_spec),
            scope=scope,
        )
    except UnstorableFormError as e:
        return f"\nThe connection is saved, but it was not added to your connectors list: {e}\n"
    except Exception:
        logger.exception("Could not save the custom connector definition for %s", connector_id)
        return "\nThe connection is saved, but it could not be added to your connectors list.\n"
    return None


async def save_confirmed(entry: PendingTest) -> tuple[str, str | None, str | None]:
    """Save the connection the member confirmed, and its definition when handcrafted.

    Returns:
        The saved slug, its user label, and a note when the definition could
        not be kept.
    """
    vault = vault_for_scope(entry.scope)
    slug = entry.grant_slug or persist_connection(
        entry.connector_id, entry.method, entry.extends_name or entry.name, entry.credentials,
        label=entry.label, user_label=entry.user_label, extends=bool(entry.extends_name),
        spec_form=entry.form_spec, vault=vault, scope=entry.scope,
    )
    record = vault.read_record(entry.connector_id, slug) or {}
    user_label = str(record.get("fields", {}).get("_user_label", "")).strip() or None
    note = None
    if entry.handcrafted and not entry.extends_name:
        note = await save_definition(entry.scope, entry.connector_id, entry.form_spec)
    return slug, user_label, note
