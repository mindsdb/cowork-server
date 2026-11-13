"""The routes the owner uses to see channel senders, approve or block them,
hand out pairing codes and set what a binding's turns may use."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_open_session
from cowork.models.channel import (
    SENDER_ALLOWED,
    SENDER_BLOCKED,
    SENDER_PENDING,
    ChannelBinding,
    ChannelPairingCode,
    ChannelSender,
)
from cowork.server import create_app
from cowork.services.channel_pairing import CODE_ALPHABET, CODE_LENGTH, UNUSED_CODES_MAX
from cowork.services.channel_senders import PENDING_TTL

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"
A = {"X-User-Id": USER_A, "X-Organization-Id": ORG_A}
A_ADMIN = {**A, "X-User-Roles": "manage-organization"}
B_ADMIN = {"X-User-Id": USER_B, "X-Organization-Id": ORG_B, "X-User-Roles": "manage-organization"}
SWITCHES = ("allow_scratchpad", "allow_connectors", "allow_memory_writes", "allow_file_replies")


@pytest.fixture(autouse=True)
def no_leftover_rows():
    """Senders, codes and bindings made by a test are removed after it."""
    yield
    with get_open_session() as session:
        session.exec(delete(ChannelSender))
        session.exec(delete(ChannelPairingCode))
        session.exec(delete(ChannelBinding).where(ChannelBinding.external_group_id.like("api-%")))
        session.commit()


@pytest.fixture(scope="module")
def local_client():
    # No `with`: skipping the lifespan skips boot migrations and background workers.
    client = TestClient(create_app())
    yield client
    client.close()


@pytest.fixture(scope="module")
def org_client():
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        client = TestClient(create_app())
        try:
            yield client
        finally:
            client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


def _sender(status: str = SENDER_PENDING, *, org_id: str | None = None, created_at: datetime | None = None,
            sender_id: str = "4242", display_name: str | None = "Stranger") -> uuid.UUID:
    with get_open_session() as session:
        row = ChannelSender(
            channel_type="telegram", external_group_id=f"api-{uuid.uuid4().hex[:8]}", external_sender_id=sender_id,
            status=status, display_name=display_name, org_id=org_id,
            created_at=created_at or datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()
        return row.id


def _status_of(row_id: uuid.UUID) -> str | None:
    with get_open_session() as session:
        row = session.get(ChannelSender, row_id)
        return row.status if row is not None else None


def _binding(*, org_id: str | None = None) -> uuid.UUID:
    with get_open_session() as session:
        row = ChannelBinding(
            channel_type="telegram", external_group_id=f"api-{uuid.uuid4().hex[:8]}", org_id=org_id,
        )
        session.add(row)
        session.commit()
        return row.id


def test_pending_requests_are_listed_and_expired_ones_are_not(local_client):
    fresh = _sender()
    _sender(created_at=datetime.now(timezone.utc) - PENDING_TTL - timedelta(hours=1), sender_id="old")
    _sender(SENDER_ALLOWED, sender_id="friend")

    res = local_client.get("/api/v1/channels/senders", params={"status": "pending"})

    assert res.status_code == 200
    assert [(row["id"], row["display_name"], row["status"]) for row in res.json()] == [
        (str(fresh), "Stranger", "pending"),
    ]


def test_approve_allows_and_block_blocks_the_sender(local_client):
    row_id = _sender()

    approved = local_client.post(f"/api/v1/channels/senders/{row_id}/approve")
    assert (approved.status_code, approved.json()["status"]) == (200, "allowed")
    assert _status_of(row_id) == SENDER_ALLOWED

    blocked = local_client.post(f"/api/v1/channels/senders/{row_id}/block")
    assert (blocked.status_code, blocked.json()["status"]) == (200, "blocked")
    assert _status_of(row_id) == SENDER_BLOCKED


def test_delete_forgets_the_sender_and_unknown_ids_are_404(local_client):
    row_id = _sender()

    assert local_client.delete(f"/api/v1/channels/senders/{row_id}").status_code == 204
    assert _status_of(row_id) is None
    assert local_client.delete(f"/api/v1/channels/senders/{row_id}").status_code == 404
    assert local_client.post(f"/api/v1/channels/senders/{uuid.uuid4()}/approve").status_code == 404


def test_the_telegram_linked_channel_account_cannot_be_approved(local_client):
    row_id = _sender(sender_id="777000")

    res = local_client.post(f"/api/v1/channels/senders/{row_id}/approve")

    assert res.status_code == 400
    assert _status_of(row_id) == SENDER_PENDING


def test_a_pairing_code_is_returned_once_and_capped(local_client):
    res = local_client.post("/api/v1/channels/telegram/pairing-codes")

    assert res.status_code == 201
    body = res.json()
    assert re.fullmatch(rf"[{CODE_ALPHABET}]{{{CODE_LENGTH}}}", body["code"])
    assert datetime.fromisoformat(body["expires_at"]) > datetime.now(timezone.utc)
    for _ in range(UNUSED_CODES_MAX - 1):
        assert local_client.post("/api/v1/channels/telegram/pairing-codes").status_code == 201
    assert local_client.post("/api/v1/channels/telegram/pairing-codes").status_code == 409
    assert local_client.post("/api/v1/channels/nosuch/pairing-codes").status_code == 404


def test_patch_sets_a_bindings_switches_and_returns_them(local_client):
    binding_id = _binding()

    before = local_client.patch(f"/api/v1/channels/bindings/{binding_id}", json={})
    assert {k: before.json()[k] for k in SWITCHES} == dict.fromkeys(SWITCHES, False)

    res = local_client.patch(
        f"/api/v1/channels/bindings/{binding_id}", json={"allow_scratchpad": True, "allow_file_replies": True},
    )

    assert res.status_code == 200
    assert {k: res.json()[k] for k in SWITCHES} == {
        "allow_scratchpad": True, "allow_connectors": False, "allow_memory_writes": False, "allow_file_replies": True,
    }


def test_binding_create_does_not_take_the_switches(local_client):
    res = local_client.post(
        "/api/v1/channels/bindings",
        json={"channel_type": "telegram", "external_group_id": f"api-{uuid.uuid4().hex[:8]}", "allow_scratchpad": True},
    )

    assert res.status_code == 201
    assert res.json()["allow_scratchpad"] is False


def test_disconnecting_a_channel_drops_its_senders_and_codes(local_client):
    assert local_client.put("/api/v1/channels/telegram/config", json={"values": {"bot_token": "T:tok"}}).status_code == 200
    row_id = _sender(SENDER_ALLOWED)
    assert local_client.post("/api/v1/channels/telegram/pairing-codes").status_code == 201

    assert local_client.delete("/api/v1/channels/telegram/config").status_code == 204

    assert _status_of(row_id) is None
    with get_open_session() as session:
        assert session.exec(select(ChannelPairingCode)).all() == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/channels/senders"),
        ("post", "/api/v1/channels/senders/{id}/approve"),
        ("post", "/api/v1/channels/senders/{id}/block"),
        ("delete", "/api/v1/channels/senders/{id}"),
        ("post", "/api/v1/channels/slack/pairing-codes"),
    ],
)
def test_sender_routes_need_an_org_admin(org_client, method, path):
    row_id = _sender(org_id=ORG_A)

    res = getattr(org_client, method)(path.format(id=row_id), headers=A)

    assert res.status_code == 403
    assert _status_of(row_id) == SENDER_PENDING


def test_setting_a_switch_needs_an_org_admin_but_other_edits_do_not(org_client):
    binding_id = _binding(org_id=ORG_A)

    refused = org_client.patch(f"/api/v1/channels/bindings/{binding_id}", json={"allow_connectors": True}, headers=A)
    renamed = org_client.patch(f"/api/v1/channels/bindings/{binding_id}", json={"display_name": "team"}, headers=A)
    allowed = org_client.patch(
        f"/api/v1/channels/bindings/{binding_id}", json={"allow_connectors": True}, headers=A_ADMIN,
    )

    assert refused.status_code == 403
    assert renamed.status_code == 200
    assert (allowed.status_code, allowed.json()["allow_connectors"]) == (200, True)


def test_an_org_admin_sees_and_acts_on_only_their_orgs_senders(org_client):
    mine = _sender(org_id=ORG_A)
    theirs = _sender(org_id=ORG_B)

    listed = org_client.get("/api/v1/channels/senders", headers=A_ADMIN)
    approve_theirs = org_client.post(f"/api/v1/channels/senders/{theirs}/approve", headers=A_ADMIN)
    approve_mine = org_client.post(f"/api/v1/channels/senders/{mine}/approve", headers=A_ADMIN)

    assert [row["id"] for row in listed.json()] == [str(mine)]
    assert approve_theirs.status_code == 404
    assert _status_of(theirs) == SENDER_PENDING
    assert approve_mine.status_code == 200
    assert [row["id"] for row in org_client.get("/api/v1/channels/senders", headers=B_ADMIN).json()] == [str(theirs)]
