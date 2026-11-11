"""The member answers a waiting custom connector test: allow a private host,
confirm the result, or cancel.

Each answer goes through a single-use id that only the member's form holds.
These run the real probe stream and the real routes, and record the stream and
answer shapes the renderer's tests load
(``tests/fixtures/connector_confirm_stream.json``). Run with
``COWORK_RECORD_FIXTURES=1`` to rewrite that file.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
from cowork.api.v1.endpoints.connectors.submissions import allow_private_route, cancel_route, confirm_route
from cowork.api.v1.permissions import SingleUserOnly
from cowork.api.v1.route_walker import declared_permissions
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorTestAnswer
from cowork.server import create_app
from cowork.services.connectors import connector_test_run, pending_tests as pending_module
from cowork.services.connectors.connector_test_flow import CANCELLED, ENDED, NOT_ALLOWED
from cowork.services.connectors.pending_tests import PendingTest, PendingTests
# The mock API and the local vault, database and saved turn are the probe stream's own fixtures.
from tests.test_probe_test_request import ISSUED, PLANNER_FORM, SECRET, VALUES, api, connect, of, saved

FIXTURE = Path(__file__).parent / "fixtures" / "connector_confirm_stream.json"
OTHER_SCOPE = TenantScope(personal=True, org_id="other-person", user_id="other")


def _entry(scope=LOCAL_SCOPE, kind="confirm") -> PendingTest:
    return PendingTest(
        kind=kind, scope=scope, connector_id="planner", method=None, name="", conversation_id=None,
        form_id="fm_aaaaaaaaaa", form_spec={}, test_request={}, test_values={}, credentials={}, secret_names=[],
    )


class TestThePendingStore:
    def test_an_id_answers_only_its_own_scope_and_kind_and_stays_for_them(self):
        store = PendingTests()
        entry_id = store.put(_entry())

        assert store.pop(entry_id, "confirm", OTHER_SCOPE) is None
        assert store.pop(entry_id, "approval", LOCAL_SCOPE) is None
        assert store.pop(entry_id, "confirm", LOCAL_SCOPE) is not None
        assert store.pop(entry_id, "confirm", LOCAL_SCOPE) is None

    def test_an_entry_expires(self, monkeypatch):
        store = PendingTests()
        entry_id = store.put(_entry())
        now = pending_module.time.monotonic()
        monkeypatch.setattr(pending_module.time, "monotonic", lambda: now + 601)
        assert store.pop(entry_id, "confirm", LOCAL_SCOPE) is None

    def test_a_sixth_entry_drops_the_scopes_oldest_only(self):
        store = PendingTests()
        other = store.put(_entry(scope=OTHER_SCOPE))
        ids = [store.put(_entry()) for _ in range(6)]
        assert store.pop(ids[0], "confirm", LOCAL_SCOPE) is None
        assert all(store.pop(i, "confirm", LOCAL_SCOPE) for i in ids[1:])
        assert store.pop(other, "confirm", OTHER_SCOPE) is not None


class TestTheRoutesRunOnlyOnASingleUserInstall:
    @pytest.mark.parametrize("path", ["/allow-private", "/confirm", "/cancel"])
    def test_each_route_declares_single_user_only(self, path):
        [route] = [r for r in create_app().routes if getattr(r, "path", "") == f"/api/v1/connectors/submissions{path}"]
        assert SingleUserOnly in declared_permissions(route)

    def test_the_guard_refuses_a_shared_deployment(self, monkeypatch):
        from cowork.api.v1.endpoints import guards

        monkeypatch.setattr("cowork.db.scoped.tenant_scope_required", lambda: True)
        with pytest.raises(HTTPException) as exc:
            guards.require_single_user()
        assert exc.value.status_code == 403


def _normal(value):
    """``value`` with every random id, timestamp and sequence replaced, for the fixture."""
    text = json.dumps(value, sort_keys=True)
    text = re.sub(r'"(confirm_id|approval_id|id|item_id|assistant_message_id)": "[^"]+"', r'"\1": "<\1>"', text)
    text = re.sub(r'"conversation_id": "[^"]+"', '"conversation_id": "<conversation_id>"', text)
    text = re.sub(r"planner-[0-9a-f]{8}", "planner-<slug>", text)
    return json.loads(text)


def _check_fixture(recorded: dict) -> None:
    """Compare with the committed fixture, or rewrite it when asked to."""
    if os.environ.get("COWORK_RECORD_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
    assert json.loads(FIXTURE.read_text()) == recorded


class TestTheMemberAnswers:
    @pytest.mark.asyncio
    async def test_a_client_credentials_api_queried_by_post_saves_once_confirmed(self, api, saved, monkeypatch):
        events = await connect(PLANNER_FORM, VALUES, saved)
        [preview] = of(events, "connector.test_preview")
        assert saved.vault.list_connections() == []

        answer = await confirm_route(ConnectorTestAnswer(confirm_id=preview["confirm_id"]), LOCAL_SCOPE)

        [connection] = saved.vault.list_connections()
        assert answer["saved_slug"] == connection["name"]
        assert saved.vault.read_record("planner", connection["name"])["fields"]["client_secret"] == SECRET
        with Session(saved.engine) as session:
            assert [r.connector_id for r in session.exec(select(CustomConnector)).all()] == ["planner"]
        assert saved.turns[-1]["text"].startswith(f"Saved as `{connection['name']}`.")
        assert preview["confirm_id"] not in json.dumps(saved.turns)
        with pytest.raises(HTTPException) as again:
            await confirm_route(ConnectorTestAnswer(confirm_id=preview["confirm_id"]), LOCAL_SCOPE)
        assert again.value.status_code == 404
        recorded = {"stream": _normal(events), "confirm": _normal(answer)}

        events = await connect(PLANNER_FORM, VALUES, saved)
        [cancelled] = of(events, "connector.test_preview")
        await cancel_route(ConnectorTestAnswer(confirm_id=cancelled["confirm_id"]), LOCAL_SCOPE)
        assert saved.turns[-1]["text"] == CANCELLED
        recorded["private"] = await self._private_flow(api, saved, monkeypatch)
        _check_fixture(recorded)

    async def _private_flow(self, api, saved, monkeypatch) -> dict:
        monkeypatch.setattr(connector_test_run, "_RESOLVER", lambda host, port, type=None: [(None, None, None, "", ("10.0.0.5", 443))])
        events = await connect(PLANNER_FORM, VALUES, saved)
        asks = []
        [ask] = of(events, "connector.private_host_request")
        for _ in range(3):
            asks.append(ask)
            answer = await allow_private_route(ConnectorTestAnswer(approval_id=ask["approval_id"]), LOCAL_SCOPE)
            if "private_host_request" not in answer:
                break
            ask = answer["private_host_request"]
        assert [a["step"] for a in asks] == ["token", "data"]
        assert {r.url.host for r in api.requests[-2:]} == {"10.0.0.5"}
        assert set(answer) == {"preview", "confirm_id"}
        return {"stream": _normal(events), "allow": _normal(answer)}

    @pytest.mark.asyncio
    async def test_another_scope_cannot_answer_and_leaves_the_test_waiting(self, api, saved):
        [preview] = of(await connect(PLANNER_FORM, VALUES, saved), "connector.test_preview")
        answer = ConnectorTestAnswer(confirm_id=preview["confirm_id"])

        with pytest.raises(HTTPException) as exc:
            await confirm_route(answer, OTHER_SCOPE)
        assert exc.value.status_code == 404
        await cancel_route(answer, OTHER_SCOPE)
        await confirm_route(answer, LOCAL_SCOPE)
        assert len(saved.vault.list_connections()) == 1

    @pytest.mark.asyncio
    async def test_dont_allow_drops_the_test_and_says_so(self, api, saved, monkeypatch):
        monkeypatch.setattr(connector_test_run, "_RESOLVER", lambda host, port, type=None: [(None, None, None, "", ("10.0.0.5", 443))])
        [ask] = of(await connect(PLANNER_FORM, VALUES, saved), "connector.private_host_request")

        await cancel_route(ConnectorTestAnswer(approval_id=ask["approval_id"]), LOCAL_SCOPE)

        assert saved.turns[-1]["text"] == NOT_ALLOWED and api.requests == []
        with pytest.raises(HTTPException):
            await allow_private_route(ConnectorTestAnswer(approval_id=ask["approval_id"]), LOCAL_SCOPE)

    @pytest.mark.asyncio
    async def test_a_failure_after_allowing_returns_the_failure_patch(self, api, saved, monkeypatch):
        monkeypatch.setattr(connector_test_run, "_RESOLVER", lambda host, port, type=None: [(None, None, None, "", ("10.0.0.5", 443))])
        api.rows = {"access_token": ISSUED, "token_type": "bearer"}
        [ask] = of(await connect(PLANNER_FORM, VALUES, saved), "connector.private_host_request")
        answer = await allow_private_route(ConnectorTestAnswer(approval_id=ask["approval_id"]), LOCAL_SCOPE)
        answer = await allow_private_route(ConnectorTestAnswer(approval_id=answer["private_host_request"]["approval_id"]), LOCAL_SCOPE)

        assert answer["patch"]["form_error"] == "The data step's answer is a token, not data."
        assert saved.vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_cancel_takes_exactly_one_id(self):
        for answer in (ConnectorTestAnswer(), ConnectorTestAnswer(approval_id="a", confirm_id="b")):
            with pytest.raises(HTTPException) as exc:
                await cancel_route(answer, LOCAL_SCOPE)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_an_unknown_id_is_a_404(self):
        with pytest.raises(HTTPException) as exc:
            await confirm_route(ConnectorTestAnswer(confirm_id="nope"), LOCAL_SCOPE)
        assert exc.value.status_code == 404


CANCEL_URL = "/api/v1/connectors/submissions/cancel"
CONFIRM_URL = "/api/v1/connectors/submissions/confirm"
# The api fixture swaps httpx.AsyncClient for the mock provider; the app is called with the real one.
_APP_CLIENT = httpx.AsyncClient


async def _post(url: str, body: dict) -> httpx.Response:
    """POST through the real app, so the query string and body are parsed as a client sends them."""
    transport = httpx.ASGITransport(app=create_app())
    async with _APP_CLIENT(transport=transport, base_url="http://cowork.test") as client:
        return await client.post(url, json=body)


class TestWhyAWaitingTestEnded:
    """The cancel route records whether the member declined or the test ended another way."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("query", "text"), [
        ("?reason=declined", CANCELLED),
        ("", CANCELLED),
        ("?reason=ended", ENDED),
        ("?reason=something_newer", ENDED),
    ], ids=["declined", "no-reason", "ended", "unknown"])
    async def test_a_waiting_confirmation(self, api, saved, query, text):
        [preview] = of(await connect(PLANNER_FORM, VALUES, saved), "connector.test_preview")
        body = {"confirm_id": preview["confirm_id"]}

        assert (await _post(CANCEL_URL + query, body)).status_code == 204

        assert saved.turns[-1]["text"] == text
        assert (await _post(CONFIRM_URL, body)).status_code == 404
        assert saved.vault.list_connections() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("query", "text"), [
        ("?reason=declined", NOT_ALLOWED),
        ("", NOT_ALLOWED),
        ("?reason=ended", ENDED),
    ], ids=["declined", "no-reason", "ended"])
    async def test_a_waiting_private_address(self, api, saved, monkeypatch, query, text):
        monkeypatch.setattr(connector_test_run, "_RESOLVER", lambda host, port, type=None: [(None, None, None, "", ("10.0.0.5", 443))])
        [ask] = of(await connect(PLANNER_FORM, VALUES, saved), "connector.private_host_request")

        assert (await _post(CANCEL_URL + query, {"approval_id": ask["approval_id"]})).status_code == 204

        assert saved.turns[-1]["text"] == text and api.requests == []
        with pytest.raises(HTTPException):
            await allow_private_route(ConnectorTestAnswer(approval_id=ask["approval_id"]), LOCAL_SCOPE)


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    """Each test starts with no waiting tests, wherever the store is read."""
    fresh = PendingTests()
    for module in (pending_module, submissions_endpoint):
        monkeypatch.setattr(module, "pending_tests", fresh)
    from cowork.services.connectors import connector_test_flow

    monkeypatch.setattr(connector_test_flow, "pending_tests", fresh)
