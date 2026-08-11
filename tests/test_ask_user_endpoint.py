"""POST /api/v1/responses/answer.

Authorization copies /cancel exactly: same _require_streaming_scope, same
_authorized_handle, same 404 shape — so a foreign-org id is
indistinguishable from an unknown one and cannot leak existence.

The router is mounted at /api/v1 (cowork/api/v1/router.py:56), which is why
every path below carries that prefix.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from anton.core.interaction.elicit import AskOption, AskRequest
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from cowork.streaming.answers import broker
from cowork.streaming.registry import RunHandle, registry

CID = "conv-answer-test"
QID = "ask:1"

_OPTIONS = (
    AskOption(value="pg", label="postgres"),
    AskOption(value="my", label="mysql"),
)

# The default shape: pick exactly one, free text welcome.
_REQUEST = AskRequest(prompt="Which database?", options=_OPTIONS)
# Pick any number of the offered options.
_MULTI_REQUEST = AskRequest(prompt="Which databases?", options=_OPTIONS, select="many")
# Buttons only — no free-form answer was invited.
_NO_CUSTOM_REQUEST = AskRequest(
    prompt="Which database?", options=_OPTIONS, allow_custom=False
)


@pytest.fixture(autouse=True)
def _clean_globals():
    """Both the registry and the broker outlive a single test."""
    yield
    broker.reset()
    registry.reset()
    get_app_settings.cache_clear()


async def _register(
    org_id: str | None = None, request: AskRequest = _REQUEST
) -> RunHandle:
    """An in-flight run for CID plus one open question on it."""

    async def _forever():
        await asyncio.sleep(3600)

    handle = RunHandle(
        conversation_id=CID,
        turn_id=1,
        buffer=None,
        task=asyncio.create_task(_forever()),
        org_id=org_id,
    )
    registry._by_cid[CID] = handle
    broker.open(CID, QID, request)
    return handle


@pytest.fixture()
async def run():
    handle = await _register()
    yield handle
    handle.task.cancel()


@pytest.fixture()
async def multi_select_run():
    handle = await _register(request=_MULTI_REQUEST)
    yield handle
    handle.task.cancel()


@pytest.fixture()
async def no_custom_run():
    handle = await _register(request=_NO_CUSTOM_REQUEST)
    yield handle
    handle.task.cancel()


@pytest.fixture()
def client():
    return TestClient(create_app())


def _post(client, **body):
    return client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID, "question_id": QID, **body},
    )


async def test_accepts_a_valid_selection(client, run):
    resp = _post(client, values=["pg"])
    assert resp.status_code == 200
    assert resp.json() == {"accepted": True}


async def test_accepts_values_and_text_together(client, multi_select_run):
    """Legitimate for a multi-select question that allows custom text: two
    buttons plus a third answer typed by hand."""
    assert _post(client, values=["pg", "my"], text="and duckdb").status_code == 200


async def test_accepts_skipped(client, run):
    assert _post(client, skipped=True).status_code == 200


@pytest.mark.parametrize(
    "body,expected_status",
    [
        ({}, "empty_answer"),
        ({"values": []}, "empty_answer"),
        ({"text": "  "}, "empty_answer"),
        ({"skipped": True, "values": ["pg"]}, "ambiguous_answer"),
    ],
    ids=["nothing", "empty-values", "blank-text", "skipped-plus-values"],
)
async def test_rejects_a_malformed_body(client, run, body, expected_status):
    resp = _post(client, **body)
    assert resp.status_code == 400
    assert resp.json() == {"status": expected_status}


async def test_rejects_an_option_that_was_never_offered(client, run):
    resp = _post(client, values=["sqlite"])
    assert resp.status_code == 400
    assert resp.json() == {"status": "invalid_option"}
    assert not broker._pending[(CID, QID)].future.done()


# ── the answer must fit the shape that was offered ────────────────────
#
# The broker rejects rather than truncating. anton's CLIElicitor truncates a
# multi-selection (`picked[:1]` when select == "one") because a human typed it
# at a terminal; the HTTP caller is a GUI that rendered this exact card, so a
# shape it never offered is a frontend bug and silently altering the answer
# would hide it. Nothing downstream compensates — handle_ask_user passes
# answer.values straight to the model.
#
# Mutation proof (recorded 2026-07-31), one clause of AnswerBroker._violation
# deleted at a time:
#   * `if len(set(values)) != len(values)` -> duplicates case returns 200
#   * `if request.select == "one" and len(values) > 1` -> single-choice case 200
#   * `if answer.get("text") and not request.allow_custom` -> no-custom case 200
# Deleting any one of them leaves the other two green, so each test carries its
# own clause.


async def test_rejects_two_values_for_a_single_choice_question(client, run):
    """select="one" means one. Accepting two would hand the model a pair of
    answers to a question that offered a single pick."""
    resp = _post(client, values=["pg", "my"])
    assert resp.status_code == 400
    assert resp.json() == {"status": "invalid_option"}
    assert not broker._pending[(CID, QID)].future.done()


async def test_rejects_duplicate_values(client, multi_select_run):
    """The offered-options check is set-based, so duplicates would otherwise
    pass it unchanged and reach the model as ["pg", "pg", "pg"]. Pinned on a
    multi-select question so the cardinality clause cannot be what rejects it."""
    resp = _post(client, values=["pg", "pg", "pg"])
    assert resp.status_code == 400
    assert resp.json() == {"status": "invalid_option"}
    assert not broker._pending[(CID, QID)].future.done()


async def test_rejects_free_form_text_when_custom_is_not_allowed(client, no_custom_run):
    """allow_custom=False means the model asked for a button press, not prose."""
    resp = _post(client, text="actually duckdb")
    assert resp.status_code == 400
    assert resp.json() == {"status": "invalid_option"}
    assert not broker._pending[(CID, QID)].future.done()


async def test_accepts_two_values_for_a_multi_select_question(client, multi_select_run):
    """The control for the cardinality clause: it must key off `select`, not
    reject every multi-value answer."""
    assert _post(client, values=["pg", "my"]).status_code == 200


async def test_accepts_a_button_press_when_custom_is_not_allowed(
    client, no_custom_run
):
    """The control for the allow_custom clause."""
    assert _post(client, values=["pg"]).status_code == 200


async def test_skipping_bypasses_the_shape_checks(client, no_custom_run):
    """Skipping is not an answer to the question, so it is never measured
    against the offer."""
    assert _post(client, skipped=True).status_code == 200


# ── input bounds ──────────────────────────────────────────────────────
# A malformed request, not one of the spec's four rejected-answer statuses, so
# Pydantic's 422 is the right shape. Mutation proof (2026-07-31): removing
# `max_length=_MAX_TEXT_LENGTH` / `max_length=_MAX_VALUES` from AnswerRequest
# turns each 422 below into 200 and 400 invalid_option respectively.


async def test_rejects_an_over_long_text(client, run):
    assert _post(client, text="x" * 8193).status_code == 422


async def test_rejects_an_over_long_value(client, run):
    assert _post(client, values=["x" * 513]).status_code == 422


async def test_rejects_too_many_values(client, run):
    assert _post(client, values=[f"v{i}" for i in range(65)]).status_code == 422


async def test_unknown_question_is_404(client, run):
    resp = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID, "question_id": "ask:nope", "values": ["pg"]},
    )
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


def test_unknown_conversation_is_404(client):
    resp = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": "no-such-conv", "question_id": QID, "values": ["pg"]},
    )
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


async def test_duplicate_answer_is_409(client, run):
    assert _post(client, values=["pg"]).status_code == 200
    second = _post(client, values=["my"])
    assert second.status_code == 409
    assert second.json() == {"accepted": False, "status": "already_answered"}


async def test_an_unmapped_submit_result_is_not_reported_as_accepted(
    client, run, monkeypatch
):
    """The SubmitResult -> HTTP mapping must be exhaustive.

    Stands in for adding a fifth enum member: a result the endpoint does not
    know must not fall through to 200 {"accepted": true}. That answer would
    tell the frontend the card was delivered while the future stayed
    unresolved, and the turn would then hang to its 300 s timeout showing a
    delivered answer. A 500 is the right direction for "the server does not
    understand its own state".

    Mutation proof (recorded 2026-07-31): replacing
    `case _: raise AssertionError(...)` with `case _: return {"accepted": True}`
    (i.e. the pre-fix implicit else) makes this return 200 and the test fails.
    """
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.responses.broker.submit",
        lambda *a, **k: "a-result-that-does-not-exist-yet",
    )
    with pytest.raises(AssertionError, match="unhandled SubmitResult"):
        _post(client, values=["pg"])


async def test_foreign_org_is_404(monkeypatch):
    """A conversation_id is not an authorization token."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    owner_org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    intruder_org = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
    user_id = "11111111-1111-1111-1111-111111111111"
    handle = await _register(org_id=owner_org)
    try:
        client = TestClient(create_app())
        resp = client.post(
            "/api/v1/responses/answer",
            json={"conversation_id": CID, "question_id": QID, "values": ["pg"]},
            headers={"X-Organization-Id": intruder_org, "X-User-Id": user_id},
        )
        assert resp.status_code == 404
        assert resp.json() == {"status": "not_found"}
    finally:
        handle.task.cancel()
