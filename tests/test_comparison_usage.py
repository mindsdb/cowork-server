"""What each side of a comparison cost: the gateway's calls filed under the side's turns."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from cowork.db.session import get_open_session
from cowork.models.comparison import ComparisonSide
from cowork.models.message import Message
from cowork.services import comparison_usage
from cowork.services.comparison_usage import _SessionUsage, side_usage

T0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)


def _session(payload: dict) -> _SessionUsage:
    """The usage service's answer, read the way the route reads it."""
    return _SessionUsage.model_validate(payload)


def _call(seconds: float, cost: float | None = 0.01, **tokens) -> dict:
    return {
        "timestamp": (T0 + timedelta(seconds=seconds)).isoformat(),
        "input_tokens": tokens.get("input", 100),
        "output_tokens": tokens.get("output", 10),
        "cached_input_tokens": tokens.get("cached", 0),
        "cache_write_tokens": tokens.get("written", 0),
        "estimated_cost_usd": cost,
    }


def _included(seconds: float, **figures) -> dict:
    """A call a plan window or the Free allowance covered whole: no tokens, no price."""
    return {**_call(seconds, None, input=0, output=0), "included": True, **figures}


def _starts(*seconds: float) -> list[datetime]:
    return [T0 + timedelta(seconds=s) for s in seconds]


def test_each_call_is_filed_under_the_turn_that_started_before_it():
    # A turn's first call usually lands in the same second as its message,
    # which is stored to the second: that call belongs to the new turn.
    payload = {"requests": [_call(0.4, 0.02), _call(20, 0.03), _call(60, 0.5), _call(90, 0.25)]}

    usage = side_usage(turn_starts=_starts(0, 60), payload=_session(payload))

    assert usage.available is True
    assert [t.turn for t in usage.turns] == [1, 2]
    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([0.05, 0.75])
    assert usage.estimated_cost_usd == pytest.approx(0.8)
    assert [t.input_tokens for t in usage.turns] == [200, 200]
    assert usage.input_tokens == 400


def test_unpriced_calls_are_counted_and_left_out_of_the_cost():
    payload = {"requests": [_call(1, 0.1), _call(2, None), _call(61, None)]}

    usage = side_usage(turn_starts=_starts(0, 60), payload=_session(payload))

    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([0.1, None])
    assert [t.unpriced_calls for t in usage.turns] == [1, 1]
    assert usage.unpriced_calls == 2
    assert usage.estimated_cost_usd == pytest.approx(0.1)
    # Tokens still count even when the price is unknown.
    assert usage.turns[1].input_tokens == 100


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(10**400, id="too large for a float"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_a_price_that_is_not_a_finite_number_reads_as_unpriced(cost):
    usage = side_usage(turn_starts=_starts(0), payload=_session({"requests": [_call(1, 0.1), _call(2, cost)]}))

    assert usage.unpriced_calls == 1
    assert usage.estimated_cost_usd == pytest.approx(0.1)


def test_a_call_that_would_make_its_turns_cost_overflow_reads_as_unpriced():
    # Each price is finite, but the two add up past the largest float.
    usage = side_usage(turn_starts=_starts(0), payload=_session({"requests": [_call(1, 1e308), _call(2, 1e308)]}))

    turn = usage.turns[0]
    assert (turn.estimated_cost_usd, turn.unpriced_calls) == (1e308, 1)
    assert (usage.estimated_cost_usd, usage.unpriced_calls) == (1e308, 1)
    # Its tokens still count, as for any unpriced call.
    assert usage.input_tokens == 200


def test_a_call_that_would_make_the_sides_cost_overflow_reads_as_unpriced():
    # Each turn's cost is finite on its own; the two add up past the largest float.
    payload = _session({"requests": [_call(1, 1e308), _call(61, 1e308)]})

    usage = side_usage(turn_starts=_starts(0, 60), payload=payload)

    assert [t.estimated_cost_usd for t in usage.turns] == [1e308, None]
    assert [t.unpriced_calls for t in usage.turns] == [0, 1]
    assert (usage.estimated_cost_usd, usage.unpriced_calls) == (1e308, 1)


def test_an_entry_that_is_not_an_object_is_skipped():
    payload = _session({"requests": [_call(1, 0.1), "junk", 7, None, [_call(2, 0.2)]]})

    usage = side_usage(turn_starts=_starts(0), payload=payload)

    assert (usage.estimated_cost_usd, usage.unpriced_calls, usage.input_tokens) == (pytest.approx(0.1), 0, 100)


@pytest.mark.parametrize(
    ("marking", "truncated"),
    [
        pytest.param({}, False, id="absent"),
        pytest.param({"truncated": None}, False, id="null"),
        pytest.param({"truncated": 0}, False, id="zero"),
        pytest.param({"truncated": 1}, True, id="one"),
        pytest.param({"truncated": "yes, at the cap"}, True, id="a string"),
    ],
)
def test_any_truthy_cap_marking_marks_the_totals_as_a_floor(marking, truncated):
    usage = side_usage(turn_starts=_starts(0), payload=_session({"requests": [_call(1)], **marking}))

    assert (usage.available, usage.truncated) == (True, truncated)


@pytest.mark.parametrize(
    ("starts", "payload", "turn_limit"),
    [
        pytest.param(
            _starts(0, 60), {"requests": [_included(61), _call(62, 0.3)]}, 1, id="every call after the owned turns"
        ),
        pytest.param(_starts(0), {"requests": []}, None, id="no calls"),
    ],
)
def test_a_side_with_no_calls_in_its_turns_is_available_and_counts_nothing(starts, payload, turn_limit):
    usage = side_usage(turn_starts=starts, payload=_session(payload), turn_limit=turn_limit)

    assert usage.available is True
    assert [(t.included_calls, t.unpriced_calls, t.estimated_cost_usd) for t in usage.turns] == [(0, 0, None)]
    assert (usage.included_calls, usage.unpriced_calls, usage.estimated_cost_usd) == (0, 0, None)


# ── Calls a plan window or the Free allowance covered ─────────────────────


def test_a_turn_whose_calls_were_all_included_has_no_cost_and_nothing_unpriced():
    usage = side_usage(turn_starts=_starts(0), payload=_session({"requests": [_included(1), _included(2)]}))

    turn = usage.turns[0]
    assert (turn.included_calls, turn.unpriced_calls, turn.estimated_cost_usd) == (2, 0, None)
    assert (turn.input_tokens, turn.output_tokens, turn.cached_input_tokens, turn.cache_write_tokens) == (0, 0, 0, 0)
    assert usage.available is True
    assert (usage.included_calls, usage.unpriced_calls, usage.estimated_cost_usd) == (2, 0, None)


def test_a_mixed_turn_sums_only_its_priced_call():
    payload = {"requests": [_included(1), _call(2, 0.25, input=300, output=20)]}

    turn = side_usage(turn_starts=_starts(0), payload=_session(payload)).turns[0]

    assert turn.estimated_cost_usd == pytest.approx(0.25)
    assert (turn.input_tokens, turn.output_tokens) == (300, 20)
    assert (turn.included_calls, turn.unpriced_calls) == (1, 0)


def test_an_included_call_beside_an_unpriced_one_leaves_only_that_one_unpriced():
    # A call the usage service could not price comes with no cost and no tokens.
    payload = {"requests": [_included(1), _call(2, None, input=0, output=0)]}

    usage = side_usage(turn_starts=_starts(0), payload=_session(payload))

    turn = usage.turns[0]
    assert (turn.included_calls, turn.unpriced_calls, turn.estimated_cost_usd) == (1, 1, None)
    assert (usage.included_calls, usage.unpriced_calls, usage.estimated_cost_usd) == (1, 1, None)


def test_an_included_call_adds_nothing_even_when_it_carries_figures():
    # Only paid figures reach the totals, so plan usage never shows.
    stray = _included(
        1, input_tokens=500, output_tokens=40, cached_input_tokens=7, cache_write_tokens=3, estimated_cost_usd=0.9
    )

    usage = side_usage(turn_starts=_starts(0), payload=_session({"requests": [stray, _call(2, 0.1)]}))

    assert (usage.included_calls, usage.unpriced_calls) == (1, 0)
    assert usage.estimated_cost_usd == pytest.approx(0.1)
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (
        100, 10, 0, 0,
    )


def test_the_side_totals_count_included_calls_over_the_turns_the_comparison_owns():
    payload = {
        "requests": [
            _included(1), _call(2, 0.1),  # turn 1
            _included(61), _included(62), _call(63, None, input=0, output=0),  # turn 2
            _included(121), _call(122, 4.0),  # turn 3, after a Continue
        ]
    }

    usage = side_usage(turn_starts=_starts(0, 60, 120), payload=_session(payload), turn_limit=2)

    assert [t.included_calls for t in usage.turns] == [1, 2]
    assert [t.unpriced_calls for t in usage.turns] == [0, 1]
    assert (usage.included_calls, usage.unpriced_calls) == (3, 1)
    assert usage.estimated_cost_usd == pytest.approx(0.1)
    assert (usage.input_tokens, usage.output_tokens) == (100, 10)


@pytest.mark.parametrize(
    "marking",
    [
        # The inference service before the field existed sends none.
        pytest.param({}, id="absent"),
        pytest.param({"included": None}, id="null"),
        pytest.param({"included": False}, id="false"),
        # Only JSON true marks a call included.
        pytest.param({"included": "true"}, id="not a boolean"),
    ],
)
def test_a_call_not_marked_included_reads_as_it_always_has(marking):
    payload = {"requests": [{**_call(1, 0.1), **marking}, {**_call(2, None), **marking}]}

    usage = side_usage(turn_starts=_starts(0), payload=_session(payload))

    assert (usage.included_calls, usage.unpriced_calls) == (0, 1)
    assert usage.estimated_cost_usd == pytest.approx(0.1)
    assert usage.input_tokens == 200


def test_a_continued_side_counts_only_the_turns_the_comparison_owns():
    payload = {"requests": [_call(1, 0.1), _call(61, 0.2), _call(120, 4.0)]}

    usage = side_usage(turn_starts=_starts(0, 60, 120), payload=_session(payload), turn_limit=2)

    assert [t.turn for t in usage.turns] == [1, 2]
    assert usage.estimated_cost_usd == pytest.approx(0.3)


def test_naive_timestamps_read_as_utc():
    # SQLite hands message times back without a zone.
    naive = [T0.replace(tzinfo=None), (T0 + timedelta(seconds=60)).replace(tzinfo=None)]

    usage = side_usage(turn_starts=naive, payload=_session({"requests": [_call(61, 0.2)]}))

    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([None, 0.2])


@pytest.mark.parametrize(
    ("starts", "payload"),
    [
        pytest.param(_starts(0), None, id="gateway had nothing"),
        pytest.param([], _session({"requests": [_call(1)]}), id="no turns yet"),
    ],
)
def test_no_usage_reads_as_unavailable(starts, payload):
    assert side_usage(turn_starts=starts, payload=payload).available is False


def test_the_gateway_cap_marks_the_totals_as_a_floor():
    payload = _session({"requests": [_call(1)], "truncated": True})

    assert side_usage(turn_starts=_starts(0), payload=payload).truncated is True


@pytest.mark.parametrize(
    ("calls_read", "truncated"),
    [
        # The cap was reached after the Continue: every call of the owned turns was read.
        ([_call(1), _call(61), _call(130)], False),
        # The cap was reached before it: owned turns may be missing calls.
        ([_call(1), _call(61)], True),
    ],
    ids=["cap-after-continue", "cap-before-continue"],
)
def test_a_continued_side_is_partial_only_when_the_cap_cuts_its_own_turns(calls_read, truncated):
    usage = side_usage(
        turn_starts=_starts(0, 60, 120), payload=_session({"requests": calls_read, "truncated": True}), turn_limit=2
    )
    assert usage.truncated is truncated


def test_a_fast_local_clock_is_corrected_before_calls_are_filed():
    # The laptop runs 30 s fast: its turn starts read 30 s late against the
    # gateway's timestamps. Uncorrected, the second turn's first calls would be
    # filed under the first turn.
    local_starts = _starts(30, 90)
    payload = {"requests": [_call(1, 0.1), _call(61, 0.2), _call(75, 0.3)]}

    uncorrected = side_usage(turn_starts=local_starts, payload=_session(payload))
    assert [t.estimated_cost_usd for t in uncorrected.turns] == pytest.approx([0.6, None])
    corrected = side_usage(turn_starts=local_starts, payload=_session(payload), clock_offset=timedelta(seconds=-30))
    assert [t.estimated_cost_usd for t in corrected.turns] == pytest.approx([0.1, 0.5])


def test_the_gateway_clock_is_read_from_its_date_header():
    from email.utils import format_datetime

    from cowork.services.comparison_usage import _clock_offset

    received = datetime(2026, 9, 29, 12, 0, 30, tzinfo=timezone.utc)
    gateway_now = format_datetime(datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc), usegmt=True)
    assert _clock_offset(httpx.Response(200, headers={"date": gateway_now}), received) == timedelta(seconds=-30)
    assert _clock_offset(httpx.Response(200), received) is None
    assert _clock_offset(httpx.Response(200, headers={"date": "not a date"}), received) is None


# ── The route ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    from cowork.server import create_app

    return TestClient(create_app())


@pytest.fixture()
def gateway():
    """The gateway, faked at the HTTP transport, recording what it was sent."""
    state = {"sent": [], "respond": lambda request: httpx.Response(404)}

    def handler(request: httpx.Request) -> httpx.Response:
        state["sent"].append(request)
        return state["respond"](request)

    real = httpx.AsyncClient

    def client_factory(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    with patch.object(comparison_usage.httpx, "AsyncClient", side_effect=client_factory):
        yield state


def _comparison(client) -> dict:
    resp = client.post(
        "/api/v1/comparisons/",
        json={"title": "t", "sides": [{"model": "kimi"}, {"model": "qwen"}]},
    )
    assert resp.status_code == 201
    return resp.json()


def _user_turn(conversation_id: str, seq: int, at: datetime, content="task") -> None:
    session = get_open_session()
    try:
        session.add(Message(conversation_id=UUID(conversation_id), role="user", content=content, seq=seq, created_at=at))
        session.commit()
    finally:
        session.close()


def test_reads_each_side_from_this_deployments_gateway_as_the_caller(client, gateway):
    comparison = _comparison(client)
    side_a, side_b = comparison["sides"]
    _user_turn(side_a["conversationId"], 1, T0)
    # A tool result is a user-role row, not a turn.
    _user_turn(side_a["conversationId"], 2, T0 + timedelta(seconds=5),
               content=[{"type": "tool_result", "tool_use_id": "t1", "content": "found"}])
    _user_turn(side_b["conversationId"], 1, T0)

    def respond(request):
        cost = 0.4 if side_a["conversationId"] in request.url.path else 0.1
        return httpx.Response(200, json={"requests": [_call(1, cost), _call(9, cost)]})

    gateway["respond"] = respond

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage",
        headers={"X-MindsHub-Authorization": "Bearer caller-jwt"},
    )

    assert resp.status_code == 200
    sides = resp.json()["sides"]
    assert [len(sides["a"]["turns"]), len(sides["b"]["turns"])] == [1, 1]
    assert sides["a"]["estimatedCostUsd"] == pytest.approx(0.8)
    assert sides["b"]["estimatedCostUsd"] == pytest.approx(0.2)
    from cowork.common.settings.app_settings import default_turn_minds_api_host

    host = urlparse(default_turn_minds_api_host()).hostname
    assert {r.url.host for r in gateway["sent"]} == {host}
    assert {r.url.path for r in gateway["sent"]} == {
        f"/v1/usage/sessions/{side_a['conversationId']}",
        f"/v1/usage/sessions/{side_b['conversationId']}",
    }
    assert {r.headers["authorization"] for r in gateway["sent"]} == {"Bearer caller-jwt"}


@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(lambda r: httpx.Response(404, json={"detail": "Not Found"}), id="gateway without the route"),
        pytest.param(lambda r: httpx.Response(500), id="gateway error"),
        pytest.param(lambda r: httpx.Response(200, text="not json"), id="not json"),
        pytest.param(
            lambda r: httpx.Response(302, headers={"location": "https://elsewhere.example/v1/usage"}),
            id="redirect",
        ),
        pytest.param(lambda r: httpx.Response(200, json=[_call(1)]), id="not an object"),
        pytest.param(lambda r: httpx.Response(200, json={"truncated": False}), id="no requests"),
        pytest.param(lambda r: httpx.Response(200, json={"requests": None}), id="requests not a list"),
    ],
)
def test_a_failed_read_hides_the_figure_without_failing(client, gateway, respond):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
    gateway["respond"] = respond

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage",
        headers={"X-MindsHub-Authorization": "Bearer caller-jwt"},
    )

    assert resp.status_code == 200
    assert [s["available"] for s in resp.json()["sides"].values()] == [False, False]
    # A redirect is not followed, so the bearer never reaches another host.
    assert {r.url.host for r in gateway["sent"]} != {"elsewhere.example"}
    assert len(gateway["sent"]) == 2


def test_without_a_hub_credential_nothing_is_sent(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)

    resp = client.get(f"/api/v1/comparisons/{comparison['id']}/usage")

    assert resp.status_code == 200
    assert [s["available"] for s in resp.json()["sides"].values()] == [False, False]
    assert gateway["sent"] == []


def test_an_unknown_comparison_is_not_found(client, gateway):
    assert client.get("/api/v1/comparisons/00000000-0000-0000-0000-000000000000/usage").status_code == 404


def test_the_route_corrects_for_the_local_clock(client, gateway):
    from email.utils import format_datetime

    comparison = _comparison(client)
    side_a = comparison["sides"][0]
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # Messages carry this machine's clock, which runs 30 s ahead of the gateway.
    _user_turn(side_a["conversationId"], 1, now - timedelta(seconds=90))
    _user_turn(side_a["conversationId"], 2, now - timedelta(seconds=30))
    gateway_clock = lambda seconds_ago: (now - timedelta(seconds=30 + seconds_ago)).isoformat()

    def respond(request):
        calls = [
            {**_call(0, 0.1), "timestamp": gateway_clock(89)},
            {**_call(0, 0.2), "timestamp": gateway_clock(29)},
        ]
        date = format_datetime(now - timedelta(seconds=30), usegmt=True)
        return httpx.Response(200, json={"requests": calls}, headers={"date": date})

    gateway["respond"] = respond
    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage", headers={"X-MindsHub-Authorization": "Bearer caller-jwt"}
    )

    assert [t["estimatedCostUsd"] for t in resp.json()["sides"]["a"]["turns"]] == pytest.approx([0.1, 0.2])


def test_the_history_shows_the_cost_last_read_and_keeps_it_when_a_read_fails(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
    headers = {"X-MindsHub-Authorization": "Bearer caller-jwt"}
    gateway["respond"] = lambda r: httpx.Response(200, json={"requests": [_call(1, 0.25, input=1000, output=50)]})
    client.get(f"/api/v1/comparisons/{comparison['id']}/usage", headers=headers)

    gateway["respond"] = lambda r: httpx.Response(500)
    client.get(f"/api/v1/comparisons/{comparison['id']}/usage", headers=headers)

    listed = next(c for c in _all_comparisons(client) if c["id"] == comparison["id"])
    assert [s["usage"] for s in listed["sides"]] == [
        {"estimatedCostUsd": 0.25, "tokens": 1050, "partial": False, "includedCalls": 0}
    ] * 2


def test_the_history_counts_a_side_whose_calls_were_all_included(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
    gateway["respond"] = lambda r: httpx.Response(200, json={"requests": [_included(1), _included(2)]})

    client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage", headers={"X-MindsHub-Authorization": "Bearer caller-jwt"}
    )

    listed = next(c for c in _all_comparisons(client) if c["id"] == comparison["id"])
    assert [s["usage"] for s in listed["sides"]] == [
        {"estimatedCostUsd": None, "tokens": 0, "partial": False, "includedCalls": 2}
    ] * 2


def test_the_route_reports_included_calls_and_the_history_does_not_mark_them_partial(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
    gateway["respond"] = lambda r: httpx.Response(
        200, json={"requests": [_included(1), _call(2, 0.25, input=1000, output=50)]}
    )

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage", headers={"X-MindsHub-Authorization": "Bearer caller-jwt"}
    )

    side_a = resp.json()["sides"]["a"]
    assert (side_a["includedCalls"], side_a["unpricedCalls"]) == (1, 0)
    assert [(t["includedCalls"], t["unpricedCalls"]) for t in side_a["turns"]] == [(1, 0)]
    listed = next(c for c in _all_comparisons(client) if c["id"] == comparison["id"])
    assert [s["usage"] for s in listed["sides"]] == [
        {"estimatedCostUsd": 0.25, "tokens": 1050, "partial": False, "includedCalls": 1}
    ] * 2


def test_costs_that_add_up_past_the_largest_float_still_answer(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
        _user_turn(side["conversationId"], 2, T0 + timedelta(seconds=60))
    # Turn 1 overflows on its own; turn 2's cost is finite but overflows the side's.
    gateway["respond"] = lambda r: httpx.Response(
        200, json={"requests": [_call(1, 1e308), _call(2, 1e308), _call(61, 1e308)]}
    )

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage", headers={"X-MindsHub-Authorization": "Bearer caller-jwt"}
    )

    assert resp.status_code == 200
    side_a = resp.json()["sides"]["a"]
    assert (side_a["estimatedCostUsd"], side_a["unpricedCalls"]) == (1e308, 2)
    assert [(t["estimatedCostUsd"], t["unpricedCalls"]) for t in side_a["turns"]] == [(1e308, 1), (None, 1)]
    listed = next(c for c in _all_comparisons(client) if c["id"] == comparison["id"])
    assert [s["usage"] for s in listed["sides"]] == [
        {"estimatedCostUsd": 1e308, "tokens": 330, "partial": True, "includedCalls": 0}
    ] * 2


def test_a_snapshot_stored_without_included_calls_serves_none_included(client):
    comparison = _comparison(client)
    session = get_open_session()
    try:
        sides = session.exec(select(ComparisonSide).where(ComparisonSide.comparison_id == UUID(comparison["id"]))).all()
        for side in sides:
            side.usage_snapshot = {"estimated_cost_usd": 0.25, "tokens": 1050, "partial": False}
            session.add(side)
        session.commit()
    finally:
        session.close()

    listed = next(c for c in _all_comparisons(client) if c["id"] == comparison["id"])
    assert [s["usage"] for s in listed["sides"]] == [
        {"estimatedCostUsd": 0.25, "tokens": 1050, "partial": False, "includedCalls": 0}
    ] * 2


def _all_comparisons(client, limit=2, max_pages=500):
    seen, offset = [], 0
    for _ in range(max_pages):
        page = client.get("/api/v1/comparisons/", params={"limit": limit, "offset": offset}).json()
        assert len(page["comparisons"]) <= limit
        seen += page["comparisons"]
        if not page["hasMore"]:
            return seen
        offset += limit
    raise AssertionError("the history never ran out of pages")


def test_the_history_pages_past_its_limit(client):
    ids = {_comparison(client)["id"] for _ in range(3)}

    first = client.get("/api/v1/comparisons/", params={"limit": 2}).json()
    everything = _all_comparisons(client)

    assert first["hasMore"] is True
    listed = [c["id"] for c in everything]
    # Every comparison once, however many pages it takes.
    assert ids <= set(listed) and len(listed) == len(set(listed))
