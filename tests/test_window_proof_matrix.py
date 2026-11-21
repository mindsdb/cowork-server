"""WindowProof asked of every route, both ways.

Builds the app as ``create_app()`` builds it and calls each method of every
HTTP route (the app has no websocket route), with a placeholder for each path
parameter and no body, from a loopback client in local tenancy. Every guard in
front of WindowProof therefore passes for real, with no dependency override:
``require_local`` sees a loopback peer, socket and host, and the tenancy
guards see local tenancy. Each request also carries the app secret, so
``require_app_secret``, which runs before WindowProof on the custom MCP
routes, lets it through too.

No endpoint runs. Each route gets one more dependency after every dependency
its router and the route itself declare, which answers 418 with the route's
path. So the walk never resolves an endpoint's own parameters or body and
never runs a handler: run for real with placeholder input, several hundred
handlers would reach the network, start runtimes and change state that later
tests in this worker read. A 418 that names the route says the request passed every
guard the route declares. Since no handler runs, the walk can't see a handler
that gives the refusal itself; ``test_no_other_code_gives_the_refusal`` reads
the source for that instead.

The cases:

- A server the desktop app launched with the hand-off, called with the bearer
  token alone: exactly the routes that declare WindowProof answer 403
  ``window_proof_required`` with the ``X-Cowork-Reason`` header, and no other
  route gives that refusal.
- The same server, called with the proof: no route gives the refusal, and
  every declared route reaches its endpoint. A wrong proof is refused on each
  declared route.
- A server that holds no hand-off, started from a terminal or by an older
  desktop app with the app secret alone: no route gives the refusal, and every
  declared route reaches its endpoint.

tests/test_window_proof_pin.py pins the declared set, with a reason per route.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.dependencies.utils import get_parameterless_sub_dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import cowork
from cowork import app_secret
from cowork.api.v1.permissions import WindowProof
from cowork.api.v1.route_walker import declared_permissions
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from tests._launch_handoff import PROOF, SECRET, TOKEN

REFUSAL = "window_proof_required"
REACHED = status.HTTP_418_IM_A_TEAPOT
#: What every request carries: the bearer token, and the app secret so that
#: require_app_secret is never the check that answers.
BEARER_AND_APP_SECRET = {"Authorization": f"Bearer {TOKEN}", app_secret.HEADER: SECRET}
WRONG_PROOF = "12" * 32
_PATH_PARAMETER = re.compile(r"\{[^}]+\}")


class Call(NamedTuple):
    path: str
    method: str


@dataclass(frozen=True)
class Answer:
    status: int
    detail: object
    reason: str | None


def _stop_before_the_endpoint(route: APIRoute) -> None:
    """Answer REACHED with the route's path once every dependency the router
    and the route declare has run, before the endpoint's own parameters."""
    declared = len(route.dependencies)
    # FastAPI puts the router's and the route's declared dependencies first,
    # in order, and the endpoint's own after them.
    assert [dep.call for dep in route.dependant.dependencies[:declared]] == [
        depends.dependency for depends in route.dependencies
    ], route.path

    def stop() -> None:
        raise HTTPException(status_code=REACHED, detail=route.path)

    route.dependant.dependencies.insert(
        declared, get_parameterless_sub_dependant(depends=Depends(stop), path=route.path_format)
    )


def _app(*, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FastAPI:
    # A held hand-off makes create_app remove the token line from
    # cowork-home's .env; keep that off the home this worker shares.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    get_app_settings.cache_clear()
    app = create_app()
    for route in app.routes:
        if isinstance(route, APIRoute):
            _stop_before_the_endpoint(route)
    return app


def _calls(app: FastAPI) -> list[Call]:
    return [
        Call(route.path, method)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods - {"HEAD"})
    ]


def _declared(app: FastAPI) -> set[Call]:
    return {
        Call(route.path, method)
        for route in app.routes
        if isinstance(route, APIRoute) and WindowProof in declared_permissions(route)
        for method in route.methods - {"HEAD"}
    }


def _detail(response: httpx.Response) -> object:
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("detail") if isinstance(body, dict) else None


def _ask(app: FastAPI, *, calls: list[Call], headers: dict[str, str]) -> dict[Call, Answer]:
    # No `with`: skipping the lifespan skips boot migrations, which would
    # collide with the schema conftest already created.
    client = TestClient(
        app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321), raise_server_exceptions=False
    )
    try:
        answers = {}
        for call in calls:
            response = client.request(call.method, _PATH_PARAMETER.sub("placeholder", call.path), headers=headers)
            answers[call] = Answer(
                status=response.status_code,
                detail=_detail(response),
                reason=response.headers.get("X-Cowork-Reason"),
            )
        return answers
    finally:
        client.close()


def _refused(answers: dict[Call, Answer]) -> set[Call]:
    return {call for call, answer in answers.items() if answer.detail == REFUSAL}


def _not_reached(answers: dict[Call, Answer], calls: set[Call]) -> dict[Call, Answer]:
    """The calls that stopped short of their own route's endpoint."""
    return {
        call: answers[call]
        for call in sorted(calls)
        if (answers[call].status, answers[call].detail) != (REACHED, call.path)
    }


def test_with_a_hand_off_exactly_the_declared_routes_refuse_the_bearer_alone(handoff_held, monkeypatch, tmp_path):
    app = _app(monkeypatch=monkeypatch, tmp_path=tmp_path)
    declared = _declared(app)
    answers = _ask(app, calls=_calls(app), headers=BEARER_AND_APP_SECRET)
    refused = _refused(answers)

    assert declared, "no route declares WindowProof"
    assert sorted(refused - declared) == [], "these routes give the refusal without declaring WindowProof"
    missed = {call: answers[call] for call in sorted(declared - refused)}
    assert missed == {}, "these routes declare WindowProof, and something else answered first"
    assert {answers[call].status for call in refused} == {status.HTTP_403_FORBIDDEN}
    assert {answers[call].reason for call in refused} == {REFUSAL}


def test_with_the_proof_no_route_refuses_and_every_declared_route_reaches_its_endpoint(
    handoff_held, monkeypatch, tmp_path
):
    app = _app(monkeypatch=monkeypatch, tmp_path=tmp_path)
    headers = {**BEARER_AND_APP_SECRET, app_secret.WINDOW_PROOF_HEADER: PROOF}
    answers = _ask(app, calls=_calls(app), headers=headers)

    assert sorted(_refused(answers)) == []
    assert _not_reached(answers, _declared(app)) == {}


def test_a_wrong_proof_is_refused_on_every_declared_route(handoff_held, monkeypatch, tmp_path):
    app = _app(monkeypatch=monkeypatch, tmp_path=tmp_path)
    declared = _declared(app)
    headers = {**BEARER_AND_APP_SECRET, app_secret.WINDOW_PROOF_HEADER: WRONG_PROOF}
    answers = _ask(app, calls=sorted(declared), headers=headers)

    assert _refused(answers) == declared


@pytest.mark.parametrize("app_secret_alone", [False, True], ids=["started-from-a-terminal", "app-secret-alone"])
def test_without_a_hand_off_no_route_refuses_and_every_declared_route_reaches_its_endpoint(
    nothing_held, monkeypatch, tmp_path, app_secret_alone
):
    if app_secret_alone:
        monkeypatch.setattr(app_secret, "_secret", SECRET)
    app = _app(monkeypatch=monkeypatch, tmp_path=tmp_path)
    answers = _ask(app, calls=_calls(app), headers=BEARER_AND_APP_SECRET)

    assert sorted(_refused(answers)) == []
    assert _not_reached(answers, _declared(app)) == {}


def test_no_other_code_gives_the_refusal():
    """The walk stops before every endpoint, so it can't see a handler that
    gives this refusal itself. The source can: only WindowProof writes it."""
    root = Path(cowork.__file__).parent
    found = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if REFUSAL in path.read_text(encoding="utf-8")
    )
    assert found == ["api/v1/permissions.py"]
