from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import coding
from cowork.coding.control_errors import ModelDiscoveryAuthenticationError, StateConflict
from cowork.coding.control_models import ComputerCapabilities
from cowork.coding.control_service import ControlPlaneService
from cowork.coding.runtime_protocol import RegistrationTokenRequest
from cowork.coding.run_recovery import NoEligibleComputer
from cowork.coding.run_state import InvalidRunTransition
from cowork.coding.workspace import WorkspaceError


@pytest.mark.parametrize(
    ("exc", "status_code"),
    [
        (KeyError("Task not found"), 404),
        (ValueError("/goal pause does not accept an objective"), 400),
        (StateConflict("Code Project already exists"), 409),
        (StateConflict("Skill source already exists"), 409),
        (StateConflict("That repository branch is already in the Skills Library"), 409),
        (InvalidRunTransition("Task Run cannot move from completed to running"), 409),
        (NoEligibleComputer("No online computer can access every resource"), 409),
        (WorkspaceError("The selected folder cannot be inspected"), 409),
    ],
)
def test_http_error_separates_invalid_input_from_state_conflicts(exc: Exception, status_code: int) -> None:
    assert coding._http_error(exc).status_code == status_code


def test_workspace_inspection_translates_typed_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_path: str) -> None:
        raise WorkspaceError("The selected folder cannot be inspected")

    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(inspect_workspace=fail))

    with pytest.raises(HTTPException) as raised:
        coding.inspect_workspace("/unavailable")

    assert raised.value.status_code == 409
    assert raised.value.detail == "The selected folder cannot be inspected"


def test_model_authentication_failure_has_actionable_copy_and_stable_code() -> None:
    error = coding._http_error(ModelDiscoveryAuthenticationError("Sign in again"))

    assert error.status_code == 401
    assert error.detail == "Sign in again"
    assert error.headers == {"X-MindsHub-Error-Code": "coding_model_authentication_failed"}


def test_unknown_upstream_failure_remains_a_safe_500() -> None:
    request = httpx.Request("GET", "https://api.mindshub.ai/v1/models")
    response = httpx.Response(503, request=request)

    error = coding._http_error(
        httpx.HTTPStatusError("unavailable", request=request, response=response)
    )

    assert error.status_code == 500
    assert error.detail == "Coding operation failed"


def _control(tmp_path) -> ControlPlaneService:
    return ControlPlaneService(
        tmp_path,
        ComputerCapabilities(platform="darwin", architecture="arm64", runtime_version="test"),
    )


def test_connect_computer_names_a_pending_computer_and_the_list_shows_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    control = _control(tmp_path)
    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(control=control))

    issued = coding.runtime_registration_token(RegistrationTokenRequest(name="Build box", platform="linux"))

    assert issued.pending is not None and issued.pending.name == "Build box"
    assert issued.expires_in_seconds == 600
    assert [item.name for item in coding.coding_computers().pending] == ["Build box"]

    coding.revoke_coding_computer(issued.pending.id)
    assert coding.coding_computers().pending == []
    with pytest.raises(HTTPException) as raised:
        coding.revoke_coding_computer(issued.pending.id)
    assert raised.value.status_code == 404


def test_registration_tokens_without_a_body_stay_anonymous(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    control = _control(tmp_path)
    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(control=control))

    issued = coding.runtime_registration_token(None)

    assert issued.pending is None
    assert coding.coding_computers().pending == []

# gemini advertises three levels; gpt is deliberately absent from the listing, so
# the gateway remains the judge for it.
LEVELS = SimpleNamespace(ids=["gemini"], efforts={"gemini": {"efforts": ["low", "medium", "high"], "default": "high"}})


def _settings_with(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coding, "_settings", lambda _session, _scope: SimpleNamespace(minds_url="https://api.mindshub.ai/v1", coding_agent_model="gpt"))
    monkeypatch.setattr(coding, "cached_minds_models", lambda _url: LEVELS)


def test_a_project_default_effort_its_model_lacks_is_refused_before_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    from cowork.coding.project_models import ProjectUpdateRequest

    _settings_with(monkeypatch)
    saved: list = []
    projects = SimpleNamespace(
        get=lambda _id: SimpleNamespace(default_model="gemini"),
        update=lambda *args: saved.append(args),
    )
    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(projects=projects))

    with pytest.raises(HTTPException) as raised:
        coding.update_code_project("p1", ProjectUpdateRequest(default_reasoning_effort="max"), session=None, scope=None)

    assert raised.value.status_code == 400
    assert raised.value.detail == 'Reasoning effort "max" isn\'t available for gemini. It offers: low, medium, high.'
    assert saved == []


def test_a_project_default_effort_its_model_offers_is_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    from cowork.coding.project_models import ProjectUpdateRequest

    _settings_with(monkeypatch)
    projects = SimpleNamespace(
        get=lambda _id: SimpleNamespace(default_model="gpt"),
        update=lambda project_id, body: {"id": project_id, "default_reasoning_effort": body.default_reasoning_effort},
    )
    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(projects=projects))

    # gemini's levels apply when the same request moves the project onto gemini.
    with pytest.raises(HTTPException):
        coding.update_code_project("p1", ProjectUpdateRequest(default_model="gemini", default_reasoning_effort="max"), session=None, scope=None)
    # gpt isn't in the listing, so the gateway remains the judge and the save goes through.
    assert coding.update_code_project("p1", ProjectUpdateRequest(default_reasoning_effort="max"), session=None, scope=None)["default_reasoning_effort"] == "max"


def test_task_controls_check_the_effort_against_the_model_they_switch_to(monkeypatch: pytest.MonkeyPatch) -> None:
    from cowork.coding.contracts import SessionUpdateRequest

    _settings_with(monkeypatch)
    applied: list = []
    service = SimpleNamespace(
        get_session=lambda _id: SimpleNamespace(model="gpt"),
        update_session_config=lambda session_id, body: applied.append((session_id, body.reasoning_effort)),
    )
    monkeypatch.setattr(coding, "_service", lambda: service)

    with pytest.raises(HTTPException) as raised:
        coding.update_session("s1", SessionUpdateRequest(model="gemini", reasoning_effort="max"), session=None, scope=None)
    assert raised.value.status_code == 400

    coding.update_session("s1", SessionUpdateRequest(model="gemini", reasoning_effort="low"), session=None, scope=None)
    assert applied == [("s1", "low")]
