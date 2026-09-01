from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import coding
from cowork.coding.control_errors import StateConflict
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
