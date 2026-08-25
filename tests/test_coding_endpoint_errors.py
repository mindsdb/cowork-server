from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import coding
from cowork.coding.workspace import WorkspaceError


def test_workspace_inspection_translates_typed_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_path: str) -> None:
        raise WorkspaceError("The selected folder cannot be inspected")

    monkeypatch.setattr(coding, "_service", lambda: SimpleNamespace(inspect_workspace=fail))

    with pytest.raises(HTTPException) as raised:
        coding.inspect_workspace("/unavailable")

    assert raised.value.status_code == 409
    assert raised.value.detail == "The selected folder cannot be inspected"
