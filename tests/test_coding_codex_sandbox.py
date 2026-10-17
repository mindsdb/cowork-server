from unittest.mock import Mock

import pytest
from openai_codex.generated.v2_all import WindowsSandboxReadinessResponse

from cowork.coding.engines import codex_sandbox


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_does_not_query_or_override_sandbox(monkeypatch, platform):
    monkeypatch.setattr(codex_sandbox.sys, "platform", platform)
    client = Mock()

    assert codex_sandbox.windows_sandbox_override(client) == ()
    client.request.assert_not_called()


@pytest.mark.parametrize("status,expected", [
    ("notConfigured", ('windows.sandbox="unelevated"',)),
    ("ready", ()),
    ("updateRequired", ()),
])
def test_windows_default_never_replaces_an_existing_sandbox(monkeypatch, status, expected):
    monkeypatch.setattr(codex_sandbox.sys, "platform", "win32")
    client = Mock()
    client.request.return_value = WindowsSandboxReadinessResponse(status=status)

    assert codex_sandbox.windows_sandbox_override(client) == expected
    client.request.assert_called_once_with(
        "windowsSandbox/readiness", {}, response_model=WindowsSandboxReadinessResponse,
    )


def test_unknown_readiness_does_not_silently_select_a_weaker_sandbox(monkeypatch):
    monkeypatch.setattr(codex_sandbox.sys, "platform", "win32")
    client = Mock()
    client.request.side_effect = RuntimeError("readiness failed")

    with pytest.raises(RuntimeError, match="readiness failed"):
        codex_sandbox.windows_sandbox_override(client)
