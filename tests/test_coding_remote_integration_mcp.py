from __future__ import annotations

import json
import os
import stat
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from cowork.coding.connector_capabilities import ConnectorCapability
from cowork.coding.contracts import utc_now
from cowork.coding.remote_integration_mcp import (
    RemoteCodeIntegrationMcp,
    RemoteIntegrationConfig,
    write_remote_integration_config,
)


def test_remote_mcp_config_is_created_owner_only_without_a_chmod(monkeypatch, tmp_path: Path) -> None:
    created: dict[str, tuple[int, int]] = {}
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        created[os.fspath(path)] = (flags, mode)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: pytest.fail("mode must be set at creation"))
    monkeypatch.setattr(Path, "chmod", lambda *_args, **_kwargs: pytest.fail("mode must be set at creation"))

    config_path = write_remote_integration_config(
        tmp_path / "run.json",
        RemoteIntegrationConfig(
            server_url="https://control.example.test",
            computer_id="computer-1",
            run_id="run-1",
            agent_token="agent-token-that-is-long-enough-for-runtime",
            project_context={"project": "Product"},
        ),
    )

    ((flags, mode),) = [item for path, item in created.items() if path.startswith(str(tmp_path))]
    assert mode == 0o600
    assert flags & os.O_EXCL
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert [path.name for path in tmp_path.iterdir()] == ["run.json"]
    assert "agent-token-that-is-long-enough-for-runtime" in config_path.read_text(encoding="utf-8")


def test_remote_mcp_uses_only_the_exact_scoped_capability(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"title": "Scoped issue"})

    config_path = write_remote_integration_config(
        tmp_path / "run.json",
        RemoteIntegrationConfig(
            server_url="https://control.example.test",
            computer_id="computer-1",
            run_id="run-1",
            agent_token="agent-token-that-is-long-enough-for-runtime",
            project_context={"project": "Product"},
            capabilities=[ConnectorCapability(
                id="grant-1",
                provider="linear",
                token="grant-token-that-is-long-enough-for-runtime",
                actions=["read_source"],
                resource_constraints={"url": "https://linear.app/acme/issue/ENG-1"},
                expires_at=utc_now() + timedelta(minutes=5),
            )],
        ),
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    mcp = RemoteCodeIntegrationMcp(
        config_path,
        httpx.Client(transport=httpx.MockTransport(respond)),
    )
    try:
        result = mcp.handle("tools/call", {
            "name": "mindshub_read_developer_context",
            "arguments": {
                "provider": "linear",
                "kind": "issue",
                "url": "https://linear.app/acme/issue/ENG-1",
            },
        })
        assert result is not None
        assert json.loads(result["content"][0]["text"])["title"] == "Scoped issue"
        assert seen[0].headers["Authorization"] == "Bearer agent-token-that-is-long-enough-for-runtime"
        assert "grant-token-that-is-long-enough-for-runtime" in seen[0].read().decode()

        with pytest.raises(RuntimeError, match="not granted"):
            mcp.handle("tools/call", {
                "name": "mindshub_read_developer_context",
                "arguments": {
                    "provider": "linear",
                    "kind": "issue",
                    "url": "https://linear.app/acme/issue/ENG-2",
                },
            })
        assert len(seen) == 1
    finally:
        mcp.close()
