from __future__ import annotations

import itertools
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines import codex, codex_config, codex_models
from cowork.coding.engines.base import EngineCredentials, EngineSessionConfig


@pytest.fixture
def catalog() -> dict:
    return {"models": [{
        "slug": "fable",
        "visibility": "list",
        "context_window": 200000,
        "base_instructions": "Version-matched upstream instructions",
        "supported_reasoning_levels": [{"effort": "high", "description": "High"}],
    }]}


def mock_catalog(monkeypatch, payload, status=200, content=None):
    requests = []

    def respond(request):
        requests.append(request)
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=payload)

    client_class = httpx.Client
    monkeypatch.setattr(
        codex_models.httpx, "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    return requests


def test_catalog_uses_scoped_proxy_and_preserves_metadata(monkeypatch, catalog):
    requests = mock_catalog(monkeypatch, catalog)
    with codex_models.model_catalog("https://gateway/inference/", "scoped-token", "0.147.0", "fable") as path:
        loaded = json.loads(path.read_text())
        expected = {**catalog["models"][0], "supports_parallel_tool_calls": False}
        assert loaded == {"models": [expected]}
        assert path.parent.stat().st_mode & 0o077 == 0
        request = requests[0]
        assert str(request.url) == "https://gateway/inference/models?client_version=0.147.0"
        assert request.headers["Authorization"] == "Bearer scoped-token"
        assert request.headers["originator"] == "codex_mindshub_cowork"
    assert not path.exists()


def test_catalog_preserves_explicit_parallel_tool_support(monkeypatch, catalog):
    catalog["models"][0]["supports_parallel_tool_calls"] = True
    mock_catalog(monkeypatch, catalog)
    with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable") as path:
        assert json.loads(path.read_text()) == catalog


@pytest.mark.parametrize("payload, message", [
    ({"data": [{"id": "fable"}]}, "native Codex model catalog"),
    ({"models": []}, "native Codex model catalog"),
    ({"models": [None]}, "native Codex model catalog"),
    ({"models": [{"slug": "other", "visibility": "list"}]}, "selected model is missing"),
    ({"models": [{"slug": "fable", "visibility": "hide"}]}, "selected model is missing"),
    ({"models": [{"slug": "fable"}]}, "selected model is missing"),
])
def test_catalog_rejects_fallback_conditions(monkeypatch, payload, message):
    mock_catalog(monkeypatch, payload)
    with pytest.raises(RuntimeError, match=message):
        with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable"):
            pytest.fail("Invalid metadata must not start Codex")


@pytest.mark.parametrize("status", [401, 403, 500])
def test_catalog_reports_fetch_failure(monkeypatch, status):
    mock_catalog(monkeypatch, {"error": "upstream error"}, status)
    with pytest.raises(RuntimeError, match="Unable to load Codex model metadata"):
        with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable"):
            pytest.fail("Failed discovery must not start Codex")


def test_catalog_files_are_isolated_and_removed_on_failure(monkeypatch, catalog):
    mock_catalog(monkeypatch, catalog)
    with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable") as first:
        with pytest.raises(RuntimeError, match="startup failed"):
            with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable") as second:
                assert first != second
                raise RuntimeError("startup failed")
        assert first.exists()
        assert not second.exists()
    assert not first.exists()


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_catalog_reports_transport_and_json_errors(monkeypatch, failure):
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("Timed out", request=request)
        return httpx.Response(200, content=b"not JSON")

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(codex_models.httpx, "Client", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="Unable to load Codex model metadata"):
        with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable"):
            pytest.fail("Failed discovery must not start Codex")


def test_catalog_deadline_bounds_a_trickled_response(monkeypatch, catalog):
    # Each chunk arrives within the per-operation timeout but advances the clock.
    clock = itertools.count(step=10)
    monkeypatch.setattr(codex_models.time, "monotonic", lambda: next(clock))
    body = json.dumps(catalog).encode()
    chunks = []

    def trickle():
        for byte in body:
            chunks.append(byte)
            yield bytes([byte])

    mock_catalog(monkeypatch, catalog, content=trickle())
    with pytest.raises(RuntimeError, match="Unable to load Codex model metadata"):
        with codex_models.model_catalog("http://proxy", "token", "0.147.0", "fable"):
            pytest.fail("A fetch past its deadline must not start Codex")
    assert len(chunks) < len(body)


@pytest.mark.parametrize("existing_session_id", [None, "existing-thread"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("failure", [None, "start", "initialize", "thread"])
def test_session_loads_catalog_before_start_and_cleans_up(
    monkeypatch, tmp_path, catalog, existing_session_id, remote, failure,
):
    requests = mock_catalog(monkeypatch, catalog)
    calls = []
    paths = []

    @dataclass
    class FakeConfig:
        cwd: str
        env: dict
        config_overrides: tuple
        client_name: str
        client_title: str
        client_version: str = "0.147.0"

    class FakeClient:
        def __init__(self, config, approval_handler):
            self.config = config

        def start(self):
            overrides = tomllib.loads("\n".join(self.config.config_overrides))
            path = Path(overrides["model_catalog_json"])
            paths.append(path)
            assert json.loads(path.read_text())["models"][0]["slug"] == "fable"
            assert overrides["model_auto_compact_token_limit"] == codex_config.auto_compact_token_limit()
            assert "real-secret" not in str(self.config)
            calls.append("start")
            if failure == "start":
                raise RuntimeError("startup failed")

        def initialize(self):
            if failure == "initialize":
                raise RuntimeError("startup failed")

        def thread_start(self, params):
            calls.append("thread_start")
            if failure == "thread":
                raise RuntimeError("startup failed")
            return SimpleNamespace(thread=SimpleNamespace(id="thread"))

        def thread_resume(self, session_id, params):
            assert session_id == existing_session_id
            calls.append("thread_resume")
            if failure == "thread":
                raise RuntimeError("startup failed")
            return SimpleNamespace(thread=SimpleNamespace(id=session_id))

        def close(self):
            assert paths[0].exists()
            calls.append("close")

    sdk = ModuleType("openai_codex.client")
    sdk.CodexConfig = FakeConfig
    sdk.CodexClient = FakeClient
    monkeypatch.setitem(sys.modules, "openai_codex.client", sdk)
    monkeypatch.setattr(codex.CodexEngineSession, "_register_skill_roots", lambda self: None)
    monkeypatch.setattr(codex.CodexEngineSession, "_route_global_notifications", lambda self: None)
    monkeypatch.setattr(codex.CodexEngineSession, "_app_server_pid", lambda self: None)
    monkeypatch.setattr(codex, "terminate_descendants", lambda pid: None)
    config = EngineSessionConfig(
        model="fable", permission_mode=PermissionMode.workspace,
        inference_base_url="https://gateway/inference" if remote else "",
        inference_api_key="scoped-token" if remote else "",
    )

    def open_session():
        return codex.CodexEngineSession(
            cowork_root=tmp_path, workspace=tmp_path, config=config,
            credentials=EngineCredentials(minds_url="https://upstream", minds_api_key="real-secret"),
            existing_session_id=existing_session_id, approval_handler=lambda *args: {},
        )

    if failure:
        with pytest.raises(RuntimeError, match="startup failed"):
            open_session()
    else:
        session = open_session()
        assert paths[0].exists()
        assert calls == ["start", "thread_resume" if existing_session_id else "thread_start"]
        session.close()
        session.close()
    assert calls[-1] == "close"
    assert not paths[0].exists()
    token = "scoped-token" if remote else codex_config.LOCAL_PROXY_TOKEN
    assert requests[0].headers["Authorization"] == f"Bearer {token}"
    assert requests[0].url.host == ("gateway" if remote else "127.0.0.1")
