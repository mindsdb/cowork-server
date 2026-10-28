"""Startup compatibility checks stay separate from routing and its budgets."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from anton.core.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, ToolCall

from cowork.db.scoped import LOCAL_SCOPE
from cowork.handlers import response_routing as routing
from cowork.services import router_probe
from tests._logging_assertions import assert_log_fields


class _Provider:
    def __init__(self):
        self.events = [StreamComplete(LLMResponse(content="Hello!", stop_reason="completed"))]
        self.calls = []
        self.waiting = asyncio.Event()
        self.hold_open = False
        self.delay = 0
        self.closed = False
        self.error = None

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        try:
            if self.error is not None:
                raise self.error
            if self.delay:
                await asyncio.sleep(self.delay)
            for event in self.events:
                yield event
            if self.hold_open:
                self.waiting.set()
                await asyncio.Event().wait()
        finally:
            self.closed = True


@pytest.fixture
def probe(monkeypatch):
    settings = SimpleNamespace(
        resolved_gate_model="router-model",
        resolved_router_provider=SimpleNamespace(value="openai_compatible"),
    )
    provider = _Provider()
    client = SimpleNamespace(router_provider=provider, coding_provider=_Provider(), aclose=AsyncMock())
    built = []

    async def read(fn, *, scope):
        assert scope is LOCAL_SCOPE
        return settings

    def build(*, settings):
        built.append(settings)
        return client

    monkeypatch.setattr(router_probe, "run_db", read)
    monkeypatch.setattr(router_probe, "build_llm_client", build)
    return SimpleNamespace(settings=settings, provider=provider, client=client, built=built)


def _outcome(caplog, outcome, *, reason=None):
    records = [
        record for record in caplog.records
        if record.name == router_probe.__name__ and getattr(record, "outcome", None) == outcome
    ]
    assert len(records) == 1
    record = records[0]
    assert record.event == f"router_startup_probe_{outcome}"
    assert record.levelno == (logging.INFO if outcome == "passed" else logging.WARNING)
    assert record.exc_info is None
    if reason:
        assert record.reason == reason
        assert "startup continues" in record.getMessage()
        assert "Recheck" in record.getMessage()
    return record


async def test_probe_uses_the_gate_contract_without_gate_latency_budget(probe, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=router_probe.__name__)
    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 0.001)
    monkeypatch.setattr(routing, "_GATE_TOTAL_SECONDS", 0.002)
    monkeypatch.setattr(router_probe, "_PROBE_TIMEOUT_SECONDS", 1)
    probe.provider.delay = 0.02
    await router_probe.run_router_startup_probe()

    assert probe.built == [probe.settings]
    assert probe.provider.calls == [{
        "model": "router-model",
        "system": routing.with_deployment_identity(routing._SYSTEM_PROMPT),
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [routing._DELEGATE_TOOL],
        "max_tokens": routing._DIRECT_MAX_TOKENS,
    }]
    assert probe.provider.closed
    probe.client.aclose.assert_awaited_once()
    _outcome(caplog, "passed")


@pytest.mark.parametrize("stop", ["completed", "tool_calls", "tool_use"])
async def test_valid_completed_delegate_is_compatible(probe, caplog, stop):
    caplog.set_level(logging.INFO, logger=router_probe.__name__)
    probe.provider.events = [StreamComplete(LLMResponse(
        content="", stop_reason=stop,
        tool_calls=[ToolCall(id="call_probe", name="delegate", input={"reason": "Needs tools"})],
    ))]
    await router_probe.run_router_startup_probe()
    _outcome(caplog, "passed")
    probe.client.aclose.assert_awaited_once()


@pytest.mark.parametrize("response", [
    LLMResponse(content="", stop_reason="completed"),
    LLMResponse(content="   ", stop_reason="stop"),
    LLMResponse(content="Partial", stop_reason=None),
    LLMResponse(content="Partial", stop_reason="length"),
    LLMResponse(content="Partial", stop_reason="max_tokens"),
    LLMResponse(content="Blocked", stop_reason="content_filter"),
    LLMResponse(content="Unexpected", stop_reason="in_progress"),
    LLMResponse(content="Hello", stop_reason="completed", tool_calls=[
        ToolCall(id="x", name="wrong_tool", input={"reason": "No"})
    ]),
    LLMResponse(content="", stop_reason="completed", tool_calls=[
        ToolCall(id="x", name="delegate", input={})
    ]),
    LLMResponse(content="", stop_reason="completed", tool_calls=[
        ToolCall(id="", name="delegate", input={"reason": "No"})
    ]),
    LLMResponse(content="", stop_reason="completed", tool_calls=[
        ToolCall(id="x", name="delegate", input={"reason": 1})
    ]),
    LLMResponse(content="", stop_reason="completed", tool_calls=[
        ToolCall(id="x", name="delegate", input={"reason": "No"}, repaired=True)
    ]),
    LLMResponse(content="", stop_reason="completed", tool_calls=[
        ToolCall(id="x", name="delegate", input={"reason": "No"}, parse_error="private parse text")
    ]),
])
async def test_invalid_completed_output_is_not_a_pass(probe, caplog, response):
    probe.provider.events = [StreamComplete(response)]
    await router_probe.run_router_startup_probe()
    _outcome(caplog, "unverified", reason="invalid_response")
    probe.client.aclose.assert_awaited_once()


async def test_stream_without_complete_does_not_pass(probe, caplog):
    probe.provider.events = [StreamTextDelta("Hello")]
    await router_probe.run_router_startup_probe()
    _outcome(caplog, "unverified", reason="invalid_response")
    assert probe.provider.closed


async def test_timeout_closes_stream_and_entire_client(probe, monkeypatch, caplog):
    probe.provider.events = []
    probe.provider.hold_open = True
    monkeypatch.setattr(router_probe, "_PROBE_TIMEOUT_SECONDS", 0.02)
    await asyncio.wait_for(router_probe.run_router_startup_probe(), 1)
    _outcome(caplog, "unverified", reason="timeout")
    assert probe.provider.closed
    probe.client.aclose.assert_awaited_once()


async def test_cancellation_propagates_after_cleanup(probe, caplog):
    probe.provider.events = []
    probe.provider.hold_open = True
    task = asyncio.create_task(router_probe.run_router_startup_probe())
    await asyncio.wait_for(probe.provider.waiting.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert probe.provider.closed
    probe.client.aclose.assert_awaited_once()
    assert not [r for r in caplog.records if r.name == router_probe.__name__]


@pytest.mark.parametrize("shutdown", [False, True], ids=["operation_timeout", "shutdown_cancel"])
async def test_stalled_stream_finally_cannot_block_probe_cleanup(probe, monkeypatch, caplog, shutdown):
    started = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    stopped = asyncio.Event()
    readers = []

    class StalledCloseProvider:
        async def stream(self, **kwargs):
            readers.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
                yield StreamTextDelta("unreachable")
            finally:
                closing.set()
                try:
                    # Like Anton closing the HTTP stream while unwinding a
                    # cancelled read: the first cancellation starts this wait.
                    await release_close.wait()
                finally:
                    stopped.set()

    probe.client.router_provider = StalledCloseProvider()
    monkeypatch.setattr(router_probe, "_PROBE_TIMEOUT_SECONDS", 1 if shutdown else 0.02)
    monkeypatch.setattr(router_probe, "_CLOSE_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(router_probe.run_router_startup_probe())
    try:
        await asyncio.wait_for(started.wait(), 1)
        if shutdown:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.3)
        assert task in done, "probe waits indefinitely for the cancelled stream's finally"
        if shutdown:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
            _outcome(caplog, "unverified", reason="timeout")
        assert closing.is_set()
        assert stopped.is_set()
        assert all(reader.done() for reader in readers)
        probe.client.aclose.assert_awaited_once()
    finally:
        # The red implementation is stuck in its finally; let it unwind so a
        # failed regression does not leave a hanging task in the test runner.
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("status,reason", [
    (400, "provider_refused"), (401, "provider_refused"),
    (403, "provider_refused"), (404, "provider_refused"),
    (429, "provider_unavailable"), (503, "provider_unavailable"),
])
async def test_provider_errors_are_safe_and_never_infer_unsupported(
    probe, caplog, owned_logger, status, reason,
):
    sentinel = "private-provider-body-and-key"
    request = httpx.Request("POST", f"https://{sentinel}.example/v1/responses")
    probe.provider.error = openai.APIStatusError(
        sentinel, response=httpx.Response(status, request=request), body={"error": sentinel},
    )
    logged = owned_logger(router_probe.__name__, level=logging.WARNING)
    await router_probe.run_router_startup_probe()
    record = _outcome(caplog, "unverified", reason=reason)
    assert (record.status_code, record.error_type) == (status, "APIStatusError")
    assert sentinel not in repr(record.__dict__)
    assert "unsupported" not in record.getMessage()
    # The serialized line, as the server writes it, carries the values.
    output = logged.output()
    assert_log_fields(
        output,
        event="router_startup_probe_unverified",
        log_level="WARNING",
        outcome="unverified",
        reason=reason,
        provider="openai_compatible",
        error_type="APIStatusError",
        status_code=status,
    )
    assert "%s" not in output
    assert sentinel not in output
    assert len(probe.provider.calls) == 1
    probe.client.aclose.assert_awaited_once()


@pytest.mark.parametrize("stage", ["load", "build", "missing_model"])
async def test_configuration_failures_are_nonfatal(probe, monkeypatch, caplog, stage):
    sentinel = "private-configuration-value"
    if stage == "load":
        monkeypatch.setattr(router_probe, "run_db", AsyncMock(side_effect=ValueError(sentinel)))
    elif stage == "build":
        def fail(**kwargs):
            raise ValueError(sentinel)
        monkeypatch.setattr(router_probe, "build_llm_client", fail)
    else:
        probe.settings.resolved_gate_model = None
    await router_probe.run_router_startup_probe()
    record = _outcome(caplog, "unverified", reason="configuration_unavailable")
    assert sentinel not in repr(record.__dict__)
    assert not probe.provider.calls


async def test_client_cleanup_is_bounded(probe, monkeypatch, caplog):
    async def hang():
        await asyncio.Event().wait()
    probe.client.aclose.side_effect = hang
    monkeypatch.setattr(router_probe, "_CLOSE_TIMEOUT_SECONDS", 0.01)
    await asyncio.wait_for(router_probe.run_router_startup_probe(), 1)
    probe.client.aclose.assert_awaited_once()


async def test_settings_load_uses_global_scope_and_no_conversation(probe, monkeypatch):
    from cowork.db.units import run_db
    from cowork.services.conversations import ConversationService

    calls = []

    class GlobalSettings:
        def __init__(self, session, scope):
            assert scope is LOCAL_SCOPE
            assert not hasattr(session, "scope")
            calls.append(scope)

        def load(self):
            return probe.settings

    def no_conversation(*args, **kwargs):
        pytest.fail("startup probe must not create a conversation")

    monkeypatch.setattr(router_probe, "run_db", run_db)
    monkeypatch.setattr(router_probe, "SettingService", GlobalSettings)
    monkeypatch.setattr(ConversationService, "__init__", no_conversation)
    await router_probe.run_router_startup_probe()
    assert calls == [LOCAL_SCOPE]
    assert probe.built == [probe.settings]


async def test_persisted_global_settings_override_environment_and_ignore_tenant_rows(probe, monkeypatch):
    from sqlmodel import Session, create_engine

    from cowork.common.settings.user_settings import Provider, UserSettings
    from cowork.db.scoped import ScopedSession
    from cowork.models.setting import Setting

    monkeypatch.setenv("ROUTER_MODEL", "environment-router")
    monkeypatch.setattr(UserSettings, "model_config", {**UserSettings.model_config, "env_file": None})
    assert UserSettings(
        router_provider=Provider.OPENAI_COMPATIBLE,
        openai_base_url="https://global-router.example/v1",
    ).router_model == "environment-router"
    engine = create_engine("sqlite://")
    Setting.__table__.create(engine)
    try:
        with Session(engine) as session:
            session.add_all([
                Setting(key="router_provider", value="openai_compatible"),
                Setting(key="router_model", value="global-router"),
                Setting(key="router_reasoning_effort", value="low"),
                Setting(key="openai_base_url", value="https://global-router.example/v1"),
                Setting(key="router_model", value="tenant-router", scope="org", org_id="test-org"),
            ])
            session.commit()

            async def read(fn, *, scope):
                assert scope is LOCAL_SCOPE
                return fn(ScopedSession(session, scope))

            monkeypatch.setattr(router_probe, "run_db", read)
            await router_probe.run_router_startup_probe()
        (settings,) = probe.built
        assert settings.router_model == "global-router"
        assert settings.resolved_gate_model == "global-router"
        assert settings.router_reasoning_effort == "low"
        assert probe.provider.calls[0]["model"] == "global-router"
    finally:
        engine.dispose()
