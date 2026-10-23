"""The gate and first take use the same trusted identity as Anton's turn."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.deployment_identity import identity_for_turn
from cowork.handlers import first_take, response_routing, responses
from cowork.schemas.responses import ResponsesRequest
from cowork.server import create_app


PRODUCT = "Acme & Co's $workbench"
# Exact prompts at staging 8b1ecf4, before deployment identity support. These
# compatibility assertions exercise provider requests, not just the constants.
DEFAULT_GATE_SHA256 = "e77c722af7fd7011d9ffc717120760c7cb369b6612bd533addb2b90628b66f6d"
DEFAULT_TAKE_SHA256 = "423cf10dd88589f84adadb158b7851a89cfd32da4bad967c473062f07a51427a"


@pytest.fixture(autouse=True)
def isolated_identity(monkeypatch):
    for name in ("COWORK_DEPLOYMENT_KIND", "COWORK_DEPLOYMENT_BRAND_FILE"):
        monkeypatch.delenv(name, raising=False)
    get_app_settings.cache_clear()
    # The shadow probe is unrelated to the user-visible gate response.
    monkeypatch.setattr(responses, "_spawn_jev_shadow_probe", lambda *args, **kwargs: None)
    yield
    get_app_settings.cache_clear()


def _initialize(monkeypatch, tmp_path, *, branded: bool, name: str = PRODUCT):
    if branded:
        pack = tmp_path / "pack.json"
        pack.write_text(json.dumps({"name": name}), encoding="utf-8")
        monkeypatch.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
        monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(pack))
        get_app_settings.cache_clear()
    else:
        def no_new_prompt_api(*args, **kwargs):
            pytest.fail("unset hosts must not call Anton's deployment prompt API")

        monkeypatch.setattr("anton.core.llm.identity.product_lines", no_new_prompt_api)
    create_app()
    if branded:
        # All fast calls must use create_app's snapshot, just like the harness.
        pack.write_text('{"name":"Changed after startup"}', encoding="utf-8")


class _Provider:
    def __init__(self, text: str, *, delegate: bool = False):
        self.text = text
        self.delegate = delegate
        self.calls: list[dict] = []

    async def stream(self, **kwargs):
        from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamToolUseStart

        self.calls.append(kwargs)
        if self.delegate:
            yield StreamToolUseStart(id="delegate-1", name="delegate")
        else:
            yield StreamTextDelta(text=self.text)
            yield StreamComplete(response=SimpleNamespace(tool_calls=[], stop_reason="end_turn"))


def _binding(provider):
    return response_routing.RouterBinding(provider=provider, model="test-fast-model", label="test")


def _assert_prompt(request, *, branded: bool, default_sha256: str):
    prompt = request["system"]
    if not branded:
        assert hashlib.sha256(prompt.encode()).hexdigest() == default_sha256
        return
    from anton.core.llm.identity import product_lines

    # Use Anton's canonical facts so this check also catches divergence when
    # its product guidance changes, including install/update/editions advice.
    canonical = "\n".join(product_lines("web", identity_for_turn()))
    assert canonical in prompt
    assert "Changed after startup" not in prompt
    assert "Member override" not in prompt
    assert "recall_skill is available only to the full agent" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("branded", [False, True])
@pytest.mark.parametrize("stream,interactive", [(False, True), (True, True), (True, False)])
async def test_actual_direct_handler_uses_startup_identity_without_building_anton(
    monkeypatch, tmp_path, granted_product_permissions, branded, stream, interactive
):
    _initialize(monkeypatch, tmp_path, branded=branded)
    answer = f"Hello from {PRODUCT if branded else 'Cowork'}."
    provider = _Provider(answer)
    monkeypatch.setattr(responses, "_settings_binding", lambda: _binding(provider))
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: False)

    def no_harness(*args, **kwargs):
        pytest.fail("a direct answer must not initialize Anton")

    monkeypatch.setattr(responses, "get_harness", no_harness)
    # The scheduler uses interactive=False and stream=True on this same path.
    handler = responses.ResponsesHandler(interactive=interactive)
    result = await handler.handle(ResponsesRequest(
        input="Hello",
        stream=stream,
        trace_metadata={"deployment_identity": "Member override"},
    ))

    output = "".join([frame async for frame in result]) if stream else result.model_dump_json()
    assert answer in output
    assert handler.harness is None
    assert len(provider.calls) == 1
    _assert_prompt(provider.calls[0], branded=branded, default_sha256=DEFAULT_GATE_SHA256)


@pytest.mark.asyncio
@pytest.mark.parametrize("branded", [False, True])
async def test_real_routing_starts_a_first_take_with_the_same_startup_identity(
    monkeypatch, tmp_path, branded
):
    _initialize(monkeypatch, tmp_path, branded=branded)
    gate_provider = _Provider("", delegate=True)
    take_provider = _Provider("I'll look into that.")
    monkeypatch.setattr(responses, "_settings_binding", lambda: _binding(gate_provider))
    monkeypatch.setattr(
        responses, "_first_take_binding", lambda minted, *, binding_factory=None: _binding(take_provider)
    )
    monkeypatch.setattr(responses, "_first_take_enabled", lambda: True)
    handler = responses.ResponsesHandler()
    handler.harness_name = "anton"
    decision, _ = await handler._route_request(
        conversation_id=None,
        gate_rows=[],
        harness_input=[{"type": "text", "text": "How do I update this installation?"}],
        has_attachments=False,
        has_disabled_connections=False,
        first_take=True,
    )
    assert decision.route == response_routing.DELEGATED_AGENTIC
    assert isinstance(handler.first_take, first_take.FirstTake)
    await handler.first_take._writer
    try:
        assert len(gate_provider.calls) == len(take_provider.calls) == 1
        _assert_prompt(take_provider.calls[0], branded=branded, default_sha256=DEFAULT_TAKE_SHA256)
        _assert_prompt(gate_provider.calls[0], branded=branded, default_sha256=DEFAULT_GATE_SHA256)
        assert "tools" not in take_provider.calls[0]
        assert "never state a fact" in " ".join(take_provider.calls[0]["system"].split())
    finally:
        handler.first_take.cancel()


@pytest.mark.parametrize("name", ["Acme Workbench", PRODUCT, "O'Brien's [R&D]+ (v2)$ #tools", "AI"])
@pytest.mark.parametrize("answer", [
    "I'm not familiar with a product called {name}.",
    "{name} does not appear to exist.",
    "I could not find any official {name} application.",
    "There is no verified {name} desktop app.",
])
async def test_configured_product_denials_delegate_and_withdraw_first_takes(monkeypatch, tmp_path, name, answer):
    _initialize(monkeypatch, tmp_path, branded=True, name=name)
    # The startup name remains authoritative after _initialize changes the
    # mounted pack. Match case-insensitively, as the stock product guard does.
    answer = answer.format(name=name.lower())
    history = [{"role": "user", "content": "What is this product?"}]
    decision = await response_routing.decide_route(
        history=history, has_non_text_input=False, has_attachments=False,
        has_disabled_connections=False, binding=_binding(_Provider(answer)),
    )

    take = first_take.FirstTake(_binding(_Provider(answer)), history)
    outcome = await take._write()
    assert (decision.route, outcome) == (response_routing.DELEGATED_AGENTIC, "withdrawn")
    assert decision.reason == "router_answer_denied_product"
    assert not decision.text
    frame = take._queue.get_nowait()
    assert "response.first_take.cleared" in frame
    assert take._queue.empty()


@pytest.mark.parametrize("name", ["Acme Workbench", PRODUCT, "O'Brien's [R&D]+ (v2)$ #tools", "AI"])
def test_configured_product_guard_keeps_the_existing_narrow_policy(monkeypatch, tmp_path, name):
    _initialize(monkeypatch, tmp_path, branded=True, name=name)
    for text in (
        f"{name} doesn't support that file type yet.",
        f"I couldn't find that setting in {name}.",
        f"I can't locate that file in your {name} workspace.",
        f"I cannot verify the checksum of the {name} download you pasted.",
        "I'm not familiar with a product called Changed after startup.",
        "I'm not familiar with a product called O'Brien's R&D v2 tools.",
        # A short name matches only as a whole word, not inside "explain".
        "I'm not familiar with that library, but I can explain the general approach.",
    ):
        assert response_routing.denies_our_product(text) is None, text
        assert first_take._shippable(text), text


@pytest.mark.parametrize("answer,route,reason", [
    ("Hi, I am Acme Copilot, made by MindsHub.", response_routing.DIRECT_CONTEXT, "router_direct_response"),
    ("I am Copilot.", response_routing.DELEGATED_AGENTIC, "router_answer_named_foreign_product"),
])
async def test_a_configured_name_holding_a_foreign_product_is_still_ours(
    monkeypatch, tmp_path, answer, route, reason
):
    _initialize(monkeypatch, tmp_path, branded=True, name="Acme Copilot")
    decision = await response_routing.decide_route(
        history=[{"role": "user", "content": "Who are you?"}], has_non_text_input=False,
        has_attachments=False, has_disabled_connections=False, binding=_binding(_Provider(answer)),
    )

    assert (decision.route, decision.reason) == (route, reason)
    assert first_take._shippable(answer) is (route == response_routing.DIRECT_CONTEXT)


def test_without_opt_in_the_denial_guard_needs_no_startup_or_new_anton_api(monkeypatch, tmp_path):
    # A file setting alone does not opt in, and the default guard can run
    # before create_app, including with an older Anton installed.
    monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(tmp_path / "missing.json"))
    get_app_settings.cache_clear()
    monkeypatch.delattr("anton.core.llm.identity.DeploymentIdentity")

    assert response_routing.denies_our_product("I'm not familiar with a product called Cowork.")
    assert response_routing.denies_our_product("I'm not familiar with a product called Acme Workbench.") is None
