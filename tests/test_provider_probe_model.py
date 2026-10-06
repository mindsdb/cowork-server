"""The MindsHub connectivity probe must use a tier-universal model (ENG-576).

MindsHub gates paid models per plan tier: a free-tier key gets a 403 for
haiku/sonnet/etc. The Settings health probe (`ping_provider`) and onboarding
validation (`validate_minds`) used to POST `CODING_MODEL_DEFAULTS["minds_cloud"]`
= "haiku" (paid) → free-tier accounts saw "MindsHub failed its last test" /
"Invalid API key" even though chat worked on mindshub_air. Both must now probe
`mindshub_air` (the free baseline, present in every tier).
"""
import asyncio

import httpx
import pytest

import cowork.services.providers as providers
from cowork.handlers import turn_errors as te
from cowork.schemas.settings import ProbeDenialCode
from cowork.services.providers import (
    MINDS_PROBE_MODEL,
    MINDS_REQUEST_KIND_HEADER,
    MINDS_REQUEST_KIND_PROBE,
    GatewayDenial,
    ProviderPing,
    is_minds_host,
    ping_provider,
    ping_providers,
    validate_minds,
    validate_provider,
)


class _CapturingClient:
    """Fake httpx.AsyncClient that records the JSON body of the probe POST."""

    captured: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        _CapturingClient.captured = {"url": url, "json": json, "headers": headers or {}}
        return _Resp(200)

    async def get(self, url, headers=None):
        return _Resp(200)


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code

    def json(self):
        return {"choices": [{"message": {"content": "pong"}}]}


def _patch(monkeypatch):
    _CapturingClient.captured = {}
    monkeypatch.setattr(providers.httpx, "AsyncClient", _CapturingClient)


def test_probe_model_is_tier_universal():
    # Guard the constant itself — this is the whole fix.
    assert MINDS_PROBE_MODEL == "mindshub_air"


def test_ping_provider_probes_universal_model(monkeypatch):
    _patch(monkeypatch)
    status = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"})).status
    assert status == "ok"
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_ping_provider_ignores_configured_paid_model(monkeypatch):
    # Even if a (paid) model is passed, the connectivity probe uses the
    # universal one — the dot reflects reachability, not model availability.
    _patch(monkeypatch)
    asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x", "model": "sonnet"}))
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_validate_minds_probes_universal_model(monkeypatch):
    _patch(monkeypatch)
    result = asyncio.run(validate_minds("mdb_x", "https://api.mindshub.ai"))
    assert result.get("ok") is True
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_ping_provider_missing_key_still_fails_fast(monkeypatch):
    _patch(monkeypatch)
    ping = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": ""}))
    status, detail = ping.status, ping.detail
    assert status == "fail" and "key" in detail.lower()


def test_ping_minds_cloud_surfaces_provider_message(monkeypatch):
    # minds-cloud is the one provider routed through _chat_probe (real chat
    # completions), so its failures carry the gateway's actionable reason
    # (wallet/allowance/model). The dot detail must show it, not a bare
    # "HTTP 429" (ENG-1145 review, ENG-576).
    class _FailResp:
        status_code = 429

        def json(self):
            return {"error": {"message": "Wallet allowance exhausted. Top up to continue."}}

    class _FailClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return _FailResp()

    monkeypatch.setattr(providers.httpx, "AsyncClient", _FailClient)
    ping = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"}))
    status, detail = ping.status, ping.detail
    assert status == "fail"
    assert "HTTP 429" in detail
    assert "Wallet allowance exhausted" in detail


# ── The omitted-model default on a MindsHub host ──────────────────────
#
# Onboarding validates the MindsHub key through validate_provider. The generic
# openai-compatible default ("gpt-5.5") is not a MindsHub alias, so it 404s
# there, and the recommended MindsHub model is paid, so it 402s for an account
# whose wallet is empty. Both come back looking like a bad key, which routed a
# brand-new user to bring-your-own-key holding a valid MindsHub key.


def test_omitted_model_on_minds_host_probes_free_model(monkeypatch):
    _patch(monkeypatch)
    result = asyncio.run(
        validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", None)
    )
    assert result.get("ok") is True
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_empty_model_on_minds_host_probes_free_model(monkeypatch):
    # The client sends `model` as an optional field, so an empty string arrives
    # as often as a missing key. Both mean "no model was chosen".
    _patch(monkeypatch)
    asyncio.run(validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", ""))
    assert _CapturingClient.captured["json"]["model"] == "mindshub_air"


def test_explicit_model_on_minds_host_is_sent_as_asked(monkeypatch):
    # The negative case that keeps the default honest: a user validating one
    # specific model must not be told a different model passed.
    _patch(monkeypatch)
    asyncio.run(
        validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", "sonnet")
    )
    assert _CapturingClient.captured["json"]["model"] == "sonnet"


def test_omitted_model_off_minds_host_keeps_generic_default(monkeypatch):
    # A real openai-compatible endpoint is unchanged by this.
    _patch(monkeypatch)
    asyncio.run(
        validate_provider("openai-compatible", "sk_x", "https://api.openai.com/v1", None)
    )
    assert _CapturingClient.captured["json"]["model"] == "gpt-5.5"


def test_is_minds_host_matches_the_host_not_a_substring():
    for url in (
        "https://api.mindshub.ai/v1",
        "https://api.staging.mindshub.ai",
        "https://api-pr-12.dev.mindshub.ai/v1",
        "https://mindshub.ai",
        "https://mdb.ai/api/v1",
        "https://llm.mdb.ai",
        "api.mindshub.ai/v1",  # no scheme, as a stored setting can be
    ):
        assert is_minds_host(url) is True, url

    for url in (
        "",
        None,
        "https://api.openai.com/v1",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        # A lookalike domain and a redirect-style parameter both defeat a
        # substring test, which is why this compares the parsed hostname.
        "https://mindshub.ai.example.test/v1",
        "https://evil-mindshub.ai/v1",
        "https://example.test/r?u=https://api.mindshub.ai/v1",
        # Unbalanced brackets: urlparse raises ValueError reading .hostname on
        # these, and this predicate is called outside the caller's except, so an
        # unguarded parse would answer 500 instead of ok:false. See the try in
        # is_minds_host.
        "https://[",
        "https://a[b].mindshub.ai/v1",
        "[",
        "https://]",
    ):
        assert is_minds_host(url) is False, url


def test_unparseable_base_url_is_a_failed_probe_not_a_500():
    # The base URL is free text off the openai-compatible card, and
    # validate_provider runs is_minds_host before validate_openai_compatible's
    # except can catch anything. A ValueError here leaves the service layer and
    # FastAPI turns it into a 500. No client patch: httpx rejects the URL locally,
    # so this makes no network call.
    result = asyncio.run(validate_provider("openai-compatible", "k", "https://[", None))
    assert result["ok"] is False


def test_minds_probe_caps_its_token_budget_and_keeps_the_host_path(monkeypatch):
    # Asserted on the wire, because both are easy to lose: without max_tokens the
    # probe asks for a full-length completion that MindsHub bills to the included
    # allowance on every onboarding attempt, and the URL is the only thing that
    # catches a base whose chat path is derived wrongly.
    _patch(monkeypatch)
    asyncio.run(
        validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", None)
    )
    assert _CapturingClient.captured["json"]["max_tokens"] == 20
    assert _CapturingClient.captured["url"] == "https://api.mindshub.ai/v1/chat/completions"


def test_a_non_minds_probe_sends_no_token_cap(monkeypatch):
    # The cap is MindsHub-only on purpose. OpenAI's reasoning models reject
    # max_tokens and want max_completion_tokens, and o3/o4-mini are both in
    # RECOMMENDED_MODELS, so sending it to any endpoint would report a working key
    # as invalid for exactly the models this ticket is about.
    _patch(monkeypatch)
    asyncio.run(validate_provider("openai-compatible", "sk_x", "https://api.openai.com/v1", "o3"))
    assert "max_tokens" not in _CapturingClient.captured["json"]


# ── The connectivity-probe marker for the Traces list ──────
#
# Every MindsHub probe hits the real /chat/completions path under the user's key,
# so it persists as an ordinary trace and is noise in the list. Each sender
# stamps X-Minds-Request-Kind: probe so the list can hide them by default. The
# marker is MindsHub-only — our own header, sent to no arbitrary endpoint.


def test_ping_provider_stamps_probe_kind_header(monkeypatch):
    _patch(monkeypatch)
    asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"}))
    assert _CapturingClient.captured["headers"].get(MINDS_REQUEST_KIND_HEADER) == MINDS_REQUEST_KIND_PROBE


def test_validate_minds_stamps_probe_kind_header(monkeypatch):
    _patch(monkeypatch)
    asyncio.run(validate_minds("mdb_x", "https://api.mindshub.ai"))
    assert _CapturingClient.captured["headers"].get(MINDS_REQUEST_KIND_HEADER) == MINDS_REQUEST_KIND_PROBE


def test_minds_fallback_stamps_probe_kind_header(monkeypatch):
    # The omitted-model MindsHub fallback routes through validate_openai_compatible,
    # so it must carry the marker too.
    _patch(monkeypatch)
    asyncio.run(validate_provider("openai-compatible", "mdb_x", "https://api.mindshub.ai/v1", None))
    assert _CapturingClient.captured["headers"].get(MINDS_REQUEST_KIND_HEADER) == MINDS_REQUEST_KIND_PROBE


def test_non_minds_probe_does_not_stamp_probe_kind_header(monkeypatch):
    # The marker is our own header and means nothing to a third-party endpoint, so
    # a non-MindsHub probe must not send it.
    _patch(monkeypatch)
    asyncio.run(validate_provider("openai-compatible", "sk_x", "https://api.openai.com/v1", "gpt-4o"))
    assert MINDS_REQUEST_KIND_HEADER not in _CapturingClient.captured["headers"]


# ── The probe names the gateway's reason ────────────────────────────────────
#
# The probe used to keep only "HTTP <status>: <message>" and drop the response
# headers, so the Settings notice matched on "429" and called a velocity limit,
# the free-Air fuse and a spent allowance all "No credits available". The probe
# now carries the reason the gateway named, origin-checked by turn_errors.

_RESET_AT = "2026-09-25T00:00:00Z"
_GATEWAY_MESSAGE = "The gateway refused this request."


def _serve_gateway(monkeypatch, status_code, response_headers, *, content=None, body=None):
    """A client whose POST answers with a real httpx.Response.

    The response carries a request built from the URL the probe actually
    posted to, as httpx's own does, so the origin check sees the probe's host.
    ``body`` replaces the default JSON error body, and ``content`` sends raw
    bytes instead of JSON.
    """
    if content is None:
        payload = {"json": body if body is not None else {"error": {"message": _GATEWAY_MESSAGE}}}
    else:
        payload = {"content": content}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return httpx.Response(
                status_code,
                headers=response_headers,
                request=httpx.Request("POST", url),
                **payload,
            )

    monkeypatch.setattr(providers.httpx, "AsyncClient", _Client)


def _gateway_base() -> str:
    """The MindsHub URL this install treats as its gateway."""
    return f"https://{te._configured_minds_host()}"


@pytest.mark.parametrize(
    ("status_code", "reason", "extra_headers", "reset_at"),
    [
        (429, "rate_limited", {"Retry-After": "12"}, None),
        (429, "free_air_daily_spend_fuse_exceeded", {"X-MindsHub-Reset-At": _RESET_AT}, _RESET_AT),
        (429, "included_allowance_exhausted", {"X-MindsHub-Reset-At": _RESET_AT}, _RESET_AT),
        (402, "wallet_empty", {}, None),
        (503, "policy_unavailable", {}, None),
    ],
)
def test_a_gateway_refusal_carries_its_reason(monkeypatch, status_code, reason, extra_headers, reset_at):
    _serve_gateway(monkeypatch, status_code, {"X-MindsHub-Reason": reason, **extra_headers})

    ping = asyncio.run(
        ping_provider({"type": "minds-cloud", "apiKey": "mdb_x", "mindsUrl": _gateway_base()})
    )

    assert ping.status == "fail"
    assert ping.detail == f"HTTP {status_code}: {_GATEWAY_MESSAGE}"
    assert ping.denial == GatewayDenial(reason=reason, reset_at=reset_at)


def test_a_reason_from_another_host_is_not_trusted(monkeypatch):
    # Anyone can send X-MindsHub-Reason. Only the configured gateway's counts,
    # so a probe aimed elsewhere keeps its detail and names no reason.
    _serve_gateway(
        monkeypatch, 429,
        {"X-MindsHub-Reason": "free_air_daily_spend_fuse_exceeded", "X-MindsHub-Reset-At": _RESET_AT},
    )

    ping = asyncio.run(
        ping_provider(
            {"type": "minds-cloud", "apiKey": "mdb_x", "mindsUrl": "https://attacker.example"}
        )
    )

    assert ping.denial is None
    assert ping.status == "fail"
    assert ping.detail == f"HTTP 429: {_GATEWAY_MESSAGE}"


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (429, "included_allowance_exhausted"),
        (429, "free_air_daily_spend_fuse_exceeded"),
        (402, "wallet_empty"),
    ],
)
def test_a_gateway_refusal_with_its_reason_header_stripped_classifies_from_the_body_code(
    monkeypatch, status_code, reason
):
    # A proxy on the way can drop X-MindsHub-Reason. The gateway sets its body
    # `code` to the same value, so the probe still names the reason from the
    # configured gateway host's body.
    _serve_gateway(
        monkeypatch, status_code, {},
        body={"error": {"message": _GATEWAY_MESSAGE, "code": reason}},
    )

    ping = asyncio.run(
        ping_provider({"type": "minds-cloud", "apiKey": "mdb_x", "mindsUrl": _gateway_base()})
    )

    assert ping.status == "fail"
    assert ping.detail == f"HTTP {status_code}: {_GATEWAY_MESSAGE}"
    assert ping.denial == GatewayDenial(reason=reason, reset_at=None)


def test_a_body_code_from_another_host_is_not_trusted(monkeypatch):
    # The body is the other host's to write, and unlike the header it counts
    # only from the configured gateway host.
    _serve_gateway(
        monkeypatch, 429, {},
        body={"error": {"message": _GATEWAY_MESSAGE, "code": "included_allowance_exhausted"}},
    )

    ping = asyncio.run(
        ping_provider(
            {"type": "minds-cloud", "apiKey": "mdb_x", "mindsUrl": "https://attacker.example"}
        )
    )

    assert ping.denial is None
    assert ping.status == "fail"
    assert ping.detail == f"HTTP 429: {_GATEWAY_MESSAGE}"


def test_a_non_json_body_still_classifies_from_the_reason_header(monkeypatch):
    # An HTML error page cannot be parsed for a body code. The probe must
    # still read the reason header rather than lose the denial altogether.
    _serve_gateway(
        monkeypatch, 429, {"X-MindsHub-Reason": "rate_limited"},
        content=b"<html><body>Too Many Requests</body></html>",
    )

    ping = asyncio.run(
        ping_provider({"type": "minds-cloud", "apiKey": "mdb_x", "mindsUrl": _gateway_base()})
    )

    assert ping.status == "fail"
    assert ping.denial == GatewayDenial(reason="rate_limited", reset_at=None)


def test_a_passing_probe_names_no_reason(monkeypatch):
    _patch(monkeypatch)
    ping = asyncio.run(ping_provider({"type": "minds-cloud", "apiKey": "mdb_x"}))
    assert ping == ProviderPing(status="ok", detail="HTTP 200", denial=None)


def test_ping_providers_reports_denials_only_for_the_types_that_have_one(monkeypatch):
    async def _fake_ping(p):
        if p["type"] == "minds-cloud":
            return ProviderPing(
                status="fail", detail="HTTP 429",
                denial=GatewayDenial(reason="rate_limited", reset_at=None),
            )
        return ProviderPing(status="ok", detail="HTTP 200")

    monkeypatch.setattr(providers, "ping_provider", _fake_ping)

    results = asyncio.run(ping_providers([{"type": "minds-cloud"}, {"type": "anthropic"}]))

    assert results.statuses == {"minds-cloud": "fail", "anthropic": "ok"}
    assert results.details == {"minds-cloud": "HTTP 429", "anthropic": "HTTP 200"}
    assert results.denials == {"minds-cloud": GatewayDenial(reason="rate_limited", reset_at=None)}


def test_the_last_card_of_a_type_decides_its_denial_too(monkeypatch):
    # Two cards of one type share one slot and the last status wins it. An
    # earlier card's reason left behind would sit beside a status it does not
    # explain, and the Settings notice would name a stop that did not happen.
    answers = iter([
        ProviderPing(
            status="fail", detail="HTTP 429",
            denial=GatewayDenial(reason="free_air_daily_spend_fuse_exceeded", reset_at=_RESET_AT),
        ),
        ProviderPing(status="fail", detail="HTTP 500"),
    ])

    async def _fake_ping(p):
        return next(answers)

    monkeypatch.setattr(providers, "ping_provider", _fake_ping)

    results = asyncio.run(ping_providers([{"type": "minds-cloud"}, {"type": "minds-cloud"}]))

    assert results.statuses == {"minds-cloud": "fail"}
    assert results.details == {"minds-cloud": "HTTP 500"}
    assert results.denials == {}


def test_the_probe_vocabulary_matches_the_wire_contract():
    # The classifier's allowlist and the response model's Literal are written
    # twice, in two layers. A reason in one and not the other would either be
    # dropped silently or fail validation and 500 the whole Settings probe.
    from typing import get_args

    assert set(get_args(ProbeDenialCode)) == te.PROBE_DENIAL_REASONS
