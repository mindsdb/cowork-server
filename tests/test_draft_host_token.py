"""The draft-preview token: one wire format, checked field by field."""
from __future__ import annotations

import base64

import pytest

from cowork.draft_host.token import (
    DRAFT_TTL_SECONDS,
    DraftTokenError,
    issue_draft_token,
    verify_draft_token,
)

from draft_host_support import forge_identity, sign_claims

USER = "11111111-1111-4111-8111-111111111111"
ORG = "22222222-2222-4222-8222-222222222222"
PROJECT = "33333333-3333-4333-8333-333333333333"
ARTIFACT = "44444444444444444444444444444444"
KEYS = {"k1": "draft-secret"}
NOW = 1_700_000_000


def _issue(**overrides) -> str:
    fields = dict(env="staging", sub=USER, org=ORG, proj=PROJECT, art=ARTIFACT, keys=KEYS, kid="k1", now=NOW)
    fields.update(overrides)
    return issue_draft_token(**fields)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _claims(**overrides) -> dict:
    claims = {
        "p": "draft", "env": "staging", "sub": USER, "org": ORG, "proj": PROJECT,
        "art": ARTIFACT, "exp": NOW + DRAFT_TTL_SECONDS, "kid": "k1",
    }
    claims.update(overrides)
    return claims


def _reason(token: str, **kwargs) -> str:
    kwargs.setdefault("env", "staging")
    kwargs.setdefault("keys", KEYS)
    kwargs.setdefault("now", NOW)
    with pytest.raises(DraftTokenError) as refused:
        verify_draft_token(token, **kwargs)
    return refused.value.reason


def test_the_wire_format_is_pinned_by_a_fixed_vector():
    # If this literal changes, the format changed: the published-artifact
    # token in mindshub_services must change with it.
    assert _issue() == (
        "v1.eyJhcnQiOiI0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NCIsImVudiI6InN0YWdpbmciLCJleHAiOjE3MDAwMDE4MDAsImtpZCI6"
        "ImsxIiwib3JnIjoiMjIyMjIyMjItMjIyMi00MjIyLTgyMjItMjIyMjIyMjIyMjIyIiwicCI6ImRyYWZ0IiwicHJvaiI6IjMzMzMzMzMzLTMzMzMt"
        "NDMzMy04MzMzLTMzMzMzMzMzMzMzMyIsInN1YiI6IjExMTExMTExLTExMTEtNDExMS04MTExLTExMTExMTExMTExMSJ9"
        ".QNm0tQAs4c7pfMfsflQzCy1joIUTm9Y8nZLLRckRVt4"
    )
    assert _issue() == sign_claims(_claims())


def test_a_valid_token_returns_its_claims():
    claims = verify_draft_token(_issue(), env="staging", keys=KEYS, now=NOW + DRAFT_TTL_SECONDS - 1)

    assert claims == _claims()


def test_the_token_lives_thirty_minutes():
    assert DRAFT_TTL_SECONDS == 1800
    assert _reason(_issue(), now=NOW + DRAFT_TTL_SECONDS) == "expired"


def test_a_changed_payload_fails_the_signature():
    forged = forge_identity(_issue(), sub="55555555-5555-4555-8555-555555555555")

    assert _reason(forged) == "signature"


def test_a_token_from_another_environment_is_refused():
    # Dev and PR environments may share one key; `env` is what separates them.
    assert _reason(_issue(env="pr-cowork-server-42")) == "env"
    assert _reason(_issue(env="prod")) == "env"


def test_a_token_for_another_purpose_is_refused():
    assert _reason(sign_claims(_claims(p="published"))) == "purpose"


def test_an_unknown_kid_is_refused_and_a_rotation_keeps_both_keys():
    ring = {"k1": "draft-secret", "k2": "next-secret"}
    assert _reason(_issue(), keys={"k2": "next-secret"}) == "kid"
    assert verify_draft_token(_issue(), env="staging", keys=ring, now=NOW)["kid"] == "k1"
    assert verify_draft_token(_issue(keys=ring, kid="k2"), env="staging", keys=ring, now=NOW)["kid"] == "k2"
    with pytest.raises(DraftTokenError):
        _issue(kid="missing")


@pytest.mark.parametrize(
    "token",
    [
        "",
        "v1",
        "v2.eyJ9.sig",
        "v1..sig",
        "v1.!!!!.sig",
        "v1." + _b64(b"not json") + ".sig",
        "v1." + _b64(b"[1, 2]") + ".sig",
        "v1." + _b64(b"[" * 3000) + ".sig",
        "v1.a.b.c",
        "v1." + "a" * 5000 + ".sig",
    ],
)
def test_malformed_tokens_are_refused(token):
    assert _reason(token) == "malformed"


def test_a_payload_that_exhausts_the_json_parser_is_malformed(monkeypatch):
    # How deep json.loads can nest before RecursionError depends on the
    # interpreter build, so the parser is made to fail the way it does there.
    def too_deep(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("cowork.draft_host.token.json.loads", too_deep)

    assert _reason("v1." + _b64(b"[]") + ".sig") == "malformed"


def test_a_signed_token_without_an_identity_is_refused():
    assert _reason(sign_claims(_claims(sub=""))) == "malformed"
    assert _reason(sign_claims({k: v for k, v in _claims().items() if k != "art"})) == "malformed"


def test_a_boolean_or_string_expiry_is_refused():
    assert _reason(sign_claims(_claims(exp=True))) == "expired"
    assert _reason(sign_claims(_claims(exp=str(NOW + 60)))) == "expired"
