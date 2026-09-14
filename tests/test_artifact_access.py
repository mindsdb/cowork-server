from __future__ import annotations

import json

import httpx
import pytest

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.services.artifact_access import (
    ArtifactAccessUnavailable,
    provision_draft_review_access,
    revoke_draft_review_access,
)


@pytest.mark.asyncio
async def test_provision_draft_review_access_uses_authenticated_org_identity():
    captured = {}

    def handler(request: httpx.Request):
        captured["request"] = request
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        key = await provision_draft_review_access(
            "11111111-1111-1111-1111-111111111111",
            TenantScope(
                org_mode=True,
                user_id="22222222-2222-2222-2222-222222222222",
                org_id="33333333-3333-3333-3333-333333333333",
            ),
            owner_user_id="44444444-4444-4444-4444-444444444444",
            settings=TurnQueueSettings(
                auth_internal_base_url="http://auth.internal",
                auth_internal_secret="secret",
            ),
            client=client,
        )

    request = captured["request"]
    assert key == "artifact/11111111-1111-1111-1111-111111111111"
    assert str(request.url) == "http://auth.internal/v1/internal/artifact-access/"
    assert request.headers["X-Internal-Auth"] == "secret"
    assert b'"artifact_id":"artifact-draft/11111111-1111-1111-1111-111111111111"' in request.content
    assert b'"org_allowed":true' in request.content
    assert b'"owner_keycloak_id":"44444444-4444-4444-4444-444444444444"' in request.content


@pytest.mark.asyncio
async def test_desktop_cannot_invent_draft_collaboration_identity():
    with pytest.raises(ArtifactAccessUnavailable, match="signed-in organization"):
        await provision_draft_review_access(
            "11111111-1111-1111-1111-111111111111",
            TenantScope(org_mode=False, user_id=None, org_id=None),
        )


@pytest.mark.asyncio
async def test_revoke_draft_review_access_deletes_the_isolated_rule():
    captured = {}

    def handler(request: httpx.Request):
        captured["request"] = request
        return httpx.Response(200, json={"removed": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        removed = await revoke_draft_review_access(
            "11111111-1111-1111-1111-111111111111",
            TenantScope(
                org_mode=True,
                user_id="22222222-2222-2222-2222-222222222222",
                org_id="33333333-3333-3333-3333-333333333333",
            ),
            settings=TurnQueueSettings(
                auth_internal_base_url="http://auth.internal",
                auth_internal_secret="secret",
            ),
            client=client,
        )

    request = captured["request"]
    assert removed is True
    assert str(request.url) == "http://auth.internal/v1/internal/artifact-access/delete/"
    assert request.headers["X-Internal-Auth"] == "secret"
    assert json.loads(request.content) == {
        "artifact_id": "artifact-draft/11111111-1111-1111-1111-111111111111",
        "owner_keycloak_id": "22222222-2222-2222-2222-222222222222",
        "organization_id": "33333333-3333-3333-3333-333333333333",
    }


@pytest.mark.parametrize("scope", [
    TenantScope(org_mode=False, user_id=None, org_id=None),
    TenantScope(org_mode=True, user_id=None, org_id="organization"),
    TenantScope(org_mode=True, user_id="owner", org_id=None),
])
@pytest.mark.asyncio
async def test_revoke_requires_an_authenticated_owner_and_organization(scope):
    def handler(request):
        pytest.fail("Incomplete identity must not reach auth")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArtifactAccessUnavailable, match="signed-in organization"):
            await revoke_draft_review_access(
                "11111111-1111-1111-1111-111111111111",
                scope,
                settings=TurnQueueSettings(
                    auth_internal_base_url="http://auth.internal",
                    auth_internal_secret="secret",
                ),
                client=client,
            )


@pytest.mark.parametrize("failure", ["unavailable", "wrong-owner", "forbidden", "timeout"])
@pytest.mark.asyncio
async def test_revoke_fails_closed_when_auth_does_not_accept_the_owner(failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("Auth unavailable", request=request)
        return httpx.Response({"wrong-owner": 409, "forbidden": 403, "unavailable": 503}[failure])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArtifactAccessUnavailable, match="Could not revoke"):
            await revoke_draft_review_access(
                "11111111-1111-1111-1111-111111111111",
                TenantScope(org_mode=True, user_id="owner", org_id="organization"),
                settings=TurnQueueSettings(
                    auth_internal_base_url="http://auth.internal",
                    auth_internal_secret="secret",
                ),
                client=client,
            )


@pytest.mark.asyncio
async def test_revoke_is_a_noop_when_draft_authorization_is_unconfigured():
    def handler(request):
        pytest.fail("An unconfigured deployment must not contact auth")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await revoke_draft_review_access(
            "11111111-1111-1111-1111-111111111111",
            TenantScope(org_mode=True, user_id="owner", org_id="organization"),
            settings=TurnQueueSettings(auth_internal_base_url="", auth_internal_secret=""),
            client=client,
        ) is False
