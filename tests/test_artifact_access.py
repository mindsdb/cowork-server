from __future__ import annotations

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
    assert b'"artifact_id":"artifact-draft/11111111-1111-1111-1111-111111111111"' in request.content
