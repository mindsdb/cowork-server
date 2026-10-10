"""The turn key that pins the desktop's LLM billing to its mounted organization."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings import runtime_credential
from cowork.common.settings.runtime_credential import InferenceKey
from cowork.services import inference_key

ORG_A = "org-a"
ORG_B = "org-b"


@pytest.fixture(autouse=True)
def clear_hand_over():
    runtime_credential.clear_minds_credential()
    yield
    runtime_credential.clear_minds_credential()


def _key(org=ORG_A, minutes_left=30, value="mdb_turn", instance_id="desktop-old"):
    return InferenceKey(
        value=value,
        organization_id=org,
        instance_id=instance_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=minutes_left),
    )


class TestInferenceCredential:
    def test_an_older_desktop_that_sends_no_org_keeps_the_session_token(self):
        runtime_credential.set_minds_credential("jwt")

        assert runtime_credential.get_inference_credential() == "jwt"

    def test_a_mounted_org_bills_its_key(self):
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)
        runtime_credential.set_inference_key(_key())

        assert runtime_credential.get_inference_credential() == "mdb_turn"

    @pytest.mark.parametrize(
        "key",
        [None, _key(minutes_left=-1), _key(org=ORG_B)],
        ids=["no-key", "expired", "other-org"],
    )
    def test_a_mounted_org_never_falls_back_to_the_session_token(self, key):
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)
        runtime_credential.set_inference_key(key)

        assert runtime_credential.get_inference_credential() is None

    def test_clearing_drops_the_org_and_key(self):
        runtime_credential.set_organization(ORG_A)
        runtime_credential.set_inference_key(_key())

        runtime_credential.clear_minds_credential()

        assert runtime_credential.get_organization() is None
        assert runtime_credential.get_inference_key() is None


@pytest.fixture
def http():
    with (
        patch.object(inference_key, "httpx") as mock,
        patch.object(inference_key, "_schedule_retry") as retry,
    ):
        mock.post.return_value.json.return_value = {"key": "mdb_new"}
        mock.schedule_retry = retry
        yield mock


class TestRefresh:
    def test_mints_for_the_org_with_the_session_token(self, http):
        inference_key.refresh_inference_key("jwt", ORG_A)

        url = http.post.call_args.args[0]
        kwargs = http.post.call_args.kwargs
        assert url.endswith("/v1/turn-keys/")
        assert kwargs["headers"] == {"Authorization": "Bearer jwt"}
        assert kwargs["json"]["organization_id"] == ORG_A
        held = runtime_credential.get_inference_key()
        assert held.value == "mdb_new"
        assert held.organization_id == ORG_A
        assert held.instance_id == kwargs["json"]["instance_id"]

    def test_a_fresh_key_is_kept(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=25))

        inference_key.refresh_inference_key("jwt", ORG_A)

        http.post.assert_not_called()

    def test_an_ageing_key_is_replaced_but_not_revoked(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=15))

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key().value == "mdb_new"
        # The scratchpad and coding sessions still hold it; it expires on its own.
        http.delete.assert_not_called()

    def test_an_org_change_mints_at_once(self, http):
        runtime_credential.set_inference_key(_key(org=ORG_A, minutes_left=29))

        inference_key.refresh_inference_key("jwt", ORG_B)

        assert runtime_credential.get_inference_key().organization_id == ORG_B

    def test_a_failed_mint_keeps_a_live_key_for_the_same_org(self, http):
        held = _key(minutes_left=15)
        runtime_credential.set_inference_key(held)
        http.post.side_effect = RuntimeError("auth down")

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() == held
        http.schedule_retry.assert_called_once()

    def test_a_failed_mint_drops_a_key_for_another_org(self, http):
        runtime_credential.set_inference_key(_key(org=ORG_A))
        http.post.return_value.raise_for_status.side_effect = RuntimeError("403")

        inference_key.refresh_inference_key("jwt", ORG_B)

        assert runtime_credential.get_inference_key() is None
        http.schedule_retry.assert_called_once()

    def test_mints_for_the_desktop_workspace_pick(self, http):
        with patch.object(inference_key, "desktop_workspace_pick", return_value="ws-1"):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert http.post.call_args.kwargs["json"]["workspace_id"] == "ws-1"
        assert runtime_credential.get_inference_key().workspace_id == "ws-1"

    def test_no_pick_sends_no_workspace(self, http):
        with patch.object(inference_key, "desktop_workspace_pick", return_value=None):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert "workspace_id" not in http.post.call_args.kwargs["json"]

    def test_a_changed_pick_mints_at_once(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=29))

        with patch.object(inference_key, "desktop_workspace_pick", return_value="ws-2"):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key().workspace_id == "ws-2"

    def test_a_stale_pick_falls_back_to_the_default(self, http):
        refused, minted = MagicMock(), MagicMock()
        refused.json.return_value = {"code": "workspace_not_found"}
        minted.json.return_value = {"key": "mdb_default"}
        http.post.side_effect = [refused, minted]

        with patch.object(
            inference_key, "desktop_workspace_pick", return_value="ws-gone"
        ):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert "workspace_id" not in http.post.call_args_list[1].kwargs["json"]
        held = runtime_credential.get_inference_key()
        assert held.value == "mdb_default"
        # Recorded against the pick, so the next hand-over doesn't re-mint for it again.
        assert held.workspace_id == "ws-gone"

    def test_a_successful_mint_schedules_no_retry(self, http):
        inference_key.refresh_inference_key("jwt", ORG_A)

        http.schedule_retry.assert_not_called()

    def test_a_failed_revoke_is_ignored(self, http):
        http.delete.side_effect = RuntimeError("auth down")

        inference_key.revoke_inference_key("jwt", _key())


@pytest.fixture
def client():
    from cowork.server import create_app

    return TestClient(create_app(), client=("127.0.0.1", 40000))


class TestRoute:
    def test_an_org_on_the_hand_over_mints(self, client):
        with patch(
            "cowork.api.v1.endpoints.runtime_credential.refresh_inference_key"
        ) as refresh:
            response = client.put(
                "/api/v1/runtime-credential/minds",
                json={"value": "jwt", "organization_id": ORG_A},
            )

        assert response.status_code == 200
        refresh.assert_called_once_with("jwt", ORG_A)
        assert runtime_credential.get_organization() == ORG_A

    def test_a_hand_entered_key_is_never_pinned(self, client):
        with patch(
            "cowork.api.v1.endpoints.runtime_credential.refresh_inference_key"
        ) as refresh:
            client.put(
                "/api/v1/runtime-credential/minds",
                json={"value": "mdb_byok", "organization_id": ORG_A},
            )

        refresh.assert_not_called()
        assert runtime_credential.get_organization() is None
        assert runtime_credential.get_inference_credential() == "mdb_byok"

    def test_sign_out_revokes_the_key(self, client):
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)
        held = _key()
        runtime_credential.set_inference_key(held)

        with patch(
            "cowork.api.v1.endpoints.runtime_credential.revoke_inference_key"
        ) as revoke:
            client.put("/api/v1/runtime-credential/minds", json={"value": ""})

        revoke.assert_called_once_with("jwt", held)
        assert runtime_credential.get_inference_key() is None

    def test_sign_out_with_no_key_revokes_nothing(self, client):
        with patch(
            "cowork.api.v1.endpoints.runtime_credential.revoke_inference_key"
        ) as revoke:
            client.put("/api/v1/runtime-credential/minds", json={"value": ""})

        revoke.assert_not_called()


def test_org_mode_holds_no_org_or_key(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        runtime_credential.set_organization(ORG_A)
        runtime_credential.set_inference_key(_key())
        assert runtime_credential.get_organization() is None
        assert runtime_credential.get_inference_key() is None
        assert runtime_credential.get_inference_credential() is None
    finally:
        get_app_settings.cache_clear()


class TestRetry:
    def test_retries_with_the_latest_hand_over(self):
        runtime_credential.set_minds_credential("jwt-latest")
        runtime_credential.set_organization(ORG_A)
        with patch.object(inference_key, "refresh_inference_key") as refresh:
            inference_key._retry()

        refresh.assert_called_once_with("jwt-latest", ORG_A)

    def test_does_nothing_after_sign_out(self):
        with patch.object(inference_key, "refresh_inference_key") as refresh:
            inference_key._retry()

        refresh.assert_not_called()

    def test_only_one_retry_is_pending(self):
        with patch.object(inference_key.threading, "Timer") as timer:
            inference_key._schedule_retry()
            first = timer.return_value
            inference_key._schedule_retry()

        first.cancel.assert_called_once()
        assert timer.call_args.args == (inference_key._RETRY_S, inference_key._retry)
        assert timer.return_value.daemon is True
