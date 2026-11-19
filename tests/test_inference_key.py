"""The turn key that pins the desktop's LLM billing to its mounted organization."""

import threading
import time
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
    inference_key._installed.clear()
    inference_key._reset_retry_backoff()
    yield
    runtime_credential.clear_minds_credential()
    inference_key._installed.clear()
    inference_key._reset_retry_backoff()


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
        # A live hand-over for ORG_A, as the mint expects to find when it installs.
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)
        yield mock


class TestRefresh:
    def test_mints_for_the_org_with_the_session_token(self, http):
        inference_key.refresh_inference_key("jwt", ORG_A)

        url = http.post.call_args.args[0]
        kwargs = http.post.call_args.kwargs
        assert url.endswith("/v1/turn-keys/")
        assert kwargs["headers"] == {"Authorization": "Bearer jwt"}
        assert kwargs["json"]["organization_id"] == ORG_A
        # A duration, so auth sets the expiry on its own clock rather than the laptop's.
        assert kwargs["json"]["ttl_seconds"] == 3600
        assert "expiry_date" not in kwargs["json"]
        held = runtime_credential.get_inference_key()
        assert held.value == "mdb_new"
        assert held.organization_id == ORG_A
        assert held.instance_id == kwargs["json"]["instance_id"]

    def test_a_fresh_key_is_kept(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=50))

        inference_key.refresh_inference_key("jwt", ORG_A)

        http.post.assert_not_called()

    def test_an_ageing_key_is_replaced_but_not_revoked(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=40))

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key().value == "mdb_new"
        # The scratchpad and coding sessions still hold it; it expires on its own.
        http.delete.assert_not_called()

    def test_an_org_change_mints_at_once(self, http):
        runtime_credential.set_inference_key(_key(org=ORG_A, minutes_left=29))
        runtime_credential.set_organization(ORG_B)

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

    def test_a_failed_mint_drops_a_key_for_another_workspace(self, http):
        runtime_credential.set_inference_key(_key(minutes_left=29))
        http.post.side_effect = RuntimeError("auth down")

        with patch.object(inference_key, "desktop_workspace_pick", return_value="ws-2"):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() is None
        http.schedule_retry.assert_called_once()

    def test_a_key_minted_after_sign_out_is_revoked_not_installed(self, http):
        runtime_credential.clear_minds_credential()

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() is None
        http.delete.assert_called_once()

    def test_a_key_minted_for_an_org_left_meanwhile_is_revoked(self, http):
        runtime_credential.set_organization(ORG_B)

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() is None
        http.delete.assert_called_once()

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

    @pytest.mark.parametrize(("status", "code"), [(403, "permission_denied"), (409, "api_key_limit_reached")])
    def test_a_permanent_refusal_is_not_retried_every_minute(self, http, status, code):
        runtime_credential.set_inference_key(_key(org=ORG_B))
        http.post.return_value.status_code = status
        http.post.return_value.json.return_value = {"code": code}

        inference_key.refresh_inference_key("jwt", ORG_A)

        http.schedule_retry.assert_not_called()
        assert runtime_credential.get_inference_key() is None

    def test_an_unreadable_pick_fails_neither_the_hand_over_nor_the_retries(self, http):
        held = _key(minutes_left=40)
        runtime_credential.set_inference_key(held)

        with patch.object(inference_key, "desktop_workspace_pick", side_effect=RuntimeError("database is locked")):
            inference_key.refresh_inference_key("jwt", ORG_A)

        http.schedule_retry.assert_called_once()
        assert runtime_credential.get_inference_key() == held

    def test_a_bad_request_is_not_retried_every_minute(self, http):
        http.post.return_value.status_code = 400
        http.post.return_value.json.return_value = {"detail": "ttl_seconds exceeds the max turn-key TTL of 1800s"}

        inference_key.refresh_inference_key("jwt", ORG_A)

        http.schedule_retry.assert_not_called()

    def test_sign_out_revokes_every_live_key_this_process_installed(self, http):
        inference_key.refresh_inference_key("jwt", ORG_A)
        first = runtime_credential.get_inference_key()
        runtime_credential.set_organization(ORG_B)
        inference_key.refresh_inference_key("jwt", ORG_B)
        inference_key._installed["desktop-expired"] = _key(minutes_left=-1, instance_id="desktop-expired")

        inference_key.revoke_all_inference_keys("jwt")

        revoked = {c.args[0].rsplit("/", 2)[-2] for c in http.delete.call_args_list}
        assert revoked == {first.instance_id, runtime_credential.get_inference_key().instance_id}
        assert inference_key._installed == {}

    def test_a_failed_revoke_is_logged(self, http, caplog):
        http.delete.return_value.is_success = False
        http.delete.return_value.status_code = 401

        inference_key.revoke_inference_key("jwt", _key())

        assert "HTTP 401" in caplog.text

    def test_sign_out_stops_the_retries_and_resets_the_backoff(self, http):
        with patch.object(inference_key.threading, "Timer") as timer:
            inference_key._retry_timer = timer.return_value
            inference_key._retry_delay = 480.0

            inference_key.revoke_all_inference_keys("jwt")

        timer.return_value.cancel.assert_called_once()
        assert inference_key._retry_delay == inference_key._RETRY_S

    def test_a_key_installed_as_sign_out_clears_the_hand_over_is_revoked(self, http, monkeypatch):
        real_set = runtime_credential.set_inference_key

        def sign_out_mid_install(key):
            real_set(key)
            if key is not None:
                runtime_credential.clear_minds_credential()

        monkeypatch.setattr(runtime_credential, "set_inference_key", sign_out_mid_install)

        inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() is None
        http.delete.assert_called_once()
        assert inference_key._installed == {}

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

    def test_a_token_without_an_org_keeps_the_pin(self, client):
        runtime_credential.set_organization(ORG_A)
        with patch(
            "cowork.api.v1.endpoints.runtime_credential.refresh_inference_key"
        ) as refresh:
            client.put("/api/v1/runtime-credential/minds", json={"value": "jwt-2"})

        refresh.assert_called_once_with("jwt-2", ORG_A)
        assert runtime_credential.get_organization() == ORG_A

    def test_the_organization_read_answers_the_pinned_org(self, client):
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)

        response = client.get("/api/v1/runtime-credential/minds/organization")

        assert response.json() == {"organizationId": ORG_A}

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
            "cowork.api.v1.endpoints.runtime_credential.revoke_all_inference_keys"
        ) as revoke:
            client.put("/api/v1/runtime-credential/minds", json={"value": ""})

        revoke.assert_called_once_with("jwt")
        assert runtime_credential.get_inference_key() is None

    def test_sign_out_revokes_with_the_desktops_live_token(self, client):
        runtime_credential.set_minds_credential("expired-jwt")

        with patch(
            "cowork.api.v1.endpoints.runtime_credential.revoke_all_inference_keys"
        ) as revoke:
            client.put("/api/v1/runtime-credential/minds", json={"value": "", "revoke_token": "live-jwt"})

        revoke.assert_called_once_with("live-jwt")

    def test_sign_out_with_no_key_revokes_nothing(self, client):
        with patch(
            "cowork.api.v1.endpoints.runtime_credential.revoke_all_inference_keys"
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


class TestTimeBudget:
    """The hand-over waits on the mint, and the desktop gives up on it at 10s."""

    def test_each_call_gets_only_the_time_left(self, http):
        inference_key.refresh_inference_key("jwt", ORG_A)

        left = http.Timeout.call_args.args[0]
        assert 0 < left <= inference_key._MINT_BUDGET_S
        assert http.Timeout.call_args.kwargs["connect"] <= inference_key._CONNECT_S
        assert http.post.call_args.kwargs["timeout"] is http.Timeout.return_value

    def test_a_spent_budget_is_a_failed_mint(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0)

        inference_key.refresh_inference_key("jwt", ORG_A)

        http.post.assert_not_called()
        http.schedule_retry.assert_called_once()
        assert runtime_credential.get_inference_key() is None

    def test_a_slow_response_is_cut_off_at_the_budget(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0.2)
        released = threading.Event()
        http.post.side_effect = lambda *a, **k: released.wait(2) and MagicMock(is_success=False)

        started = time.monotonic()
        inference_key.refresh_inference_key("jwt", ORG_A)

        assert time.monotonic() - started < 1
        assert runtime_credential.get_inference_key() is None
        http.schedule_retry.assert_called_once()
        released.set()

    def test_a_mint_that_lands_after_the_budget_is_revoked(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0.1)
        landed = MagicMock(is_success=True)
        http.post.side_effect = lambda *a, **k: time.sleep(0.3) or landed

        inference_key.refresh_inference_key("jwt", ORG_A)
        instance_id = http.post.call_args.kwargs["json"]["instance_id"]
        deadline = time.monotonic() + 2
        while not http.delete.called and time.monotonic() < deadline:
            time.sleep(0.02)

        assert http.delete.call_args.args[0].endswith(f"/v1/turn-keys/{instance_id}/")
        assert runtime_credential.get_inference_key() is None

    def test_a_busy_lock_does_not_hold_up_the_hand_over(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0.05)

        with inference_key._mint_lock:
            inference_key.refresh_inference_key("jwt", ORG_A)

        http.post.assert_not_called()
        http.schedule_retry.assert_called_once()

    def test_a_busy_lock_with_an_unreadable_pick_keeps_the_orgs_key(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0.05)
        held = _key(minutes_left=40)
        runtime_credential.set_inference_key(held)

        with (
            inference_key._mint_lock,
            patch.object(inference_key, "desktop_workspace_pick", side_effect=RuntimeError("database is locked")),
        ):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() == held
        http.schedule_retry.assert_called_once()

    def test_a_busy_lock_drops_a_key_for_another_workspace(self, http, monkeypatch):
        monkeypatch.setattr(inference_key, "_MINT_BUDGET_S", 0.05)
        runtime_credential.set_inference_key(_key(minutes_left=29))

        with (
            inference_key._mint_lock,
            patch.object(inference_key, "desktop_workspace_pick", return_value="ws-2"),
        ):
            inference_key.refresh_inference_key("jwt", ORG_A)

        assert runtime_credential.get_inference_key() is None

    def test_revoke_has_a_short_timeout(self, http):
        inference_key.revoke_inference_key("jwt", _key())

        assert (
            http.delete.call_args.kwargs["timeout"] == inference_key._REVOKE_TIMEOUT_S
        )


class TestConcurrentMints:
    def test_a_slow_older_mint_cannot_install_over_a_newer_pick(self, http):
        import threading

        pick = {"ws": "ws-1"}
        first_post_started, release_first = threading.Event(), threading.Event()

        def post(url, json, **kwargs):
            response = MagicMock()
            response.json.return_value = {"key": f"mdb_{json.get('workspace_id')}"}
            if json.get("workspace_id") == "ws-1":
                first_post_started.set()
                release_first.wait(5)
            return response

        http.post.side_effect = post
        with patch.object(
            inference_key, "desktop_workspace_pick", side_effect=lambda: pick["ws"]
        ):
            old = threading.Thread(
                target=inference_key.refresh_inference_key, args=("jwt", ORG_A)
            )
            old.start()
            first_post_started.wait(5)
            pick["ws"] = "ws-2"  # the user switches workspace mid-mint
            new = threading.Thread(
                target=inference_key.refresh_inference_key, args=("jwt", ORG_A)
            )
            new.start()
            release_first.set()
            old.join(5)
            new.join(5)

        held = runtime_credential.get_inference_key()
        assert (held.workspace_id, held.value) == ("ws-2", "mdb_ws-2")


class TestRetry:
    def test_retries_with_the_latest_hand_over(self):
        runtime_credential.set_minds_credential("jwt-latest")
        runtime_credential.set_organization(ORG_A)
        with patch.object(inference_key, "refresh_inference_key") as refresh:
            inference_key._retry()

        refresh.assert_called_once_with("jwt-latest", ORG_A)

    def test_retries_back_off_to_ten_minutes_and_reset_on_a_mint(self):
        runtime_credential.set_minds_credential("jwt")
        runtime_credential.set_organization(ORG_A)
        delays = []
        with patch.object(inference_key, "httpx") as http, patch.object(inference_key.threading, "Timer") as timer:
            http.post.return_value.json.return_value = {"key": "mdb_new"}
            timer.side_effect = lambda delay, fn: delays.append(delay) or MagicMock()
            for _ in range(6):
                inference_key._schedule_retry()
            inference_key.refresh_inference_key("jwt", ORG_A)
            inference_key._schedule_retry()

        assert delays == [60.0, 120.0, 240.0, 480.0, 600.0, 600.0, 60.0]

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
        assert timer.call_args.args == (inference_key._RETRY_S * 2, inference_key._retry)
        assert timer.return_value.daemon is True
