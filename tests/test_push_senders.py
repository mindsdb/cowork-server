"""What a push says, and how APNs and FCM are spoken to, against mocked
transports: provider-token headers, the sandbox fallback, dead-token cleanup,
the cached FCM access token, and a dispatcher that never raises."""
from __future__ import annotations

import base64
import json

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.push import dispatch, payloads
from cowork.services.push.apns import PRODUCTION_HOST, SANDBOX_HOST, ApnsSender, Delivery, read_key
from cowork.services.push.devices import Device
from cowork.services.push.fcm import FcmSender, ServiceAccount, read_service_account

CONVERSATION = "0b9f7c1e-2f7d-4d35-8b8e-4f8f2c0a1d11"


def _ec_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ).decode()


def _rsa_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ).decode()


# ── payloads ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("succeeded", "body"), [(True, "Finished."), (False, "Didn't finish. Tap to see what happened.")])
def test_task_finished_payload(succeeded, body):
    payload = payloads.task_finished(conversation_id=CONVERSATION, title="Q3 numbers", succeeded=succeeded)
    assert (payload.title, payload.body) == ("Q3 numbers", body)
    assert payload.data() == {"kind": "task_finished", "path": f"/c/{CONVERSATION}"}


def test_task_finished_title_falls_back():
    assert payloads.task_finished(conversation_id=CONVERSATION, title=None, succeeded=True).title == "Your task"
    assert payloads.task_finished(conversation_id=CONVERSATION, title="   ", succeeded=True).title == "Your task"


@pytest.mark.parametrize(("succeeded", "body"), [(True, "Scheduled run finished."), (False, "Scheduled run didn't finish.")])
def test_scheduled_run_payload(succeeded, body):
    payload = payloads.scheduled_run_finished(
        conversation_id=CONVERSATION, schedule_name="Daily report", succeeded=succeeded,
    )
    assert (payload.title, payload.body) == ("Daily report", body)
    assert payload.data() == {"kind": "scheduled_run_finished", "path": f"/c/{CONVERSATION}"}


def test_scheduled_run_without_a_conversation_has_no_path():
    payload = payloads.scheduled_run_finished(conversation_id=None, schedule_name="Daily", succeeded=False)
    assert payload.data() == {"kind": "scheduled_run_finished"}


def test_artifact_shared_payload():
    payload = payloads.artifact_shared(
        sharer="jorge@mindsdb.com", artifact_title="Q3 report", url="https://share.example.com/r/1",
    )
    assert payload.title == "jorge@mindsdb.com shared “Q3 report” with you"
    assert payload.body == "Tap to open."
    assert payload.data() == {"kind": "artifact_shared", "url": "https://share.example.com/r/1"}
    assert "path" not in payload.data()


def test_artifact_shared_payload_without_names_or_url():
    payload = payloads.artifact_shared(sharer=None, artifact_title="", url="")
    assert payload.title == "Someone shared an artifact with you"
    assert payload.data() == {"kind": "artifact_shared"}


def test_a_url_that_isnt_https_is_left_out():
    payload = payloads.artifact_shared(sharer="a", artifact_title="b", url="http://view.example/r/1")
    assert "url" not in payload.data()


def test_long_titles_are_clipped():
    payload = payloads.task_finished(conversation_id=CONVERSATION, title="x" * 500, succeeded=True)
    assert len(payload.title) <= 120 and payload.title.endswith("…")


def test_platform_shapes():
    payload = payloads.task_finished(conversation_id=CONVERSATION, title="T", succeeded=True)
    assert payload.apns_body() == {
        "aps": {"alert": {"title": "T", "body": "Finished."}, "sound": "default", "thread-id": "task_finished"},
        "kind": "task_finished",
        "path": f"/c/{CONVERSATION}",
    }
    assert payload.fcm_message("fcm-token") == {"message": {
        "token": "fcm-token",
        "notification": {"title": "T", "body": "Finished."},
        "data": {"kind": "task_finished", "path": f"/c/{CONVERSATION}"},
        "android": {"priority": "HIGH", "notification": {"channel_id": "cowork_updates"}},
    }}


# ── APNs ──────────────────────────────────────────────────────────────


class _Recorder:
    def __init__(self, respond):
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._respond(request)

        return httpx.MockTransport(handler)


def _apns(recorder: _Recorder, pem: str | None = None, clock=None) -> ApnsSender:
    kwargs = {"clock": clock} if clock else {}
    return ApnsSender(
        key_pem=pem or _ec_pem(), key_id="KEY123", team_id="TEAM456", topic="ai.mindshub.cowork",
        transport=recorder.transport(), **kwargs,
    )


_PAYLOAD = payloads.task_finished(conversation_id=CONVERSATION, title="T", succeeded=True)


async def test_apns_sends_with_a_provider_token_and_alert_headers():
    recorder = _Recorder(lambda request: httpx.Response(200))
    result = await _apns(recorder).send(Device("ios", "abc123", None), _PAYLOAD)
    assert result.delivery == Delivery.SENT and result.environment is None
    (request,) = recorder.requests
    assert str(request.url) == f"{PRODUCTION_HOST}/3/device/abc123"
    assert request.headers["apns-topic"] == "ai.mindshub.cowork"
    assert request.headers["apns-push-type"] == "alert"
    assert request.headers["apns-priority"] == "10"
    scheme, token = request.headers["authorization"].split(" ", 1)
    assert scheme == "bearer"
    assert jwt.get_unverified_header(token) == {"alg": "ES256", "kid": "KEY123", "typ": "JWT"}
    assert jwt.decode(token, options={"verify_signature": False})["iss"] == "TEAM456"
    assert json.loads(request.content) == _PAYLOAD.apns_body()


async def test_apns_reuses_its_provider_token_for_50_minutes():
    now = [1_000_000.0]
    recorder = _Recorder(lambda request: httpx.Response(200))
    sender = _apns(recorder, clock=lambda: now[0])
    await sender.send(Device("ios", "a", None), _PAYLOAD)
    now[0] += 49 * 60
    await sender.send(Device("ios", "a", None), _PAYLOAD)
    now[0] += 2 * 60
    await sender.send(Device("ios", "a", None), _PAYLOAD)
    tokens = [request.headers["authorization"] for request in recorder.requests]
    assert tokens[0] == tokens[1] != tokens[2]


async def test_apns_sandbox_device_goes_straight_to_sandbox():
    recorder = _Recorder(lambda request: httpx.Response(200))
    await _apns(recorder).send(Device("ios", "abc", "sandbox"), _PAYLOAD)
    assert [request.url.host for request in recorder.requests] == ["api.sandbox.push.apple.com"]


async def test_apns_bad_device_token_in_production_is_retried_once_in_sandbox():
    def respond(request):
        if request.url.host == "api.push.apple.com":
            return httpx.Response(400, json={"reason": "BadDeviceToken"})
        return httpx.Response(200)

    recorder = _Recorder(respond)
    result = await _apns(recorder).send(Device("ios", "abc", None), _PAYLOAD)
    assert result.delivery == Delivery.SENT and result.environment == "sandbox"
    assert [str(r.url) for r in recorder.requests] == [
        f"{PRODUCTION_HOST}/3/device/abc", f"{SANDBOX_HOST}/3/device/abc",
    ]


async def test_apns_bad_token_in_both_environments_is_dead():
    recorder = _Recorder(lambda request: httpx.Response(400, json={"reason": "BadDeviceToken"}))
    result = await _apns(recorder).send(Device("ios", "abc", None), _PAYLOAD)
    assert result.delivery == Delivery.DEAD
    assert len(recorder.requests) == 2


async def test_apns_410_is_dead():
    recorder = _Recorder(lambda request: httpx.Response(410, json={"reason": "Unregistered"}))
    assert (await _apns(recorder).send(Device("ios", "abc", None), _PAYLOAD)).delivery == Delivery.DEAD
    assert len(recorder.requests) == 1


async def test_apns_wrong_topic_fails_without_forgetting_the_device():
    # Every device answers this when the configured topic is wrong; deleting
    # them would wipe every registration over a configuration mistake.
    recorder = _Recorder(lambda request: httpx.Response(400, json={"reason": "DeviceTokenNotForTopic"}))
    assert (await _apns(recorder).send(Device("ios", "abc", None), _PAYLOAD)).delivery == Delivery.FAILED
    assert len(recorder.requests) == 1


async def test_apns_reuses_one_client_across_sends_until_closed():
    recorder = _Recorder(lambda request: httpx.Response(200))
    sender = _apns(recorder)
    await sender.send(Device("ios", "a", None), _PAYLOAD)
    first = sender._client()
    await sender.send(Device("ios", "b", None), _PAYLOAD)
    assert sender._client() is first and not first.is_closed
    await sender.aclose()
    assert first.is_closed
    await sender.send(Device("ios", "c", None), _PAYLOAD)
    assert len(recorder.requests) == 3


async def test_apns_other_errors_fail_and_an_expired_provider_token_is_resigned():
    now = [1_000_000.0]
    responses = iter([httpx.Response(403, json={"reason": "ExpiredProviderToken"}), httpx.Response(200)])
    recorder = _Recorder(lambda request: next(responses))
    sender = _apns(recorder, clock=lambda: now[0])
    assert (await sender.send(Device("ios", "abc", None), _PAYLOAD)).delivery == Delivery.FAILED
    now[0] += 1
    assert (await sender.send(Device("ios", "abc", None), _PAYLOAD)).delivery == Delivery.SENT
    first, second = (request.headers["authorization"] for request in recorder.requests)
    assert first != second


def test_apns_key_reads_pem_or_base64():
    pem = _ec_pem()
    assert read_key(pem) == pem.strip()
    assert read_key(base64.b64encode(pem.encode()).decode()) == pem.strip()
    with pytest.raises(ValueError):
        read_key("not a key")


# ── FCM ───────────────────────────────────────────────────────────────


def _account() -> ServiceAccount:
    return ServiceAccount(
        project_id="cowork-mobile", client_email="push@cowork-mobile.iam.gserviceaccount.com",
        private_key=_rsa_pem(), token_uri="https://oauth2.googleapis.com/token",
    )


def _fcm_responder(send_response=lambda request: httpx.Response(200, json={"name": "x"})):
    def respond(request):
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "ya29.token", "expires_in": 3599})
        return send_response(request)

    return respond


async def test_fcm_fetches_its_access_token_once_and_sends_v1_messages():
    recorder = _Recorder(_fcm_responder())
    sender = FcmSender(_account(), transport=recorder.transport())
    assert await sender.send(Device("android", "reg-1", None), _PAYLOAD) == Delivery.SENT
    assert await sender.send(Device("android", "reg-2", None), _PAYLOAD) == Delivery.SENT
    token_requests = [r for r in recorder.requests if r.url.host == "oauth2.googleapis.com"]
    sends = [r for r in recorder.requests if r.url.host == "fcm.googleapis.com"]
    assert len(token_requests) == 1, "the access token is cached"
    form = dict(pair.split("=", 1) for pair in token_requests[0].content.decode().split("&"))
    assert form["grant_type"] == "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
    claims = jwt.decode(form["assertion"], options={"verify_signature": False})
    assert claims["scope"] == "https://www.googleapis.com/auth/firebase.messaging"
    assert claims["iss"] == "push@cowork-mobile.iam.gserviceaccount.com"
    assert [str(r.url) for r in sends] == [
        "https://fcm.googleapis.com/v1/projects/cowork-mobile/messages:send"
    ] * 2
    assert sends[0].headers["authorization"] == "Bearer ya29.token"
    assert json.loads(sends[0].content) == _PAYLOAD.fcm_message("reg-1")


async def test_fcm_refetches_a_token_near_expiry():
    now = [1_000_000.0]
    recorder = _Recorder(_fcm_responder())
    sender = FcmSender(_account(), transport=recorder.transport(), clock=lambda: now[0])
    await sender.send(Device("android", "r", None), _PAYLOAD)
    now[0] += 3599 - 200
    await sender.send(Device("android", "r", None), _PAYLOAD)
    assert len([r for r in recorder.requests if r.url.host == "oauth2.googleapis.com"]) == 2


@pytest.mark.parametrize("response", [
    httpx.Response(404, json={"error": {"status": "NOT_FOUND"}}),
    httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT", "details": [
        {"@type": "type.googleapis.com/google.firebase.fcm.v1.FcmError", "errorCode": "UNREGISTERED"},
    ]}}),
])
async def test_fcm_unregistered_is_dead(response):
    recorder = _Recorder(_fcm_responder(lambda request: response))
    sender = FcmSender(_account(), transport=recorder.transport())
    assert await sender.send(Device("android", "r", None), _PAYLOAD) == Delivery.DEAD


async def test_fcm_other_errors_fail():
    recorder = _Recorder(_fcm_responder(lambda request: httpx.Response(500, json={"error": {"status": "INTERNAL"}})))
    sender = FcmSender(_account(), transport=recorder.transport())
    assert await sender.send(Device("android", "r", None), _PAYLOAD) == Delivery.FAILED


def test_service_account_reads_json_or_base64():
    raw = json.dumps({"project_id": "p", "client_email": "e@x", "private_key": "k"})
    assert read_service_account(raw).project_id == "p"
    assert read_service_account(base64.b64encode(raw.encode()).decode()).client_email == "e@x"
    with pytest.raises(ValueError):
        read_service_account(json.dumps({"project_id": "p"}))


# ── dispatch ──────────────────────────────────────────────────────────


ORG = "11111111-1111-1111-1111-111111111111"
USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()
    dispatch.set_senders(None)


@pytest.fixture
def stored_devices(monkeypatch):
    """The person's devices, and what the dispatcher writes back to them."""
    state = {"devices": [], "forgotten": [], "environments": []}

    async def fake_run_db(fn, *, scope):
        name = fn.func.__name__
        if name == "devices_for":
            return list(state["devices"])
        if name == "forget_token":
            state["forgotten"].append(fn.keywords["token"])
        if name == "set_environment":
            state["environments"].append((fn.keywords["token"], fn.keywords["environment"]))
        return None

    monkeypatch.setattr(dispatch, "run_db", fake_run_db)
    return state


async def test_notify_user_sends_to_every_device_and_cleans_up(org_mode, stored_devices):
    def respond(request):
        token = request.url.path.rsplit("/", 1)[-1]
        if token == "gone":
            return httpx.Response(410, json={"reason": "Unregistered"})
        if token == "dev" and request.url.host == "api.push.apple.com":
            return httpx.Response(400, json={"reason": "BadDeviceToken"})
        return httpx.Response(200)

    apns_recorder = _Recorder(respond)
    fcm_recorder = _Recorder(_fcm_responder(
        lambda request: httpx.Response(404, json={"error": {"status": "NOT_FOUND"}})
    ))
    dispatch.set_senders(dispatch.Senders(
        apns=_apns(apns_recorder), fcm=FcmSender(_account(), transport=fcm_recorder.transport()),
    ))
    stored_devices["devices"] = [
        Device("ios", "ok", None), Device("ios", "gone", None), Device("ios", "dev", None),
        Device("android", "old-android", None),
    ]
    await dispatch.notify_user(ORG, USER, _PAYLOAD)
    assert sorted(stored_devices["forgotten"]) == ["gone", "old-android"]
    assert stored_devices["environments"] == [("dev", "sandbox")]


async def test_notify_user_never_raises(org_mode, stored_devices, caplog):
    def explode(request):
        raise httpx.ConnectError("down")

    dispatch.set_senders(dispatch.Senders(apns=_apns(_Recorder(explode))))
    stored_devices["devices"] = [Device("ios", "ok", None)]
    await dispatch.notify_user(ORG, USER, _PAYLOAD)
    assert any(getattr(r, "event", None) == "push_send_failed" for r in caplog.records)


async def test_notify_user_survives_a_database_failure(org_mode, monkeypatch):
    async def broken(*_args, **_kwargs):
        raise RuntimeError("db down")

    dispatch.set_senders(dispatch.Senders(apns=_apns(_Recorder(lambda r: httpx.Response(200)))))
    monkeypatch.setattr(dispatch, "run_db", broken)
    await dispatch.notify_user(ORG, USER, _PAYLOAD)


async def test_push_is_off_outside_org_mode_or_with_no_credentials(monkeypatch, stored_devices):
    sent = _Recorder(lambda request: httpx.Response(200))
    dispatch.set_senders(dispatch.Senders(apns=_apns(sent)))
    stored_devices["devices"] = [Device("ios", "ok", None)]
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        assert dispatch.push_enabled() is False
        await dispatch.notify_user(ORG, USER, _PAYLOAD)
        assert sent.requests == []
        monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
        get_app_settings.cache_clear()
        assert dispatch.push_enabled() is True
        dispatch.set_senders(dispatch.Senders())
        assert dispatch.push_enabled() is False
    finally:
        get_app_settings.cache_clear()
        dispatch.set_senders(None)


def test_senders_are_built_from_settings(monkeypatch):
    from cowork.common.settings.app_settings import get_push_settings

    account = json.dumps({"project_id": "p", "client_email": "e@x", "private_key": _rsa_pem()})
    monkeypatch.setenv("COWORK_PUSH_APNS_KEY_P8", base64.b64encode(_ec_pem().encode()).decode())
    monkeypatch.setenv("COWORK_PUSH_APNS_KEY_ID", "KEY")
    monkeypatch.setenv("COWORK_PUSH_APNS_TEAM_ID", "TEAM")
    monkeypatch.setenv("COWORK_PUSH_FCM_SERVICE_ACCOUNT_JSON", account)
    get_push_settings.cache_clear()
    dispatch.set_senders(None)
    try:
        senders = dispatch.get_senders()
        assert senders.apns is not None and senders.fcm is not None
        monkeypatch.delenv("COWORK_PUSH_APNS_KEY_ID")
        monkeypatch.setenv("COWORK_PUSH_FCM_SERVICE_ACCOUNT_JSON", "garbage")
        get_push_settings.cache_clear()
        dispatch.set_senders(None)
        senders = dispatch.get_senders()
        assert senders.apns is None and senders.fcm is None
    finally:
        get_push_settings.cache_clear()
        dispatch.set_senders(None)


async def test_the_kill_switch_turns_push_off(org_mode, monkeypatch):
    from cowork.common.settings.app_settings import get_push_settings

    dispatch.set_senders(dispatch.Senders(apns=_apns(_Recorder(lambda r: httpx.Response(200)))))
    monkeypatch.setenv("COWORK_PUSH_ENABLED", "false")
    get_push_settings.cache_clear()
    try:
        assert dispatch.push_enabled() is False
    finally:
        get_push_settings.cache_clear()
