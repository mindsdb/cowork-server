"""/health proves to the desktop app that it is the server the app spawned.

The owner token /health echoes can be read and replayed by any local caller,
so a listener another OS user puts on the app's port could pass for the app's
server and receive its credentials. The proof is an HMAC of the app's nonce
and the port the request arrived on, keyed by a per-install key only the app
and the server it spawned hold.
"""

import hashlib
import hmac
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from cowork.api.v1.endpoints import health
from cowork.common.settings.user_settings import Provider

KEY = "f3" * 32
NONCE = "0123456789abcdef" * 4


def _hmac(port: int, nonce: str = NONCE) -> str:
    return hmac.new(KEY.encode(), f"cowork-health-proof:{port}:{nonce}".encode(), hashlib.sha256).hexdigest()


@pytest.fixture
def probe():
    def get(*, key: str = KEY, params: dict | None = None):
        app = FastAPI()
        app.include_router(health.router, prefix="/api/v1/health")
        user = SimpleNamespace(
            config_status={},
            resolved_planning_provider=Provider.OPENAI,
            resolved_coding_provider=Provider.OPENAI,
            resolved_router_provider=Provider.OPENAI,
        )
        settings = SimpleNamespace(
            owner="install-token", tenancy_mode="local", draft_links_enabled=lambda: False,
            health_proof_key=SecretStr(key),
        )
        with (
            patch.object(health, "get_app_settings", return_value=settings),
            patch.object(health, "get_user_settings", return_value=user),
            TestClient(app) as client,
        ):
            return client.get("/api/v1/health/", params=params)

    return get


def test_the_proof_is_an_hmac_of_the_nonce_and_the_port_the_request_arrived_on(probe):
    answer = probe(params={"nonce": NONCE})

    assert answer.status_code == 200
    body = answer.json()
    # TestClient's requests arrive on port 80.
    assert body["proof"] == _hmac(80)
    assert "health-proof" in body["capabilities"]
    assert KEY not in answer.text
    assert body["owner"] == "install-token", "apps that predate the proof still adopt by the owner"


def test_a_proof_names_its_port_so_a_relay_to_the_real_server_on_another_port_does_not_pass():
    """A listener on the app's port that relays the app's nonce to the app's
    real server on another port gets back a proof for that other port."""
    on_the_apps_port = health.health_proof(key=KEY, nonce=NONCE, port=47100)
    relayed = health.health_proof(key=KEY, nonce=NONCE, port=47101)

    assert on_the_apps_port == _hmac(47100)
    assert relayed != on_the_apps_port


@pytest.mark.parametrize("params", [
    None,
    {"nonce": "zz" * 16},
    {"nonce": "ab" * 15},
    {"nonce": "ab" * 65},
    {"nonce": "AB" * 16},
])
def test_no_nonce_or_a_malformed_one_gets_no_proof_and_the_probe_still_answers(probe, params):
    answer = probe(params=params)

    assert answer.status_code == 200
    assert answer.json()["proof"] is None


def test_a_server_started_with_no_proof_key_proves_nothing(probe):
    """The web server and older desktop launches pass no key."""
    answer = probe(key="", params={"nonce": NONCE})

    assert answer.status_code == 200
    assert answer.json()["proof"] is None
