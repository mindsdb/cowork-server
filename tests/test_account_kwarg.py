"""The turn carries the account that ran it (ENG-2121).

anton's `turn_completed` was keyed on the install fingerprint only, so
completed work joined nothing: not sign-up, not payment, not the staff/test
exclusion. anton now keys the event on `user_id` when the host supplies one.
This server is the host on both surfaces:

- **desktop**: the in-process harness, where the only identity is the MindsHub
  JWT the desktop app hands over (`runtime_credential`);
- **web**: the remote pod, where the identity is the gateway-verified
  principal, sent in the job's trace block.

Only opaque ids travel, never the email the JWT also carries.
"""

import base64
import dataclasses
import json

import pytest

from cowork import build_info
from cowork.build_info import account_kwargs, desktop_account
from cowork.turnqueue.producer import _trace_block

SUB = "0f2b5c71-9e3a-4d18-bb44-7c6a1d2e5f30"
ORG = "7c6a1d2e-5f30-4d18-bb44-0f2b5c719e3a"


def _jwt(payload: dict) -> str:
    def seg(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{seg({'alg': 'RS256', 'typ': 'JWT'})}.{seg(payload)}.signature"


_PAYLOAD = {
    "sub": SUB,
    "email": "someone@example.com",
    "name": "Some One",
    "activate_organization": {"id": ORG, "name": "Acme"},
}


@pytest.fixture
def held(monkeypatch):
    """Set what the desktop app has handed over."""

    def _set(value):
        monkeypatch.setattr(
            "cowork.common.settings.runtime_credential.get_minds_credential",
            lambda: value,
        )

    return _set


@dataclasses.dataclass
class _NewAnton:
    surface: str | None = None
    user_id: str | None = None
    organization_id: str | None = None


@dataclasses.dataclass
class _OldAnton:
    """The pinned anton, from before ENG-2121."""

    surface: str | None = None


class TestDesktopAccount:
    def test_the_held_jwt_yields_the_sub_and_active_org(self, held):
        held(_jwt(_PAYLOAD))
        assert desktop_account() == {"user_id": SUB, "organization_id": ORG}

    def test_nothing_but_the_two_ids_is_taken(self, held):
        # The JWT carries the email and name too. Neither may leave here.
        held(_jwt(_PAYLOAD))
        values = json.dumps(desktop_account())
        assert "someone@example.com" not in values
        assert "Some One" not in values

    def test_a_token_without_an_org_still_yields_the_user(self, held):
        held(_jwt({"sub": SUB}))
        assert desktop_account() == {"user_id": SUB}

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "mdb_0123456789abcdef",  # a user-supplied API key: no identity in it
            "not.a.jwt",
            _jwt({"email": "someone@example.com"}),  # no sub
            _jwt({"sub": "someone@example.com"}),  # a sub that is not a UUID
        ],
    )
    def test_anything_else_yields_no_account(self, held, value):
        assert desktop_account() == {}

    def test_a_broken_holder_never_propagates(self, monkeypatch):
        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "cowork.common.settings.runtime_credential.get_minds_credential", _boom
        )
        assert desktop_account() == {}


class TestAccountKwargs:
    def test_an_old_anton_gets_no_kwargs_at_all(self, held):
        # The pinned anton has no such fields; passing them would raise
        # TypeError on every turn.
        held(_jwt(_PAYLOAD))
        assert account_kwargs(_OldAnton) == {}

    def test_a_current_anton_gets_the_account(self, held):
        held(_jwt(_PAYLOAD))
        kwargs = account_kwargs(_NewAnton)
        assert kwargs == {"user_id": SUB, "organization_id": ORG}
        _NewAnton(**kwargs)  # must construct

    def test_no_account_is_omitted_rather_than_sent_empty(self, held):
        held(None)
        assert account_kwargs(_NewAnton) == {}


class TestTheWebTraceBlock:
    """Web turns run in a pod that cannot know who the user is."""

    def test_the_principal_rides_the_trace_block(self):
        block = _trace_block(user_id=SUB, org_id=ORG)
        assert block["user_id"] == SUB
        assert block["organization_id"] == ORG

    def test_no_principal_adds_no_keys(self):
        block = _trace_block()
        assert "user_id" not in block
        assert "organization_id" not in block

    def test_a_malformed_id_is_not_sent(self):
        block = _trace_block(user_id="someone@example.com", org_id="acme")
        assert "user_id" not in block
        assert "organization_id" not in block

    def test_the_producer_passes_the_principal(self):
        # The seam the unit tests above cannot see: the call site in the
        # producer must hand the turn's identity to the block builder.
        import inspect

        from cowork.turnqueue import producer

        src = inspect.getsource(producer)
        assert '"trace": _trace_block(user_id=user_id, org_id=org_id)' in src


@pytest.fixture(autouse=True)
def _clear_build_info_caches():
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()
    yield
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()
