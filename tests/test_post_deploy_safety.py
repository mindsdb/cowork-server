"""Regression tests for post-deploy test helpers that must fail safely."""

from tests.integration import test_post_deploy


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _Peer:
    def __init__(self, probes: list[dict]) -> None:
        self.probes = probes
        self.calls = 0

    def get(self, path: str, *, params: dict) -> _Response:
        assert path == "/api/v1/responses/in-flight"
        assert params == {"conversation_id": "conversation-1"}
        payload = self.probes[min(self.calls, len(self.probes) - 1)]
        self.calls += 1
        return _Response(payload)


def test_identity_repr_never_renders_the_api_key():
    identity = test_post_deploy._Identity({
        "api_key": "mdb_do-not-print-me",
        "user_id": "user-1",
        "organization_id": "org-1",
    })

    rendered = repr(identity)

    assert "mdb_do-not-print-me" not in rendered
    assert "'api_key': '***'" in rendered
    assert "user-1" in rendered


def test_peer_probe_retries_until_the_shared_index_is_visible(monkeypatch):
    peer = _Peer([
        {"has_buffer": False},
        {"has_buffer": False},
        {"has_buffer": True, "latest_seq": 1},
    ])
    monkeypatch.setattr(test_post_deploy.time, "sleep", lambda _seconds: None)

    probe = test_post_deploy._await_shared_buffer(peer, "conversation-1")

    assert probe == {"has_buffer": True, "latest_seq": 1}
    assert peer.calls == 3


def test_peer_probe_returns_the_last_response_at_timeout(monkeypatch):
    peer = _Peer([{"has_buffer": False, "latest_seq": 0}])
    monkeypatch.setattr(test_post_deploy.time, "sleep", lambda _seconds: None)

    probe = test_post_deploy._await_shared_buffer(
        peer, "conversation-1", timeout_s=-1.0,
    )

    assert probe == {"has_buffer": False, "latest_seq": 0}
    assert peer.calls == 1
