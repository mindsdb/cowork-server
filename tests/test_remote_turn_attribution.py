"""The remote (web) turn must carry attribution to the pod (ENG-1459).

Web turns do not run in this process. `COWORK_TURN_BACKEND=remote` routes them
over Redis to a scratchpad-controller pod, and `AntonHarness.stream_response`
*refuses* to run in-process under org tenancy — so everything the in-process
path stamps via `build_trace_metadata` was simply absent on web. Measured on
prod: 0 of 68 `harness=cloud` traces carried `surface`,
`cowork_server_version` or `install_channel`, against 28,993 that did.

The pod cannot derive any of it: cowork-server is not installed in the
`minds-anton-scratchpad` image, and only the deployment knows its surface.
"""

import pytest

from cowork import build_info
from cowork.turnqueue.producer import _trace_block


@pytest.fixture(autouse=True)
def _clear_build_info_caches():
    # `install_channel` and `_dist_version` are memoized for the process
    # lifetime, so without this these tests read whatever an earlier test's
    # monkeypatched settings cached — passing in isolation and failing in the
    # full suite. Same fixture `test_build_info.py` already carries.
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()
    yield
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()


def _patch(monkeypatch, *, tenancy="org", override=""):
    class _Settings:
        tenancy_mode = tenancy
        install_channel_override = ""
        surface_override = override

    monkeypatch.setattr(
        "cowork.common.settings.app_settings.get_app_settings", lambda: _Settings()
    )


class TestTheBlockSentToThePod:
    def test_a_web_deployment_sends_surface_web(self, monkeypatch):
        # The whole point: `web` was previously unreachable, because the only
        # consumer of it raised under the very condition that produced it.
        _patch(monkeypatch, tenancy="org")
        assert _trace_block()["surface"] == "web"

    def test_it_also_carries_the_build_attribution(self, monkeypatch):
        # These are ENG-1279's fields, absent on web for the same reason.
        _patch(monkeypatch, tenancy="org")
        block = _trace_block()
        assert "cowork_server_version" in block
        assert block["install_channel"] == "hosted"

    def test_the_same_helper_as_the_in_process_path(self, monkeypatch):
        # Reusing build_trace_metadata is deliberate: two hand-rolled blocks
        # would drift, and the drift would be invisible until someone compared
        # a desktop trace against a web one.
        #
        # The remote block is that helper's output with exactly two deltas:
        # `surface` added, and `anton_version` removed (the pod runs a different
        # anton and self-reports). Pinning the delta rather than a subset means
        # a NEW key appearing in the helper reaches the pod automatically, and
        # a key silently going missing fails here.
        _patch(monkeypatch, tenancy="local")
        from cowork.build_info import build_trace_metadata

        expected = (set(build_trace_metadata()) - {"anton_version"}) | {"surface"}
        assert set(_trace_block()) == expected

    def test_an_unresolvable_surface_still_sends_the_build_attribution(self, monkeypatch):
        # An invalid override yields no surface. The build half must survive —
        # losing it too would make one bad env var erase all attribution.
        _patch(monkeypatch, tenancy="org", override="nonsense")
        block = _trace_block()
        assert "surface" not in block
        assert "cowork_server_version" in block

    def test_broken_settings_degrade_rather_than_raise(self, monkeypatch):
        """Telemetry must not be the reason a turn fails to start.

        It degrades to PARTIAL attribution rather than none: the versions come
        from package metadata and are still knowable when settings are broken,
        so they are still worth sending. Only the settings-derived values drop
        out. (My first version of this test asserted `{}` and was wrong about
        its own code — the layered try/except below already salvages more.)
        """

        def _boom():
            raise RuntimeError("settings exploded")

        monkeypatch.setattr(
            "cowork.common.settings.app_settings.get_app_settings", _boom
        )
        block = _trace_block()  # must not raise
        assert "surface" not in block, "settings-derived values drop out"
        assert "cowork_server_version" in block, "versions survive broken settings"
        # Not asserting the channel VALUE: with settings unreadable it falls
        # through to pip metadata inference, so the answer legitimately differs
        # per install (`local` in a checkout, `pypi` from a wheel). What matters
        # is that it is still reported rather than silently dropped.
        assert "install_channel" in block


class TestItActuallyReachesTheJob:
    """The seam the unit tests above cannot see: `params` must carry it."""

    def test_the_job_params_include_the_trace_block(self):
        import ast
        import inspect

        from cowork.turnqueue import producer

        src = inspect.getsource(producer)
        tree = ast.parse(src)
        # Find the params dict literal and assert "trace" is one of its keys.
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                literal = {k.value for k in node.keys
                           if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if "input" in literal and "history" in literal:
                    keys |= literal
        assert keys, "could not find the job params dict"
        assert "trace" in keys, (
            "the remote job's params omit the trace block, so web turns reach "
            f"Langfuse unattributed (ENG-1459). keys={sorted(keys)}"
        )

    def test_our_anton_version_is_not_sent(self, monkeypatch):
        # The pod runs a DIFFERENT anton — its own pinned scratchpad image,
        # bumped independently of this server's vendored dep. Sending ours
        # would put a wrong value on the wire, harmless today only because
        # anton overwrites it when building its headers. Don't rely on that.
        _patch(monkeypatch, tenancy="org")
        assert "anton_version" not in _trace_block()
        # …while the values that ARE ours still go.
        assert "cowork_server_version" in _trace_block()
