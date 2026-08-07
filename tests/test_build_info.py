"""Build attribution on traces (ENG-1279).

Locks in the two things a release comparison depends on: the versions are
actually present in every turn's trace metadata, and the install channel is
either classified correctly or reported as ``unknown`` — never silently
bucketed with PyPI, which would put hosted/dev traces into the population a
release measurement is comparing.
"""

import json

import pytest

from cowork import build_info
from cowork.build_info import (
    KEY_ANTON_VERSION,
    KEY_INSTALL_CHANNEL,
    KEY_SERVER_VERSION,
    build_trace_metadata,
)


@pytest.fixture(autouse=True)
def _clear_build_info_caches():
    # Every value here is memoized for the process lifetime; tests need to see
    # their own monkeypatched state.
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()
    yield
    build_info._dist_version.cache_clear()
    build_info.install_channel.cache_clear()


class _FakeDist:
    def __init__(self, direct_url: str | None):
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


def _patch_channel_inputs(
    monkeypatch, *, tenancy: str = "local", direct_url: str | None = None, override: str = ""
):
    class _Settings:
        tenancy_mode = tenancy
        install_channel_override = override

    monkeypatch.setattr(
        "cowork.common.settings.app_settings.get_app_settings", lambda: _Settings()
    )
    monkeypatch.setattr(build_info, "distribution", lambda name: _FakeDist(direct_url))


class TestInstallChannel:
    def test_explicit_override_beats_tenancy_and_install_source(self, monkeypatch):
        # The hub snapshot case: local tenancy (org would change auth
        # semantics) with a git/PyPI-installed cowork-server inside the docker
        # image. Inference from inside the process is wrong there — the
        # deployer declares the channel via COWORK_INSTALL_CHANNEL.
        _patch_channel_inputs(
            monkeypatch,
            tenancy="local",
            direct_url=json.dumps({"vcs_info": {"commit_id": "abc123"}}),
            override="hosted",
        )
        assert build_info.install_channel() == "hosted"

    def test_invalid_override_degrades_to_inference(self, monkeypatch, caplog):
        # A typo'd override must not mint a new channel population; it is
        # warned about and inference proceeds as if it were unset.
        _patch_channel_inputs(
            monkeypatch,
            direct_url=json.dumps({"vcs_info": {"commit_id": "abc123"}}),
            override="lightsail",
        )
        with caplog.at_level("WARNING"):
            assert build_info.install_channel() == "git"
        assert "COWORK_INSTALL_CHANNEL" in caplog.text

    def test_override_is_normalized(self, monkeypatch):
        _patch_channel_inputs(monkeypatch, direct_url=None, override="  Hosted ")
        assert build_info.install_channel() == "hosted"

    def test_hosted_wins_over_install_source(self, monkeypatch):
        # A hosted deployment's provenance belongs to the snapshot image, not
        # to how pip happened to fetch the package inside it.
        _patch_channel_inputs(
            monkeypatch,
            tenancy="org",
            direct_url=json.dumps({"vcs_info": {"commit_id": "abc123"}}),
        )
        assert build_info.install_channel() == "hosted"

    def test_vcs_record_is_a_git_install(self, monkeypatch):
        _patch_channel_inputs(
            monkeypatch, direct_url=json.dumps({"vcs_info": {"commit_id": "abc123"}})
        )
        assert build_info.install_channel() == "git"

    def test_no_direct_url_is_a_pypi_install(self, monkeypatch):
        # pip only writes direct_url.json for non-index installs.
        _patch_channel_inputs(monkeypatch, direct_url=None)
        assert build_info.install_channel() == "pypi"

    def test_editable_checkout_is_local_not_pypi(self, monkeypatch):
        # A developer's `-e .` install has a direct_url with no vcs_info;
        # calling it "pypi" would mix dev traffic into the released population.
        _patch_channel_inputs(
            monkeypatch,
            direct_url=json.dumps({"url": "file:///src/cowork-server", "dir_info": {"editable": True}}),
        )
        assert build_info.install_channel() == "local"

    def test_malformed_direct_url_is_unknown(self, monkeypatch):
        _patch_channel_inputs(monkeypatch, direct_url="{not json")
        assert build_info.install_channel() == "unknown"

    def test_missing_distribution_is_unknown(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        class _Settings:
            tenancy_mode = "local"
            install_channel_override = ""

        monkeypatch.setattr(
            "cowork.common.settings.app_settings.get_app_settings", lambda: _Settings()
        )

        def _boom(name):
            raise PackageNotFoundError(name)

        monkeypatch.setattr(build_info, "distribution", _boom)
        assert build_info.install_channel() == "unknown"


class TestBuildTraceMetadata:
    def test_stamps_versions_and_channel(self, monkeypatch):
        monkeypatch.setattr(build_info, "install_channel", lambda: "pypi")
        monkeypatch.setattr(
            build_info,
            "_dist_version",
            lambda name: {"cowork-server": "0.26.8.2.1", "anton-agent": "2.26.8.2.1"}[name],
        )
        assert build_trace_metadata({"harness": "anton"}) == {
            "harness": "anton",
            KEY_SERVER_VERSION: "0.26.8.2.1",
            KEY_ANTON_VERSION: "2.26.8.2.1",
            KEY_INSTALL_CHANNEL: "pypi",
        }

    def test_works_without_a_base(self, monkeypatch):
        # Local (desktop) mode has no principal, so the base is None — the
        # build stamp must not be gated on tenancy the way identity is.
        monkeypatch.setattr(build_info, "install_channel", lambda: "git")
        monkeypatch.setattr(build_info, "_dist_version", lambda name: None)
        assert build_trace_metadata(None) == {KEY_INSTALL_CHANNEL: "git"}

    def test_server_values_win_over_client_supplied(self, monkeypatch):
        monkeypatch.setattr(build_info, "install_channel", lambda: "git")
        monkeypatch.setattr(build_info, "_dist_version", lambda name: "1.2.3")
        merged = build_trace_metadata(
            {KEY_SERVER_VERSION: "spoof", KEY_ANTON_VERSION: "spoof", KEY_INSTALL_CHANNEL: "spoof"}
        )
        assert merged == {
            KEY_SERVER_VERSION: "1.2.3",
            KEY_ANTON_VERSION: "1.2.3",
            KEY_INSTALL_CHANNEL: "git",
        }

    def test_does_not_mutate_base(self, monkeypatch):
        monkeypatch.setattr(build_info, "install_channel", lambda: "git")
        monkeypatch.setattr(build_info, "_dist_version", lambda name: "1.2.3")
        base = {"harness": "anton"}
        build_trace_metadata(base)
        assert base == {"harness": "anton"}

    def test_a_broken_stamp_never_fails_the_turn(self, monkeypatch):
        # Telemetry on the hot path: an unattributable turn is acceptable, a
        # failed turn is not. Anything unexpected degrades to the base dict.
        def _boom(*_args, **_kwargs):
            raise RuntimeError("dist metadata exploded")

        monkeypatch.setattr(build_info, "_dist_version", _boom)
        monkeypatch.setattr(build_info, "install_channel", _boom)
        assert build_trace_metadata({"harness": "anton"}) == {"harness": "anton"}

    def test_real_process_reports_its_own_versions(self):
        # No monkeypatching: proves the keys resolve against real installed
        # metadata in the test environment (cowork-server + anton are both
        # installed), not just against fakes.
        merged = build_trace_metadata(None)
        assert merged[KEY_SERVER_VERSION]
        assert merged[KEY_ANTON_VERSION]
        assert merged[KEY_INSTALL_CHANNEL] in build_info.VALID_CHANNELS


class TestStampReachesTheHarness:
    """The function being correct is not the same as the turn carrying it —
    this drives a real POST /responses so a future refactor of the handler
    can't quietly drop the stamp."""

    def test_responses_turn_forwards_build_metadata(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        captured: dict = {}

        class _CapturingHarness:
            id = "stub"

            def stream_response(self, **kwargs):
                captured.update(kwargs)
                return None

            async def formatter(self, stream, model, event_sink):
                event_sink(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "ok"},
                )
                if False:
                    yield

        from cowork.server import create_app

        with patch("cowork.handlers.responses.get_harness", return_value=_CapturingHarness()):
            client = TestClient(create_app())
            res = client.post("/api/v1/responses/", json={"input": "hello", "stream": False})
        assert res.status_code == 200, res.text

        metadata = captured["trace_metadata"]
        assert metadata[KEY_SERVER_VERSION]
        assert metadata[KEY_ANTON_VERSION]
        assert metadata[KEY_INSTALL_CHANNEL]
