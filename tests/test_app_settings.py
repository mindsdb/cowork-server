import pytest
from pydantic import ValidationError

from cowork.common.settings.app_settings import AppSettings, TurnQueueSettings


def test_app_settings_ignores_generic_server_port(monkeypatch):
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


def test_app_settings_reads_cowork_listen_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9999")
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_rejects_invalid_cowork_listen_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "invalid")
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_app_settings_reads_desktop_cowork_server_port(monkeypatch):
    # The desktop app hands the derived per-user port to the sidecar as
    # COWORK_SERVER_PORT. Regression: dropping this alias made the server
    # bind :26866 while the app health-polled the derived port.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "27735")

    settings = AppSettings(_env_file=None)

    assert settings.port == 27735


def test_app_settings_listen_port_wins_over_server_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9999")
    monkeypatch.setenv("COWORK_SERVER_PORT", "27735")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_ignores_k8s_injected_server_port_uri(monkeypatch):
    # K8s auto-injects COWORK_SERVER_PORT=tcp://<ip>:<port> on any pod
    # colocated with a `cowork-server` Service — must fall back to the
    # default, not fail int parsing.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "tcp://10.0.0.5:26866")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


def test_app_settings_listen_port_wins_over_service_link_uri(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9010")
    monkeypatch.setenv("COWORK_SERVER_PORT", "tcp://10.3.0.12:26866")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9010


def test_app_settings_rejects_invalid_legacy_server_port(monkeypatch):
    # Only the K8s tcp:// URI shape is discarded; other malformed values on
    # the legacy alias must still fail loudly.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "invalid")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_app_settings_ignores_generic_server_host(monkeypatch):
    monkeypatch.delenv("COWORK_SERVER_HOST", raising=False)
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")

    settings = AppSettings(_env_file=None)

    assert settings.host == "127.0.0.1"


def test_app_settings_tenancy_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.tenancy_mode == "local"


def test_app_settings_reads_tenancy_mode_org(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    settings = AppSettings(_env_file=None)

    assert settings.tenancy_mode == "org"


def test_app_settings_rejects_invalid_tenancy_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "multi")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_hermes_hidden_from_harness_options_in_org_mode(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.common.settings.user_settings import _harness_options
    import cowork.harnesses.anton_harness.harness  # noqa: F401  register anton
    import cowork.harnesses.hermes_harness.harness  # noqa: F401  register hermes

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        options = _harness_options()
        assert "anton" in options
        assert "hermes" not in options
    finally:
        get_app_settings.cache_clear()


def test_hermes_available_in_local_mode(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.common.settings.user_settings import _harness_options
    import cowork.harnesses.anton_harness.harness  # noqa: F401
    import cowork.harnesses.hermes_harness.harness  # noqa: F401

    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    try:
        options = _harness_options()
        assert "anton" in options
        assert "hermes" in options
    finally:
        get_app_settings.cache_clear()


def test_app_settings_identity_enforce_defaults_to_enforce(monkeypatch):
    """A deployment that loses the env var must not quietly start letting
    identity-less requests through, so the closed value is the default and
    audit has to be asked for."""
    monkeypatch.delenv("COWORK_IDENTITY_ENFORCE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.identity_enforce == "enforce"


def test_app_settings_identity_enforce_can_opt_into_audit(monkeypatch):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")

    settings = AppSettings(_env_file=None)

    assert settings.identity_enforce == "audit"


def test_app_settings_rejects_invalid_identity_enforce(monkeypatch):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "strict")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_turn_queue_settings_is_remote(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    assert TurnQueueSettings().is_remote is True

    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess")
    assert TurnQueueSettings().is_remote is False

    monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    assert TurnQueueSettings().is_remote is False  # default is "inprocess"


def test_app_settings_organization_boundary_defaults_to_enforce(monkeypatch):
    monkeypatch.delenv("COWORK_ORGANIZATION_BOUNDARY_MODE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.organization_boundary_mode == "enforce"


@pytest.mark.parametrize("mode", ["audit", "enforce"])
def test_app_settings_reads_organization_boundary_mode(monkeypatch, mode):
    monkeypatch.setenv("COWORK_ORGANIZATION_BOUNDARY_MODE", mode)

    settings = AppSettings(_env_file=None)

    assert settings.organization_boundary_mode == mode


def test_app_settings_rejects_invalid_organization_boundary_mode(monkeypatch):
    monkeypatch.setenv("COWORK_ORGANIZATION_BOUNDARY_MODE", "strict")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_app_settings_organization_switch_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("COWORK_ORGANIZATION_SWITCH_ENABLED", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.organization_switch_enabled is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_app_settings_reads_organization_switch_enabled(monkeypatch, raw, expected):
    monkeypatch.setenv("COWORK_ORGANIZATION_SWITCH_ENABLED", raw)

    settings = AppSettings(_env_file=None)

    assert settings.organization_switch_enabled is expected


def test_app_settings_rejects_invalid_organization_switch_enabled(monkeypatch):
    monkeypatch.setenv("COWORK_ORGANIZATION_SWITCH_ENABLED", "sometimes")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)
