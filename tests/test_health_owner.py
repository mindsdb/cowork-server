"""The /health endpoint echoes the per-install owner token (ENG-439).

The desktop app only adopts a running server whose /health `owner` matches
the token it generated, so it can never drive another OS user's sidecar on a
shared loopback port.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings


def _user_settings(**overrides):
    """A stand-in with every attribute `health()` actually reads.

    `_minds_runtime_credential_required` reads the resolved provider properties
    directly rather than through `getattr(..., None)`: that default would turn a
    future rename into a silent `False`, and `False` is what disables the
    desktop's wake barrier. Fakes therefore have to carry the real shape.
    """
    fields = {
        "config_status": {},
        "resolved_planning_provider": Provider.OPENAI,
        "resolved_coding_provider": Provider.OPENAI,
        "resolved_router_provider": Provider.OPENAI,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_health_reports_owner_from_app_settings():
    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="install-token-abc", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
    ):
        body = health.health()

    assert body["owner"] == "install-token-abc"
    assert body["status"] == "ok"


def test_health_owner_empty_when_unset():
    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
    ):
        body = health.health()

    # Empty owner = server advertises no identity → the app must not adopt it.
    assert body["owner"] == ""


def _health_with(tenancy_mode: str) -> dict:
    from cowork.api.v1.endpoints import health

    with (
        patch.object(
            health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode=tenancy_mode)
        ),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
    ):
        return health.health()


def test_health_reports_org_mode():
    """The client can't tell a hosted org deployment from an authenticated
    standalone one from ``config_ready`` alone — that says the deployment can
    run, not who may configure it. Onboarding needs the difference: in org mode
    provider config is admin-owned, so finalizing by writing it 403s a member.
    """
    assert _health_with("org")["org_mode"] is True


def test_health_org_mode_false_on_desktop():
    assert _health_with("local")["org_mode"] is False


@pytest.mark.parametrize(
    ("planning", "coding", "expected"),
    [
        (Provider.MINDS_CLOUD, Provider.OPENAI, True),
        (Provider.OPENAI, Provider.MINDS_CLOUD, True),
        (Provider.OPENAI, Provider.OPENAI, False),
    ],
)
def test_health_reports_when_required_roles_need_the_runtime_minds_credential(
    planning, coding, expected
):
    from cowork.api.v1.endpoints import health

    settings = _user_settings(
        resolved_planning_provider=planning,
        resolved_coding_provider=coding,
    )
    with (
        patch.object(
            health,
            "get_app_settings",
            return_value=SimpleNamespace(owner="", tenancy_mode="local"),
        ),
        patch.object(health, "get_user_settings", return_value=settings),
    ):
        body = health.health()

    assert body["minds_runtime_credential_required"] is expected


def test_health_runtime_requirement_uses_resolved_provider_fallback():
    from cowork.api.v1.endpoints import health

    settings = UserSettings(
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        minds_api_key=SecretStr("mdb-runtime"),
        openai_api_key=None,
        minds_url="https://api.mindshub.ai",
    )

    assert settings.resolved_planning_provider is Provider.MINDS_CLOUD
    assert settings.resolved_coding_provider is Provider.MINDS_CLOUD
    assert health._minds_runtime_credential_required(settings, org_mode=False) is True


def test_health_router_only_minds_use_does_not_gate_required_direct_roles():
    from cowork.api.v1.endpoints import health

    settings = _user_settings(resolved_router_provider=Provider.MINDS_CLOUD)

    assert health._minds_runtime_credential_required(settings, org_mode=False) is False


def test_health_org_mode_never_requires_the_desktop_runtime_credential():
    from cowork.api.v1.endpoints import health

    settings = _user_settings(
        resolved_planning_provider=Provider.MINDS_CLOUD,
        resolved_coding_provider=Provider.MINDS_CLOUD,
    )

    assert health._minds_runtime_credential_required(settings, org_mode=True) is False


# ─── aid: the join key the desktop app cannot compute itself (ENG-1689) ───────
#
# `turn_completed` carries `aid` on 100% of events and an identified person on
# 0%; cowork's own events are the mirror image. Neither side can reach the
# other, so per-user cost is unanswerable — we can name the 78 people who hit
# the spend ceiling but not what any of their turns cost. Serving anton's id
# here is what lets the app stamp it on an event that already knows the user.


def test_health_serves_antons_install_id_on_desktop():
    from types import SimpleNamespace
    from unittest.mock import patch

    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
        patch.object(health, "_anton_install_id", return_value="a1b2c3d4e5f60718"),
    ):
        body = health.health()

    assert body["aid"] == "a1b2c3d4e5f60718"


def test_health_withholds_the_id_in_org_mode():
    """In org mode the id fingerprints the SERVER, not the user — identical for
    every caller, and useless for the join because a web turn runs in a
    scratchpad pod rather than on this host. Publishing it would expose one
    machine's fingerprint to every user of the deployment for no benefit.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode="org")),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
        patch.object(health, "_anton_install_id", return_value="a1b2c3d4e5f60718"),
    ):
        body = health.health()

    assert body["aid"] == ""


def test_health_still_answers_when_the_id_cannot_be_resolved():
    """This is the readiness probe the app polls before mounting the renderer.
    An unresolvable analytics identifier must degrade to "" — never a 500, and
    never a raise that leaves the app waiting on a server it thinks is down.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from cowork.api.v1.endpoints import health

    # Patched on `anton.analytics`, NOT on the health module: `_anton_install_id`
    # imports it locally, so a module-level patch here is a no-op and the real
    # resolver runs — which is exactly how the first version of this test passed
    # while asserting nothing.
    import anton.analytics

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=_user_settings()),
        patch.object(
            anton.analytics, "get_installation_id", side_effect=RuntimeError("anton not importable")
        ),
    ):
        body = health.health()

    assert body["status"] == "ok"
    assert body["aid"] == ""


def test_the_real_resolver_returns_a_string_not_none():
    """Exercises `_anton_install_id` itself rather than a stub, so a change to
    anton's `get_installation_id` contract (it can return "unknown") surfaces
    here instead of putting None into the payload.
    """
    from cowork.api.v1.endpoints import health

    got = health._anton_install_id()
    assert isinstance(got, str)


def test_health_does_not_serve_the_unknown_sentinel():
    """`get_installation_id` returns the literal "unknown" when it cannot
    fingerprint the machine — a container with stripped networking whose
    fallback file is unwritable, or any getnode exception.

    anton stamps that SAME string on its own events, so passing it through
    would produce a join key that matches across every unfingerprintable
    machine and merges them into one identity. That is ENG-713's over-merge
    outcome reached without an alias, and worse than an absent key because the
    value looks valid.
    """
    from unittest.mock import patch

    import anton.analytics

    from cowork.api.v1.endpoints import health

    with patch.object(anton.analytics, "get_installation_id", return_value="unknown"):
        assert health._anton_install_id() == ""


def test_health_serves_a_real_looking_id_unchanged():
    """Control for the test above: the filter must reject only the sentinel,
    not mangle a genuine id.
    """
    from unittest.mock import patch

    import anton.analytics

    from cowork.api.v1.endpoints import health

    with patch.object(anton.analytics, "get_installation_id", return_value="a1b2c3d4e5f60718"):
        assert health._anton_install_id() == "a1b2c3d4e5f60718"
