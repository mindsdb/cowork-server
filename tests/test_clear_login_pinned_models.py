"""Tests for clear_login_pinned_models — heal login-written model pins.

The desktop SSO flow historically pinned ANTON_PLANNING_MODEL=latest:sonnet /
ANTON_CODING_MODEL=latest:haiku on every sign-in and synced them to the DB,
making every login an explicit model pick. That defeats enabled-aware default
resolution, so a free-tier user is stuck on a locked model with no self-serve
recovery (ENG-597 / ENG-739). The DB is authoritative, so dropping the pin
desktop-side does not heal an already-pinned user; this boot-time cleanup does.
"""

from cowork.common.settings.user_settings import Provider
from cowork.db.session import get_open_session
from cowork.migrations import clear_login_pinned_models
from cowork.services.settings import SettingService

MINDS = Provider.MINDS_CLOUD.value


def _seed(session, key, value):
    SettingService(session).upsert_setting(key, value)


def _value(session, key):
    row = SettingService(session)._fetch_row(key)
    return row.value if row else None


def _cleanup(session, *keys):
    svc = SettingService(session)
    for key in keys:
        svc.delete_setting(key)


def test_clears_latest_pin_on_minds_cloud(tmp_path):
    session = get_open_session()
    try:
        _seed(session, "planning_provider", MINDS)
        _seed(session, "coding_provider", MINDS)
        _seed(session, "planning_model", "latest:sonnet")
        _seed(session, "coding_model", "latest:haiku")

        assert clear_login_pinned_models(session) is True

        # Pins removed → the enabled-aware default can now resolve per tier.
        assert _value(session, "planning_model") is None
        assert _value(session, "coding_model") is None
        # Provider rows are left intact.
        assert _value(session, "planning_provider") == MINDS
    finally:
        _cleanup(session, "planning_provider", "coding_provider",
                 "planning_model", "coding_model")


def test_preserves_explicit_bare_alias_pick(tmp_path):
    """A deliberate picker/user choice writes a bare alias — never cleared."""
    session = get_open_session()
    try:
        _seed(session, "planning_provider", MINDS)
        _seed(session, "planning_model", "mindshub_air")

        assert clear_login_pinned_models(session) is False
        assert _value(session, "planning_model") == "mindshub_air"
    finally:
        _cleanup(session, "planning_provider", "planning_model")


def test_does_not_touch_byok_provider(tmp_path):
    """A latest: value on a direct provider (shouldn't happen, but be safe)."""
    session = get_open_session()
    try:
        _seed(session, "planning_provider", Provider.ANTHROPIC.value)
        _seed(session, "planning_model", "latest:sonnet")

        assert clear_login_pinned_models(session) is False
        assert _value(session, "planning_model") == "latest:sonnet"
    finally:
        _cleanup(session, "planning_provider", "planning_model")


def test_idempotent_and_noop_when_clean(tmp_path):
    session = get_open_session()
    try:
        _seed(session, "planning_provider", MINDS)
        _seed(session, "planning_model", "latest:sonnet")

        assert clear_login_pinned_models(session) is True
        # Second boot: nothing left to clear.
        assert clear_login_pinned_models(session) is False

        # A stale re-login re-introduces the pin; a later boot self-heals.
        _seed(session, "planning_model", "latest:sonnet")
        assert clear_login_pinned_models(session) is True
        assert clear_login_pinned_models(session) is False
    finally:
        _cleanup(session, "planning_provider", "planning_model")
