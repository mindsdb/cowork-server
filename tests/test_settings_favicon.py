"""The browser tab icon setting (ENG-3330): a per-user data URI, like nav_logo."""

from cowork.common.settings.user_settings import UserSettings, setting_is_org_scoped

ICON = "data:image/png;base64,iVBORw0KGgo="


def test_defaults_to_empty_so_the_built_in_icon_shows():
    assert UserSettings.model_fields["favicon"].default == ""


def test_is_a_personal_setting_like_the_sidebar_logo():
    assert setting_is_org_scoped("favicon") is setting_is_org_scoped("nav_logo")
    assert setting_is_org_scoped("favicon") is False


def test_round_trips_a_data_uri_and_is_not_a_secret():
    assert UserSettings.model_validate({"favicon": ICON}).favicon == ICON
    assert UserSettings.field_is_sensitive("favicon") is False
