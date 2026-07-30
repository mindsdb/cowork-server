"""ENG-1125: UserSettings is the single source of truth for the settings
contract. These guards fail CI the moment the migration's env map or the
provider normalization could drift from the model.
"""
from cowork.common.settings.env_boundary import (
    ENV_ALIAS_TO_SETTING,
    SETTING_ENV_ALIASES,
    env_to_db_updates,
    normalize_provider_value,
)
from cowork.common.settings.user_settings import UserSettings


def test_env_aliases_reference_only_real_fields():
    for setting_key, env_var in SETTING_ENV_ALIASES.items():
        assert setting_key in UserSettings.model_fields, setting_key
        assert env_var.startswith("ANTON_"), env_var
    # Model keys are CLI-only and must never be aliased into a bulk .env sync.
    assert "planning_model" not in SETTING_ENV_ALIASES
    assert "coding_model" not in SETTING_ENV_ALIASES


def test_env_alias_inverse_is_exact():
    assert ENV_ALIAS_TO_SETTING == {v: k for k, v in SETTING_ENV_ALIASES.items()}
    assert len(ENV_ALIAS_TO_SETTING) == len(SETTING_ENV_ALIASES)


def test_migration_env_map_is_derived_from_the_canonical_alias_map():
    # The migration must not keep its own hand-maintained copy of the map.
    from cowork.migrations import _ENV_TO_SETTING

    assert _ENV_TO_SETTING is ENV_ALIAS_TO_SETTING


def test_show_dots_default_is_true():
    # The client (App.jsx) had seeded show_dots False while the model default
    # is True — the exact drift ENG-1125 removes. Pin the canonical value so a
    # future flip has to be deliberate.
    assert UserSettings.model_fields["show_dots"].get_default() is True


def test_normalize_provider_value_single_implementation():
    # openai-compatible + a Minds key really means minds_cloud.
    assert (
        normalize_provider_value("openai-compatible", minds_key_present=True)
        == "minds_cloud"
    )
    # Without a Minds key it is a genuine custom endpoint.
    assert (
        normalize_provider_value("openai-compatible", minds_key_present=False)
        == "openai_compatible"
    )
    # Hyphen → underscore canonicalization for everything else.
    assert normalize_provider_value("anthropic", minds_key_present=True) == "anthropic"
    assert (
        normalize_provider_value("openai_compatible", minds_key_present=False)
        == "openai_compatible"
    )


def test_env_to_db_updates_normalizes_providers():
    # The inbound .env->DB conversion: a Minds key alongside an openai-compatible
    # provider means minds_cloud; without it, a genuine custom endpoint.
    with_key = env_to_db_updates(
        {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_MINDS_API_KEY": "sk-x"}
    )
    assert with_key["planning_provider"] == "minds_cloud"
    without_key = env_to_db_updates({"ANTON_PLANNING_PROVIDER": "openai-compatible"})
    assert without_key["planning_provider"] == "openai_compatible"
    # Unmapped / empty vars are skipped.
    assert env_to_db_updates({"ANTON_TERMS_CONSENT": "true", "ANTON_MINDS_URL": ""}) == {}
