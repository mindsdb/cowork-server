"""ENG-1125: UserSettings is the single source of truth for the settings
contract. These guards fail CI the moment the migration's env map or the
provider normalization could drift from the model.
"""
from cowork.common.settings.user_settings import (
    ENV_ALIAS_TO_SETTING,
    SETTING_ENV_ALIASES,
    UserSettings,
    normalize_provider_value,
)


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


def test_migration_normalizer_delegates_to_the_canonical_one():
    from cowork.migrations import _normalize_provider_value

    assert (
        _normalize_provider_value(
            "openai-compatible", {"ANTON_MINDS_API_KEY": "sk-x"}
        )
        == "minds_cloud"
    )
    assert _normalize_provider_value("openai-compatible", {}) == "openai_compatible"


# ─── which env var actually overrides a setting ────────────────────────────
#
# There are two env paths onto a UserSettings field and they are easy to confuse
# — the artifact_autopublish_enabled docstring got it wrong until these tests
# were written:
#
#   SETTING_ENV_ALIASES  (ANTON_*)  seeds DB rows from a .env file, and org
#     deployments skip that seed entirely (dev_setup._migrate_env_to_db_if_local),
#     so on the cloud it overrides nothing.
#   the FIELD NAME itself                is read by pydantic-settings whenever no
#     DB row supplies the field. Process-global: it applies to every tenant of
#     the deployment, with no row for an operator to find afterwards.
#
# Anything that documents "set X to turn this on" has to name the second one.

# Both cases drive the value AWAY from whatever the field currently defaults to,
# rather than asserting a literal. The default is not a fixed part of the
# contract these tests are about — it is flipped on deploy branches — and a test
# that hardcodes it reports a deliberate default change as an env-plumbing bug.
_DEFAULT = UserSettings.model_fields["artifact_autopublish_enabled"].default
_OPPOSITE = str(not _DEFAULT).lower()


def test_field_name_env_var_overrides_a_setting(monkeypatch):
    monkeypatch.delenv("ANTON_ARTIFACT_AUTOPUBLISH", raising=False)
    monkeypatch.setenv("ARTIFACT_AUTOPUBLISH_ENABLED", _OPPOSITE)

    assert UserSettings().artifact_autopublish_enabled is not _DEFAULT


def test_anton_alias_alone_does_not_override_a_setting(monkeypatch):
    """The ANTON_* alias is a .env→DB seed key, not a live override. Asserted so
    the field's own description cannot drift back to naming it."""
    monkeypatch.delenv("ARTIFACT_AUTOPUBLISH_ENABLED", raising=False)
    monkeypatch.setenv("ANTON_ARTIFACT_AUTOPUBLISH", _OPPOSITE)

    assert UserSettings().artifact_autopublish_enabled is _DEFAULT


def test_autopublish_description_names_the_override_that_works(monkeypatch):
    text = UserSettings.model_fields["artifact_autopublish_enabled"].description
    assert "ARTIFACT_AUTOPUBLISH_ENABLED" in text
    # Naming the alias is fine — saying it is the override is not. The
    # description must mark it as NOT the bypass.
    assert "NOT that" in text or "not that" in text
