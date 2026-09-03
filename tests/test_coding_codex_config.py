from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from cowork.coding.contracts import PermissionMode
from cowork.coding.engines import codex_config
from cowork.coding.engines.base import EngineSessionConfig
from cowork.coding.runtime_protocol import RuntimeExecutionConfig

INJECTED_EFFORT = 'high"\nmodel="x'


def test_runtime_execution_config_rejects_values_outside_the_codex_vocabulary() -> None:
    for field in ("reasoning_effort", "service_tier", "personality"):
        with pytest.raises(ValidationError):
            RuntimeExecutionConfig.model_validate({
                "engine_id": "codex",
                "model": "fable",
                "permission_mode": "workspace",
                field: INJECTED_EFFORT,
            })


@pytest.mark.parametrize("value", ["a\tb", "nul\x00byte", 'quote " and \\ slash', "line\nbreak\x7f"])
def test_toml_string_round_trips_control_characters(value: str) -> None:
    encoded = codex_config.toml_string(value)

    assert "\n" not in encoded
    assert "\x00" not in encoded
    assert tomllib.loads(f"value = {encoded}")["value"] == value


def test_every_codex_override_is_a_single_toml_assignment() -> None:
    launch = codex_config.prepare_launch(
        EngineSessionConfig(
            model="fable",
            permission_mode=PermissionMode.workspace,
            reasoning_effort=INJECTED_EFFORT,
            service_tier=INJECTED_EFFORT,
            personality=INJECTED_EFFORT,
            web_search=True,
        ),
        Path("/workspace"),
        "http://127.0.0.1:26866/api/v1/coding/inference",
    )

    assert all("\n" not in override for override in launch.config_overrides)
    parsed = tomllib.loads("\n".join(launch.config_overrides))
    assert parsed["model_reasoning_effort"] == INJECTED_EFFORT
    assert parsed["service_tier"] == INJECTED_EFFORT
    assert parsed["personality"] == INJECTED_EFFORT
    assert parsed["model"] == "fable"
