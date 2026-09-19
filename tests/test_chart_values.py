"""The cowork-server chart's datasource values: off by default, one identity, no duplicates.

Cloud datasource grants for hosted turns sit behind ``COWORK_TURN_DATASOURCE_ENABLED``.
Three facts about the values files hold the rollout together, and each fails
quietly on its own:

- The flag must be ``"false"`` in the base values and ``"true"`` only in the
  environments this file lists. A PR environment composes the base alone.
- The producer identity must come from the shared ``datasource-service-keys``
  bundle, required, and never from the legacy shared internal secret. auth's
  chart guard covers auth's own references only, so a renamed bundle or key is
  caught here on this side.
- No datasource name may be declared twice across the base and a
  per-environment file: a duplicate renders fine and then either fails the
  upgrade or silently empties the variable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "deployment" / "cowork-server"
BASE = CHART / "values.yaml"
ENVIRONMENTS = sorted(CHART.glob("values-*.yaml"))
ENV_KEYS = ("globalEnvs", "globalEnvsSecondary", "extraEnvs")
BUNDLE = "datasource-service-keys"
FLAG = "COWORK_TURN_DATASOURCE_ENABLED"
# Environments the release gate has switched datasource grants on in. Empty
# until it does; the gate's own change adds the environment here and in its
# values file.
ENABLED_ENVIRONMENTS: frozenset[str] = frozenset()


def _env(path: Path) -> list[dict]:
    deployment = (yaml.safe_load(path.read_text()) or {}).get("deployment") or {}
    return [entry for key in ENV_KEYS for entry in deployment.get(key) or [] if isinstance(entry, dict)]


def _entry(entries: list[dict], name: str) -> dict:
    matches = [entry for entry in entries if entry.get("name") == name]
    assert len(matches) == 1, f"expected {name} once, found {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def base() -> list[dict]:
    return _env(BASE)


def test_datasource_grants_are_off_in_the_base_and_on_only_where_the_gate_passed(base):
    assert _entry(base, FLAG)["value"] == "false"
    for env_values in ENVIRONMENTS:
        environment = env_values.stem.removeprefix("values-")
        overrides = [entry.get("value") for entry in _env(env_values) if entry.get("name") == FLAG]
        expected = ["true"] if environment in ENABLED_ENVIRONMENTS else []
        assert overrides == expected, f"{env_values.name} sets {FLAG}={overrides}"


def test_the_producer_identity_is_the_producer_role_of_the_shared_bundle(base):
    references = {
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID": "DATASOURCE_PRODUCER_KEY_ID",
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY": "DATASOURCE_PRODUCER_KEY",
    }
    for name, key in references.items():
        ref = _entry(base, name)["valueFrom"]["secretKeyRef"]
        assert ref == {"name": BUNDLE, "key": key}, f"{name} must be a required reference to {BUNDLE}/{key}"


@pytest.mark.parametrize("env_values", ENVIRONMENTS, ids=lambda p: p.name)
def test_no_datasource_name_is_declared_twice_per_environment(base, env_values):
    names = [entry["name"] for entry in base + _env(env_values) if str(entry.get("name", "")).startswith("COWORK_TURN_DATASOURCE")]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"{env_values.name} renders {duplicates} more than once with values.yaml"
