"""The cowork-server chart's datasource values: off by default, one identity, no duplicates.

Cloud datasource grants for hosted turns sit behind ``COWORK_TURN_DATASOURCE_ENABLED``.
Three facts about the values files hold the rollout together, and each fails
quietly on its own:

- The flag must be ``"false"`` in the base values and ``"true"`` only in the
  environments this file lists. A PR environment composes the base alone.
- The producer identity must come from the shared ``datasource-service-keys``
  bundle and never from the legacy shared internal secret. auth's chart guard
  covers auth's own references only, so a renamed bundle or key is caught here
  on this side. The references are optional: without the Secret the producer
  raises ``ProductPermissionUnavailable`` on the first datasource turn, which is
  the same closed door, and a namespace that has no bundle yet still deploys.
- No env name may be declared twice across the base and a per-environment
  file: a duplicate renders fine and then either fails the upgrade or silently
  empties the variable. The controller's guard exists because that emptied its
  worker image in prod; this chart had no such guard.
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


def test_no_datasource_method_is_offered_in_the_released_values(base):
    """The list of methods a deployment may run, empty until a gate says
    otherwise. Declared here so turning one on is an edit to a value rather
    than a new line nobody remembers to add."""
    assert _entry(base, "COWORK_DATASOURCE_CAPABILITIES")["value"] == ""
    for env_values in ENVIRONMENTS:
        declared = [e for e in _env(env_values) if e.get("name") == "COWORK_DATASOURCE_CAPABILITIES"]
        assert declared == [], f"{env_values.name} declares the manifest; the base already does"


def test_the_gateway_address_is_the_service_name_and_is_set(base):
    """An address, not a permission: it does nothing while the feature is off,
    and without it a saved connection is never checked and waits forever. The
    Service name resolves namespace-relative, so one value is right everywhere."""
    assert _entry(base, "COWORK_TURN_DATASOURCE_GATEWAY_BASE_URL")["value"] == "http://mindshub-inference"


def test_the_producer_identity_is_the_producer_role_of_the_shared_bundle(base):
    references = {
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID": "DATASOURCE_PRODUCER_KEY_ID",
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY": "DATASOURCE_PRODUCER_KEY",
    }
    for name, key in references.items():
        ref = _entry(base, name)["valueFrom"]["secretKeyRef"]
        assert ref == {"name": BUNDLE, "key": key, "optional": True}, f"{name} must reference {BUNDLE}/{key}, optional"


@pytest.mark.parametrize("env_values", ENVIRONMENTS, ids=lambda p: p.name)
def test_no_env_name_is_declared_twice_per_environment(base, env_values):
    names = [entry["name"] for entry in base + _env(env_values) if "name" in entry]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"{env_values.name} renders {duplicates} more than once with values.yaml"
