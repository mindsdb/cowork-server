"""The cowork-server chart's datasource values: off by default, one identity, no duplicates.

Cloud datasource grants for hosted turns sit behind ``COWORK_TURN_DATASOURCE_ENABLED``
and ``COWORK_DATASOURCE_CAPABILITIES``. Three facts about the values files hold
the rollout together, and each fails quietly on its own:

- Each switch is a scalar under ``deployment:``, ``datasourceTurnsEnabled`` and
  ``datasourceCapabilities``, rendered into one env entry. The base leaves both
  off, and only the environments this file lists may set them. A PR environment
  composes the base alone, and an environment declaring the env name itself
  would render it twice.
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
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "deployment" / "cowork-server"
BASE = CHART / "values.yaml"
ENVIRONMENTS = sorted(CHART.glob("values-*.yaml"))
ENV_KEYS = ("globalEnvs", "globalEnvsSecondary", "extraEnvs")
BUNDLE = "datasource-service-keys"
FLAG = "COWORK_TURN_DATASOURCE_ENABLED"
CAPABILITIES = "COWORK_DATASOURCE_CAPABILITIES"
# Environments the release gate has switched datasource grants on in. Empty
# until it does; the gate's own change adds the environment here and sets
# both deployment scalars in its values file.
ENABLED_ENVIRONMENTS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Env:
    name: str
    value: str | None = None
    secret: str | None = None
    key: str | None = None
    optional: bool = False


def _deployment(path: Path) -> dict:
    """The ``deployment:`` block of a values file, the subchart's own values."""
    return (yaml.safe_load(path.read_text()) or {}).get("deployment") or {}


def _env(path: Path) -> list[_Env]:
    """Every entry of the three env lists the subchart renders into the container."""
    entries: list[_Env] = []
    deployment = _deployment(path)
    for key in ENV_KEYS:
        for raw in deployment.get(key) or []:
            if not isinstance(raw, dict) or "name" not in raw:
                continue
            ref = (raw.get("valueFrom") or {}).get("secretKeyRef") or {}
            entries.append(
                _Env(
                    name=raw["name"],
                    value=raw.get("value"),
                    secret=ref.get("name"),
                    key=ref.get("key"),
                    optional=ref.get("optional") is True,
                )
            )
    return entries


def _entry(entries: list[_Env], name: str) -> _Env:
    matches = [entry for entry in entries if entry.name == name]
    assert len(matches) == 1, f"expected {name} once, found {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def base() -> list[_Env]:
    return _env(BASE)


def test_datasource_grants_are_off_in_the_base_and_on_only_where_the_gate_passed(base):
    assert _deployment(BASE)["datasourceTurnsEnabled"] == "false"
    assert _entry(base, FLAG).value == "{{ $.Values.datasourceTurnsEnabled }}"
    for env_values in ENVIRONMENTS:
        environment = env_values.stem.removeprefix("values-")
        expected = "true" if environment in ENABLED_ENVIRONMENTS else None
        assert _deployment(env_values).get("datasourceTurnsEnabled") == expected, f"{env_values.name} flips the grants"
        assert not [e for e in _env(env_values) if e.name == FLAG], (
            f"{env_values.name} declares {FLAG}; set deployment.datasourceTurnsEnabled instead"
        )


def test_no_datasource_method_is_offered_in_the_released_values(base):
    """The list of methods a deployment may run, empty until a gate says
    otherwise, and set only where grants are on."""
    assert _deployment(BASE)["datasourceCapabilities"] == ""
    assert _entry(base, CAPABILITIES).value == "{{ $.Values.datasourceCapabilities }}"
    for env_values in ENVIRONMENTS:
        environment = env_values.stem.removeprefix("values-")
        if environment not in ENABLED_ENVIRONMENTS:
            assert "datasourceCapabilities" not in _deployment(env_values), (
                f"{env_values.name} offers a method where grants are off"
            )
        assert not [e for e in _env(env_values) if e.name == CAPABILITIES], (
            f"{env_values.name} declares {CAPABILITIES}; set deployment.datasourceCapabilities instead"
        )


def test_the_gateway_address_is_the_service_name_and_is_set(base):
    """An address, not a permission: it does nothing while the feature is off,
    and without it a saved connection is never checked and waits forever. The
    Service name resolves namespace-relative, so one value is right everywhere."""
    assert _entry(base, "COWORK_TURN_DATASOURCE_GATEWAY_BASE_URL").value == "http://mindshub-inference"


def test_the_producer_identity_is_the_producer_role_of_the_shared_bundle(base):
    references = {
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID": "DATASOURCE_PRODUCER_KEY_ID",
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY": "DATASOURCE_PRODUCER_KEY",
    }
    for name, key in references.items():
        entry = _entry(base, name)
        assert (entry.secret, entry.key, entry.optional) == (BUNDLE, key, True), (
            f"{name} must reference {BUNDLE}/{key}, optional"
        )


@pytest.mark.parametrize("env_values", ENVIRONMENTS, ids=lambda p: p.name)
def test_no_env_name_is_declared_twice_per_environment(base, env_values):
    names = [entry.name for entry in base + _env(env_values)]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"{env_values.name} renders {duplicates} more than once with values.yaml"
