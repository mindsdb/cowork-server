"""The cowork-server chart's values: datasource switches and the pod's shutdown.

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

The shutdown values hold answers in progress through a rollout
(README.md, "What a shutdown does to answers in progress"). Each environment
renders the base values with its own file merged over them, and a PR
environment renders the base alone, so each is checked as rendered:

- No values file sets the image's command or args, which would replace the
  Dockerfile's CMD and its uvicorn drain bound, and none sets a ``UVICORN_*``
  variable, which uvicorn reads as an option outside the command.
- The preStop sleep keeps the pod serving while the ingress stops routing to it.
- The grace period covers the preStop, the pre-drain, uvicorn's bound and the
  lifespan teardown's bounded waits, each read from where it is set.
- A rollout starts every new pod before it stops an old one.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from _hosted_command import graceful_shutdown_seconds, hosted_command

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


# What a pod's teardown may run past the waits that have a bound: the
# closes of the scratchpads, the launched backends and the HTTP clients.
UNBOUNDED_TEARDOWN_MARGIN_SECONDS = 10
PRESTOP = ["/bin/sh", "-c", "sleep 15"]
# Each file as an environment renders it; "values.yaml" alone is a PR environment.
RENDERED = [BASE, *ENVIRONMENTS]


def _merged(base: dict, overlay: dict) -> dict:
    """Helm's merge of an overlay onto the base: maps merge key by key, any
    other value replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merged(merged[key], value)
        else:
            merged[key] = value
    return merged


def _rendered(path: Path) -> dict:
    """The ``deployment:`` values an environment renders with."""
    return _deployment(BASE) if path == BASE else _merged(_deployment(BASE), _deployment(path))


def _prestop_seconds(deployment: dict) -> int:
    command = deployment["lifecycle"]["preStop"]["exec"]["command"]
    return int(re.fullmatch(r"sleep (\d+)", command[-1]).group(1))


def _timeout_default(function) -> float:
    return inspect.signature(function).parameters["timeout"].default


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_no_values_file_replaces_the_images_command(values):
    """The Dockerfile's CMD carries uvicorn's drain bound; a chart command or
    args would replace it (tests/_hosted_command.py)."""
    image = _deployment(values).get("image") or {}
    assert "command" not in image and "args" not in image, f"{values.name} sets the container command"
    uvicorn_options = [
        entry.name for entry in _env(values)
        if entry.name.startswith("UVICORN_") or entry.name == "WEB_CONCURRENCY"
    ]
    assert not uvicorn_options, f"{values.name} sets uvicorn options outside the command: {uvicorn_options}"


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_the_pod_leaves_the_ingress_before_sigterm(values):
    assert _rendered(values)["lifecycle"] == {"preStop": {"exec": {"command": PRESTOP}}}
    assert _prestop_seconds(_rendered(values)) == 15


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_the_grace_period_covers_the_whole_shutdown(values):
    """Counted from the pod's deletion: the preStop sleep, the pre-drain that
    ends remote turns before uvicorn sees SIGTERM, uvicorn's drain bound, then
    the lifespan teardown's bounded waits (another pass over turns, channel
    tasks, scratchpad closes), and a margin for its unbounded closes."""
    from cowork.channels.webhooks import drain_background_tasks
    from cowork.common.chat_session import drain_scratchpad_closes
    from cowork.streaming.registry import TURN_SHUTDOWN_GRACE_SECONDS

    deployment = _rendered(values)
    teardown = (
        TURN_SHUTDOWN_GRACE_SECONDS
        + _timeout_default(drain_background_tasks)
        + _timeout_default(drain_scratchpad_closes)
    )
    needed = (
        _prestop_seconds(deployment)
        + TURN_SHUTDOWN_GRACE_SECONDS
        + graceful_shutdown_seconds(hosted_command())
        + teardown
        + UNBOUNDED_TEARDOWN_MARGIN_SECONDS
    )
    assert deployment["terminationGracePeriodSeconds"] >= needed, (
        f"{values.name}: grace {deployment['terminationGracePeriodSeconds']}s, shutdown needs {needed}s"
    )


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_a_rollout_starts_every_new_pod_before_it_stops_an_old_one(values):
    assert _rendered(values)["deploymentStrategy"] == {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxSurge": "100%", "maxUnavailable": 0},
    }
