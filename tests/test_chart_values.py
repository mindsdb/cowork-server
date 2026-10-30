"""The cowork-server chart's values: datasource switches, the draft-preview
host and the pod's shutdown.

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

The draft-preview host is one Ingress, one container port and the
``COWORK_DRAFT_*`` variables, all in the base and templated from two
per-environment names, ``draftHost`` and ``webOrigin``. None of them fails at
render time when they disagree: a wrong port answers 502, a host the listener
does not know answers 404 (``draft_host_unknown``), a host outside the web
build's list is refused by the page, and a wrong web origin blocks the frame
(CSP frame-ancestors). So the tests pin every reference to its one name, keep
the per-environment files to the names alone, and pin the names themselves.

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

The probes keep a slow dependency from restarting or unrouting the pod:
readiness reads ``/api/v1/health/ready`` and liveness ``/api/v1/health/live``,
neither of which touches the database, the file store or settings, and the
liveness delay leaves the boot room to finish. The ``deployment`` subchart
that renders them is pinned to one version, so a new release of it can't
change them unreviewed.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from _hosted_command import graceful_shutdown_seconds, hosted_command, image_cmd

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
# The draft host of each environment. cowork's web build frames exactly these
# (src/shared/draft-link-origins.ts there) and refuses a link to any other.
DRAFT_HOSTS = {
    "dev": "draft-dev.mindshub.co",
    "staging": "draft-staging.mindshub.co",
    "prod": "draft.mindshub.co",
}
DRAFT_HOST = "{{ $.Values.draftHost }}"
# The Ingress refuses to render without a host, which would make it a catch-all.
REQUIRED_DRAFT_HOST = '{{ required "set deployment.draftHost to enable ingress_secondary" $.Values.draftHost }}'


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


def test_the_draft_host_is_off_in_the_base():
    """A PR environment renders the base alone: no draft Ingress, and an empty
    COWORK_DRAFT_PORT, which starts no listener."""
    deployment = _deployment(BASE)
    assert (deployment["draftHost"], deployment["webOrigin"]) == ("", "")
    assert deployment["ingress_secondary"]["enabled"] is False


def test_every_draft_host_reference_is_the_draft_host_name(base):
    """The host the links point at, the hosts the listener answers for, the
    Ingress host and the upstream Host the Ingress pins are one name, so a typo
    cannot split them. The frame origin is the other name."""
    secondary = _deployment(BASE)["ingress_secondary"]
    assert _entry(base, "COWORK_DRAFT_HOSTS").value == DRAFT_HOST
    assert _entry(base, "COWORK_DRAFT_ORIGIN").value == "{{ with $.Values.draftHost }}https://{{ . }}{{ end }}"
    assert [host["host"] for host in secondary["hosts"]] == [REQUIRED_DRAFT_HOST]
    assert secondary["annotations"]["nginx.ingress.kubernetes.io/upstream-vhost"] == DRAFT_HOST
    assert _entry(base, "COWORK_WEB_ORIGIN").value == "{{ $.Values.webOrigin }}"


def test_the_draft_port_is_the_draft_listener(base):
    """The draft Ingress, the Service port it targets, the container port and
    COWORK_DRAFT_PORT are one port, and the Service sends it to the container
    port named draft, never to the API's. The listener binds
    COWORK_SERVER_HOST, not the command's --host. A drift answers 502 or puts
    the API app behind the draft host."""
    deployment = _deployment(BASE)
    port = {p["name"]: p["containerPort"] for p in deployment["additionalPorts"]}["draft"]
    service = {p["name"]: p for p in deployment["service"]["additionalPorts"]}["draft"]
    paths = [path for host in deployment["ingress_secondary"]["hosts"] for path in host["paths"]]
    assert port != deployment["service"]["targetPort"]
    assert _entry(base, "COWORK_DRAFT_PORT").value == f"{{{{ if $.Values.draftHost }}}}{port}{{{{ end }}}}"
    assert (service["port"], service["targetPort"]) == (port, "draft")
    assert [(p["path"], p["serviceName"], p["servicePort"]) for p in paths] == [("/_t/", deployment["name"], port)]
    assert _entry(base, "COWORK_SERVER_HOST").value == "0.0.0.0"


@pytest.mark.parametrize("env_values", ENVIRONMENTS, ids=lambda p: p.name)
def test_an_environment_names_its_draft_host_and_nothing_more(env_values):
    """The rest stays in the base, so an override cannot pull one reference
    away from the others. Helm replaces a list wholesale, so an environment
    that set the ports or the Ingress hosts would drop the base's. The web app
    is served from the primary Ingress host, the only page allowed to frame a
    draft. An environment the web build has no draft host for keeps the base,
    which is off."""
    environment = env_values.stem.removeprefix("values-")
    deployment = _deployment(env_values)
    expected = DRAFT_HOSTS.get(environment)
    assert deployment.get("draftHost") == expected, f"{env_values.name} names a draft host the web build does not frame"
    if expected is None:
        assert "webOrigin" not in deployment and "ingress_secondary" not in deployment, env_values.name
    else:
        primary = [host["host"] for host in (deployment.get("ingress") or {}).get("hosts") or []]
        assert primary and deployment.get("webOrigin") == f"https://{primary[0]}", env_values.name
        assert deployment.get("ingress_secondary") == {"enabled": True}, env_values.name
    assert "additionalPorts" not in deployment
    assert "additionalPorts" not in (deployment.get("service") or {})


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


def test_the_image_command_is_the_final_stages_last_cmd(monkeypatch, tmp_path):
    """Docker runs the final stage's last CMD; a CMD in an earlier stage, or
    an earlier one in the final stage, never reaches the image."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python AS builder\n"
        'CMD ["builder", "--timeout-graceful-shutdown", "2"]\n'
        "FROM python AS final\n"
        'CMD ["replaced"]\n'
        'CMD ["final", "--timeout-graceful-shutdown", "20"]\n'
    )
    monkeypatch.setattr("_hosted_command.DOCKERFILE", dockerfile)

    assert image_cmd() == ["final", "--timeout-graceful-shutdown", "20"]


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_the_pod_leaves_the_ingress_before_sigterm(values):
    assert _rendered(values)["lifecycle"] == {"preStop": {"exec": {"command": PRESTOP}}}
    assert _prestop_seconds(_rendered(values)) == 15


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_the_grace_period_covers_the_whole_shutdown(values):
    """Counted from the pod's deletion: the preStop sleep, the pre-drain that
    ends remote turns before uvicorn sees SIGTERM, uvicorn's drain bound, then
    the lifespan teardown's bounded waits (the draft listener's own drain,
    another pass over turns, channel tasks, scratchpad closes), and a margin
    for its unbounded closes.

    The channel drain cancels the tasks still running at its timeout, then
    waits up to its cancel_timeout for them to unwind: a cancelled channel turn
    writes its controller cancel flag, which takes up to
    CANCEL_REQUEST_SECONDS, before it ends."""
    from cowork.channels.webhooks import drain_background_tasks
    from cowork.common.chat_session import drain_scratchpad_closes
    from cowork.draft_host.listener import (
        _GRACEFUL_SHUTDOWN_SECONDS as DRAFT_LISTENER_DRAIN_SECONDS,
    )
    from cowork.streaming.cancellation import CANCEL_REQUEST_SECONDS
    from cowork.streaming.registry import TURN_SHUTDOWN_GRACE_SECONDS

    channel_cancel_wait = (
        inspect.signature(drain_background_tasks).parameters["cancel_timeout"].default
    )
    assert channel_cancel_wait >= CANCEL_REQUEST_SECONDS
    deployment = _rendered(values)
    teardown = (
        DRAFT_LISTENER_DRAIN_SECONDS
        + TURN_SHUTDOWN_GRACE_SECONDS
        + _timeout_default(drain_background_tasks) + channel_cancel_wait
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


# The liveness delay the boot has: kill comes at this plus failureThreshold *
# periodSeconds, and staging boots under load took about 120s on 28 Oct.
LIVENESS_DELAY_SECONDS = 180
DEPLOYMENT_SUBCHART_VERSION = "1.0.47"


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_readiness_reads_the_route_that_does_no_io(values):
    probe = _rendered(values)["readinessProbe"]
    assert probe["enabled"] is True
    assert probe["path"] == "/api/v1/health/ready"


@pytest.mark.parametrize("values", RENDERED, ids=lambda p: p.name)
def test_liveness_reads_live_and_leaves_the_boot_its_delay(values):
    probe = _rendered(values)["livenessProbe"]
    assert probe["enabled"] is True
    assert probe["path"] == "/api/v1/health/live"
    assert probe["initialDelaySeconds"] == LIVENESS_DELAY_SECONDS
    # The process may still be killed for a real hang: 60s of failures.
    assert probe["failureThreshold"] * probe["periodSeconds"] == 60


def test_both_probe_routes_are_async_so_they_never_wait_for_a_thread():
    """A sync route runs on anyio's default threads, which sync dependencies
    and file reads fill, so a probe there would fail with the pool and not
    with the process."""
    from cowork.api.v1.endpoints import health

    assert inspect.iscoroutinefunction(health.live)
    assert inspect.iscoroutinefunction(health.ready)


def test_the_deployment_subchart_is_pinned_to_one_version():
    requirements = yaml.safe_load((CHART / "requirements.yaml").read_text())
    (subchart,) = [dep for dep in requirements["dependencies"] if dep["alias"] == "deployment"]
    assert subchart["version"] == DEPLOYMENT_SUBCHART_VERSION
