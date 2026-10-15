# cowork-server

This helm chart is just using a subchart of our standardized deployment helm charts.

## Introduction

This chart bootstraps a highly available deployment on a [Kubernetes](http://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

## Prerequisites

- Kubernetes 1.10+ with Beta APIs enabled
- The kubectl binary
- The helm binary
- Helm diff plugin installed

## Installing the Chart

```bash
# dev
export SERVICE_NAME="cowork-server"
export CI_ENVIRONMENT_SLUG="dev"
export K8S_NAMESPACE="dev"
export HELM_CHART=$SERVICE_NAME
export CURRENT_HELM_CHART=$SERVICE_NAME
export HELM_IMG_TAG="latest" # Change this to the tag of the image you want to deploy


# Go into our deployment folder
cd deployment
# Update our helm subchart (fetches the pinned deployment subchart into charts/)...
helm dependencies update $SERVICE_NAME/
# View the diff of what you want to do
helm diff upgrade --namespace $K8S_NAMESPACE --allow-unreleased $CURRENT_HELM_CHART $HELM_CHART     -f $CURRENT_HELM_CHART/values.yaml     -f $CURRENT_HELM_CHART/values-${CI_ENVIRONMENT_SLUG}.yaml --set global.namespace="$K8S_NAMESPACE" --set global.image.tag="$HELM_IMG_TAG"
# Actually do it...
helm upgrade --namespace $K8S_NAMESPACE --install $CURRENT_HELM_CHART $HELM_CHART     -f $CURRENT_HELM_CHART/values.yaml     -f $CURRENT_HELM_CHART/values-${CI_ENVIRONMENT_SLUG}.yaml  --set global.namespace="$K8S_NAMESPACE" --set global.image.tag="$HELM_IMG_TAG"
```

Swap `CI_ENVIRONMENT_SLUG` / `K8S_NAMESPACE` for `staging` or `prod` to target those environments.

## Required cluster secrets

The chart references these Secrets in the target namespace:

- `cowork-db` — key `database_uri`, a Postgres SQLAlchemy URI. Consumed by the
  `db-migrate` initContainer (`alembic upgrade head`) and the app's
  `DATABASE_URI`.
- `mindsdb-secrets` — provider API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`.
- `datasource-service-keys` — the producer role of the cloud datasource
  identities: `DATASOURCE_PRODUCER_KEY_ID` and `DATASOURCE_PRODUCER_KEY`,
  declared and provisioned by the auth repository's secrets inventory. The
  references are optional: without the Secret the pod starts and every
  datasource turn is refused, so the bundle must exist before
  `COWORK_TURN_DATASOURCE_ENABLED` turns on, not before this deploys. PR
  environments get it from the keycloak chart's ephemeral secrets.

## Datasource grants for hosted turns

Two switches, both off in `values.yaml`, and both set per environment by the
release gate as scalars under `deployment:` in `values-<env>.yaml`. Never add
the env names themselves there: `extraEnvs` appends to the base list, so the
variable would render twice. `tests/test_chart_values.py` pins which
environments are on.

- `deployment.datasourceTurnsEnabled`, rendered as
  `COWORK_TURN_DATASOURCE_ENABLED`: `"true"` registers grants and queues the
  datasource block for hosted turns.
- `deployment.datasourceCapabilities`, rendered as
  `COWORK_DATASOURCE_CAPABILITIES`: the methods this deployment may run, as a
  versioned manifest such as
  `{"manifest_version": 1, "enabled": ["postgres:host-port"]}`. Empty offers
  none, so with only the first switch on no method is available. A method
  whose own spec says the adapters cannot run it stays unavailable whatever
  this lists.

`COWORK_TURN_DATASOURCE_GATEWAY_BASE_URL` is not a switch. It is where this
server reaches the gateway to check a connection someone just saved, the
inference Service by name, and the same in every environment.

Turn the switches on only after auth serves the datasource endpoints with the
bundle above, mindshub_inference serves `/v1/datasources/`, and a scratchpad
image with the datasource helper is deployed and older pods are recycled.
Rollback is `datasourceTurnsEnabled` off: no new grants are registered and no
datasource block is queued; encrypted records stay in auth and OAuth
connections are unaffected.

## Required cluster permissions

In `staging` and `prod` this chart renders a NetworkPolicy (`networkPolicy.enabled`
is true in those values files), so whoever runs the upgrade needs
`create networkpolicies` in the target namespace. Without it the release fails and
`--atomic` rolls the whole deploy back. CI runs as
`system:serviceaccount:infrastructure:<env>-gha-runner`, which holds the verb only
in the namespaces listed under `runner_deploy_namespaces` in
Kubernetes-Foundational-Services. See the `networkPolicy` block in `values.yaml`
for the two selector checks to run alongside it, since getting either one wrong
drops every real request without failing the release.

## Browser organization boundary

Every environment refuses a browser request whose tab names a different
organization than the auth gateway resolved: 426 for a missing
`X-Cowork-Expected-Organization-Id`, 409 for a malformed or mismatched one. This
is not configurable. An overlay that still carries the retired
`COWORK_ORGANIZATION_BOUNDARY_MODE` key boots normally and ignores it, so a
stale entry is safe to leave and safe to delete.

`COWORK_ORGANIZATION_SWITCH_ENABLED` shows or hides the organization picker.
`values-dev.yaml`, `values-staging.yaml`, and `values-prod.yaml` set it to
`"true"`, so the picker is on wherever an overlay applies. PR environments take
base values, where the code default `false` keeps it off. It is the product
enable and the lever to reach for first, because it is a values entry the
pipeline reapplies on every deploy. To hide the picker, set its `value` to
`"false"` under `deployment.extraEnvs` in that environment's values file and
ship the commit through CI. Restarting the pods does not apply a values edit,
and `kubectl set env` lasts only until the next deploy.
`COWORK_IDENTITY_ENFORCE=audit` also hides the picker, by dropping
`expectedOrganizationEnforced` from the capability, but it reopens the
no-principal path and leaves the boundary refusing anyway, so it is never the
right lever here.

Deploy this server only with the capability-aware Cowork client image. An older
client does not send the expected-organization header and receives 426. The
current client also sends no header until its access token carries a readable
`activate_organization` id, so a `missing expected organization` count alone
does not prove an old client is serving.

### Back out enforcement

No values entry turns enforcement off on the current image. Backing it out
means running code from before
[cowork-server#524](https://github.com/mindsdb/cowork-server/pull/524) together
with that code's values files. An earlier image deployed with today's values
files still enforces, because that code defaults the boundary mode to `enforce`
and today's overlays no longer set it.

| Environment | Kube context | Namespace | Helm release |
| --- | --- | --- | --- |
| Staging | `newdev` | `staging` | `cowork-server` |
| Production | `newprod` | `prod` | `cowork-server` |

**A Helm rollback cannot back this change out in either environment.**

- CI runs `helm upgrade` without `--history-max`, so Helm keeps only the last
  ten revisions of each release. Every upgrade and every rollback adds one and
  prunes the oldest. `helm history --max 256` only caps the listing and cannot
  bring a pruned revision back. Staging retains no revision from before the
  September 13 boundary change.
- Each pod's `db-migrate` initContainer runs `alembic upgrade head` from its
  own image. Every production image built before 2026-09-13 lacks
  `e2262a14c001_artifact_identities.py`, which reached production in the same
  release as the boundary change. A production rollback to one of those
  revisions fails with `Can't locate revision identified by 'e2262a14c001'`.
  The new pod never becomes ready, `--wait` times out, and the old pods keep
  serving with the boundary on.

To check a candidate yourself, read the history, then list the migrations its
image lacks, taking each SHA from its image tag:

```bash
helm --kube-context <context> -n <namespace> history cowork-server --max 256
git diff --name-only --diff-filter=A <candidate-sha> <deployed-sha> -- cowork/db/alembic/versions
```

**Revert in Git and ship the revert through CI.** Revert the cowork-server#524
merge and keep its `values-*.yaml` hunks. They restore
`COWORK_ORGANIZATION_BOUNDARY_MODE: "audit"` together with the code that reads
it, and they set the picker back to `"false"`:

```bash
git revert -m 1 dc79a531
```

Later commits touched `cowork/principal.py`, so expect conflicts there. The
revert and its conflict resolution go through review like any other change. A
revert merged to `staging` deploys through `publish-staging.yml`, which has no
approval gate. Production takes the revert only from `main`: `publish.yml` then
waits at the `prod` GitHub Environment for a Devops approval before
`build-deploy / deploy` runs. A push to `main` also syncs into `staging`. Unlike
an image rollback, the revert backs out nothing else, and later deploys keep it.

After the backout, check every replica's image and repeat the probes below.
Expect the pre-change behavior. Missing, malformed, and mismatched expectations
reach the route and return its normal status, and each still logs
`organization boundary: <reason> on <METHOD> <path> (audit mode)` at WARNING.
The capability reports `expectedOrganizationEnforced: false` and
`enabled: false`, and a valid `mdb_` API key still succeeds. A documented
command is not a completed rehearsal: record the revert commit, image tag, UTC
timestamps, and results when an operator exercises it.

**Disabling the picker is a separate rehearsal.** In staging, merge a change
that sets `COWORK_ORGANIZATION_SWITCH_ENABLED` to `"false"` in
`values-staging.yaml`, verify the result, then merge the restore the same way.
Each merge runs all of `publish-staging.yml`, which rebuilds the image from
unchanged code. The capability's `enabled` becomes false, while
`expectedOrganizationEnforced` stays true and the 426/409 refusals remain. This
tests the picker switch only; enforcement stays on.

### Verify replicas and the gateway separately

Check the running image and effective environment on every serving replica,
using the contexts above. Each must use org tenancy, enforced identity, and
`COWORK_ORGANIZATION_SWITCH_ENABLED=true` outside a picker-disable rehearsal.
The retired boundary-mode setting cannot change enforcement.

Then probe each replica directly with a browser-shaped bearer and the identity
headers the gateway would inject, `X-User-Id` and `X-Organization-Id`. A
browser-shaped bearer has three dot-separated parts, does not start with
`mdb_`, and has a first part that decodes to a JSON header with a non-empty
`alg`, such as `eyJhbGciOiJSUzI1NiJ9.e30.sig`. A placeholder such as `x.y.z`
skips the boundary, so every probe returns the route's normal answer. Vary
`X-Cowork-Expected-Organization-Id`: a matching organization must pass, a
missing header must return 426, and malformed and mismatched headers must
return 409. Both refusals must carry `organization_reload_required`,
`X-Cowork-Organization-Reload: required`, and `Cache-Control: no-store`.
`GET /api/v1/capabilities/organization-switch` must report `protocolVersion: 1`
and `expectedOrganizationEnforced: true`. Its `enabled` must be `true` outside a
picker-disable rehearsal.

These probes test the server after identity resolution. They do not prove that
the ingress authenticates a real credential or replaces caller-supplied identity
headers. Verify that path separately through the public hostname with a valid
browser session and an `mdb_` API key. Keep the credential out of recorded
commands and output. Record the environment, UTC time, image digest, replica,
request class, status, and response headers with each result.

### Count refusals by reason

The server still logs every boundary refusal at WARNING after removal of the
mode setting. In OpenSearch, filter `kubernetes.container_name.keyword` to
`cowork-server` and `kubernetes.namespace_name.keyword` to the environment's
namespace, match `organization boundary` in `message`, and count each fragment
below as its own `message` phrase:

| Reason | Message fragment | HTTP status |
| --- | --- | --- |
| Missing | `missing expected organization` | 426 |
| Malformed | `malformed expected organization` | 409 |
| Mismatch | `organization mismatch` | 409 |

Use the same explicit UTC start and end for staging and production. Record the
namespace and application filter, retained time range, and total matching
application records. Verify that logging includes WARNING and every serving
replica reaches the log store. A zero count is meaningful only when that
environment has other records in the window. Status 409 alone cannot distinguish
malformed and mismatched expectations, or separate these refusals from unrelated
conflicts.

Label counts from a retained 48-hour window after deployment as post-deploy
evidence. They cannot reconstruct an unavailable pre-deploy baseline. Keep
deliberate verification probes identifiable by timestamp and path when comparing
traffic before and after a change. The 2026-09-23 counts for both environments
are recorded on [ENG-2701](https://linear.app/mindsdb/issue/ENG-2701).

## Configuration

For configuration options possible, please see our [helm-charts](#todo) repository.
