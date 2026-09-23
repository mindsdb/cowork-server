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

The chart references two Secrets that must exist in the target namespace:

- `cowork-db` — key `database_uri`, a Postgres SQLAlchemy URI. Consumed by the
  `db-migrate` initContainer (`alembic upgrade head`) and the app's
  `DATABASE_URI`.
- `mindsdb-secrets` — provider API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`.

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

`COWORK_ORGANIZATION_SWITCH_ENABLED` shows or hides the organization picker. It
is the product enable and the lever to reach for first, because it is a values
entry the pipeline reapplies on every deploy. To hide the picker, set it to
`false` and roll the pods. `COWORK_IDENTITY_ENFORCE=audit` also hides the
picker, by dropping `expectedOrganizationEnforced` from the capability, but it
reopens the no-principal path and leaves the boundary refusing anyway, so it is
never the right lever here.

Deploy this server only with the capability-aware Cowork client image. An older
client does not send the expected-organization header and receives 426.

### Back out through an operator

Backing out the boundary requires an earlier implementation; there is no value
to edit. A Helm rollback works only if a suitable release revision remains in
history. Record the current revision, then inspect a candidate's image and
values. Restoring an earlier image also backs out unrelated changes shipped
since that revision.

```bash
helm --kube-context <context> -n <namespace> history cowork-server --max 256
helm --kube-context <context> -n <namespace> get manifest cowork-server --revision <revision>
helm --kube-context <context> -n <namespace> get values cowork-server --revision <revision> --all
```

**Staging history checked on 2026-09-23 has no pre-change revision.** The
read-only query used context `newdev` and namespace `staging`. It returned ten
revisions, 219 through 228, despite requesting up to 256. The earliest retained
revision is dated 2026-09-21 08:03 UTC; deployed revision 228 is dated
2026-09-23 04:13 UTC. None predates the September 13 boundary change.

Do not invent a rollback revision or choose one solely because its number is
lower. For an enforcement backout without a suitable retained revision, deploy
a reviewed previous image or a reviewed code revert through CI. Verify its
compatibility with changes made since that image shipped.

Where a suitable revision is retained, an operator selects it, coordinates the
deployment hold, and runs:

```bash
helm --kube-context <context> -n <namespace> rollback cowork-server <verified-revision> --wait
```

**The next deployment can overwrite the rollback.** A push to `staging` starts
`publish-staging.yml`; a push to `main` starts `publish.yml` and also syncs into
`staging`. Check queued and running deployments before rolling back. A staging
branch freeze alone does not stop production deployments or a run already in
progress. To keep the backout through later deployments, revert the change in
Git and ship that revert through CI.

After rollback, verify every replica's image and repeat the capability,
missing-expectation, malformed-expectation, mismatch, and valid API-key checks
through ingress. Record the responses against the selected revision's intended
behavior. A documented command is not a completed rehearsal: record the rollback
and restore revisions, timestamps, and results when an operator exercises it.

**Disabling the picker is a separate rehearsal.** In staging, deploy
`COWORK_ORGANIZATION_SWITCH_ENABLED=false` through CI, verify the result, then
restore its intended value through CI. The capability's `enabled` becomes
false, while `expectedOrganizationEnforced` stays true and the 426/409 refusals
remain. That exercises picker availability, not an enforcement backout. The
2026-09-23 history check performed no rollback or picker-disable rehearsal.

### Verify replicas and the gateway separately

Check the running image and effective environment on every serving replica.
Each must use org tenancy, enforced identity, and the intended picker setting.
The retired boundary-mode setting cannot change enforcement.

Then probe each replica with a browser-shaped bearer and controlled identity
headers. Matching organizations must pass; a missing expectation must return
426; malformed and mismatched expectations must return 409. Both refusals must
carry `organization_reload_required`, `X-Cowork-Organization-Reload: required`,
and `Cache-Control: no-store`. The capability must report protocol version 1
and enforcement enabled. Its `enabled` value follows the picker setting.

These probes test the server after identity resolution. They do not prove that
the ingress authenticates a real credential or replaces caller-supplied identity
headers. Verify that path separately through the public hostname with a valid
browser session and an `mdb_` API key. Keep the credential out of recorded
commands and output. Record the environment, UTC time, image digest, replica,
request class, status, and response headers with each result.

### Count refusals by reason

The server still logs every boundary refusal at WARNING after removal of the
mode setting. Search the application logs for `organization boundary:` and
count these message fragments separately:

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
traffic before and after a change.

#### Observed 48-hour window, 2026-09-23

Retained OpenSearch logs cover **2026-09-21 08:30:00 UTC inclusive through
2026-09-23 08:30:00 UTC exclusive**. The queries selected container
`cowork-server` and namespace `staging` or `prod`, then counted the boundary
message and each reason above.

| Environment | Missing | Malformed | Mismatch | Application log records |
| --- | ---: | ---: | ---: | ---: |
| Staging | 0 | 0 | 0 | 148,937 |
| Production | 0 | 0 | 1 | 127,446 |

Both queries completed without timeout or failed shards, and every hourly bucket
contained records. These are post-deploy observations, not a pre-deploy baseline
or proof that every replica has the intended image. Application log records are
not a count of requests, so they cannot supply a refusal rate. Hourly coverage
does not prove that no individual record was dropped.

## Configuration

For configuration options possible, please see our [helm-charts](#todo) repository.
