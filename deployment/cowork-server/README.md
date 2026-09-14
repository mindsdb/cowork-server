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

To back the boundary itself out, roll the release back; there is no value to
edit. Find the revision that predates the change and roll to it:

```bash
helm history cowork-server -n <namespace>
helm rollback cowork-server <revision> -n <namespace> --wait
```

**That rollback lasts only until the next deploy into the namespace, and nobody
has to trigger one.** A push to `main` fires `sync-main-to-staging.yml`, which
pushes `staging`, which fires `publish-staging.yml` and its `helm upgrade`. So
an unrelated hotfix puts the boundary back hours later. Hold the rollback by
freezing the branch through `staging-freeze.yml`, or revert the commit and let
the pipeline deploy the revert, which is the only backout that survives a
redeploy.

Deploy this server only with the capability-aware Cowork client image. An older
client does not send the expected-organization header and receives 426.

## Configuration

For configuration options possible, please see our [helm-charts](#todo) repository.
