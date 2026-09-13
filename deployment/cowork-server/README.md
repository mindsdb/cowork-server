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
is the product enable and the only lever that changes browser organization
behavior without a rebuild. To hide the picker, set it to `false` and roll the
pods.

To back the boundary itself out, roll the release back; there is no value to
edit. Find the revision that predates the change and roll to it:

```bash
helm history cowork-server -n <namespace>
helm rollback cowork-server <revision> -n <namespace> --wait
```

Deploy this server only with the capability-aware Cowork client image. An older
client does not send the expected-organization header and receives 426.

## Configuration

For configuration options possible, please see our [helm-charts](#todo) repository.
