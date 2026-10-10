# Measure CPU spent on answers

The opt-in workload records answer throughput, latency, errors, stored event
counts, server-process CPU, and a py-spy SVG flamegraph for each server process.
It does not deploy a candidate or claim a capacity improvement from unit tests.

## Prepare staging

Use a reserved staging window. CPU counters cover every request the process
serves, including unrelated traffic. They exclude CPU used by PostgreSQL, Redis,
the scratchpad controller, and remote Anton workers. Profile overhead is included;
keep the profiler and sampling rate identical on both sides of a comparison.

The workflow requires two Ready `cowork-server` pods running the supplied exact
`staging-<40-character-sha>` image. It sums both processes because the ingress can
route a conversation to either replica. Pod UID, container ID, process PID, start
time, and kernel boot ID must remain unchanged throughout measurement. A rollout
or process restart invalidates the result.

The workflow builds a separate, hash-pinned py-spy image through CI. It adds one
ephemeral debugger to each existing API pod, targeting that API container's process
namespace. The normal server image and chart do not change. Each debugger has
`SYS_PTRACE`, no added mounts or environment secrets, and a 45-minute absolute
expiry. It verifies the serving PID and start time before checking ptrace access.
Preflight fails before creating conversations if the runtime or admission policy
does not permit this inspection.

An infrastructure operator applies the reviewed staging-only permission manifest
once. These commands do not change the active kubectl context:

```sh
kubectl --context newdev diff -f deployment/performance/staging-profiler-rbac.yaml
kubectl --context newdev apply -f deployment/performance/staging-profiler-rbac.yaml
kubectl --context newdev -n staging auth can-i get pods \
  --as=system:serviceaccount:infrastructure:newdev-gha-runner
kubectl --context newdev -n staging auth can-i list pods \
  --as=system:serviceaccount:infrastructure:newdev-gha-runner
kubectl --context newdev -n staging auth can-i create pods/exec \
  --as=system:serviceaccount:infrastructure:newdev-gha-runner
kubectl --context newdev -n staging auth can-i patch pods/ephemeralcontainers \
  --as=system:serviceaccount:infrastructure:newdev-gha-runner
```

The manifest grants only pod metadata reads, exec, and ephemeral-container patches
in `staging` to the existing `mdb-dev` runner service account. The workflow cannot
grant itself these permissions. Review any cluster-policy denial before changing
that policy; the tooling does not relax it automatically.

Supply a dedicated staging performance user and organization with permission to
execute turns and enough allowance for the complete run. The workload does not
provision or modify an identity. It rejects the nightly suite's `@emailsink.dev`
users and verifies the key's user and organization through staging auth before
starting. Never copy the nightly suite's key.

After the identity exists, configure its values without putting the key on a
command line or in this document:

```sh
gh secret set COWORK_PERF_API_KEY --repo mindsdb/cowork-server --env staging
gh variable set COWORK_PERF_ORG_ID --repo mindsdb/cowork-server --env staging
gh variable set COWORK_PERF_USER_ID --repo mindsdb/cowork-server --env staging
```

These commands are operator-run. No prerequisite is considered complete until
its preflight passes. The profiler executes only from the reviewed workflow;
local invocation against staging is refused. The job installs driver dependencies
on its runner and builds the debugger image in CI. It never installs into a serving
container. The job stops its recorders, downloads the SVGs, then terminates only
its own debuggers. `report.json` records their image digests, expiry, and cleanup
state, including a failed start.

Kubernetes retains terminated ephemeral-container records until the pod's normal
rollout. The workflow never deletes or restarts API pods to remove those records.
To revoke this access, the infrastructure operator runs:

```sh
kubectl --context newdev delete -f deployment/performance/staging-profiler-rbac.yaml
```

Revocation does not stop already-running debuggers. Their watchdogs still expire;
check the cleanup state and pod metadata after an interrupted run. A pending image
pull cannot be removed without replacing the pod. Cleanup waits 30 seconds, then
fails with `termination_unconfirmed` if needed. The absolute expiry makes a late
start exit instead of granting it another 45 minutes.

## Select a workload

The `text` scenario asks for 80 numbered squares without tools. The `scratchpad`
scenario asks for one local arithmetic execution. It requires a successful
`scratchpad` result with action `exec` in both live and stored events; another tool,
a dump/reset, or a failed cell does not satisfy it. Both use the requested model for every turn. Model output still varies;
repeat baseline and candidate runs instead of treating one answer as a benchmark.

The driver creates fresh conversations through `POST /api/v1/conversations/`.
It seeds each history through real streamed `POST /api/v1/responses/` turns,
checks stored events through `GET /items`, then measures further answers on those
same conversations. The history therefore contains real delta and tool events.
No database seeding or API for importing fabricated events is involved.

History setup is excluded from CPU and timing counters but consumes real model
turns. A run allows at most eight simultaneous conversations, 500 history turns
per conversation, 50 measured answers per conversation, and 1,200 total turns.
The default global deadline is 30 minutes and every turn has a 180-second deadline
plus a bounded stream size. A timeout sends cancellation and cleans up only the
conversation IDs created by that invocation. Cleanup failures make the run fail
and leave the IDs in `report.json` for operator follow-up. A forcibly terminated
CI runner may leave those temporary conversations; inspect its artifact before
removing them. Setup failures and timeouts never count as a successful comparison.

Record actual visible-message and event counts rather than assuming two rows per
turn. Tool history may add rows. A 500-turn history is expensive; establish small
runs and the available allowance before requesting it.

## Run and compare

Use the same workload configuration, model, capacity, profile settings, and
background traffic for the baseline and candidate. Hold the controller/worker
image constant while measuring the server-only change. Then explicitly change
Anton and record that separate comparison. The server wheel's Anton pin is not
proof of the remote worker's installed version.

The registered `nightly-staging-integration.yml` manual entry calls
`measure-staging-answers.yml` only when `measure-answers=true`. Dispatch the trusted
feature ref to capture a baseline before merging or deploying the candidate.
Ordinary manual runs and scheduled nightlies retain the functional suite. The
performance mode runs no functional load or nightly recovery notification. Its inputs
are the exact already-deployed server SHA, model, scenario, concurrency, history
turns, measured answers, remote-worker provenance, and an operator's confirmation
that the prerequisites and reserved window are ready. It does not merge or deploy.
For the baseline, read the deployed image tag immediately before dispatch. The
driver independently checks both Ready pods against this SHA:

```sh
COWORK_BASELINE_IMAGE=$(kubectl --context newdev -n staging get deployment cowork-server \
  -o 'jsonpath={.spec.template.spec.containers[?(@.name=="cowork-server")].image}')
COWORK_BASELINE_SHA=${COWORK_BASELINE_IMAGE##*:staging-}
kubectl --context newdev -n staging get deployment scratchpad-controller \
  -o 'jsonpath={.spec.template.spec.containers[*].image}{"\n"}'
kubectl --context newdev -n staging get pods \
  -o 'custom-columns=NAME:.metadata.name,IMAGES:.spec.containers[*].image,IMAGE_IDS:.status.containerStatuses[*].imageID'
```

Set `COWORK_WORKER_PROVENANCE` to the controller and remote worker image digests
plus the Anton revision verified by their build records. The deployment lookup
shows the controller image; the pod inventory shows each running image's resolved
digest. A controller image does not establish the remote worker's image or Anton
revision. Workers may exist only during turns, so an empty worker inventory is
not evidence of a particular build. Obtain that mapping from the worker's build
and deployment records. The server's lockfile cannot establish it. Keep that exact
value for the server-only pair; retain the supporting build links with the reports. Missing or unknown provenance
blocks comparison.

```sh
gh workflow run nightly-staging-integration.yml --repo mindsdb/cowork-server \
  --ref perf/eng-3362-cpu-per-answer \
  -f measure-answers=true -f server-sha="$COWORK_BASELINE_SHA" \
  -f model=mindshub_air -f scenario=text -f concurrency=1 \
  -f history-turns=0 -f answers=5 \
  -f remote-worker-provenance="$COWORK_WORKER_PROVENANCE" \
  -f quiet-window-confirmed=true
```

After the operator deploys the candidate through CI, repeat the same command with
its verified SHA. Keep the feature ref containing the measurement driver constant
for both runs. Start with the small case above, then explicitly choose larger
histories and the scratchpad scenario within the run budget.

Retain the baseline image and evidence before candidate deployment. The operator
uses the existing staging release workflow for image changes; this measurement
workflow never changes a deployment. Record the controller and worker image plus
Anton revision in `remote-worker-provenance`; an unverified value is not evidence
that both runs had the intended Anton build.

Download each workflow artifact and compare its reports:

```sh
python -m scripts.performance.compare_answers baseline/report.json candidate/report.json
```

The comparison refuses failed answers, incomplete cleanup, mismatched workload
hashes, profiler versions, or different server resource limits. Server-only
comparisons also require the same known remote-worker provenance. Pass
`--scope full-stack` explicitly when comparing a deliberate Anton/worker change;
the output labels that scope and does not attribute the result to server changes
alone. It reports CPU seconds per completed answer and relative reduction, alongside throughput and latency. It does not assert
a pass threshold or prove an unchanged external service. Inspect flamegraphs,
versions, real history counts, background load, and repeated-run variance before
claiming improvement. Flamegraphs contain stack symbols, not request payloads;
the JSON report retains counts and timing, not prompts, answer text, or API keys.

## Linux self-hosted use

The same modules can measure an already running Linux server through a loopback
HTTP origin. Set `base_url` to that origin in a JSON `WorkloadConfig`, keep the
remaining fields the same, and pass the verified server process ID with `--pid`.
The driver discovers a single Python or Uvicorn process whose argv names
`cowork.server:app`, `spa_wrapper:app`, or the `cowork-server` entry point. It
refuses to profile an init process or an ambiguous match.
Run with the server's Python environment and an already installed `py-spy`, with
permission to inspect that PID. `COWORK_PERF_API_KEY` is optional for a local
server with authentication disabled. No server is started or reconfigured.

```sh
python -m scripts.performance.profile_answers --config measurement-config.json \
  --output measurement-results --pid "$COWORK_SERVER_PID" \
  --remote-worker-provenance 'in-process; no remote worker'
```

The output directory must be new. Results from macOS require another counter
collector; this collector deliberately uses Linux `/proc` in both self-hosted and
staging runs. Profiling or workload evidence has not been produced merely by
adding this tooling.
