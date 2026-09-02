# MindsHub Code runtime architecture

Status: implemented protocol v1 for the desktop control plane and an outbound code-only runtime.

## Product model

MindsHub Code keeps durable work separate from the computer that happens to execute it:

- A `CodeProject` owns shared resources, connector references, skills and task defaults.
- A `CodeTask` owns the prompt, source contexts, resource scope and delivery history.
- A `TaskRun` is one fenced execution attempt assigned to a `Computer`.
- An `ExecutionWorkspace` records the execution machine's claim for one scoped resource. Its path is execution data, not Project identity.
- A `ConnectorGrant` is short-lived authority for one action against one external resource.

Projects are never assigned to a computer. Repository resources are portable when they have a clone URL; local-folder resources are explicitly bound to their owning computer. A no-project folder task is represented by a Task and Task Run on the local computer without manufacturing a Project.

`CodeProject.schema_version == 2` is the typed-resource format. The local project store reads schema 1, classifies each legacy folder, preserves its path and commands, binds local-only resources to the current desktop computer, and persists the upgrade idempotently. Existing `CodingSession` records are projected into Task/Run records without replacing their existing identity or event history.

## Control plane

`ControlPlaneService` owns Computers, Tasks, Task Runs, workspace claims, commands, credentials, grants, leasing and recovery. Storage is accessed through `ControlPlaneStore`; `LocalControlPlaneStore` is the development/desktop adapter. Creating a Task, its first Run and initial workspace claims is journalled as a single recoverable transaction.

The desktop adapter serializes multi-record lease, command and Task/Run mutations within the process. A tenant-scoped production store must preserve the same contract with database transactions and compare-and-swap fencing; replacing the adapter must not turn the current process lock into a distributed-systems assumption.

The existing `CodingSession` remains a compatibility/read model for the current desktop event timeline, approvals and renderer. Canonical Computer and Run state is projected onto it at read time. This lets the UI migrate without making an undocumented Codex process the durable Task identity.

## Execution plane

`CodeOnlyRuntime` is a separately runnable, UI-free execution process. It only makes outbound HTTPS requests to the control plane. It:

1. registers with a single-use, ten-minute registration token;
2. advertises protocol, platform, shell, Git, terminal, agent and capacity capabilities;
3. heartbeats and long-polls for an eligible Task Run;
4. prepares isolated repository workspaces and ports;
5. opens an interchangeable `EngineSession` (Codex today);
6. runs turns, approvals, steering, cancellation, Git and terminal operations;
7. publishes ordered events/checkpoints and releases the workspace explicitly.

One current worker advertises one concurrent run because it owns one retained engine/workspace loop. Parallelism comes from multiple runtime processes or computers, not from overstating capacity.

## Protocol and fencing

The protocol version is `1.0`. Runtime requests carry a computer identity, lease ID, execution epoch and monotonic per-run event sequence where applicable.

- A lease has an expiry and is renewed by events/checkpoints.
- Commands are persisted, idempotent by key, claimed with a timeout and explicitly acknowledged.
- Recovery increments the execution epoch and changes the lease.
- A stale computer, lease, epoch, duplicate sequence or old command is rejected.
- UI timeline sequence numbers never advance the runtime protocol cursor.

An expired runtime therefore cannot publish late events, approvals, commits or command results into a recovered task.

## Credential boundary

Long-lived MindsHub, GitHub and Linear credentials remain in the control plane.

- The runtime has a hashed-at-rest computer token for registration/leases only.
- Each leased run receives a new per-run agent token; only its digest is durable.
- Inference is proxied centrally and authenticates that run/computer/epoch/lease token.
- Linked GitHub/Linear items receive exact-URL, action-scoped `ConnectorGrant`s with short expiry.
- The runtime writes these short-lived capabilities to a mode `0600` file, exposes them through an agent-neutral MCP server and deletes the file when the run closes.
- OAuth tokens never enter a lease, event, Task, Project, command, log or execution config.

Remote Git checkouts use ordinary Git credentials configured on the execution computer. Branch publication is split deliberately: the runtime pushes a normal branch without receiving central OAuth, then the control plane creates the draft PR with the configured GitHub connector. A future hosted runtime can replace that checkout transport with a central scoped Git proxy without changing Task, Run or engine contracts.

## State and recovery

Run transitions are explicit in `run_state.py`. A normal turn completion returns the Run to `ready`, retaining its workspace and agent session for follow-ups. Explicit release completes the Run and clears workspace paths and connector grants. Lease/computer loss increments the fencing epoch and moves the Run to `recovering`, surfaced to users as **Ready to resume**, while preserving Project, Task, source contexts, events and delivery metadata.

Queued follow-ups are durable in the compatibility Task read model. A completed remote turn atomically removes the oldest queued instruction from the UI queue and persists an idempotent `start` command for the same fenced Run.

## Extraction path

The local store is intentionally an adapter, not the architecture:

- replace `ControlPlaneStore` with a tenant-scoped database implementation;
- host the runtime router on the central service;
- retain the versioned runtime and engine contracts;
- supervise multiple `CodeOnlyRuntime` workers for higher per-computer capacity;
- add a scoped Git smart-HTTP proxy for hosted/private-repository checkout;
- move compatibility `CodingSession` projection into a query/read-model layer once all clients consume Tasks and Runs directly.

No cloud scheduler, Kubernetes provisioner or multi-agent coordinator is part of protocol v1.
