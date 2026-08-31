# MindsHub Code security model

This document records the security decisions enforced by Code Mode's Task/Run architecture. It is a release boundary, not a claim that an unfinished hosted deployment is safe to expose.

## Assets and trust boundaries

The control plane holds Task metadata, organization-scoped connector references and the authority to schedule work. A Computer holds repository workspaces, shells, processes and a Code Runtime. The coding agent is treated as untrusted input within the permissions selected for its Run. GitHub, Linear and inference credentials remain control-plane assets.

The important boundaries are:

1. a user to their organization/control namespace;
2. the control plane to a registered Computer;
3. a Computer to one fenced Task Run;
4. an agent to one narrowly delegated connector action;
5. a Task to its immutable resource and external-work-item scope.

## Enforced controls

### Runtime identity and scheduling

- Computer pairing uses a cryptographically random, one-use, ten-minute registration credential. Only its digest is stored.
- A Computer receives a private runtime credential bound to its registration epoch. Revocation advances the epoch and invalidates the credential.
- A Run lease is bound to Computer, lease ID, execution epoch and expiry. Runtime events also require a monotonic sequence number.
- Run credentials are separate from Computer credentials and become stale when ownership or epoch changes.
- SQL run claiming uses a row lock and `SKIP LOCKED`; local claiming is serialized by the desktop store.
- Task, first Run and workspace claims are persisted atomically.

### Task and workspace authority

- Resource selection is validated at Task creation and copied into an immutable execution snapshot.
- Later Project edits cannot silently add repositories, folders, connectors or skills to an existing Task.
- The runtime must publish exactly the workspace resource IDs claimed by the Run.
- A local folder is eligible only on its owning Computer. A repository can move only when it has a clone URL.
- Same-computer recovery restores the existing workspace. Cross-computer recovery requires explicit confirmation and creates a fresh workspace; unpushed changes are never represented as transferred.

### Connector and delivery authority

- OAuth and long-lived connector credentials are never included in a lease, Project snapshot, runtime environment, event, command or log.
- The control plane issues short-lived capability tokens scoped to provider, connection, Run, Computer, epoch, action, exact resource URL and use count.
- Grant validation and use-count increments are atomic. Grants are revoked when the Run ends or is superseded.
- Capabilities are generated from the Task's immutable execution snapshot.
- A PR mutation is allowed only for a published PR recorded on that Task. Progress/result/completion updates are allowed only for a source item linked to that Task.
- Agent-facing capability files are mode `0600` and removed when the runtime session closes.

### Persistence and audit

- Hosted control records are structurally namespaced; the SQL adapter exposes no unscoped read or write operation.
- Sensitive control-plane decisions write append-only, secret-free audit events with action, outcome, actor type, target, Run and Computer.
- User-facing errors and partial delivery failures pass through redaction before persistence or display.

## Fail-closed deployment gates

The current desktop Code API is loopback-only and disabled in organization tenancy mode. Do not remove that guard until all of the following are complete:

- Projects, Tasks/events and their query/read models use the authenticated organization namespace—not only control records;
- organization roles are mapped to explicit create, run, approve, connect, publish, merge and administer-computer permissions;
- runtime ingress is HTTPS-only with production certificate validation, request limits and abuse controls;
- private repository checkout uses a short-lived scoped Git proxy or equivalent credential broker;
- audit events are exported to the production security/retention sink with alerting for repeated denials and stale-runtime traffic;
- runtime packages are signed, version policy is enforced, and upgrade/revocation procedures are exercised;
- backup/restore, disaster recovery and cross-region scheduling are tested under failover;
- threat modelling, dependency scanning, secret scanning and an independent penetration test have release sign-off.

Until those gates are met, the correct response in an organization deployment is `403`, not a partially isolated Code surface.
