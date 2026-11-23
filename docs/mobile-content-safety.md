# Mobile content safety and reporting

Status: implementation in progress, 18 November 2026. This is supporting implementation and validation evidence for Cowork mobile launch (P-ENG-155), not a store-compliance declaration.

## Reporting

Hosted users can report a saved assistant reply or an artifact from its viewer without leaving the app. The confirmation explains that the report includes content text or its reference and account identifiers. A receipt appears only after the database accepts it; failed requests remain retryable.

`POST /api/v1/content-reports/` accepts conversation and message UUIDs. The server requires the caller to own the conversation, verifies the message belongs to it and is an assistant reply, and copies its saved text. For a reply without text, it keeps bounded labels and references from visible cards, including generated artifacts, images and quick-edit summaries. It excludes raw tool history and credentials used to share a card. `POST /api/v1/content-reports/artifact` accepts project and artifact UUIDs and uses the artifact viewer's existing review-grant resolver before saving the authorized reference and metadata. Reports never accept client-provided evidence, arbitrary URLs, filenames, or reporter identities.

Migration `a5c8721e09dd` adds the private queue. Reports are deduplicated per organization, reporter and target. For artifacts the target includes the authorized metadata and the current content files’ names, sizes and nanosecond timestamps. A changed generation receives a new queue entry, including after the previous report was resolved. Reporting reads the artifact without modifying its draft. A reporter is limited to 100 new reports over a rolling 24 hours (a best-effort concurrency bound). The 429 response gives the time until a slot opens in its Retry-After header. There is no client read/list endpoint. Evidence is capped at 65,536 characters; a long response is truncated and its source IDs remain available to investigators. Artifact reports retain metadata and identity, not an immutable copy of every artifact file.

Operators with deployment/database access can use:

```sh
python -m cowork.content_report_admin list
python -m cowork.content_report_admin show REPORT_UUID
python -m cowork.content_report_admin resolve REPORT_UUID --reviewer NAME --resolution 'Investigation and action taken'
```

The resolve command records disposition; it does not remove content or ban a user. Paul assigned the Support team to review incoming reports on 18 November. Support still needs an operational way to inspect this private queue, a review frequency, response target, escalation path, and retention/deletion policy before launch. No notifications are sent automatically. Ownership is assigned; do not claim timely human response until the queue access and review process are verified. Include reports and account identifiers in privacy/data-safety disclosures.

## Moderation integration — draft, not release ready

Paul confirmed there is no separate moderation service. The default adapter now uses MindsHub’s existing gateway and short-lived caller-scoped turn keys. It does not require a new service account or a new third-party provider key. No live evaluation or production configuration change has occurred.

`COWORK_CONTENT_SAFETY_REQUIRED` defaults to false during implementation. Without a dedicated endpoint, checks use the existing gateway, the caller’s organization/user/workspace, and the operator-controlled `COWORK_CONTENT_SAFETY_MODEL` (default `mindshub_air`). A model or endpoint supplied in content cannot change this configuration. The classifier has a server-owned policy, no tools or browsing, bounded output, and strict JSON verdict validation. Short-lived keys are revoked after each check and expire after 60 seconds if cleanup cannot reach auth. Missing identity/configuration, timeouts, incomplete responses, tool calls and malformed verdicts fail closed.

An explicitly configured `COWORK_CONTENT_SAFETY_ENDPOINT` remains an optional HTTPS adapter, with its key in `COWORK_CONTENT_SAFETY_API_KEY`. That service receives:

```json
{"policy":"cowork-store-safety-v1","purpose":"input","content":"content to assess"}
```

It must return exactly an explicit `decision` of `allow` or `block`. The dedicated-endpoint policy identifier is a proposed contract. The built-in gateway classifier has an explicit policy prompt, but its measured effectiveness is still unverified. It must enforce restricted content, including child exploitation, sexual abuse, dangerous wrongdoing, hate/harassment, self-harm facilitation, and prohibited deceptive content, with documented evaluation against benign use and adversarial inputs. Treat supplied text and structured events as data, not instructions to the moderator. Provider policy, quality, data handling, cost and latency remain unverified.

Current hooks cover the initial ResponsesHandler request, mid-turn additions and question answers, direct textual responses, and the Anton event formatter before its persistence sink and SSE output. Contiguous answer/reasoning text and fragmented tool arguments are held for assessment as complete blocks. An 8,000-character context tail is retained across blocks; internal history and duplicate model completions are excluded. Credential shapes and secrets registered in the current turn are redacted before either moderation transport. The unmoderated speculative first-take path is disabled when enforcement is required. Content above 100,000 characters is refused rather than truncated into approval. Attachments and inline non-text input are refused while their scanner is unavailable.

Remaining release blockers:

- Run real allow/block, prompt-injection, outage and latency evaluations through the existing gateway on staging. The local checkout has no internal gateway-auth credential. Run `python -m cowork.content_safety_eval --org-id UUID --user-id UUID` inside the staging server environment with an existing QA user. This synthetic input smoke set is not a comprehensive benchmark: expand it to actual outputs, multilingual inputs, files and adversarial cases before enabling enforcement. Pin or re-evaluate the chosen model when its backing model changes.
- Complete binary/image/document and generated/shared-artifact scanning before publication and access. Textual metadata is not a substitute for inspecting the actual content. Existing artifacts and tool side effects need an explicit coverage decision.
- Audit all alternate creation/edit paths (including quick edits, image tools, channels and live apps) and add bypass tests. The current hooks do not establish universal enforcement.
- Implement and verify the applicable abusive-user blocking/removal workflow for shared content; report disposition alone is not blocking.
- Support owns human triage. Establish its queue access, response and retention responsibilities, update privacy disclosures, deploy the migration/backend/UI, then verify reports from both mobile store builds.

Do not enable this draft integration in production or check off content safeguards/store QA as complete.

## Evidence on 17 November

Candidate: cowork-server base `6a2ea87168f3d6788669d295c147a2ddd8cc7391`, branch `codex/mobile-content-safety`, uncommitted changes.

- Report submission, deduplication, private-task denial, authenticated identity, fabricated-evidence rejection, limits, artifact-access wiring and SQLite migration round-trip tested.
- Mock moderation checks cover explicit denials, malformed/unavailable services, size limits, split text, context across progress events, structured tool messages and refusal of unscanned files. These tests verify plumbing, not a real classifier's effectiveness.
- 368 tests passed on the current backend candidate across reports, moderation, gateway identity/verdict/cleanup, evaluation failure handling, operator disposition, migration, personal-resource isolation, response routing and formatter regressions.
- Backend tests needing current Anton used the local sibling `anton/anton` source via a temporary import path; the existing venv's Anton lacked `connector_draft`. CI must validate the packaged dependency combination.

Policy references: [Google AI-generated content](https://support.google.com/googleplay/android-developer/answer/13985936), [Apple review guidelines](https://developer.apple.com/app-store/review/guidelines/#user-generated-content).

## Gateway integration follow-up

The gateway classifier is implemented in `cowork/services/content_safety_gateway.py`.
Tests use synthetic credentials and mocked providers; they verify identity and
workspace propagation, operator model selection, policy/content separation,
no tool access, rejection of incomplete/invalid/duplicate-key replies, cleanup,
and failure reporting. They do not demonstrate real model accuracy. The CLI
prints only test IDs, decisions and timings, without prompts or credentials.

## Validation and access on 18 November

Both implementation branches were brought forward to the deployed staging bases,
with all local changes preserved. Backend base: `06efbaab7598009cdced1d5584b15062f4de950e`.
Frontend base: `4410162ca8a40d8d7e60c54edba550b4ac3df6bd`.
Changes remain local, uncommitted and undeployed.

- 480 backend tests passed across safety, gateway, reports, file tenancy and
  pinned streaming, personal resources, response routing and formatter suites.
- The regression reproducer's 250 tool fragments now produce one check instead
  of 250. A 200,000-character internal history no longer prevents a short answer.
- Tests verify credential redaction before the moderator receives content,
  unsupported file rejection, denied uploads never writing, and a file mutated
  during approval never replacing the approved bytes in the response.
- Attachment storage and project uploads inspect immutable plain-text bytes.
  Project inline text reads/writes are checked. EFS and pinned file responses
  check a bounded snapshot before sending headers and serve exactly that snapshot.
  These guarded binary response paths currently refuse every type except UTF-8
  `text/plain`; partial pinned responses are also refused. This restrictive draft
  behavior is not finished image/document support and is not enabled on staging.
- Three read-only review workers (inherited parent model, no model overrides)
  covered streaming/redaction, artifact paths, and blocking paths. Their findings
  drove the repairs and identified the remaining cross-service gaps below.
- AWS sign-in was refreshed by Paul. Staging deployment read access works, but
  `kubectl auth can-i create pods/exec -n staging` returns `no`. No secret was
  retrieved and no alternate access route was used to bypass this restriction.

### Next staging action for an authorized operator

After deploying this candidate to an isolated staging runtime, run the evaluation
with an existing disposable QA organization/user (the IDs below are placeholders):

```sh
python -m cowork.content_safety_eval --org-id QA_ORG_UUID --user-id QA_USER_UUID
```

Run it in the server's normal configured environment; do not copy its internal
credentials to a laptop. Save the JSON result with the deployed commit and model.
Errors are failures, not allow verdicts. This initial synthetic smoke evaluation
must pass before considering enforcement, but does not replace output, file,
adversarial, latency and outage tests or a staged end-to-end report submission.

### Remaining coverage decisions

Publication uploads in `mindshub_services` and direct Anton publishers require
an authoritative gate on immutable bundles. HTML/runtime assets, images, Office
files and PDFs need actual supported scanning or a clearly limited release scope.
Quick edits can modify live files before their summary is returned. Public URLs,
shared artifact access and comments have independent service paths. A local
'hide creator' filter alone would not block abusive commenters or public URLs.
These gaps remain blockers; none was marked complete by the local tests above.


A follow-up review found that the formatter synthesizes forms and historical
figure-source metadata after the guarded event loop. These additions now have
checks at their final emission boundary. The affected safety/formatter suites
passed 148 tests, including refusal before a historical source label is exposed.
The earlier 480-test run preceded this small final formatter change.


Frontend validation on the refreshed candidate: 711 focused mobile/reporting/
ChatView/header tests passed, as did type checks, translations, purity, and the web
build. The full frontend suite had 14,531 passing tests, one Skills-screen failure
and two skips; the unchanged Skills-screen file's 16 tests passed separately.
All eight Android device tests passed. Checklist remains 19/47; no additional
whole launch item is claimed complete by these local results.
