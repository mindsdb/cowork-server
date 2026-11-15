# Test evidence and execution cohorts

The broad Linux, desktop export and disposable-service jobs save `inventory.json` and pytest's `junit.xml` as artifacts for 14 days. Artifact names include the cohort, workflow run ID and attempt; desktop export names also include the runner OS so Windows and macOS upload distinct artifacts. The existing test command determines success; missing or inconsistent evidence also fails its reporting step. Reports are uploaded after failing tests when available. Each job still runs its selection once.

## Which tests run

| Cohort | Existing command/selection | Execution |
| --- | --- | --- |
| Broad PR | `make test/unit/coverage`: `tests/ --ignore=tests/integration`, coverage of `cowork` | One local xdist worker per logical CPU; `PYTEST_WORKERS=0` is the serial control |
| Disposable services | `make test/services`: `tests/ -m "redis_service or postgres_service"` | Serial; real disposable Redis/PostgreSQL, required prerequisites, exclusive Redis failover operations |
| Desktop export | The ten files below, with `-rs` | Serial on each Windows/macOS matrix runner; cohort `desktop-export` |
| Deployed | `make test/integration`: `tests/integration -m "not production_read_only and not redis_service"` | Existing deployment job; documented here, not profiled by this tooling change |
| Production read-only | `make test/integration-production-read-only`: `tests/integration/test_production_read_only.py` | Existing controlled smoke; documented here, not profiled by this tooling change |

Windows and macOS each run `test_artifact_files.py`, `test_markdown_render.py`, `test_artifact_inline.py`, `test_docx_writer.py`, `test_export_hostile.py`, `test_no_fetch.py`, `test_artifact_export_contracts.py`, `test_artifact_export_service.py`, `test_export_child.py` and `test_publish_markdown.py`, all under `tests/`. Keep separate platform inventories: skips and filesystem/process behavior differ between operating systems.

`test/unit` is the broad suite, including API and stateful tests. It does not mean every case is a pure unit test. The personal-skills pilot has 74 API cases in `tests/api/coding/test_personal_skills.py` and 15 model-validation cases in `tests/component/coding/test_personal_skill_import.py`. Both inherit root resource fixtures. Nine API wiring, catalogue and caller-boundary cases mount the full API; the other 65 mount the real personal-skills router with its dependencies and file store. Every API case gets fresh application and filesystem state. Only the three coding-runtime consumers request a fake engine and service. The component cases exercise raw YAML name types and newline handling directly, without an application or client fixture; HTTP cases retain readable errors, no-write checks and successful imports for each newline style. Shared coding fakes live in `tests/support/coding/service_fakes.py`; their callers and root cleanup use the same module. Other layer assignments remain unclassified until inspected. Test function definitions, parameterized cases and semantic behaviors are different counts.

Artifact-export route tests retain all 117 cases. Twenty-eight mounting, validation, PDF-success and concurrent-request cases use the full server; 66 other local HTTP cases use the real artifact-workspace router on a fresh application. The 23 remaining cases keep their existing direct checks, including one self-mounted permission-denial application. Clients are closed without starting the server lifespan. Root resource fixtures and project cleanup remain active.

Comparison tests retain all 147 cases. Eighteen wiring, validation, dispatch and cross-feature cases use the full server; 43 other local HTTP cases mount the real comparison and project routers. Four cases keep their existing full-server applications, and 82 direct or hosted-database cases remain unchanged. All 61 fixture clients get fresh applications and harnesses and close without starting the server lifespan. Root resource fixtures remain active.

## Record a run

Use a new output path outside the checkout for each invocation. The plugin rejects an existing inventory file or a path inside the source tree. Reports fingerprint current tracked and non-ignored untracked source, so uncommitted code is distinguishable. An edit during the run makes the evidence incomplete.

```sh
REPORT_DIR="$(mktemp -d)"
uv run --locked python -m pytest tests/ --ignore=tests/integration --collect-only -q \
  -p scripts.pytest_inventory --test-cohort=broad \
  --test-inventory="$REPORT_DIR/collection.json"
make test/unit/coverage MAKEOVERRIDES= \
  PYTEST="uv run --locked python -m pytest -p scripts.pytest_inventory --test-cohort=broad --test-inventory='$REPORT_DIR/parallel.json' --junitxml='$REPORT_DIR/parallel.xml'"
uv run --locked python scripts/test_inventory.py check "$REPORT_DIR/parallel.json" \
  --junit "$REPORT_DIR/parallel.xml"
```

For a matched serial control, use the same Make invocation with `PYTEST_WORKERS=0` and new `serial.json`/`serial.xml` paths, then run `scripts/test_inventory.py compare serial.json parallel.json`. Paths embedded in the Make command above use a controlled directory without single quotes; quote other paths appropriately.

Pass reporter options directly. Exporting them in `PYTEST_ADDOPTS` makes nested pytest invocations inherit the parent's report destination. `MAKEOVERRIDES=` prevents the outer Make overrides from leaking into nested Make calls, including command-contract tests. Ordinary `make test/unit` and direct pytest invocations remain uninstrumented unless explicitly requested.

## Read the evidence

Schema version 1 records:

- Checkout SHA, start/end source-content fingerprints, dirty state, lock/configuration hashes and reporter hash.
- Python/platform/package versions and an explicit allowlist of CI run, attempt and runner-image identifiers.
- Invocation and selection, ordered parameterized IDs, relative module paths, effective marker names, actual deselections and module-level collection skips/errors.
- Actual coverage enablement, append mode, effective run settings and hashes of the coverage configuration files read.
- Setup/call/teardown events, their outcomes, durations, skip reasons, xfail classification and worker identities.
- Worker collection agreement, completion state, final pytest exit status and elapsed time.

The reporter does not copy captured output, tracebacks, marker argument values or arbitrary environment variables. JUnit is produced by pytest separately. Phase duration sums measure accumulated work, which can exceed wall time under parallel execution. They do not isolate every individual fixture or measure billed runner minutes. `check` prints case outcomes and the slowest 20 cases/modules; JSON retains all phase timings.

A complete report of failing tests remains useful evidence. `check` validates accounting and reports the original exit code; it does not turn those tests into a passing run. An interrupted run, lost worker, changed source or missing phase cannot establish complete execution. A controller finish-hook exception prevents report publication because pytest has not yet determined the final exit status. Module-level skips are not invented test cases. JUnit can contain an extra row when both a test call and teardown fail, so XML row totals alone are not a case inventory.

`compare` requires the same revision/content, configuration, collector, selection, environment, markers, outcomes and skip reasons. Worker counts and timing values may differ. It does not establish that similar assertions are redundant, and it makes no suite-size or speedup claim.

## Preserve cases across revisions

To audit a measurement change that preserves existing node IDs, load identical collector bytes externally in both isolated source trees. Copy `scripts/pytest_inventory.py` into a separate directory, put that directory on `PYTHONPATH`, and use `-p pytest_inventory` for both collection-only runs. Do not load application code from the candidate into the base.

```sh
uv run --locked python scripts/test_inventory.py preserve base-collection.json candidate-collection.json \
  --base-sha "$BASE_SHA" --candidate-sha "$CANDIDATE_SHA" \
  --allowed-additions reviewed-new-case-ids.json
```

The additions file is a reviewed JSON array of exact new node IDs, checked against the authored tests. It is not generated by accepting whatever cases appeared. `preserve` permits the declared source revisions to differ while requiring every old ID and marker set to survive, the same dependency lock and collector, and the same selection and collection skips. This collection audit makes no claim about executed outcomes across revisions. Keep strict execution comparison separate.

A file move changes node IDs, so the command above deliberately rejects it. Before moving cases, record a reviewed one-to-one map of exact old and new IDs, including parameter IDs. Validate each original report with `check`, then use a separate migration audit to apply only that map when comparing case identities, module paths and markers. Require every unmoved case to keep its ID and every addition to match the reviewed additions file. Reject missing cases, duplicate destinations, changed parameters and unexpected markers. Keep the original reports unchanged and verify that deliberately damaged mappings are rejected.

Compare mapped setup/call/teardown outcomes and skip reasons separately using executed reports from the same environment, lock, collector, selection and coverage settings. A successful mapped collection audit does not establish execution equivalence; neither `preserve` nor the same-source `compare` command performs this migration audit.
