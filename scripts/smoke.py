#!/usr/bin/env python3
"""Bounded smoke test against a RUNNING cowork-server.

Asserts against http://127.0.0.1:26866 (does NOT start a server):
  1. /api/v1/health/                     -> status ok, config_ready true
  2. /api/v1/harnesses/                  -> expected harness ids present;
                                            claude-code has a model-picker
                                            with non-empty options
  3. /api/v1/harnesses/claude-code/status -> if installed+loggedIn, POST a
                                            streaming /api/v1/responses/ turn
                                            and expect an output_text.delta
                                            SSE event within 90s (else SKIP)
  4. /api/v1/connectors/specs            -> gmail spec exists and its
                                            browser_oauth_builtin method is
                                            not hidden

Prints one PASS/FAIL/SKIP line per check. Exit 0 only if no FAIL.
Stdlib only (urllib) - no dependencies. Run: uv run python scripts/smoke.py
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:26866"
HTTP_TIMEOUT = 20      # seconds, per plain request
SSE_TIMEOUT = 90       # seconds, overall budget for the streamed response

results = []  # list of (verdict, name)


def report(verdict, name, detail=""):
    results.append((verdict, name))
    line = f"{verdict:4s} {name}"
    if detail:
        line += f" - {detail}"
    print(line, flush=True)


def get_json(path):
    """GET BASE+path, follow redirects (urllib default), parse JSON."""
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_health():
    data = get_json("/api/v1/health/")
    problems = []
    if data.get("status") != "ok":
        problems.append(f"status={data.get('status')!r}")
    if data.get("config_ready") is not True:
        problems.append(f"config_ready={data.get('config_ready')!r}")
    if problems:
        report("FAIL", "health", ", ".join(problems))
    else:
        report("PASS", "health", f"status ok, config_ready true (server {data.get('server_version', '?')})")


def check_harnesses():
    data = get_json("/api/v1/harnesses/")
    by_id = {h.get("id"): h for h in data}
    required = ["claude-code", "antigravity", "codex", "anton", "hermes"]
    missing = [r for r in required if r not in by_id]
    if missing:
        report("FAIL", "harnesses", f"missing ids: {', '.join(missing)}")
        return
    schema = by_id["claude-code"].get("configurationSchema") or []
    pickers = [f for f in schema if f.get("type") == "model-picker"]
    if not pickers:
        report("FAIL", "harnesses", "claude-code has no model-picker in configurationSchema")
        return
    options = pickers[0].get("options") or []
    if not options:
        report("FAIL", "harnesses", "claude-code model-picker has empty options")
        return
    report("PASS", "harnesses", f"all 5 ids present; claude-code models: {', '.join(options)}")


def check_claude_response():
    status = get_json("/api/v1/harnesses/claude-code/status")
    if not (status.get("installed") is True and status.get("loggedIn") is True):
        report(
            "SKIP",
            "claude-code response",
            f"installed={status.get('installed')!r}, loggedIn={status.get('loggedIn')!r}",
        )
        return

    body = json.dumps({
        "input": "Reply with the single word: smoke",
        "model": None,
        "harness": "claude-code",
        "stream": True,
        "conversation": None,
        "project": "general",
        "attachment_ids": [],
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/v1/responses/",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    deadline = time.monotonic() + SSE_TIMEOUT
    try:
        # The timeout is per socket read; the deadline bounds the whole stream.
        with urllib.request.urlopen(req, timeout=SSE_TIMEOUT) as resp:
            for raw in resp:
                if time.monotonic() > deadline:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith(("event:", "data:")):
                    continue
                if "response.output_text.delta" in line:
                    report("PASS", "claude-code response", "output_text.delta received")
                    return
        report("FAIL", "claude-code response",
               f"stream ended without response.output_text.delta within {SSE_TIMEOUT}s")
    except TimeoutError:
        report("FAIL", "claude-code response", f"no delta event within {SSE_TIMEOUT}s (socket timeout)")


def check_connector_specs():
    specs = get_json("/api/v1/connectors/specs")  # 307 -> /specs/ followed by urllib
    ids = {s.get("id") for s in specs}
    if "gmail" not in ids:
        report("FAIL", "connector specs", "gmail spec not in listing")
        return
    gmail = get_json("/api/v1/connectors/specs/gmail")
    methods = (gmail.get("form") or {}).get("methods") or []
    builtin = next((m for m in methods if m.get("id") == "browser_oauth_builtin"), None)
    if builtin is None:
        report("FAIL", "connector specs", "gmail has no browser_oauth_builtin method")
        return
    if builtin.get("hidden"):
        report("FAIL", "connector specs", "gmail browser_oauth_builtin has hidden=true "
               "(running server may be stale vs local specs)")
        return
    report("PASS", "connector specs", "gmail browser_oauth_builtin visible (hidden false)")


def main():
    # Reachability gate: fail loudly (and early) if no server is up.
    try:
        urllib.request.urlopen(BASE + "/api/v1/health/", timeout=5).close()
    except (urllib.error.URLError, OSError) as e:
        print(f"FAIL server not reachable at {BASE} - start it first "
              f"(this script never starts one). Error: {e}")
        sys.exit(2)

    checks = [check_health, check_harnesses, check_claude_response, check_connector_specs]
    for check in checks:
        try:
            check()
        except Exception as e:  # a broken endpoint must not abort the run
            report("FAIL", check.__name__, f"{type(e).__name__}: {e}")

    fails = sum(1 for v, _ in results if v == "FAIL")
    skips = sum(1 for v, _ in results if v == "SKIP")
    passes = sum(1 for v, _ in results if v == "PASS")
    print(f"\nsmoke: {passes} passed, {fails} failed, {skips} skipped")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
