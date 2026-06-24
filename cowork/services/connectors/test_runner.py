"""Re-runnable "Test connection" for a *saved* connection.

Slice-1 introduced a connection probe that runs once, at submit time
(:class:`cowork.services.connectors.probe.CredentialProbe`, driven for new
submissions by :class:`cowork.handlers.probe.ProbeHandler`). That probe spins
up a headless Anton turn, hands the agent the credentials in a temp ``.env``,
and has it run a tiny live query against the service, ending in a
pass/fail/needs-input verdict.

This module promotes that one-shot probe into a first-class action you can run
again at any time against an *already-saved* connection: it loads the stored
(decrypted) credentials from the vault, reconstructs the probe's form context
from the registry spec + the saved ``_method``, runs the same
``CredentialProbe`` to its verdict, and returns a plain pass/fail + the real
error. It does **not** persist anything itself — the endpoint stamps the
result via the vault's ``record_test_result`` so the credential ciphertext is
never round-tripped.

Distinct from ``ProbeHandler`` (which streams SSE form-patches into a live
chat and *saves* credentials on success): this runner is non-streaming,
operates on existing credentials, and never writes them back.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from cowork.schemas.connectors import TestConnectionResponse
from cowork.services.connectors import health as health_mod
from cowork.services.connectors.probe import CredentialProbe, ProbeOutcome
from cowork.services.connectors.specs._registry import registry

logger = logging.getLogger(__name__)

# Internal credential markers stored on the vault record that are not real
# credential fields and must be stripped before probing.
_META_FIELDS = {"_connector_id", "_method"}


async def run_test(engine: str, name: str, credentials: dict) -> TestConnectionResponse:
    """Run the connection probe against saved credentials; return pass/fail.

    The verdict is mapped to ``health`` so the caller can stamp + return a
    fresh status in one shot. Never raises for a *connection* failure — those
    come back as ``ok=False`` with the real error; it only propagates if the
    probe runner itself can't be constructed (caught and surfaced as a failure
    too, so the endpoint always gets a terminal result).
    """
    method = str(credentials.get("_method") or "").strip() or None
    probe_credentials = {
        k: v for k, v in credentials.items()
        if k not in _META_FIELDS and v is not None and v != ""
    }

    spec = registry.get_connector(engine)
    if spec is not None:
        form_spec = spec.form.model_dump()
    else:
        # Handcrafted (non-registry) connector — there is no engine to probe
        # against. Mirror the submit-time behavior: no live probe. Report
        # ``result=""`` (untestable) so the endpoint does NOT stamp this as a
        # failure — stamping "fail" would make compute_health permanently mark
        # an otherwise-fine connection broken.
        return TestConnectionResponse(
            ok=False,
            result="",
            error="This connector has no live test — its credentials can't be verified automatically.",
            follow_up="Re-enter the credentials if the connection stops working.",
            health=health_mod.UNKNOWN,
            health_detail="No live test available for this connector.",
        )
    if method:
        form_spec["selected_method"] = method

    workspace, workspace_dir, llm_client = _build_probe_context()
    if workspace is None or llm_client is None:
        # Couldn't even start the probe — that's an environment problem on our
        # side, not the connection's. Don't persist it as a connection failure
        # (result=""); surface it so the user knows to configure a provider.
        if workspace_dir:
            shutil.rmtree(workspace_dir, ignore_errors=True)
        return TestConnectionResponse(
            ok=False,
            result="",
            error="Could not initialize the connection test (workspace or model unavailable).",
            follow_up="Check that a model provider is configured, then try again.",
            health=health_mod.UNKNOWN,
            health_detail="Connection could not be tested.",
        )

    outcome = ProbeOutcome(status="failure", error="Test ended without a verdict.")
    try:
        probe = CredentialProbe(
            engine=engine,
            credentials=probe_credentials,
            llm_client=llm_client,
            workspace=workspace,
            form_spec=form_spec,
            skipped=[],
        )
        async for kind, payload in probe.run():
            if kind == "verdict":
                outcome = payload
                break
    except Exception as exc:
        logger.exception("Connection test crashed for %s/%s", engine, name)
        outcome = ProbeOutcome(
            status="failure",
            error=f"Connection test crashed: {exc}",
            follow_up="Try again; if it persists, restart the app.",
        )
    finally:
        if workspace_dir:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    return _outcome_to_response(outcome, credentials=credentials)


def _outcome_to_response(outcome: ProbeOutcome, *, credentials: dict) -> TestConnectionResponse:
    """Map a ProbeOutcome to the test response + recomputed health.

    ``needs_input`` (the probe wants more fields) is treated as a failure for
    test purposes — from the user's standpoint the connection as saved does
    not currently work — but its follow-up is preserved as the actionable hint.
    """
    if outcome.status == "success":
        status = health_mod.compute_health(credentials, last_test_result=health_mod.TEST_PASS)
        return TestConnectionResponse(
            ok=True,
            result=health_mod.TEST_PASS,
            summary=outcome.summary or "Connection works.",
            health=status.status,
            health_detail=status.detail,
        )

    error = outcome.error or "Connection failed."
    follow_up = outcome.follow_up or "Update the connection's credentials and try again."
    if outcome.status == "needs_input" and not outcome.error:
        error = "The saved credentials are not sufficient to connect."
    status = health_mod.compute_health(credentials, last_test_result=health_mod.TEST_FAIL)
    return TestConnectionResponse(
        ok=False,
        result=health_mod.TEST_FAIL,
        error=error,
        follow_up=follow_up,
        health=status.status,
        health_detail=status.detail,
    )


def _build_probe_context():
    """Build the (workspace, temp_dir, llm_client) a probe needs.

    Mirrors :meth:`cowork.handlers.probe.ProbeHandler.run` — a throwaway temp
    workspace (the probe writes a temp ``.env`` and runs scratchpad cells in
    it) and the app's configured LLM client. Returns ``(None, dir, None)`` on
    failure so the caller degrades to an honest error instead of a 500.
    """
    workspace_dir = tempfile.mkdtemp(prefix="cowork-conntest-")
    workspace = None
    try:
        from anton.workspace import Workspace
        workspace = Workspace(Path(workspace_dir))
    except Exception:
        logger.exception("Could not build workspace for connection test")

    llm_client = None
    try:
        from cowork.services.providers import build_llm_client
        llm_client = build_llm_client()
    except Exception:
        logger.exception("Could not build LLM client for connection test")

    return workspace, workspace_dir, llm_client
