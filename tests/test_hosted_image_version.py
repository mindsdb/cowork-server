"""The hosted image must resolve a real version, and must not ship .git (ENG-1796).

The web deployment reported ``cowork_server_version=0.0.0.dev1+g873b59f87`` while
the wheel built from the *same commit* reported ``0.26.8.25.1``. Nothing was
hardcoded: ``actions/checkout`` defaults to ``fetch-depth: 1``, so the clone had
no tags, so hatch-vcs had nothing to describe against — and the build succeeded
anyway with a version no query can group by.

``.dockerignore`` already keeps ``.git`` in the context on purpose (hatch-vcs
needs it), and the publish job already fetches tags for exactly this reason. The
image build was the one path that did neither.

Both assertions here are build-level for the same reason as the version itself:
there is no runtime symptom. A shallow checkout produces a working image with a
wrong version, and a retained ``.git`` produces a working image that is merely
tens of megabytes larger and carries the repo's full history into production.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = (_ROOT / ".github/workflows/build-deploy.yml").read_text()


def _strip_comments(text: str) -> str:
    """Drop comment-only lines before matching. Load-bearing, not cosmetic.

    An earlier revision asserted ``"fetch-depth: 0" in build`` over raw text.
    The explanatory comment added alongside it contained that exact literal --
    ``# already checks out with fetch-depth: 0 for exactly this reason`` -- and
    it sits inside the build job, so the assertion was satisfied by prose. Both
    mutations passed at that head: deleting the setting, and changing it to
    ``fetch-depth: 1``. The guard against the precise bug this file exists to
    catch was switched off by a sentence explaining the guard.

    A comment must never be able to satisfy an assertion about configuration,
    so the stripping happens here rather than at each call site -- a future
    comment mentioning any asserted string then cannot silence anything.
    """
    # Truncate each line at its first `#` rather than dropping comment-only
    # lines. A prefix check misses the likelier disabling pattern -- keeping the
    # guard's text as a trailing note beside a no-op, e.g.
    # `true # test ! -e /app/.git`, which reads as intact and is not. Verified:
    # that exact mutation passed until this truncated instead.
    #
    # A `#` inside a quoted string would truncate a real line early, and that is
    # the safe direction: the assertion fails loudly rather than passing on
    # prose. None of the lines asserted on here contain one.
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


_DOCKERFILE = _strip_comments((_ROOT / "Dockerfile").read_text())


def _job_block(name: str) -> str:
    """The text of one job, so an assertion cannot be satisfied by a sibling job.

    Parsed by hand rather than with PyYAML: this repo does not declare pyyaml,
    and it is only importable here transitively through anton-agent. A test that
    depends on someone else's dependency graph breaks for reasons that have
    nothing to do with what it is guarding.
    """
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)", _WORKFLOW, re.M | re.S)
    assert match, f"no `{name}:` job in build-deploy.yml"
    return _strip_comments(match.group(1))


def test_the_image_build_fetches_tags() -> None:
    """Without tags hatch-vcs yields 0.0.0.dev1+g<sha> and the build still passes.

    Scoped to the build job on purpose, though not for the reason an earlier
    revision of this docstring gave. It claimed the publish job in this file
    already fetched tags, so an unscoped search would pass while the build
    stayed shallow. That was wrong twice: `publish` lives in `publish.yml`, and
    before this change `build-deploy.yml` contained no `fetch-depth: 0` at all,
    so an unscoped search would have *failed* here, not passed.

    The scoping still earns its keep, for the jobs that do live in this file:
    `scan`, `deploy-pr-env` and `deploy` each check out too, and none of them
    needs tags. An unscoped assertion would go green the day any of those gains
    a `fetch-depth: 0` for an unrelated reason, while the image build regressed
    to shallow.
    """
    build = _job_block("build")
    assert "actions/checkout" in build, "the build job no longer checks out the repo"
    # Exact line, not a substring: `in build` would also accept
    # `fetch-depth: 10`, and matching the whole stripped line is what makes
    # `fetch-depth: 1` a failure rather than a pass.
    assert any(line.strip() == "fetch-depth: 0" for line in build.splitlines()), (
        "the image build must fetch tags — with fetch-depth 1 the version "
        "silently resolves to 0.0.0.dev1+g<sha> (ENG-1796)"
    )


def test_git_metadata_is_not_shipped_in_the_image() -> None:
    """Fetching full history is only safe because the builder deletes it again.

    ``COPY --from=builder /app /app`` in the final stage copies whatever the
    builder left behind, so this is what keeps the runtime image from carrying
    every commit of this repo.
    """
    assert "rm -rf /app/.git" in _DOCKERFILE, (
        "the builder no longer deletes .git — with fetch-depth: 0 the runtime "
        "image would carry the repo's entire history"
    )


def test_an_unresolvable_version_fails_the_build() -> None:
    """The strongest guard here: a shallow checkout now FAILS instead of shipping.

    The YAML assertion above pins the workflow, but only this survives someone
    building the image by another route. It is also what makes the version
    visible in the build log at all -- had it existed, this bug would have been
    caught the day it was introduced rather than months later in a trace.
    """
    assert 'case "$v" in 0.0.0*)' in _DOCKERFILE, (
        "the 0.0.0 assertion is gone; a tagless checkout would silently ship a "
        "version no query can group by, which is the whole bug (ENG-1796)"
    )
    assert "exit 1" in _DOCKERFILE
    assert 'echo "cowork-server version: $v"' in _DOCKERFILE, (
        "the resolved version is no longer echoed, so a wrong one is invisible "
        "in the build log"
    )


def test_the_echoed_version_is_the_builder_stage_only() -> None:
    """States the guard's limit, so it is not mistaken for end-to-end proof.

    The build-time echo and the 0.0.0 rejection both read the **builder's**
    venv, which proves the version at sync time. The final stage re-copies
    ``/app`` and no job in this repo starts the resulting image, so nothing in
    CI shows what the *shipped* container reports.

    In practice the two cannot diverge -- the version is baked into the
    installed dist-info during ``uv sync`` and the final stage copies that
    directory verbatim -- but "cannot diverge" is an argument, not a
    measurement. The measurement is ``GET /api/v1/health``, whose
    ``server_version`` comes from the same ``importlib.metadata`` call the trace
    stamp uses (``cowork/api/v1/endpoints/health.py``), read off a PR
    environment or a deployed instance. That is the ticket's "verified against
    the image's own pip show", and it belongs in the QA plan rather than here.
    """
    assert "server_version" in (
        _ROOT / "cowork/api/v1/endpoints/health.py"
    ).read_text(), (
        "health no longer exposes server_version, which is the only way to read "
        "the version off a running image rather than off the builder (ENG-1796)"
    )


def test_the_version_is_resolved_before_git_is_deleted() -> None:
    """Ordering is the whole contract: delete first and the version is lost.

    hatch-vcs reads .git during ``uv sync``, which bakes the version into the
    installed dist-info. Deleting earlier would put us back to a fallback
    version -- the same bug, reached from the other side.
    """
    sync_at = _DOCKERFILE.find("uv sync --frozen --no-dev")
    delete_at = _DOCKERFILE.find("rm -rf /app/.git")
    assert sync_at != -1 and delete_at != -1
    assert sync_at < delete_at, "uv sync must resolve the version before .git is removed"
