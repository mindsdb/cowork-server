"""Refresh the bot PR from the checked-out staging commit without repeated pushes."""

import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anton_final_lock as selection  # noqa: E402


class StagingMoved(RuntimeError):
    """The prepared lock no longer belongs to the current staging commit."""


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def prepare_commit(base: str, previous: str, title: str) -> bool:
    """Keep CI intact only when both the staging parent and proposed tree match."""
    run("git", "add", "--", "pyproject.toml", "uv.lock", "cowork/release-requirements.txt")
    tree = run("git", "write-tree")
    if previous:
        parent = run("git", "rev-parse", f"{previous}^")
        old_tree = run("git", "rev-parse", f"{previous}^{{tree}}")
        if parent == base and old_tree == tree:
            return False
    if tree == run("git", "rev-parse", f"{base}^{{tree}}"):
        raise RuntimeError("The relock produced no change; refusing to publish an empty bump")
    run("git", "commit", "-m", title)
    return True


def relock(version: str) -> None:
    """Lock the final, raise a candidate floor to it, drop the candidate's allowance, and refresh the export."""
    pyproject = Path("pyproject.toml")
    pyproject.write_text(selection.drop_allowance(selection.raise_floor(pyproject.read_text(), version)))
    run("uv", "lock", "--upgrade-package", f"anton-agent=={version}")
    run(sys.executable, "scripts/release_contract.py")
    run(sys.executable, "scripts/release_contract.py", "--check")


def remote_head(branch: str) -> str:
    value = run("git", "ls-remote", "origin", f"refs/heads/{branch}")
    return value.split()[0] if value else ""


def main() -> None:
    repo, branch = os.environ["REPO"], os.environ["BRANCH"]
    locked, version = os.environ["LOCKED"], os.environ.get("VERSION", "")
    base = run("git", "rev-parse", "HEAD")
    # Never publish a resolution made from staging that has already moved.
    # The workflow retries this condition from fresh staging in the same run.
    if remote_head("staging") != base:
        raise StagingMoved("Staging moved during relock; retry from its new head")
    prs = json.loads(run("gh", "pr", "list", "--repo", repo, "--base", "staging",
                         "--head", branch, "--state", "open", "--json", "number"))
    number = str(prs[0]["number"]) if prs else ""
    if not version:
        # Nothing to propose: staging locks a final already, or it moved to a
        # newer candidate than any final. Either way an open proposal is stale.
        if number:
            reason = ("staging already locks a final" if os.environ.get("LOCKED_FINAL") == "true"
                      else f"staging now locks `{locked}`, newer than any final anton")
            run("gh", "pr", "close", number, "--repo", repo, "--comment",
                f"Closing: {reason}. The bot reopens a proposal when a newer final is published.")
        return

    title = f"fix(deps): lock anton-agent's final {version} instead of {locked} (ENG-4355)"
    # A person who closed this exact proposal without merging declined it, for
    # example because the final lacks the candidate's code. Don't reopen it.
    # A newer final or a newly locked candidate changes the title and is proposed.
    bot = os.environ["APP_SLUG"] + "[bot]"
    if not number:
        closed = json.loads(run("gh", "pr", "list", "--repo", repo, "--base", "staging", "--head", branch,
                                "--state", "closed", "--limit", "50", "--json", "number,title,mergedAt"))
        for proposal in closed:
            if proposal["title"] != title or proposal["mergedAt"]:
                continue
            # The issue endpoint records the latest closing actor, including
            # after a reopen. Bot cleanup is not a person's rejection. Reading
            # the actor also handles proposals closed before this distinction.
            closer = run("gh", "api", f"repos/{repo}/issues/{proposal['number']}",
                         "--jq", '.closed_by.login // ""')
            if closer != bot:
                print(f"#{proposal['number']} was closed outside bot cleanup; not reopening",
                      file=sys.stderr)
                return
    previous = remote_head(branch)
    if previous:
        run("git", "fetch", "--depth=2", "origin", previous)
    bot_id = run("gh", "api", f"/users/{bot}", "--jq", ".id")
    run("git", "config", "user.name", bot)
    run("git", "config", "user.email", f"{bot_id}+{bot}@users.noreply.github.com")
    changed = prepare_commit(base, previous, title)
    if remote_head("staging") != base:
        raise StagingMoved("Staging moved during relock; retry from its new head")
    if changed:
        # An explicit lease also handles an absent branch. Never overwrite a
        # concurrent human update or a branch recreated since it was inspected.
        run("git", "push", f"--force-with-lease=refs/heads/{branch}:{previous}",
            "origin", f"HEAD:refs/heads/{branch}")
    body = (
        f"Staging locks `{locked}`. This proposes final `{version}` for ENG-4355, raises "
        "a release-candidate `anton-agent` floor in `pyproject.toml` to it, and removes "
        "anton-agent's production pre-release allowance.\n\n"
        "Before merging, check that the final contains the anton changes staging needs, "
        "and let CI finish. Merge it before the staging → main release PR, which stays red "
        "until staging locks a final. anton's final usually lands during the release freeze, "
        "when only the Devops team can merge into staging.\n\n"
        "The bot refreshes this branch when staging or the proposed lock changes, and closes "
        "it once there is nothing to propose. If this final is wrong for staging, close this PR "
        "without merging: the bot won't propose this final again for the same candidate."
    )
    # A body file avoids shell interpolation and preserves literal Markdown.
    body_path = Path(os.environ["RUNNER_TEMP"]) / "anton-final-lock-pr.md"
    body_path.write_text(body)
    if number:
        run("gh", "pr", "edit", number, "--repo", repo, "--title", title, "--body-file", str(body_path))
    else:
        run("gh", "pr", "create", "--repo", repo, "--base", "staging", "--head", branch,
            "--title", title, "--body-file", str(body_path))


def refresh_staging() -> None:
    """Discard this run's generated lock/commit in its disposable CI checkout."""
    run("git", "fetch", "--depth=1", "origin", "refs/heads/staging")
    run("git", "reset", "--hard", "FETCH_HEAD")
    locked = selection.locked_version(selection.CONTRACT.read_text())
    final = selection.is_final(locked)
    version = None
    if not final:
        announced = selection.announced_final(os.environ.get("ANNOUNCED_VERSION", ""))
        version = selection.replacement(locked, selection.releases_listing(announced))
    # Replace every old decision, including clearing a previously selected final.
    os.environ.update(LOCKED=locked, VERSION=version or "", LOCKED_FINAL=str(final).lower())
    if version:
        relock(version)


def retry_staging(*, attempts: int = 3) -> None:
    """Retry staging races only; API, resolution and push-lease errors still fail."""
    for attempt in range(attempts):
        try:
            main()
            return
        except StagingMoved:
            if attempt + 1 == attempts:
                raise
            print(f"Staging moved; refreshing for attempt {attempt + 2}/{attempts}", file=sys.stderr)
            refresh_staging()


if __name__ == "__main__":
    # This opt-in resets generated files. The workflow uses a disposable checkout;
    # direct/manual calls retain the non-destructive, single-attempt behavior.
    if sys.argv[1:] == ["--retry-staging"]:
        retry_staging()
    elif sys.argv[1:2] == ["--relock"] and len(sys.argv) == 3:
        relock(sys.argv[2])
    else:
        main()
