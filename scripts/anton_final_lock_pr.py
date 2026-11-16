"""Refresh the bot PR from the checked-out staging commit without repeated pushes."""

import json
import os
from pathlib import Path
import subprocess
import sys


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def prepare_commit(base: str, previous: str, title: str) -> bool:
    """Keep CI intact only when both the staging parent and proposed tree match."""
    run("git", "add", "--", "uv.lock", "cowork/release-requirements.txt")
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


def remote_head(branch: str) -> str:
    value = run("git", "ls-remote", "origin", f"refs/heads/{branch}")
    return value.split()[0] if value else ""


def main() -> None:
    repo, branch = os.environ["REPO"], os.environ["BRANCH"]
    locked, version = os.environ["LOCKED"], os.environ.get("VERSION", "")
    base = run("git", "rev-parse", "HEAD")
    # Never publish a resolution made from staging that has already moved.
    # The next poll starts again from its new lock and requirements.
    if remote_head("staging") != base:
        raise RuntimeError("Staging moved during relock; retry from its new head")
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
    closed = json.loads(run("gh", "pr", "list", "--repo", repo, "--base", "staging", "--head", branch,
                            "--state", "closed", "--limit", "50", "--json", "number,title,mergedAt"))
    declined = next((p for p in closed if p["title"] == title and not p["mergedAt"]), None)
    if declined:
        print(f"#{declined['number']} proposed this and was closed without merging; not reopening",
              file=sys.stderr)
        return
    previous = remote_head(branch)
    if previous:
        run("git", "fetch", "--depth=2", "origin", previous)
    bot = os.environ["APP_SLUG"] + "[bot]"
    bot_id = run("gh", "api", f"/users/{bot}", "--jq", ".id")
    run("git", "config", "user.name", bot)
    run("git", "config", "user.email", f"{bot_id}+{bot}@users.noreply.github.com")
    changed = prepare_commit(base, previous, title)
    if remote_head("staging") != base:
        raise RuntimeError("Staging moved during relock; retry from its new head")
    if changed:
        # An explicit lease also handles an absent branch. Never overwrite a
        # concurrent human update or a branch recreated since it was inspected.
        run("git", "push", f"--force-with-lease=refs/heads/{branch}:{previous}",
            "origin", f"HEAD:refs/heads/{branch}")
    body = (
        f"Staging locks `{locked}`. This proposes final `{version}` for ENG-4355.\n\n"
        "Before merging, check that the final contains the anton changes staging needs, "
        "and let CI finish. Merge it before the staging → main release PR, which stays red "
        "until staging locks a final. anton's final usually lands during the release freeze.\n\n"
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


if __name__ == "__main__":
    main()
