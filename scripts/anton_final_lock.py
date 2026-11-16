"""Name the final anton-agent that should replace a release candidate in the lock.

Staging may lock an anton release candidate to test new agent code. A
production release may not (ENG-4355): desktop installs exactly the anton the
lock names, and `release_contract.py --require-final` refuses a candidate on
every PR into main. Once anton's weekly release publishes a final, staging
should lock it before cowork-server's release is cut.

This prints the newest final anton on PyPI that is newer than the locked
candidate, or nothing when the lock already names a final or no newer final
exists yet. `.github/workflows/anton-final-lock.yml` relocks to it and opens
the PR. A newer version number does not prove the final carries the
candidate's code; the PR's CI is that check, and a person merges it.
"""

import json
import os
from pathlib import Path
import sys
import urllib.request

from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_contract import CONTRACT, PIN  # noqa: E402

PACKAGE = "anton-agent"
PYPI = f"https://pypi.org/pypi/{PACKAGE}/json"


def sort_key(version: str) -> Version:
    return Version(version)


def is_final(version: str) -> bool:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    return not parsed.is_prerelease and not parsed.is_devrelease


def locked_version(export: str) -> str:
    for line in export.splitlines():
        match = PIN.match(line)
        if match and match.group(1) == PACKAGE:
            return match.group(2)
    raise ValueError(f"{PACKAGE} is not pinned in {CONTRACT.name}")


def replacement(locked: str, releases: dict[str, list[dict]]) -> str | None:
    """The newest installable final newer than `locked`, if `locked` is a candidate."""
    if is_final(locked):
        return None
    floor = sort_key(locked)
    finals = [
        version
        for version, files in releases.items()
        if is_final(version)
        and files
        and not all(file.get("yanked") for file in files)
        and sort_key(version) > floor
    ]
    return max(finals, key=sort_key, default=None)


def main() -> int:
    locked = locked_version(CONTRACT.read_text())
    version = None
    if not is_final(locked):
        with urllib.request.urlopen(PYPI, timeout=30) as response:
            releases = json.load(response)["releases"]
        version = replacement(locked, releases)
    print(f"locked {PACKAGE}=={locked}; replacement: {version or 'none'}", file=sys.stderr)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write(f"locked={locked}\nversion={version or ''}\nfinal={str(is_final(locked)).lower()}\n")
    elif version:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
