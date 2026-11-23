"""Name the final anton-agent that should replace a release candidate in the lock.

Staging may lock an anton release candidate to test new agent code. A
production release may not: desktop installs exactly the anton the
lock names, and `release_contract.py --require-final` refuses a candidate on
every PR into main. Once anton's weekly release publishes a final, staging
should lock it before cowork-server's release is cut.

This prints the newest final anton on PyPI that is newer than the locked
candidate, or nothing when the lock already names a final or no newer final
exists yet. `.github/workflows/anton-final-lock.yml` relocks to it, raises a
candidate `anton-agent` floor in pyproject.toml to it, removes anton-agent's
production pre-release allowance, and opens the PR. A
newer version number does not prove the final carries the candidate's code;
the PR's CI is that check, and a person merges it.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_contract import ALLOWANCE_TABLE, CONTRACT, PIN, allowed_prereleases  # noqa: E402

PACKAGE = "anton-agent"
PYPI = f"https://pypi.org/pypi/{PACKAGE}/json"
# The lower bound of anton-agent in pyproject.toml's dependencies.
FLOOR = re.compile(rf'"{PACKAGE}>=([^,"]+)')
# anton-agent's line in pyproject.toml's table of allowed production pre-releases,
# with the table's header and the lines before it kept in group 1.
ALLOWANCE = re.compile(
    rf'(^\[{re.escape(".".join(ALLOWANCE_TABLE))}\]\n(?:(?!\[)[^\n]*\n)*?){re.escape(PACKAGE)} = "[^"\n]*"\n',
    re.MULTILINE,
)
# How long a run started by anton's release waits for PyPI to list the final.
WAIT_ATTEMPTS = 20
WAIT_SECONDS = 30.0


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
        and installable(files)
        and sort_key(version) > floor
    ]
    return max(finals, key=sort_key, default=None)


def raise_floor(pyproject: str, final: str) -> str:
    """Raise a release-candidate anton-agent floor to `final`.

    A final floor stays: it names the oldest anton the server starts with. A
    candidate floor lets pip and uv pick newer candidates, and
    `release_contract.py --require-final` refuses it in a production release.
    """
    floors = FLOOR.findall(pyproject)
    if len(floors) != 1:
        raise ValueError(f"Expected one {PACKAGE}>= floor in pyproject.toml, found {len(floors)}")
    if is_final(floors[0]):
        return pyproject
    if sort_key(final) <= sort_key(floors[0]):
        raise ValueError(f"{PACKAGE} {final} is not newer than the floor {floors[0]}")
    return FLOOR.sub(f'"{PACKAGE}>={final}', pyproject)


def drop_allowance(pyproject: str) -> str:
    """Remove anton-agent's production pre-release allowance, since the lock names a final.

    The table and its other entries stay. An entry this can't remove, such as a
    quoted key, is refused rather than left to wave a later candidate through.
    """
    dropped = ALLOWANCE.sub(r"\1", pyproject)
    if PACKAGE in allowed_prereleases(dropped):
        raise ValueError(f"Could not remove {PACKAGE}'s entry from [{'.'.join(ALLOWANCE_TABLE)}]")
    return dropped


def installable(files: list[dict]) -> bool:
    return bool(files) and not all(file.get("yanked") for file in files)


def fetch_releases() -> dict[str, list[dict]]:
    with urllib.request.urlopen(PYPI, timeout=30) as response:
        return json.load(response)["releases"]


def announced_final(value: str) -> str | None:
    """The final anton's release says it published, if the value is one.

    The value only decides how long to wait for PyPI; what gets locked is
    still read from PyPI, so a wrong or missing value can't choose a version.
    """
    if value and is_final(value):
        return value
    if value:
        print(f"ignoring announced version {value!r}: not a final release", file=sys.stderr)
    return None


def releases_listing(announced: str | None, *, fetch=fetch_releases, sleep=time.sleep,
                     attempts: int = WAIT_ATTEMPTS, interval: float = WAIT_SECONDS) -> dict[str, list[dict]]:
    """PyPI's releases, once they list the announced final as installable.

    PyPI's JSON API can trail an upload by a minute or two, so a run started
    by anton's release retries before deciding there is nothing to relock.
    """
    releases = fetch()
    for _ in range(attempts - 1):
        if announced is None or installable(releases.get(announced, [])):
            return releases
        print(f"PyPI doesn't list {PACKAGE}=={announced} yet; retrying in {interval:.0f}s", file=sys.stderr)
        sleep(interval)
        releases = fetch()
    if announced is not None and not installable(releases.get(announced, [])):
        raise RuntimeError(f"PyPI still doesn't list {PACKAGE}=={announced}; the periodic run will retry")
    return releases


def main() -> int:
    locked = locked_version(CONTRACT.read_text())
    version = None
    if not is_final(locked):
        announced = announced_final(os.environ.get("ANNOUNCED_VERSION", ""))
        version = replacement(locked, releases_listing(announced))
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
