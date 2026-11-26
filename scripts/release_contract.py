"""Export the reviewed runtime lock without resolving or upgrading packages."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "cowork" / "release-requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

# One `name==version` pin per requirement line of the uv export; hash and
# marker continuations follow on indented lines.
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
# A canonical PEP 440 version, as uv writes it. The pre-release (`a`, `b`,
# `rc`) and `.dev` segments are captured so a candidate can be refused.
VERSION = re.compile(
    r"^(?:\d+!)?\d+(?:\.\d+)*(?P<pre>(?:a|b|rc)\d+)?(?:\.post\d+)?(?P<dev>\.dev\d+)?(?:\+[a-z0-9.]+)?$"
)
# Each version a declared requirement names, such as the `2.0rc1` in `>=2.0rc1,<3`.
SPECIFIER = re.compile(r"(?:===|~=|==|!=|<=|>=|<|>)\s*([^\s,;]+)")
# The project name a declared requirement starts with, before extras and bounds.
REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
# pyproject.toml's table of the exact pre-releases a production release may name.
ALLOWANCE_TABLE = ("tool", "cowork", "release", "allow-prerelease")


def canonical(name: str) -> str:
    """A project name as PEP 503 compares it, so `Anton_Agent` matches `anton-agent`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def allowed_prereleases(pyproject: str) -> dict[str, str]:
    """Return the one pre-release each listed project may name in a production release.

    Each entry names an exact version, so a newer candidate of the same project
    is still refused. The anton relock removes anton-agent's entry when it locks
    a final.
    """
    table = tomllib.loads(pyproject)
    for key in ALLOWANCE_TABLE:
        table = table.get(key, {})
    allowed = {}
    for name, version in table.items():
        if not isinstance(version, str):
            raise ValueError(f"[{'.'.join(ALLOWANCE_TABLE)}] {name} must name one version as a string")
        allowed[canonical(name)] = version
    return allowed


def unreleased_pins(export: str, allowed: dict[str, str] | None = None) -> list[str]:
    """Return every pin in the export that names a pre-release or dev build.

    Desktop installs exactly what this export names, so a production release
    must name only final releases, apart from the exact versions `allowed`
    names. A version this cannot parse is refused too, rather than waved through.
    """
    allowed = allowed or {}
    refused = []
    for line in export.splitlines():
        match = PIN.match(line)
        if not match:
            continue
        name, version = match.groups()
        if allowed.get(canonical(name)) == version:
            continue
        parsed = VERSION.match(version)
        if parsed is None or parsed["pre"] or parsed["dev"]:
            refused.append(f"{name}=={version}")
    return refused


def unreleased_requirements(pyproject: str) -> list[str]:
    """Return every declared requirement that names a pre-release or dev build.

    The wheel's metadata carries these requirements, and a pre-release in one
    lets pip and uv pick newer pre-releases for anyone who installs from PyPI
    without the lock. A version this cannot parse is refused too. The exact
    version an allowance names for that project passes.
    """
    allowed = allowed_prereleases(pyproject)
    project = tomllib.loads(pyproject)["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    refused = []
    for requirement in requirements:
        versions = requirement.split(";", 1)[0]
        if "@" in versions:
            # A direct reference names a URL, not a version.
            continue
        name = REQUIREMENT_NAME.match(versions)
        allowance = allowed.get(canonical(name.group(1))) if name else None
        for version in SPECIFIER.findall(versions):
            if version == allowance:
                continue
            parsed = VERSION.match(version.removesuffix(".*"))
            if parsed is None or parsed["pre"] or parsed["dev"]:
                refused.append(requirement)
                break
    return refused


def check_distributions(directory: Path, expected: str) -> None:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    member = CONTRACT.relative_to(ROOT).as_posix()
    with zipfile.ZipFile(wheels[0]) as wheel:
        if wheel.namelist().count(member) != 1 or wheel.read(member).decode() != expected:
            raise ValueError("Wheel runtime contract differs from the reviewed export")
    with tarfile.open(sdists[0]) as sdist:
        members = [item for item in sdist.getmembers() if item.name.endswith("/" + member)]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("Sdist must contain one regular runtime contract file")
        stream = sdist.extractfile(members[0])
        if stream is None or stream.read().decode() != expected:
            raise ValueError("Sdist runtime contract differs from the reviewed export")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reject a stale export")
    parser.add_argument("--dist-dir", type=Path, help="also verify the built wheel and sdist")
    parser.add_argument(
        "--require-final", action="store_true",
        help="reject an export that pins a pre-release or dev build (production releases)",
    )
    args = parser.parse_args()
    result = subprocess.run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--no-header", "--no-annotate"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    if args.check:
        if not CONTRACT.is_file() or CONTRACT.read_text() != result.stdout:
            print("Runtime export is stale; run python scripts/release_contract.py and review the diff.", file=sys.stderr)
            return 1
    else:
        CONTRACT.write_text(result.stdout)
    if args.require_final:
        pyproject = PYPROJECT.read_text()
        allowed = allowed_prereleases(pyproject)
        if allowed:
            print(
                "pyproject.toml allows these pre-releases in a production release:\n  "
                + "\n  ".join(f"{name}=={version}" for name, version in sorted(allowed.items())),
                file=sys.stderr,
            )
        refused = unreleased_pins(result.stdout, allowed)
        if refused:
            print(
                "A production release must lock final releases only. The lock pins:\n  "
                + "\n  ".join(refused)
                + "\nLock the final release that carries the same code, then run"
                " python scripts/release_contract.py.",
                file=sys.stderr,
            )
        declared = unreleased_requirements(pyproject)
        if declared:
            print(
                "A production release must declare final releases only. pyproject.toml requires:\n  "
                + "\n  ".join(declared)
                + "\nRaise each bound to the final release the lock names, then run uv lock and"
                " python scripts/release_contract.py.",
                file=sys.stderr,
            )
        if refused or declared:
            return 1
    if args.dist_dir:
        try:
            check_distributions(args.dist_dir, result.stdout)
        except (ValueError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
