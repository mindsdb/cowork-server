"""Export the reviewed runtime lock without resolving or upgrading packages."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "cowork" / "release-requirements.txt"

# One `name==version` pin per requirement line of the uv export; hash and
# marker continuations follow on indented lines.
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
# A canonical PEP 440 version, as uv writes it. The pre-release (`a`, `b`,
# `rc`) and `.dev` segments are captured so a candidate can be refused.
VERSION = re.compile(
    r"^(?:\d+!)?\d+(?:\.\d+)*(?P<pre>(?:a|b|rc)\d+)?(?:\.post\d+)?(?P<dev>\.dev\d+)?(?:\+[a-z0-9.]+)?$"
)


def unreleased_pins(export: str) -> list[str]:
    """Return every pin in the export that names a pre-release or dev build.

    Desktop installs exactly what this export names, so a production release
    must name only final releases. ENG-4101 and ENG-4355 shipped an
    anton-agent release candidate to production because nothing checked. A
    version this cannot parse is refused too, rather than waved through.
    """
    refused = []
    for line in export.splitlines():
        match = PIN.match(line)
        if not match:
            continue
        name, version = match.groups()
        parsed = VERSION.match(version)
        if parsed is None or parsed["pre"] or parsed["dev"]:
            refused.append(f"{name}=={version}")
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
        refused = unreleased_pins(result.stdout)
        if refused:
            print(
                "A production release must lock final releases only. The lock pins:\n  "
                + "\n  ".join(refused)
                + "\nLock the final release that carries the same code, then run"
                " python scripts/release_contract.py.",
                file=sys.stderr,
            )
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
