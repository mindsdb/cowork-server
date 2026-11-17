"""Regenerate cowork/skills_builtin/shipped.json from git history.

For each packaged builtin it records the fingerprint of every version committed
on the release branches (and the working tree), plus the BUILTIN_SKILLS_VERSION
the skill first shipped in. Seeding uses it to update a store's copy only when
that copy is exactly a version we shipped, and to leave a deleted builtin
deleted.

    python scripts/builtin_skills_shipped.py          # rewrite the file
    python scripts/builtin_skills_shipped.py --check  # exit 1 if it is stale

Run it in a full clone (it reads history), after any change to a builtin.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cowork.services.skills import (  # noqa: E402
    BUILTIN_SKILLS_DIR,
    BUILTIN_SKILLS_SHIPPED,
    CODE_ONLY_BUILTIN_SKILL_NAMES,
    SKILL_FILE,
    builtin_fingerprint,
    fingerprint_skill_files,
)

REL_DIR = BUILTIN_SKILLS_DIR.relative_to(ROOT).as_posix()
REFS = ("HEAD", "origin/staging", "origin/main")


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def existing_refs() -> list[str]:
    refs = []
    for ref in REFS:
        if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", ref],
                          capture_output=True).returncode == 0:
            refs.append(ref)
    return refs


def committed_fingerprints(slug: str, refs: list[str]) -> set[str]:
    path = f"{REL_DIR}/{slug}"
    commits = git("log", "--format=%H", *refs, "--", path).decode().split()
    found = set()
    for commit in commits:
        listing = git("ls-tree", "-r", "-z", commit, "--", f"{path}/").split(b"\0")
        files = {}
        for line in filter(None, listing):
            meta, name = line.split(b"\t", 1)
            _mode, kind, blob = meta.split()
            if kind != b"blob":
                continue
            rel = name.decode()[len(path) + 1:]
            files[rel] = git("cat-file", "blob", blob.decode())
        if SKILL_FILE in files:
            found.add(fingerprint_skill_files(files))
    return found


def version_when_added(slug: str, refs: list[str]) -> int:
    added = git("log", "--format=%H", "--diff-filter=A", *refs, "--",
                f"{REL_DIR}/{slug}/{SKILL_FILE}").decode().split()
    if not added:
        return 0
    source = git("show", f"{added[-1]}:cowork/services/skills.py").decode()
    # The Code store counts its own versions.
    name = "CODE_BUILTIN_SKILLS_VERSION" if slug in CODE_ONLY_BUILTIN_SKILL_NAMES else "BUILTIN_SKILLS_VERSION"
    match = re.search(rf"^{name} = (\d+)", source, re.M)
    return int(match.group(1)) if match else 1


def build() -> dict:
    refs = existing_refs()
    previous = {}
    if BUILTIN_SKILLS_SHIPPED.exists():
        previous = json.loads(BUILTIN_SKILLS_SHIPPED.read_text(encoding="utf-8"))
    current_version = int(re.search(
        r"^BUILTIN_SKILLS_VERSION = (\d+)",
        (ROOT / "cowork/services/skills.py").read_text(), re.M).group(1))
    manifest = {}
    for src in sorted(p for p in BUILTIN_SKILLS_DIR.iterdir() if (p / SKILL_FILE).is_file()):
        slug = src.name
        shipped = committed_fingerprints(slug, refs) | set(previous.get(slug, {}).get("shipped", ()))
        shipped.add(builtin_fingerprint(src))
        introduced = previous.get(slug, {}).get("introduced") or version_when_added(slug, refs) or current_version
        manifest[slug] = {"introduced": introduced, "shipped": sorted(shipped)}
    return manifest


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    text = render(build())
    if "--check" in sys.argv:
        if not BUILTIN_SKILLS_SHIPPED.exists() or BUILTIN_SKILLS_SHIPPED.read_text(encoding="utf-8") != text:
            print(f"{BUILTIN_SKILLS_SHIPPED} is stale: run python scripts/builtin_skills_shipped.py")
            return 1
        return 0
    BUILTIN_SKILLS_SHIPPED.write_text(text, encoding="utf-8")
    print(f"wrote {BUILTIN_SKILLS_SHIPPED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
