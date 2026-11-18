"""Regenerate cowork/skills_builtin/shipped.json from git history.

For each packaged builtin it records the fingerprint of every version committed
on the release branches (and the working tree), plus the BUILTIN_SKILLS_VERSION
the skill first shipped in. Seeding uses it to update a store's copy only when
that copy is exactly a version we shipped, and to leave a deleted builtin
deleted.

    python scripts/builtin_skills_shipped.py          # rewrite the file
    python scripts/builtin_skills_shipped.py --check  # exit 1 if it is stale

Run it in a full clone (it reads history), after any change to a builtin.
Bump BUILTIN_SKILLS_VERSION (CODE_BUILTIN_SKILLS_VERSION for a Code skill)
first: a new or changed builtin is refused until the version is higher than the
one the manifest on origin/staging recorded for that store, because a store
already at that version would never see the change. Comparing with staging, not
with the working copy, lets a branch edit a builtin more than once under one bump. A new builtin records the bumped version as
the one it was introduced in.
"""
from __future__ import annotations

import json
import os
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


class NeedsVersionBump(Exception):
    pass


def entry_for(slug: str, fingerprint: str, version: int, previous: dict, *,
              store_last: int, bootstrap_introduced: int = 0) -> dict:
    """The manifest fields for one builtin, except `shipped`.

    `version` is its store's version now; `store_last` is the highest version
    the previous manifest recorded for that store (0 when there is none, i.e.
    the manifest is being created). A new or changed builtin needs `version`
    above what was recorded, or stores already there would never get it.
    """
    prev = previous.get(slug)
    if prev is None:
        if store_last and version <= store_last:
            raise NeedsVersionBump(f"{slug} is new: bump its store's version above {store_last} first")
        introduced = version if store_last else (bootstrap_introduced or version)
        return {"introduced": introduced, "version": version, "current": fingerprint}
    if fingerprint == prev.get("current"):
        return {"introduced": prev["introduced"], "version": prev.get("version", version), "current": fingerprint}
    # Compared with the store's last version, not this skill's: a store at
    # store_last never re-seeds, even if this skill was recorded lower.
    if "current" in prev and version <= store_last:
        raise NeedsVersionBump(f"{slug} changed: bump its store's version above {store_last} first")
    return {"introduced": prev["introduced"], "version": version, "current": fingerprint}


def store_versions() -> dict[str, int]:
    source = (ROOT / "cowork/services/skills.py").read_text()
    return {name: int(re.search(rf"^{name} = (\d+)", source, re.M).group(1))
            for name in ("BUILTIN_SKILLS_VERSION", "CODE_BUILTIN_SKILLS_VERSION")}


def store_of(slug: str) -> str:
    return "CODE_BUILTIN_SKILLS_VERSION" if slug in CODE_ONLY_BUILTIN_SKILL_NAMES else "BUILTIN_SKILLS_VERSION"


def released_manifest() -> dict:
    """shipped.json as origin/staging has it, or {} before it existed there.
    BUILTIN_SKILLS_BASELINE_REF names another ref, for trying the rules out."""
    rel = BUILTIN_SKILLS_SHIPPED.relative_to(ROOT).as_posix()
    baseline = os.environ.get("BUILTIN_SKILLS_BASELINE_REF", "origin/staging")
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{baseline}:{rel}"],
                           capture_output=True)
    return json.loads(shown.stdout) if shown.returncode == 0 else {}


def build() -> dict:
    refs = existing_refs()
    previous = {}
    if BUILTIN_SKILLS_SHIPPED.exists():
        previous = json.loads(BUILTIN_SKILLS_SHIPPED.read_text(encoding="utf-8"))
    released = released_manifest()
    versions = store_versions()
    store_last = {store: 0 for store in versions}
    for slug, entry in released.items():
        recorded = entry.get("version", entry.get("introduced", 0))
        store_last[store_of(slug)] = max(store_last[store_of(slug)], recorded)
    manifest = {}
    for src in sorted(p for p in BUILTIN_SKILLS_DIR.iterdir() if (p / SKILL_FILE).is_file()):
        slug = src.name
        fingerprint = builtin_fingerprint(src)
        entry = entry_for(
            slug, fingerprint, versions[store_of(slug)], released,
            store_last=store_last[store_of(slug)],
            bootstrap_introduced=0 if released else version_when_added(slug, refs),
        )
        shipped = committed_fingerprints(slug, refs) | set(previous.get(slug, {}).get("shipped", ()))
        shipped.add(fingerprint)
        manifest[slug] = {**entry, "shipped": sorted(shipped)}
    return manifest


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    try:
        text = render(build())
    except NeedsVersionBump as exc:
        print(f"refused: {exc}")
        return 1
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
