#!/usr/bin/env bash
# Fail if the versions this package ACTUALLY resolves to cannot be installed
# from wheels on any platform we ship to.
#
# Why this exists: the desktop app installs cowork-server with `uv tool
# install`, on machines with no compiler. When a dependency stops publishing a
# wheel for an architecture, uv silently falls back to building it from source
# and first run dead-ends at "Install cowork-server" (ENG-1629, ENG-2864).
# Nothing about that failure is visible in this repo — it surfaces weeks later
# as a blocked user on hardware no one here runs.
#
# TWO STEPS, AND THE ORDER IS THE WHOLE POINT.
#
# Resolving once with `--only-binary :all:` does NOT work as a gate, however
# obvious it looks. That flag lets the resolver BACKTRACK: told it may not
# build from source, it quietly selects an older version that has a wheel and
# reports success. Run against the exact metadata that shipped the ENG-2864
# regression it printed "ok" for every platform — it answers "could this be
# installed wheel-only", which is not the question. The question is "does the
# version we are actually going to install have a wheel".
#
# So: resolve normally to get the pinned set (step 1), then re-resolve THAT
# pinned set wheel-only (step 2). Exact `==` pins leave nothing to backtrack
# to, so a missing wheel has to fail.
#
# `--no-config` for the same reason as `--no-sources`. Without it uv applies
# this project's `[tool.uv] override-dependencies` (currently `openai>=3.0`,
# `pillow>=12.3.0`) while resolving — and those overrides are NOT part of the
# published wheel, so a user's `uv tool install` never sees them. Measured: with
# config, a requirement of `openai<3` still resolves to 3.16.1; with
# `--no-config` it resolves to 2.54.0, which is what a user actually gets. The
# two agree for today's dependency set, so this closes a latent divergence
# rather than a live break — but a gate that resolves under constraints users do
# not have is the same defect class this file exists to catch.
#
# `--no-sources` on both steps, for a related reason. `[tool.uv.sources]` points
# anton-agent and hermes-agent at git, which is a source build by definition and
# has no wheel to find — without the flag this gate fails on every platform for a
# reason no user will ever hit. More importantly, those sources do not appear in
# the published wheel's metadata at all: users resolve `anton-agent>=...,<3` from
# PyPI. Checking the dev tree would be checking something we do not ship.
#
# `aarch64-pc-windows-msvc` is deliberately absent: it is unsatisfiable
# regardless of this check, because psycopg-binary publishes no win_arm64
# wheel, and the installer ships `nsis x64` only. Supporting Windows ARM is its
# own piece of work, not this gate's business.
#
# TARGET. With no argument this checks `pyproject.toml`, which is what the PR
# and pre-tag gates want: it blames the commit that broke it, before anything is
# tagged. Pass a BUILT WHEEL instead where the published metadata is not the
# checked-in metadata — `publish-staging.yml` rewrites the anton-agent
# requirement to an exact rc before building, so the wheel users install can
# carry a dependency set this file never described. Checking the wheel checks
# what ships.
#
# Run locally with:  bash scripts/check_wheels_available.sh
#                    bash scripts/check_wheels_available.sh dist/cowork_server-*.whl
set -uo pipefail

TARGET="${1:-pyproject.toml}"
if [ ! -e "$TARGET" ]; then
  echo "error: target '$TARGET' does not exist" >&2
  exit 2
fi

PLATFORMS=(
  x86_64-apple-darwin
  aarch64-apple-darwin
  x86_64-pc-windows-msvc
  x86_64-unknown-linux-gnu
)

# Both interpreters in `requires-python`. The desktop installer passes a RANGE
# (`--python '>=3.12,<3.14'`) and uv picks the newest managed CPython it has, so
# checking only the floor would miss a dependency whose wheel matrix stops at an
# older ABI — which is exactly how a user ends up on 3.13 with no wheel.
PYTHON_VERSIONS=(3.12 3.13)

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# `uv pip compile` takes a requirements file or a pyproject.toml, not a wheel
# path, so a wheel is wrapped in a one-line requirements file. Resolving it
# pulls the wheel's own Requires-Dist, which is the exact set a user's
# `uv tool install` sees.
case "$TARGET" in
  *.whl)
    printf '%s\n' "$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")" > "$workdir/target.txt"
    SOURCE="$workdir/target.txt"
    echo "Checking built wheel: $(basename "$TARGET")"
    ;;
  *)
    SOURCE="$TARGET"
    echo "Checking $TARGET"
    ;;
esac
echo

failed=0
for plat in "${PLATFORMS[@]}"; do
  for pyver in "${PYTHON_VERSIONS[@]}"; do
    printf '%-28s py%s  ' "$plat" "$pyver"
    pinned="$workdir/$plat-$pyver.txt"

    # Step 1 — what would we actually install here?
    if ! out=$(uv pip compile "$SOURCE" \
                 --python-platform "$plat" \
                 --python-version "$pyver" \
                 --no-sources --no-config \
                 --quiet --output-file "$pinned" 2>&1); then
      echo "FAILED (could not resolve at all)"
      echo "$out" | sed 's/^/    /'
      failed=1
      continue
    fi

    # Step 2 — is every one of those exact versions installable from a wheel?
    if out=$(uv pip compile "$pinned" \
               --python-platform "$plat" \
               --python-version "$pyver" \
               --no-sources --no-config \
               --only-binary :all: \
               --quiet 2>&1); then
      echo "ok"
    else
      echo "FAILED (resolves to a version with no wheel)"
      echo "$out" | sed 's/^/    /'
      failed=1
    fi
  done
done

if [ "$failed" -ne 0 ]; then
  cat <<'MSG'

---
A dependency resolves to a version that publishes no wheel for a platform we
ship to.

The desktop app installs this package on machines with no compiler, so this
reaches users as "cowork-server installation failed" at first run, not as a
slow install.

Fix it by constraining the dependency to its last version that publishes a
wheel for the affected platform. Scope the constraint with an environment
marker so the other platforms stay current, e.g.

    "cryptography>=48.0.0,<51",
    "cryptography<49; sys_platform == 'darwin' and platform_machine == 'x86_64'",

Then run `uv lock` and commit the lockfile.
MSG
  exit 1
fi

echo
echo "All platforms resolve to versions that ship wheels."
