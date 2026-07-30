#!/usr/bin/env bash
set -euo pipefail

readonly VERSION="${1:?usage: use-anton-package.sh <anton-agent-version>}"

if [[ ! "${VERSION}" =~ ^[0-9]+(\.[0-9]+)*(rc[0-9]+)?$ ]]; then
  echo "Refusing invalid anton-agent version: ${VERSION}" >&2
  exit 1
fi

# The committed project keeps its main-line dependency source. Staging jobs
# replace that source ephemerally with one immutable registry artifact. Removing
# first also removes any [tool.uv.sources] entry, then `uv add` writes both the
# exact requirement and a registry-backed lockfile.
uv remove --frozen anton-agent
uv add --no-sync "anton-agent==${VERSION}"
