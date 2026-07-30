#!/usr/bin/env bash
set -euo pipefail

readonly REVISION="${1:?usage: use-anton-revision.sh <40-character-commit-sha>}"

if [[ ! "${REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Refusing invalid Anton revision: ${REVISION}" >&2
  exit 1
fi

# Replace the committed Anton source only inside this CI checkout. Resolving
# the branch once in the caller and passing its immutable SHA to every consumer
# keeps unit tests and the PR image on the same upstream snapshot.
uv remove --frozen anton-agent
uv add --no-sync \
  git+https://github.com/mindsdb/anton.git \
  --rev "${REVISION}"
