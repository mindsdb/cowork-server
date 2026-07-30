#!/usr/bin/env bash
set -euo pipefail

# Resolve the immutable PyPI artifact produced by the current Anton release
# branch. A stale "latest" package is not good enough: downstream tests, images,
# and wheels must consume the package built from the exact branch HEAD.

readonly ANTON_REPO_URL="https://github.com/mindsdb/anton.git"
readonly ANTON_PYPI_URL="https://pypi.org/pypi/anton-agent/json"
readonly CHANNEL="${1:?usage: resolve-anton-package.sh <main|staging>}"
readonly ATTEMPTS="${ANTON_PACKAGE_RESOLVE_ATTEMPTS:-12}"
readonly DELAY_SECONDS="${ANTON_PACKAGE_RESOLVE_DELAY_SECONDS:-10}"

if [[ "${CHANNEL}" != "main" && "${CHANNEL}" != "staging" ]]; then
  echo "Unsupported Anton package channel: ${CHANNEL}" >&2
  exit 1
fi

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  branch_sha=$(
    git ls-remote "${ANTON_REPO_URL}" "refs/heads/${CHANNEL}" |
      awk 'NR == 1 { print $1 }'
  )
  pypi_json=$(curl -fsSL --retry 3 "${ANTON_PYPI_URL}")
  if [[ "${CHANNEL}" == "main" ]]; then
    version=$(jq -r '.info.version // empty' <<<"${pypi_json}")
    if [[ -n "${branch_sha}" && -n "${version}" ]]; then
      tag_sha=$(
        git ls-remote "${ANTON_REPO_URL}" "refs/tags/v${version}" |
          awk 'NR == 1 { print $1 }'
      )
    else
      tag_sha=""
    fi
  else
    # Search every live rc, newest upload first, rather than assuming the
    # globally newest upload belongs to the newest branch commit. This remains
    # correct even if queued publisher runs complete out of order.
    versions=$(
      jq -r '
        [
          .releases
          | to_entries[]
          | select(.key | test("^[0-9]+(\\.[0-9]+)*rc[0-9]+$"))
          | select((.value | length) > 0)
          | select(any(.value[]; .yanked != true))
          | {
              version: .key,
            uploaded: ([.value[].upload_time_iso_8601] | max)
          }
        ]
        | sort_by(.uploaded)
        | reverse
        | .[].version
      ' <<<"${pypi_json}"
    )
    rc_tags=$(git ls-remote "${ANTON_REPO_URL}" "refs/tags/v*rc*")
    version=""
    tag_sha=""
    while IFS= read -r candidate; do
      [[ -n "${candidate}" ]] || continue
      candidate_sha=$(
        awk -v ref="refs/tags/v${candidate}" '$2 == ref { print $1; exit }' <<<"${rc_tags}"
      )
      if [[ -n "${candidate_sha}" && "${candidate_sha}" == "${branch_sha}" ]]; then
        version="${candidate}"
        tag_sha="${candidate_sha}"
        break
      fi
    done <<<"${versions}"
  fi

  if [[ -n "${branch_sha}" && -n "${version}" ]]; then
    if [[ -n "${tag_sha}" && "${tag_sha}" == "${branch_sha}" ]]; then
      echo "Resolved anton-agent==${version} from anton/${CHANNEL}@${branch_sha}" >&2
      printf '%s\n' "${version}"
      exit 0
    fi
  fi

  if ((attempt < ATTEMPTS)); then
    echo "Anton ${CHANNEL} package is not published for the current ${CHANNEL} HEAD yet; retrying (${attempt}/${ATTEMPTS})..." >&2
    sleep "${DELAY_SECONDS}"
  fi
done

echo "No published anton-agent package matches the current anton/${CHANNEL} HEAD." >&2
exit 1
