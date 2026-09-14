# Builder: resolve + install dependencies and the project into a venv with uv.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS builder

# uv binary (pinned to match the repo's lockfile tooling).
COPY --from=ghcr.io/astral-sh/uv:0.6.14@sha256:3362a526af7eca2fcd8604e6a07e873fb6e4286d8837cb753503558ce1213664 /uv /uvx /bin/

# git: pyproject sources anton-agent / hermes-agent from GitHub, and hatch-vcs
# reads git metadata for the version. build-essential is NOT needed — psycopg
# is installed as psycopg[binary] (prebuilt wheels).
RUN --mount=target=/var/lib/apt,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached layer) without the project itself, so a
# source-only change doesn't invalidate the dependency install.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then the project source and a full sync (installs cowork-server into the venv).
#
# .git rides along in the context on purpose — hatch-vcs reads it to derive the
# version, which is baked into the installed dist-info by this sync. It is then
# deleted: the build workflow now checks out full history for that version
# (ENG-1796), and the final stage COPYs this whole directory, so leaving it
# would ship every commit of this repo inside the runtime image.
#
# What keeps it out is the STAGE boundary, not this layer — an earlier revision
# of this comment credited the wrong mechanism. `.git` does exist in the builder
# layer created by `COPY . /app` above, and this `rm` is a later layer, so it
# does not erase that history. The final stage's `COPY --from=builder /app /app`
# copies the builder's final filesystem STATE rather than its layer history,
# which is what actually leaves `.git` behind. Worth stating precisely: the
# classic form of this mistake — deleting in a later layer of the SHIPPED stage —
# does leak, and reads identically to this.
#
# Order matters: the sync must resolve the version before .git is gone.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && v="$(.venv/bin/python -c 'from importlib.metadata import version; print(version("cowork-server"))')" \
    && echo "cowork-server version: $v" \
    && case "$v" in 0.0.0*) \
         echo "ERROR: version resolved to $v - the checkout has no tags, so hatch-vcs" >&2; \
         echo "       had nothing to describe against. Build with fetch-depth: 0 (ENG-1796)." >&2; \
         exit 1 ;; \
       esac \
    && rm -rf /app/.git


# Final: slim runtime with just the venv + source.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS final

# Bring the base packages up to the current trixie versions. The digest above
# pins a rootfs Debian builds from a dated snapshot.debian.org pin, so it ships
# whatever the archive held on that date and never moves afterwards, and nothing
# else in this stage runs apt. Bumping the digest is not an alternative: every
# published digest of this image pins a snapshot of its own, and the newest one
# carries the same superseded packages as this one. The fixes ship in trixie
# main rather than trixie-security, so narrowing this to a security-only source
# would miss them.
#
# IMAGE_TAG is declared to put its value into this layer's cache key. BuildKit
# keys a RUN on the parent state plus the command text, and a digest-pinned base
# holds both constant forever, so a long-lived builder would serve this layer
# from cache on every later build and the image would drift back to the
# snapshot's packages while the file still read as though it upgrades. Every ARG
# in scope joins the environment of the RUN steps below it, so a changed value
# misses the cache, and the build passes
# `--build-arg IMAGE_TAG=<environment>-<commit sha>`: the upgrade re-runs once
# per commit. Runs before `USER app` below, because apt needs root.
ARG IMAGE_TAG
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    # apt-get upgrade exits 0 when it holds a package back, so a fix that needs
    # a new dependency is skipped and the image ships the CVE this layer exists
    # to remove. Re-simulate and fail the build if anything is still pending.
    && apt-get -s upgrade | grep -q "and 0 not upgraded" \
    && rm -rf /var/lib/apt/lists/*

# Non-root: this image runs in per-PR dev environments (see
# .github/workflows/build-deploy.yml) and is a candidate for staging/prod via
# Helm. A real home dir matters, not just a UID — cowork/common/paths.py
# defaults all app state (db, uploads, memory, connector vault) under
# Path.home()/".cowork", so the user needs a writable HOME for that to work.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

# Put the venv on PATH; run the app directly from it.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

USER app

EXPOSE 9010

CMD ["python", "-m", "uvicorn", "cowork.server:app", "--host", "0.0.0.0", "--port", "9010", "--forwarded-allow-ips", "*"]
