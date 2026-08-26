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
# would ship every commit of this repo inside the runtime image. Deleting it in
# the same layer keeps it out of the image rather than merely out of the last
# one. Order matters: the sync must resolve the version before .git is gone.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && rm -rf /app/.git


# Final: slim runtime with just the venv + source.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS final

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
