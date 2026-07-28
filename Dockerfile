# Builder: resolve + install dependencies and the project into a venv with uv.
FROM python:3.12-slim AS builder

# uv binary (pinned to match the repo's lockfile tooling).
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

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
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# Final: slim runtime with just the venv + source.
FROM python:3.12-slim AS final

WORKDIR /app

COPY --from=builder /app /app

# Put the venv on PATH; run the app directly from it.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 9010

CMD ["python", "-m", "uvicorn", "cowork.server:app", "--host", "0.0.0.0", "--port", "9010", "--forwarded-allow-ips", "*"]
