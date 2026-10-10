FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

COPY requirements.txt /tmp/profiler-requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r /tmp/profiler-requirements.txt \
    && rm /tmp/profiler-requirements.txt \
    && py-spy --version

# The measurement helper explicitly sets the debugger's inspection privileges.
USER 1000
CMD ["py-spy", "--version"]
