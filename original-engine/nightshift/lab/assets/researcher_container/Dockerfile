# Image for the CLI-backed researcher runtime (nightshift/lab/runtime_coral_cli.py).
# python:3.12-slim already ships bash and python3; CORAL needs git. Everything else is mounted at run time:
# the CORAL virtual environment (read-only), the Claude Code package (read-only) and uv (read-only).
# Build without a context:  docker build -t glimmer-researcher:cli-1 - < Dockerfile
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git config --system --add safe.directory '*'
