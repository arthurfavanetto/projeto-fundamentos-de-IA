# ==============================================================================
# RBAC AI Assistant — Multi-Stage Dockerfile (Python 3.12+ / uv / Non-Root)
# ==============================================================================

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PTB_TIMEDELTA=1

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/documents/raw_pdfs /app/data/documents/extracted /app/data/documents/reviews /app/logs /app/config /app/scripts \
    && chown -R 10001:10001 /app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 config/settings.yaml ./config/settings.yaml
COPY --chown=10001:10001 config/regulations_catalog.yaml ./config/regulations_catalog.yaml
COPY --chown=10001:10001 config/prompts ./config/prompts
COPY --chown=10001:10001 config/evaluations ./config/evaluations
COPY --chown=10001:10001 scripts ./scripts

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-m", "rbac_assistant.healthcheck"]

CMD ["python", "-m", "rbac_assistant.main"]
