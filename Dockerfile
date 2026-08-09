# =============================================================================
# 本番エージェント用コンテナ
# - 評価 (deepeval) / 最適化 (gepa) の依存は含めない (バッチ側の責務)
# - 非 root ユーザーで実行
# =============================================================================

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 依存だけ先に解決してレイヤキャッシュを効かせる
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# プロジェクト本体
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    AGENTOPS_PROJECT_ROOT=/app

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid agent --create-home --shell /usr/sbin/nologin agent

WORKDIR /app

COPY --from=builder --chown=agent:agent /app/.venv /app/.venv

# エージェントの挙動を決める資産 (GEPA 最適化の成果物がそのまま反映される)
COPY --chown=agent:agent prompts ./prompts
COPY --chown=agent:agent skills ./skills
COPY --chown=agent:agent config ./config
COPY --chown=agent:agent data/seed ./data/seed

USER agent

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).status==200 else 1)"

CMD ["uvicorn", "agentops_loop.service:app", "--host", "0.0.0.0", "--port", "8080"]
