"""Langfuse (OpenTelemetry ベース) によるトレース/メトリクス送信.

Langfuse Python SDK v3 以降は内部が OpenTelemetry で実装されており、
`Langfuse` クライアントを初期化した時点で OTel の TracerProvider / OTLP エクスポータが
セットアップされる。LangGraph の実行は `CallbackHandler` により自動的に
ノード単位の span + LLM generation (トークン数・レイテンシー・コスト) として記録される。

本モジュールはそれに加えて、オフライン評価データセットに必要な業務属性
(ユーザー入力 / 分類結果 / MCP tool 選択 / ツール入出力 / 検証結果 / 最終回答) を
トレースの metadata として明示的に添付する。

Langfuse 未設定時はすべて no-op になり、エージェント本体は動作し続ける。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..settings import langfuse_settings

logger = logging.getLogger(__name__)

_client: Any | None = None
_initialized = False


def init_tracing(force: bool = False) -> Any | None:
    """Langfuse クライアント (= OTel パイプライン) を初期化する."""
    global _client, _initialized
    if _initialized and not force:
        return _client

    _initialized = True
    settings = langfuse_settings()
    if not settings.is_configured:
        logger.info("Langfuse は未設定のためトレースを送信しません")
        _client = None
        return None

    # v3 (LANGFUSE_HOST) / v4 (LANGFUSE_BASE_URL) 双方の SDK に合わせて両方を設定する
    os.environ.setdefault("LANGFUSE_HOST", settings.resolved_base_url)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.resolved_base_url)
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", settings.tracing_environment)

    try:
        from langfuse import Langfuse, get_client

        Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            **_base_url_kwarg(settings.resolved_base_url),
        )
        _client = get_client()
        logger.info("Langfuse トレースを有効化しました (%s)", settings.resolved_base_url)
    except Exception:  # noqa: BLE001 - 観測基盤の障害でエージェントを止めない
        logger.exception("Langfuse の初期化に失敗しました。トレースなしで継続します")
        _client = None
    return _client


def _base_url_kwarg(base_url: str) -> dict[str, str]:
    """SDK バージョン差 (`base_url` / `host`) を吸収する."""
    try:
        import inspect

        from langfuse import Langfuse

        params = inspect.signature(Langfuse.__init__).parameters
        if "base_url" in params:
            return {"base_url": base_url}
        if "host" in params:
            return {"host": base_url}
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse コンストラクタのシグネチャ判定に失敗しました", exc_info=True)
    return {}


def get_tracing_client() -> Any | None:
    return _client if _initialized else init_tracing()


def langchain_callbacks() -> list[Any]:
    """LangGraph / LangChain 実行に渡すコールバック."""
    if get_tracing_client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        logger.exception("Langfuse CallbackHandler の生成に失敗しました")
        return []


@dataclass
class TraceHandle:
    """1 リクエスト分のトレースへの参照."""

    trace_id: str | None = None
    _span: Any | None = field(default=None, repr=False)

    def update_trace(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update_trace(**kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("update_trace に失敗しました", exc_info=True)

    def update(self, **kwargs: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update(**kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("span.update に失敗しました", exc_info=True)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        client = get_tracing_client()
        if client is None or not self.trace_id:
            return
        try:
            client.create_score(
                trace_id=self.trace_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception:  # noqa: BLE001
            logger.debug("create_score に失敗しました", exc_info=True)


@contextmanager
def start_trace(
    name: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    input_payload: Any = None,
) -> Iterator[TraceHandle]:
    """ルート span を張り、トレース属性を設定する."""
    client = get_tracing_client()
    if client is None:
        yield TraceHandle()
        return

    try:
        with client.start_as_current_observation(as_type="span", name=name) as span:
            handle = TraceHandle(trace_id=getattr(span, "trace_id", None), _span=span)
            handle.update(input=input_payload)
            handle.update_trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                tags=tags or [],
                input=input_payload,
            )
            yield handle
    except Exception:  # noqa: BLE001
        logger.exception("トレース開始に失敗しました。トレースなしで継続します")
        yield TraceHandle()


def flush_tracing() -> None:
    client = get_tracing_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.debug("flush に失敗しました", exc_info=True)


def shutdown_tracing() -> None:
    global _client, _initialized
    client = _client
    _client = None
    _initialized = False
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:  # noqa: BLE001
        logger.debug("shutdown に失敗しました", exc_info=True)
