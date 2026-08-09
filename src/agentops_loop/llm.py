"""チャットモデルのファクトリ.

エージェント / 評価 (Judge) / GEPA reflection の 3 系統でモデルを分離できるよう、
provider・model を引数化した薄いラッパーを提供する。
`init_chat_model` を使うため OpenAI 以外 (Azure OpenAI, Anthropic, Bedrock 等) にも
依存パッケージを追加するだけで切り替えられる。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=32)
def build_chat_model(
    model: str,
    provider: str = "openai",
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> BaseChatModel:
    """チャットモデルを生成する (同一パラメータならキャッシュを再利用)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "model_provider": provider,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    try:
        return init_chat_model(**kwargs)
    except Exception:  # noqa: BLE001 - temperature 非対応モデル (o系/gpt-5系) へのフォールバック
        kwargs.pop("temperature", None)
        return init_chat_model(**kwargs)
