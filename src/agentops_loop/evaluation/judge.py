"""LLM-as-a-Judge (DeepEval 用) モデル.

**エージェント用モデルとは必ず別のモデル/別の設定を使う。**
同一モデルで自己評価すると、自分の出力形式に有利なバイアスが乗り
スコアが実運用品質と乖離するため。

`DeepEvalBaseLLM` を LangChain の `init_chat_model` でラップしているので、
OpenAI 以外のプロバイダ (Azure OpenAI / Anthropic / Bedrock など) へも
環境変数だけで切り替えられる。
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import yaml
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..llm import build_chat_model
from ..paths import CONFIG_DIR
from ..settings import judge_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_evaluation_config() -> dict[str, Any]:
    path = CONFIG_DIR / "evaluation.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class LangChainJudge(DeepEvalBaseLLM):
    """LangChain チャットモデルを DeepEval の Judge として使うアダプタ."""

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._model_name = model
        self._provider = provider
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries
        super().__init__(model_name=model)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return build_chat_model(
            model=self._model_name,
            provider=self._provider,
            temperature=self._temperature,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def get_model_name(self) -> str:
        return f"{self._provider}:{self._model_name}"

    # ------------------------------------------------------------------ 生成

    def _client(self, schema: type[BaseModel] | None) -> Any:
        client = self.load_model()
        return client.with_structured_output(schema) if schema is not None else client

    @staticmethod
    def _to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        response = self._client(schema).invoke([HumanMessage(content=prompt)])
        return response if schema is not None else self._to_text(response)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        response = await self._client(schema).ainvoke([HumanMessage(content=prompt)])
        return response if schema is not None else self._to_text(response)


@lru_cache(maxsize=1)
def build_judge_model() -> LangChainJudge:
    """設定ファイル + 環境変数から Judge モデルを構築する (環境変数が優先)."""
    config = load_evaluation_config().get("judge", {}) or {}
    settings = judge_settings()

    fields_set = settings.model_fields_set
    model = settings.model if "model" in fields_set else str(config.get("model", settings.model))
    provider = (
        settings.provider if "provider" in fields_set else str(config.get("provider", settings.provider))
    )
    temperature = (
        settings.temperature
        if "temperature" in fields_set
        else float(config.get("temperature", settings.temperature))
    )

    logger.info("Judge モデル: %s:%s (temperature=%s)", provider, model, temperature)
    return LangChainJudge(
        model=model,
        provider=provider,
        temperature=temperature,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )
