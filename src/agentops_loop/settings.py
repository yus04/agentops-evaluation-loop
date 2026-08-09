"""環境変数ベースの設定.

エージェント用モデルと評価 (LLM-as-a-Judge) 用モデルは必ず分離する。
評価者がエージェントと同一モデル/同一プロンプトだと自己評価バイアスが乗るため。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """本番エージェント (コンテナ) の設定."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    # 実際に使うモデルは config/agent.yaml の model.policy (GEPA 最適化対象) で決まる。
    # ここではプロバイダ接続とガードレールのみを持つ。
    provider: str = Field(default="openai", description="langchain init_chat_model の provider")
    request_timeout_s: float = 60.0
    max_retries: int = 2

    # ツール実行モード。CI や単体テストでは "local" を使い MCP プロセス起動を省略する。
    tools_mode: Literal["mcp", "local"] = "mcp"

    # 検証ノードでのリトライ上限 (無限ループ防止)
    max_validation_retries: int = 2
    # 1 リクエストあたりのツール呼び出し上限 (コスト暴走防止)
    max_tool_calls: int = 4

    # HTTP サービスの API キー。設定した場合 X-API-Key ヘッダーが必須になる。
    api_key: str | None = None

    environment: str = "production"
    service_name: str = "order-support-agent"


class JudgeSettings(BaseSettings):
    """DeepEval (LLM-as-a-Judge) 用の設定. エージェントとは別モデルを使う."""

    model_config = SettingsConfigDict(
        env_prefix="JUDGE_", env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    request_timeout_s: float = 120.0
    max_retries: int = 3
    # 並列評価数
    max_concurrency: int = 4


class ReflectionSettings(BaseSettings):
    """GEPA の reflection LM (改善候補を書く側) の設定."""

    model_config = SettingsConfigDict(
        env_prefix="REFLECTION_", env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 1.0
    request_timeout_s: float = 180.0


class LangfuseSettings(BaseSettings):
    """Langfuse 接続設定 (OpenTelemetry のバックエンドとして使う)."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_", env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    public_key: str | None = None
    secret_key: str | None = None
    # v4 は LANGFUSE_BASE_URL, v3 は LANGFUSE_HOST。両方サポートする。
    base_url: str = "https://cloud.langfuse.com"
    host: str | None = None
    # トレースに付与する環境タグ (production / staging / eval など)
    tracing_environment: str = "production"
    enabled: bool = True

    @property
    def resolved_base_url(self) -> str:
        return self.host or self.base_url

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.public_key and self.secret_key)


@lru_cache(maxsize=1)
def agent_settings() -> AgentSettings:
    return AgentSettings()


@lru_cache(maxsize=1)
def judge_settings() -> JudgeSettings:
    return JudgeSettings()


@lru_cache(maxsize=1)
def reflection_settings() -> ReflectionSettings:
    return ReflectionSettings()


@lru_cache(maxsize=1)
def langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()
