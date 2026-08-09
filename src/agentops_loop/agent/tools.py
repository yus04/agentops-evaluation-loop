"""ツール実行ランタイム.

2 つのモードを持つ。

- ``mcp``   : MCP サーバー (stdio) を起動し、langchain-mcp-adapters 経由でツールを取得する。本番既定。
- ``local`` : MCP を介さず同一プロセスの実装を直接呼ぶ。CI / 単体テスト / GEPA の高速ループ用。

どちらのモードでも、有効なツールとその説明文は `ToolPolicy` (GEPA 最適化対象) が決める。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from ..mcp_server import backend
from .candidate import ToolPolicy

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_S = float(os.environ.get("AGENT_TOOL_TIMEOUT_S", "20"))


def _mcp_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name != "PS1"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: tuple[str, ...]

    def render(self) -> str:
        args = ", ".join(self.args) if self.args else "(なし)"
        return f"- {self.name}(引数: {args}): {self.description}"


class ToolRuntime:
    """有効化されたツールの一覧提供と実行を担う."""

    def __init__(
        self,
        policy: ToolPolicy,
        catalog: dict[str, Any],
        mode: str = "local",
    ) -> None:
        self._policy = policy
        self._catalog = catalog or {}
        self._mode = mode if mode in {"mcp", "local"} else "local"
        self._mcp_tools: dict[str, Any] | None = None
        self._mcp_client: Any | None = None

    # ------------------------------------------------------------------ ライフサイクル

    async def astart(self) -> None:
        if self._mode != "mcp" or self._mcp_tools is not None:
            return
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._mcp_client = MultiServerMCPClient(
                {
                    "order_support": {
                        "command": sys.executable,
                        "args": ["-m", "agentops_loop.mcp_server.order_support"],
                        "transport": "stdio",
                        "env": _mcp_environment(),
                    }
                }
            )
            tools = await self._mcp_client.get_tools()
            self._mcp_tools = {tool.name: tool for tool in tools}
            logger.info("MCP ツールを %d 件ロードしました", len(self._mcp_tools))
        except Exception:  # noqa: BLE001 - MCP 起動失敗時はローカル実装へ縮退する
            logger.exception("MCP サーバーの起動に失敗したため local モードへフォールバックします")
            self._mode = "local"
            self._mcp_tools = None

    async def aclose(self) -> None:
        client = self._mcp_client
        self._mcp_client = None
        self._mcp_tools = None
        if client is None:
            return
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001
            logger.debug("MCP クライアントのクローズに失敗しました", exc_info=True)

    # ------------------------------------------------------------------ カタログ

    @property
    def mode(self) -> str:
        return self._mode

    def specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name in self._policy.enabled:
            entry = self._catalog.get(name, {})
            default_description = str(entry.get("default_description", ""))
            specs.append(
                ToolSpec(
                    name=name,
                    description=self._policy.description_for(name, default_description),
                    args=tuple(entry.get("args", []) or []),
                )
            )
        return specs

    def render_catalog(self) -> str:
        specs = self.specs()
        if not specs:
            return "(利用可能なツールはありません)"
        return "\n".join(spec.render() for spec in specs)

    def is_enabled(self, name: str) -> bool:
        return name in self._policy.enabled

    def sanitize_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """カタログに定義された引数だけを通す (未知キーの注入を防ぐ)."""
        allowed = set(self._catalog.get(name, {}).get("args", []) or [])
        if not allowed:
            return {}
        if not isinstance(arguments, dict):
            return {}
        return {key: value for key, value in arguments.items() if key in allowed}

    # ------------------------------------------------------------------ 実行

    async def acall(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str | None]:
        """ツールを実行し ``(出力, エラー文字列)`` を返す."""
        if not self.is_enabled(name):
            return None, f"tool_disabled: {name} は現在の設定では利用できません"

        args = self.sanitize_arguments(name, arguments)
        try:
            if self._mode == "mcp" and self._mcp_tools and name in self._mcp_tools:
                raw = await asyncio.wait_for(
                    self._mcp_tools[name].ainvoke(args), timeout=TOOL_TIMEOUT_S
                )
                return _coerce_output(raw), None

            impl = backend.TOOL_IMPLEMENTATIONS.get(name)
            if impl is None:
                return None, f"tool_not_found: {name}"
            return await asyncio.to_thread(impl, **args), None
        except TimeoutError:
            return None, f"tool_timeout: {name} が {TOOL_TIMEOUT_S} 秒以内に応答しませんでした"
        except TypeError as exc:
            return None, f"invalid_arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - ツール失敗は評価対象なので握って記録する
            logger.warning("ツール実行に失敗しました: %s", name, exc_info=True)
            return None, f"tool_error: {type(exc).__name__}: {exc}"


def _coerce_output(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


async def timed_call(
    runtime: ToolRuntime, name: str, arguments: dict[str, Any]
) -> tuple[Any, str | None, float]:
    started = time.perf_counter()
    output, error = await runtime.acall(name, arguments)
    return output, error, (time.perf_counter() - started) * 1000.0
