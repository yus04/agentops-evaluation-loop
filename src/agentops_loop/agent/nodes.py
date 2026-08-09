"""LangGraph の各ノード実装.

    ユーザー入力
        ↓
    classify        問い合わせ分類
        ↓
    select_tools    必要な MCP tool を選択  ←──┐
        ↓                                      │ 不足があれば再計画
    execute_tools   tool 実行                  │
        ↓                                      │
    validate        tool 結果を検証  ───────────┘
        ↓
    respond         回答生成
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..llm import build_chat_model
from ..settings import AgentSettings
from .candidate import AgentCandidate, ModelPolicy, ToolPolicy
from .state import AgentState, Classification, ToolExecution, ToolPlan, Validation
from .tools import ToolRuntime, timed_call

logger = logging.getLogger(__name__)

#: カテゴリごとに参照するスキル (skills/*.md)。スキル自体が GEPA 最適化対象。
CATEGORY_SKILLS: dict[str, tuple[str, ...]] = {
    "order_status": ("order_status",),
    "shipping": ("shipping",),
    "returns": ("returns", "refund"),
    "refund": ("refund", "returns"),
    "billing": ("billing",),
    "other": ("order_status",),
}
#: 常に読み込むスキル
ALWAYS_SKILLS: tuple[str, ...] = ("escalation",)


@dataclass
class AgentContext:
    """ノードが共有する依存関係."""

    candidate: AgentCandidate
    tool_runtime: ToolRuntime
    settings: AgentSettings
    model_policy: ModelPolicy
    tool_policy: ToolPolicy
    _models: dict[str, BaseChatModel] = field(default_factory=dict, repr=False)

    def chat_model(self, node: str) -> BaseChatModel:
        if node not in self._models:
            self._models[node] = build_chat_model(
                model=self.model_policy.for_node(node),
                provider=self.settings.provider,
                temperature=self.model_policy.temperature,
                timeout=self.settings.request_timeout_s,
                max_retries=self.settings.max_retries,
            )
        return self._models[node]

    def skills_for(self, category: str) -> str:
        names = list(CATEGORY_SKILLS.get(category, CATEGORY_SKILLS["other"]))
        names.extend(name for name in ALWAYS_SKILLS if name not in names)
        available = set(self.candidate.skill_names)
        return self.candidate.skills_digest([n for n in names if n in available])

    @property
    def effective_max_tool_calls(self) -> int:
        return max(1, min(self.tool_policy.max_tool_calls, self.settings.max_tool_calls))


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _usage_entry(node: str, model: str, raw: Any) -> dict[str, Any]:
    usage = getattr(raw, "usage_metadata", None) or {}
    return {
        "node": node,
        "model": model,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


async def _structured_call(
    ctx: AgentContext,
    node: str,
    schema: type[BaseModel],
    system: str,
    user_payload: Any,
) -> tuple[BaseModel | None, dict[str, Any], str | None]:
    """構造化出力の LLM 呼び出し. ``(パース結果, usage, エラー)`` を返す."""
    model = ctx.chat_model(node)
    model_name = ctx.model_policy.for_node(node)
    messages = [SystemMessage(content=system), HumanMessage(content=_dumps(user_payload))]
    try:
        result = await model.with_structured_output(schema, include_raw=True).ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 - LLM 障害はトレース/評価対象として記録する
        logger.warning("ノード %s の LLM 呼び出しに失敗しました", node, exc_info=True)
        return None, _usage_entry(node, model_name, None), f"{node}_llm_error: {exc}"

    if isinstance(result, dict):
        parsed = result.get("parsed")
        raw = result.get("raw")
        parsing_error = result.get("parsing_error")
    else:  # include_raw 非対応の実装向けフォールバック
        parsed, raw, parsing_error = result, None, None

    usage = _usage_entry(node, model_name, raw)
    if parsed is None:
        return None, usage, f"{node}_parse_error: {parsing_error}"
    return parsed, usage, None


# --------------------------------------------------------------------------- ノード


async def classify_node(state: AgentState, ctx: AgentContext) -> dict[str, Any]:
    started = time.perf_counter()
    parsed, usage, error = await _structured_call(
        ctx,
        "classify",
        Classification,
        ctx.candidate.instruction("classify"),
        {"user_input": state["user_input"]},
    )
    classification = (
        parsed
        if isinstance(parsed, Classification)
        else Classification(category="other", priority="medium", summary=state["user_input"][:200])
    )
    return {
        "classification": classification.model_dump(),
        "token_usage": [usage],
        "node_timings": [
            {"node": "classify", "latency_ms": (time.perf_counter() - started) * 1000.0}
        ],
        "errors": [error] if error else [],
    }


async def select_tools_node(state: AgentState, ctx: AgentContext) -> dict[str, Any]:
    started = time.perf_counter()
    classification = state.get("classification") or {}
    category = str(classification.get("category", "other"))
    executions = state.get("tool_executions") or []
    validation = state.get("validation") or {}

    system = "\n\n".join(
        [
            ctx.candidate.instruction("select_tools"),
            "## 利用可能なツール\n" + ctx.tool_runtime.render_catalog(),
            f"## 1 リクエストあたりのツール呼び出し上限\n{ctx.effective_max_tool_calls} 回",
            "## 参照スキル\n" + ctx.skills_for(category),
        ]
    )
    payload = {
        "user_input": state["user_input"],
        "classification": classification,
        "already_called": [
            {"name": e.get("name"), "arguments": e.get("arguments"), "error": e.get("error")}
            for e in executions
        ],
        "validation_issues": validation.get("issues", []),
        "remaining_tool_call_budget": max(0, ctx.effective_max_tool_calls - len(executions)),
    }

    parsed, usage, error = await _structured_call(ctx, "select_tools", ToolPlan, system, payload)
    plan = parsed if isinstance(parsed, ToolPlan) else ToolPlan()

    # ポリシーで無効化されたツール・カタログ外のツールを除外する
    filtered = [call for call in plan.tool_calls if ctx.tool_runtime.is_enabled(call.name)]
    dropped = [call.name for call in plan.tool_calls if call not in filtered]
    plan = ToolPlan(
        tool_calls=filtered, missing_arguments=plan.missing_arguments, plan=plan.plan
    )

    errors = [error] if error else []
    if dropped:
        errors.append(f"tool_not_available: {', '.join(sorted(set(dropped)))}")

    return {
        "tool_plan": plan.model_dump(),
        "iteration": int(state.get("iteration", 0)) + 1,
        "token_usage": [usage],
        "node_timings": [
            {"node": "select_tools", "latency_ms": (time.perf_counter() - started) * 1000.0}
        ],
        "errors": errors,
    }


async def execute_tools_node(state: AgentState, ctx: AgentContext) -> dict[str, Any]:
    started = time.perf_counter()
    plan = ToolPlan.model_validate(state.get("tool_plan") or {})
    already = state.get("tool_executions") or []
    budget = max(0, ctx.effective_max_tool_calls - len(already))
    iteration = int(state.get("iteration", 1))

    seen = {(e.get("name"), _dumps(e.get("arguments"))) for e in already}
    executions: list[ToolExecution] = []
    errors: list[str] = []

    for call in plan.tool_calls[:budget]:
        arguments = ctx.tool_runtime.sanitize_arguments(call.name, call.arguments)
        key = (call.name, _dumps(arguments))
        if key in seen:
            continue
        seen.add(key)

        output, error, latency_ms = await timed_call(ctx.tool_runtime, call.name, arguments)
        executions.append(
            ToolExecution(
                name=call.name,
                arguments=arguments,
                output=output,
                error=error,
                latency_ms=latency_ms,
                iteration=iteration,
            )
        )
        if error:
            errors.append(f"{call.name}: {error}")

    if len(plan.tool_calls) > budget:
        errors.append(
            f"tool_budget_exceeded: 計画 {len(plan.tool_calls)} 件のうち {budget} 件のみ実行しました"
        )

    return {
        "tool_executions": executions,
        "node_timings": [
            {"node": "execute_tools", "latency_ms": (time.perf_counter() - started) * 1000.0}
        ],
        "errors": errors,
    }


async def validate_node(state: AgentState, ctx: AgentContext) -> dict[str, Any]:
    started = time.perf_counter()
    executions = state.get("tool_executions") or []
    plan = state.get("tool_plan") or {}

    payload = {
        "user_input": state["user_input"],
        "classification": state.get("classification"),
        "missing_arguments": plan.get("missing_arguments", []),
        "tool_results": [
            {
                "name": e.get("name"),
                "arguments": e.get("arguments"),
                "output": e.get("output"),
                "error": e.get("error"),
            }
            for e in executions
        ],
    }
    parsed, usage, error = await _structured_call(
        ctx, "validate", Validation, ctx.candidate.instruction("validate"), payload
    )
    validation = (
        parsed
        if isinstance(parsed, Validation)
        else Validation(sufficient=bool(executions), issues=["validation_unavailable"])
    )
    return {
        "validation": validation.model_dump(),
        "node_timings": [
            {"node": "validate", "latency_ms": (time.perf_counter() - started) * 1000.0}
        ],
        "token_usage": [usage],
        "errors": [error] if error else [],
    }


async def respond_node(state: AgentState, ctx: AgentContext) -> dict[str, Any]:
    started = time.perf_counter()
    classification = state.get("classification") or {}
    validation = state.get("validation") or {}
    plan = state.get("tool_plan") or {}
    executions = state.get("tool_executions") or []

    system = "\n\n".join(
        [
            ctx.candidate.instruction("respond"),
            "## 参照スキル\n" + ctx.skills_for(str(classification.get("category", "other"))),
        ]
    )
    payload = {
        "user_input": state["user_input"],
        "classification": classification,
        "grounded_facts": validation.get("grounded_facts", []),
        "open_issues": validation.get("issues", []),
        "needs_user_input": validation.get("needs_user_input", False),
        "missing_arguments": plan.get("missing_arguments", []),
        "tool_results": [
            {"name": e.get("name"), "output": e.get("output"), "error": e.get("error")}
            for e in executions
        ],
    }

    model = ctx.chat_model("respond")
    model_name = ctx.model_policy.for_node("respond")
    messages = [SystemMessage(content=system), HumanMessage(content=_dumps(payload))]
    errors: list[str] = []
    try:
        response = await model.ainvoke(messages)
        answer = response.content if isinstance(response.content, str) else _dumps(response.content)
        usage = _usage_entry("respond", model_name, response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("回答生成に失敗しました", exc_info=True)
        answer = (
            "申し訳ございません。ただいまシステムの不具合により回答を生成できませんでした。"
            "お手数ですが時間をおいて再度お問い合わせください。"
        )
        usage = _usage_entry("respond", model_name, None)
        errors.append(f"respond_llm_error: {exc}")

    return {
        "final_answer": answer.strip(),
        "token_usage": [usage],
        "node_timings": [
            {"node": "respond", "latency_ms": (time.perf_counter() - started) * 1000.0}
        ],
        "errors": errors,
    }


# --------------------------------------------------------------------------- ルーティング


def route_after_validation(state: AgentState, max_retries: int) -> str:
    validation = state.get("validation") or {}
    iteration = int(state.get("iteration", 1))

    if validation.get("sufficient"):
        return "respond"
    if validation.get("needs_user_input"):
        return "respond"
    if validation.get("retry_tools") and iteration <= max_retries:
        return "select_tools"
    return "respond"
