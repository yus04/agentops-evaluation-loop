"""LangGraph グラフの構築と実行."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..observability import langchain_callbacks, start_trace
from ..pricing import estimate_cost_usd
from ..settings import AgentSettings, agent_settings
from .candidate import AgentCandidate
from .nodes import (
    AgentContext,
    classify_node,
    execute_tools_node,
    respond_node,
    route_after_validation,
    select_tools_node,
    validate_node,
)
from .state import AgentState
from .tools import ToolRuntime


@dataclass
class AgentRunResult:
    """1 リクエスト分の実行結果. そのまま評価データセットの 1 レコードになる."""

    user_input: str
    final_answer: str
    classification: dict[str, Any] = field(default_factory=dict)
    tool_plan: dict[str, Any] = field(default_factory=dict)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    latency_ms: float = 0.0
    node_timings: list[dict[str, Any]] = field(default_factory=list)
    token_usage: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    trace_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [str(execution.get("name")) for execution in self.tool_executions]

    @property
    def tool_outputs(self) -> list[str]:
        """根拠性 (faithfulness) 評価の retrieval_context として使うツール出力."""
        contexts: list[str] = []
        for execution in self.tool_executions:
            if execution.get("error"):
                contexts.append(f"{execution.get('name')} -> ERROR: {execution['error']}")
            else:
                contexts.append(f"{execution.get('name')} -> {execution.get('output')}")
        return contexts

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tools_called"] = self.tools_called
        return payload


def _build_graph(ctx: AgentContext, settings: AgentSettings):
    graph = StateGraph(AgentState)

    # lambda で包むと LangGraph が async 関数として認識できないため partial を使う
    graph.add_node("classify", partial(classify_node, ctx=ctx))
    graph.add_node("select_tools", partial(select_tools_node, ctx=ctx))
    graph.add_node("execute_tools", partial(execute_tools_node, ctx=ctx))
    graph.add_node("validate", partial(validate_node, ctx=ctx))
    graph.add_node("respond", partial(respond_node, ctx=ctx))

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "select_tools")
    graph.add_edge("select_tools", "execute_tools")
    graph.add_edge("execute_tools", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: route_after_validation(state, settings.max_validation_retries),
        {"select_tools": "select_tools", "respond": "respond"},
    )
    graph.add_edge("respond", END)
    return graph.compile()


class OrderSupportAgent:
    """注文サポートエージェント.

    `AgentCandidate` (インストラクション / スキル / ツールポリシー / モデルポリシー) を
    差し替えるだけで挙動が変わるため、GEPA による最適化・回帰テストで再利用できる。
    """

    def __init__(
        self,
        candidate: AgentCandidate | None = None,
        settings: AgentSettings | None = None,
        tools_mode: str | None = None,
        seed_candidate: AgentCandidate | None = None,
    ) -> None:
        self.settings = settings or agent_settings()
        self.seed_candidate = seed_candidate or AgentCandidate.from_repo()
        self.candidate = candidate or self.seed_candidate

        tool_policy = self.candidate.tool_policy(fallback=self.seed_candidate)
        model_policy = self.candidate.model_policy(fallback=self.seed_candidate)
        self.tool_runtime = ToolRuntime(
            policy=tool_policy,
            catalog=self.candidate.meta.get("tool_catalog") or self.seed_candidate.meta.get("tool_catalog", {}),
            mode=tools_mode or self.settings.tools_mode,
        )
        self.context = AgentContext(
            candidate=self.candidate,
            tool_runtime=self.tool_runtime,
            settings=self.settings,
            model_policy=model_policy,
            tool_policy=tool_policy,
        )
        self._graph = _build_graph(self.context, self.settings)

    async def __aenter__(self) -> OrderSupportAgent:
        await self.tool_runtime.astart()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.tool_runtime.aclose()

    async def arun(
        self,
        user_input: str,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AgentRunResult:
        session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        user_id = user_id or "anonymous"
        started = time.perf_counter()

        trace_tags = [self.settings.environment, *(tags or [])]
        with start_trace(
            self.settings.service_name,
            session_id=session_id,
            user_id=user_id,
            tags=trace_tags,
            input_payload={"user_input": user_input},
        ) as trace:
            state: AgentState = {
                "user_input": user_input,
                "session_id": session_id,
                "user_id": user_id,
                "iteration": 0,
                "tool_executions": [],
                "errors": [],
                "node_timings": [],
                "token_usage": [],
            }
            final_state = await self._graph.ainvoke(
                state,
                config={
                    "callbacks": langchain_callbacks(),
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id,
                        "langfuse_tags": trace_tags,
                    },
                    "run_name": self.settings.service_name,
                    "recursion_limit": 25,
                },
            )

            latency_ms = (time.perf_counter() - started) * 1000.0
            token_usage = list(final_state.get("token_usage") or [])
            result = AgentRunResult(
                user_input=user_input,
                final_answer=str(final_state.get("final_answer", "")),
                classification=dict(final_state.get("classification") or {}),
                tool_plan=dict(final_state.get("tool_plan") or {}),
                tool_executions=list(final_state.get("tool_executions") or []),
                validation=dict(final_state.get("validation") or {}),
                errors=[e for e in (final_state.get("errors") or []) if e],
                latency_ms=round(latency_ms, 3),
                node_timings=list(final_state.get("node_timings") or []),
                token_usage=token_usage,
                total_tokens=sum(int(u.get("total_tokens", 0) or 0) for u in token_usage),
                estimated_cost_usd=estimate_cost_usd(token_usage),
                trace_id=trace.trace_id,
                session_id=session_id,
                user_id=user_id,
            )

            # オフライン評価データセット生成に必要な業務属性をトレースへ添付する
            trace.update(output={"final_answer": result.final_answer})
            trace.update_trace(
                output={"final_answer": result.final_answer},
                metadata={
                    "agentops": {
                        "classification": result.classification,
                        "tool_plan": result.tool_plan,
                        "tool_executions": result.tool_executions,
                        "validation": result.validation,
                        "errors": result.errors,
                        "metrics": {
                            "latency_ms": result.latency_ms,
                            "total_tokens": result.total_tokens,
                            "estimated_cost_usd": result.estimated_cost_usd,
                            "tool_call_count": len(result.tool_executions),
                            "error_count": len(result.errors),
                            "node_timings": result.node_timings,
                        },
                        "candidate_keys": sorted(self.candidate.components),
                        "models": self.context.model_policy.per_node,
                        "tools_enabled": list(self.context.tool_policy.enabled),
                    }
                },
            )
            return result


def build_agent(candidate: AgentCandidate | None = None, **kwargs: Any) -> OrderSupportAgent:
    return OrderSupportAgent(candidate=candidate, **kwargs)
