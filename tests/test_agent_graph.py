"""LangGraph グラフのエンドツーエンド検証 (LLM はスタブに差し替える)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentops_loop.agent import nodes as nodes_module
from agentops_loop.agent.graph import OrderSupportAgent
from agentops_loop.agent.state import Classification, PlannedToolCall, ToolPlan, Validation


@dataclass
class FakeAIMessage:
    content: str
    usage_metadata: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )


class FakeStructuredRunnable:
    def __init__(self, schema: type, script: dict[type, list[Any]]) -> None:
        self._schema = schema
        self._script = script

    async def ainvoke(self, _messages: Any, **_: Any) -> dict[str, Any]:
        queue = self._script[self._schema]
        parsed = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"parsed": parsed, "raw": FakeAIMessage(content="{}"), "parsing_error": None}


class FakeChatModel:
    """`build_chat_model` の戻り値を差し替えるスタブ."""

    def __init__(self, script: dict[type, list[Any]], answer: str) -> None:
        self._script = script
        self._answer = answer

    def with_structured_output(self, schema: type, **kwargs: Any) -> FakeStructuredRunnable:
        assert kwargs == {"include_raw": True, "method": "function_calling"}
        return FakeStructuredRunnable(schema, self._script)

    async def ainvoke(self, _messages: Any, **_: Any) -> FakeAIMessage:
        return FakeAIMessage(content=self._answer)


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    def install(script: dict[type, list[Any]], answer: str = "ご案内します。") -> None:
        model = FakeChatModel(script, answer)
        monkeypatch.setattr(nodes_module, "build_chat_model", lambda **_: model)

    return install


async def test_happy_path_calls_tool_and_answers(stub_llm) -> None:
    stub_llm(
        {
            Classification: [
                Classification(
                    category="order_status",
                    priority="low",
                    summary="注文内容の確認",
                    confidence=0.95,
                )
            ],
            ToolPlan: [
                ToolPlan(
                    tool_calls=[
                        PlannedToolCall(
                            name="lookup_order",
                            arguments={"order_id": "ORD-1001"},
                            reason="注文内容の取得",
                        )
                    ],
                    plan="注文を照会する",
                )
            ],
            Validation: [Validation(sufficient=True, grounded_facts=["合計 12,800 円"])],
        },
        answer="ORD-1001 の合計金額は 12,800 円です。",
    )

    async with OrderSupportAgent(tools_mode="local") as agent:
        result = await agent.arun("ORD-1001 の内容を教えてください")

    assert result.classification["category"] == "order_status"
    assert result.tools_called == ["lookup_order"]
    assert result.tool_executions[0]["output"]["total_jpy"] == 12800
    assert result.validation["sufficient"] is True
    assert "12,800" in result.final_answer
    assert result.errors == []
    assert result.total_tokens > 0
    assert result.estimated_cost_usd > 0


async def test_validation_failure_triggers_retry_then_responds(stub_llm) -> None:
    stub_llm(
        {
            Classification: [Classification(category="shipping", summary="配送照会")],
            ToolPlan: [
                ToolPlan(
                    tool_calls=[
                        PlannedToolCall(name="track_shipment", arguments={"order_id": "ORD-1002"})
                    ]
                ),
                ToolPlan(
                    tool_calls=[
                        PlannedToolCall(name="lookup_order", arguments={"order_id": "ORD-1002"})
                    ]
                ),
            ],
            # 1 回目は不十分 -> 再計画、2 回目は十分
            Validation: [
                Validation(sufficient=False, retry_tools=True, issues=["出荷状況が不明"]),
                Validation(sufficient=True),
            ],
        }
    )

    async with OrderSupportAgent(tools_mode="local") as agent:
        result = await agent.arun("ORD-1002 はどこにありますか")

    assert result.tools_called == ["track_shipment", "lookup_order"]


async def test_tool_call_budget_is_enforced(stub_llm) -> None:
    many_calls = [
        PlannedToolCall(name=name, arguments={"order_id": "ORD-1001"})
        for name in (
            "lookup_order",
            "track_shipment",
            "check_return_eligibility",
            "get_refund_status",
        )
    ] + [PlannedToolCall(name="search_policy", arguments={"query": "返品"})]

    stub_llm(
        {
            Classification: [Classification(category="returns", summary="返品相談")],
            ToolPlan: [ToolPlan(tool_calls=many_calls)],
            Validation: [Validation(sufficient=True)],
        }
    )

    async with OrderSupportAgent(tools_mode="local") as agent:
        result = await agent.arun("ORD-1001 を返品したい")

    # config/agent.yaml の max_tool_calls = 4 でクランプされる
    assert len(result.tool_executions) == 4
    assert any("tool_budget_exceeded" in error for error in result.errors)


async def test_disabled_tool_is_filtered_out(stub_llm) -> None:
    from agentops_loop.agent.candidate import AgentCandidate

    seed = AgentCandidate.from_repo()
    restricted = seed.with_components(
        {"policy/tools": "enabled:\n  - lookup_order\nmax_tool_calls: 2\n"}
    )

    stub_llm(
        {
            Classification: [Classification(category="billing", summary="請求確認")],
            ToolPlan: [
                ToolPlan(
                    tool_calls=[
                        PlannedToolCall(name="escalate_to_human", arguments={"reason": "調査"}),
                        PlannedToolCall(name="lookup_order", arguments={"order_id": "ORD-1001"}),
                    ]
                )
            ],
            Validation: [Validation(sufficient=True)],
        }
    )

    async with OrderSupportAgent(candidate=restricted, tools_mode="local") as agent:
        result = await agent.arun("請求内容を確認してください ORD-1001")

    assert result.tools_called == ["lookup_order"]
    assert any("tool_not_available" in error for error in result.errors)
