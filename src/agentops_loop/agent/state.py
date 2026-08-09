"""LangGraph の状態定義と、ノード間でやり取りする構造化データ."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["order_status", "shipping", "returns", "refund", "billing", "other"]
Priority = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- LLM 構造化出力


class Entities(BaseModel):
    order_id: str | None = Field(default=None, description="注文 ID (例: ORD-1001)")
    tracking_number: str | None = Field(default=None, description="追跡番号")


class Classification(BaseModel):
    """問い合わせ分類ノードの出力."""

    category: Category
    priority: Priority = "medium"
    entities: Entities = Field(default_factory=Entities)
    summary: str = Field(default="", description="ユーザーが解決したいことの 1 文要約")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PlannedToolCall(BaseModel):
    name: str = Field(description="呼び出すツール名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="ツール引数")
    reason: str = Field(default="", description="このツールを選んだ理由")


class ToolPlan(BaseModel):
    """ツール選択ノードの出力."""

    tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    missing_arguments: list[str] = Field(default_factory=list)
    plan: str = ""


class Validation(BaseModel):
    """検証ノードの出力."""

    sufficient: bool = False
    issues: list[str] = Field(default_factory=list)
    retry_tools: bool = False
    needs_user_input: bool = False
    grounded_facts: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- 実行記録


class ToolExecution(TypedDict, total=False):
    name: str
    arguments: dict[str, Any]
    output: Any
    error: str | None
    latency_ms: float
    iteration: int


class AgentState(TypedDict, total=False):
    """LangGraph の状態.

    Langfuse へ送るトレース属性 (分類結果 / ツール選択 / ツール入出力 / 検証結果 /
    最終回答) はすべてこの state から取り出す。
    """

    user_input: str
    session_id: str
    user_id: str

    classification: dict[str, Any]
    tool_plan: dict[str, Any]
    tool_executions: Annotated[list[ToolExecution], operator.add]
    validation: dict[str, Any]
    final_answer: str

    iteration: int
    errors: Annotated[list[str], operator.add]
    #: {"node": ..., "latency_ms": ...}
    node_timings: Annotated[list[dict[str, Any]], operator.add]
    #: {"node": ..., "model": ..., "input_tokens": ..., "output_tokens": ...}
    token_usage: Annotated[list[dict[str, Any]], operator.add]
