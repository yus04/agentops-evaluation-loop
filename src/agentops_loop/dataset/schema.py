"""評価用データセットのスキーマ (JSONL 1 行 = 1 レコード).

このファイルが「人間レビューの対象」になる。GitHub Actions が候補を生成し、
Pull Request でレビュー → 修正 (expected_output / expected_tools の付与) → マージ、
という流れで `data/datasets/golden.jsonl` が育っていく。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ReviewStatus = Literal["pending", "approved", "rejected"]


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error: str | None = None
    latency_ms: float | None = None


class RunMetrics(BaseModel):
    latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tool_call_count: int = 0
    error_count: int = 0


class ReviewMeta(BaseModel):
    """人間によるキュレーション結果."""

    status: ReviewStatus = "pending"
    reviewer: str | None = None
    notes: str = ""
    #: 候補として抽出された理由 (レビュー効率のためのヒント)
    sampling_reason: str = ""


class EvalRecord(BaseModel):
    """1 件の評価対象インタラクション."""

    id: str
    source: Literal["trace", "manual", "synthetic"] = "trace"
    trace_id: str | None = None
    timestamp: str | None = None

    #: DeepEval LLMTestCase.input
    input: str
    #: DeepEval LLMTestCase.actual_output (本番トレース時点の回答)
    actual_output: str = ""
    #: DeepEval LLMTestCase.expected_output (人間が付与する理想回答。任意)
    expected_output: str | None = None
    #: DeepEval LLMTestCase.expected_tools (人間が付与する期待ツール列)
    expected_tools: list[str] = Field(default_factory=list)

    tools_called: list[ToolCallRecord] = Field(default_factory=list)
    #: DeepEval LLMTestCase.retrieval_context (ツール出力 = 回答の根拠)
    retrieval_context: list[str] = Field(default_factory=list)

    classification: dict[str, Any] = Field(default_factory=dict)
    expected_category: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)

    tags: list[str] = Field(default_factory=list)
    review: ReviewMeta = Field(default_factory=ReviewMeta)

    @property
    def dedup_key(self) -> str:
        normalized = " ".join(self.input.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def is_usable(self) -> bool:
        return bool(self.input.strip()) and self.review.status != "rejected"


def write_jsonl(records: Iterable[EvalRecord], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[EvalRecord]:
    path = Path(path)
    if not path.is_file():
        return []
    records: list[EvalRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(EvalRecord.model_validate(json.loads(stripped)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{line_no} のパースに失敗しました: {exc}") from exc
    return records
