"""評価バッチの実行 (DeepEval).

- `evaluate_records`: データセット (EvalRecord) をそのまま採点する (本番トレースの事後評価)
- `refresh_records` : 現在のエージェント候補で再実行してから採点する (最適化 / 回帰テスト)
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall

from ..agent.candidate import AgentCandidate
from ..agent.graph import OrderSupportAgent
from ..dataset.schema import EvalRecord, RunMetrics, ToolCallRecord
from ..observability.tracing import get_tracing_client
from .judge import build_judge_model, load_evaluation_config
from .metrics import MetricSpec, build_metric_specs

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    key: str
    label: str
    category: str
    score: float | None
    reason: str
    threshold: float
    weight: float
    success: bool
    skipped: bool = False
    error: str | None = None


@dataclass
class RecordEvaluation:
    record_id: str
    trace_id: str | None
    input: str
    actual_output: str
    tools_called: list[str]
    metrics: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False

    def failure_reasons(self) -> list[str]:
        return [
            f"[{m.label}] score={m.score} (threshold {m.threshold}): {m.reason}"
            for m in self.metrics
            if not m.skipped and not m.success
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_reasons"] = self.failure_reasons()
        return payload


@dataclass
class EvaluationReport:
    label: str
    created_at: str
    dataset_size: int
    evaluations: list[RecordEvaluation] = field(default_factory=list)
    metric_averages: dict[str, float] = field(default_factory=dict)
    category_averages: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    overall_score: float = 0.0
    run_metrics: dict[str, float] = field(default_factory=dict)
    candidate_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "created_at": self.created_at,
            "dataset_size": self.dataset_size,
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
            "metric_averages": self.metric_averages,
            "category_averages": self.category_averages,
            "run_metrics": self.run_metrics,
            "candidate_keys": self.candidate_keys,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def render_markdown(self) -> str:
        lines = [
            f"# 評価レポート: {self.label}",
            "",
            f"- 実行日時 (UTC): {self.created_at}",
            f"- データセット件数: **{self.dataset_size}**",
            f"- 総合スコア (加重平均): **{self.overall_score:.3f}**",
            f"- 合格率: **{self.pass_rate:.1%}**",
            "",
            "## 観点別スコア",
            "",
            "| 観点 | 平均スコア |",
            "| --- | --- |",
        ]
        for category, value in sorted(self.category_averages.items()):
            lines.append(f"| {category} | {value:.3f} |")

        lines += ["", "## メトリクス別スコア", "", "| メトリクス | 平均スコア |", "| --- | --- |"]
        for key, value in sorted(self.metric_averages.items()):
            lines.append(f"| {key} | {value:.3f} |")

        if self.run_metrics:
            lines += ["", "## 実行メトリクス", "", "| 指標 | 値 |", "| --- | --- |"]
            for key, value in sorted(self.run_metrics.items()):
                lines.append(f"| {key} | {value:.4f} |")

        failures = [e for e in self.evaluations if not e.passed]
        lines += ["", f"## 不合格レコード ({len(failures)} 件)", ""]
        for evaluation in failures[:20]:
            snippet = " ".join(evaluation.input.split())[:80]
            lines.append(f"- **{evaluation.record_id}** ({snippet})")
            for reason in evaluation.failure_reasons()[:4]:
                lines.append(f"  - {reason}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- テストケース


def build_test_case(record: EvalRecord) -> LLMTestCase:
    tools_called = [
        ToolCall(
            name=tool.name,
            input_parameters=tool.arguments or {},
            output=tool.error or tool.output,
        )
        for tool in record.tools_called
    ]
    expected_tools = [ToolCall(name=name) for name in record.expected_tools]

    kwargs: dict[str, Any] = {
        "input": record.input,
        "actual_output": record.actual_output or "",
        "retrieval_context": record.retrieval_context or None,
        "tools_called": tools_called,
        "expected_tools": expected_tools or None,
    }
    if record.expected_output:
        kwargs["expected_output"] = record.expected_output

    metadata = {
        "run_metrics": record.metrics.model_dump(),
        "record_id": record.id,
        "trace_id": record.trace_id,
    }
    try:
        return LLMTestCase(**kwargs, additional_metadata=metadata)
    except TypeError:  # pragma: no cover - 古い DeepEval 互換
        test_case = LLMTestCase(**kwargs)
        test_case.additional_metadata = metadata
        return test_case


def _has_required(record: EvalRecord, requires: tuple[str, ...]) -> bool:
    for field_name in requires:
        value = getattr(record, field_name, None)
        if not value:
            return False
    return True


def _evaluate_one(record: EvalRecord, specs: list[MetricSpec]) -> RecordEvaluation:
    test_case = build_test_case(record)
    results: list[MetricResult] = []

    for spec in specs:
        if not _has_required(record, spec.requires):
            results.append(
                MetricResult(
                    key=spec.key,
                    label=spec.label,
                    category=spec.category,
                    score=None,
                    reason=f"必要なフィールドが未設定のためスキップ: {', '.join(spec.requires)}",
                    threshold=spec.threshold,
                    weight=spec.weight,
                    success=True,
                    skipped=True,
                )
            )
            continue

        try:
            spec.metric.measure(test_case)
            score = float(getattr(spec.metric, "score", 0.0) or 0.0)
            reason = str(getattr(spec.metric, "reason", "") or "")
            success = bool(getattr(spec.metric, "success", score >= spec.threshold))
            results.append(
                MetricResult(
                    key=spec.key,
                    label=spec.label,
                    category=spec.category,
                    score=round(score, 4),
                    reason=reason,
                    threshold=spec.threshold,
                    weight=spec.weight,
                    success=success,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 1 メトリクスの失敗で全体を落とさない
            logger.warning("メトリクス %s の実行に失敗しました", spec.key, exc_info=True)
            results.append(
                MetricResult(
                    key=spec.key,
                    label=spec.label,
                    category=spec.category,
                    score=0.0,
                    reason=f"評価に失敗しました: {exc}",
                    threshold=spec.threshold,
                    weight=spec.weight,
                    success=False,
                    error=str(exc),
                )
            )

    scored = [r for r in results if r.score is not None and not r.skipped]
    total_weight = sum(r.weight for r in scored) or 1.0
    overall = sum((r.score or 0.0) * r.weight for r in scored) / total_weight

    return RecordEvaluation(
        record_id=record.id,
        trace_id=record.trace_id,
        input=record.input,
        actual_output=record.actual_output,
        tools_called=[tool.name for tool in record.tools_called],
        metrics=results,
        overall_score=round(overall, 4),
        passed=all(r.success for r in results),
    )


def _push_scores_to_langfuse(evaluations: list[RecordEvaluation]) -> None:
    client = get_tracing_client()
    if client is None:
        return
    for evaluation in evaluations:
        if not evaluation.trace_id:
            continue
        for metric in evaluation.metrics:
            if metric.score is None or metric.skipped:
                continue
            try:
                client.create_score(
                    trace_id=evaluation.trace_id,
                    name=f"deepeval::{metric.key}",
                    value=metric.score,
                    data_type="NUMERIC",
                    comment=metric.reason[:1000],
                )
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse へのスコア書き戻しに失敗しました", exc_info=True)
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.debug("flush に失敗しました", exc_info=True)


def evaluate_records(
    records: list[EvalRecord],
    label: str = "evaluation",
    specs: list[MetricSpec] | None = None,
    push_scores: bool | None = None,
    max_concurrency: int | None = None,
    candidate_keys: list[str] | None = None,
) -> EvaluationReport:
    """データセットを DeepEval で採点し、スコアと理由を含むレポートを返す."""
    usable = [record for record in records if record.is_usable()]
    if not usable:
        raise ValueError("評価対象のレコードがありません")
    missing_outputs = [record.id for record in usable if not record.actual_output.strip()]
    if missing_outputs:
        examples = ", ".join(missing_outputs[:5])
        suffix = " ..." if len(missing_outputs) > 5 else ""
        raise ValueError(
            f"actual_output が未設定の評価レコードが {len(missing_outputs)} 件あります "
            f"({examples}{suffix})。現在のエージェントで実行してから評価するには "
            "--rerun を指定してください。"
        )

    config = load_evaluation_config()
    run_config = config.get("run", {}) or {}
    specs = specs if specs is not None else build_metric_specs(config, build_judge_model())
    workers = max_concurrency or int(run_config.get("max_concurrency", 4))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        evaluations = list(pool.map(lambda r: _evaluate_one(r, specs), usable))

    metric_scores: dict[str, list[float]] = {}
    category_scores: dict[str, list[float]] = {}
    for evaluation in evaluations:
        for metric in evaluation.metrics:
            if metric.score is None or metric.skipped:
                continue
            metric_scores.setdefault(metric.key, []).append(metric.score)
            category_scores.setdefault(metric.category, []).append(metric.score)

    report = EvaluationReport(
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset_size=len(usable),
        evaluations=evaluations,
        metric_averages={k: round(statistics.fmean(v), 4) for k, v in metric_scores.items()},
        category_averages={k: round(statistics.fmean(v), 4) for k, v in category_scores.items()},
        pass_rate=round(sum(1 for e in evaluations if e.passed) / len(evaluations), 4),
        overall_score=round(statistics.fmean([e.overall_score for e in evaluations]), 4),
        run_metrics={
            "avg_latency_ms": round(
                statistics.fmean([r.metrics.latency_ms for r in usable] or [0.0]), 2
            ),
            "avg_total_tokens": round(
                statistics.fmean([float(r.metrics.total_tokens) for r in usable] or [0.0]), 2
            ),
            "avg_cost_usd": round(
                statistics.fmean([r.metrics.estimated_cost_usd for r in usable] or [0.0]), 6
            ),
            "avg_tool_calls": round(
                statistics.fmean([float(r.metrics.tool_call_count) for r in usable] or [0.0]), 2
            ),
            "error_rate": round(
                sum(1 for r in usable if r.errors) / len(usable), 4
            ),
        },
        candidate_keys=candidate_keys or [],
    )

    should_push = (
        push_scores if push_scores is not None else bool(run_config.get("push_scores_to_langfuse", False))
    )
    if should_push:
        _push_scores_to_langfuse(evaluations)

    return report


# --------------------------------------------------------------------------- 再実行


async def _arefresh(
    records: list[EvalRecord],
    candidate: AgentCandidate,
    tools_mode: str | None,
    concurrency: int,
) -> list[EvalRecord]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with OrderSupportAgent(candidate=candidate, tools_mode=tools_mode) as agent:

        async def run_one(record: EvalRecord) -> EvalRecord:
            async with semaphore:
                result = await agent.arun(record.input, tags=["offline-eval"])
            refreshed = record.model_copy(deep=True)
            refreshed.actual_output = result.final_answer
            refreshed.tools_called = [
                ToolCallRecord(
                    name=str(execution.get("name", "")),
                    arguments=execution.get("arguments") or {},
                    output=execution.get("output"),
                    error=execution.get("error"),
                    latency_ms=execution.get("latency_ms"),
                )
                for execution in result.tool_executions
            ]
            refreshed.retrieval_context = result.tool_outputs
            refreshed.classification = result.classification
            refreshed.validation = result.validation
            refreshed.errors = result.errors
            refreshed.trace_id = result.trace_id
            refreshed.metrics = RunMetrics(
                latency_ms=result.latency_ms,
                total_tokens=result.total_tokens,
                estimated_cost_usd=result.estimated_cost_usd,
                tool_call_count=len(result.tool_executions),
                error_count=len(result.errors),
            )
            return refreshed

        return list(await asyncio.gather(*(run_one(record) for record in records)))


def refresh_records(
    records: list[EvalRecord],
    candidate: AgentCandidate,
    tools_mode: str | None = None,
    concurrency: int = 4,
) -> list[EvalRecord]:
    """指定 candidate でエージェントを再実行し、出力を差し替えたレコードを返す."""
    return asyncio.run(_arefresh(records, candidate, tools_mode, concurrency))
