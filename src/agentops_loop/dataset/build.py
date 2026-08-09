"""Langfuse に蓄積されたテレメトリから評価用データセット候補を生成する.

パイプライン (GitHub Actions `01-dataset-curation.yml`) から実行される。

    Langfuse traces --> EvalRecord[] --> サンプリング/重複排除 --> JSONL 候補 + レビュー用サマリ
                                                                        |
                                                                        v
                                                            Pull Request (人間レビュー)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..observability.tracing import init_tracing
from ..paths import DATASET_CANDIDATES_DIR, GOLDEN_DATASET
from .schema import EvalRecord, ReviewMeta, RunMetrics, ToolCallRecord, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


class TraceFetchError(RuntimeError):
    """Langfuse からのトレース取得に失敗した."""


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """camelCase / snake_case / dict の差異を吸収して属性を取り出す."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def fetch_traces(
    lookback_hours: int = 24,
    limit: int = 500,
    tags: list[str] | None = None,
    environment: str | None = None,
) -> list[Any]:
    """Langfuse v4 Observations API から指定期間のルート span を取得する."""
    client = init_tracing()
    if client is None:
        raise TraceFetchError(
            "Langfuse が未設定です。LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY を設定してください。"
        )

    to_ts = datetime.now(UTC)
    from_ts = to_ts - timedelta(hours=lookback_hours)

    collected: list[Any] = []
    cursor: str | None = None
    page_size = min(1000, limit)
    required_tags = set(tags or [])

    while len(collected) < limit:
        kwargs: dict[str, Any] = {
            "fields": "core,basic,io,metadata,usage,metrics,trace_context",
            "expand_metadata": "agentops",
            "limit": page_size,
            "cursor": cursor,
            "is_root_observation": True,
            "from_start_time": from_ts,
            "to_start_time": to_ts,
        }
        if environment:
            kwargs["environment"] = environment

        try:
            response = client.api.observations.get_many(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TraceFetchError(f"トレース取得に失敗しました: {exc}") from exc

        batch = list(_attr(response, "data", default=[]) or [])
        if not batch:
            break
        collected.extend(
            _observation_to_trace(observation)
            for observation in batch
            if required_tags.issubset(set(_attr(observation, "tags", default=[]) or []))
        )
        next_cursor = _attr(_attr(response, "meta", default={}), "cursor", default=None)
        if not next_cursor or next_cursor == cursor:
            break
        cursor = str(next_cursor)

    logger.info("Langfuse から %d 件のトレースを取得しました", len(collected))
    return collected[:limit]


def _observation_to_trace(observation: Any) -> dict[str, Any]:
    """v4 root observation を既存の trace 変換処理が読める形へ正規化する."""
    latency_seconds = float(_attr(observation, "latency", default=0) or 0)
    return {
        "id": str(_attr(observation, "trace_id", "traceId", "id", default="") or ""),
        "timestamp": _attr(observation, "start_time", "startTime", default=None),
        "input": _attr(observation, "input", default={}),
        "output": _attr(observation, "output", default={}),
        "metadata": _attr(observation, "metadata", default={}),
        "tags": list(_attr(observation, "tags", default=[]) or []),
        "latency": latency_seconds * 1000.0,
        "total_cost": float(_attr(observation, "total_cost", "totalCost", default=0) or 0),
    }


def trace_to_record(trace: Any) -> EvalRecord | None:
    """Langfuse トレース 1 件を `EvalRecord` へ変換する."""
    trace_id = str(_attr(trace, "id", default="") or "")
    metadata = _as_dict(_attr(trace, "metadata", default={}))
    agentops = _as_dict(metadata.get("agentops"))

    trace_input = _as_dict(_attr(trace, "input", default={}))
    trace_output = _as_dict(_attr(trace, "output", default={}))

    user_input = str(trace_input.get("user_input") or _attr(trace, "input", default="") or "").strip()
    actual_output = str(trace_output.get("final_answer") or "").strip()
    if not user_input:
        return None

    executions = agentops.get("tool_executions") or []
    tools_called = [
        ToolCallRecord(
            name=str(execution.get("name", "")),
            arguments=execution.get("arguments") or {},
            output=execution.get("output"),
            error=execution.get("error"),
            latency_ms=execution.get("latency_ms"),
        )
        for execution in executions
        if isinstance(execution, dict)
    ]
    retrieval_context = [
        f"{tool.name} -> {tool.error or tool.output}" for tool in tools_called
    ]

    raw_metrics = _as_dict(agentops.get("metrics"))
    metrics = RunMetrics(
        latency_ms=float(raw_metrics.get("latency_ms") or _attr(trace, "latency", default=0) or 0),
        total_tokens=int(raw_metrics.get("total_tokens") or 0),
        estimated_cost_usd=float(
            raw_metrics.get("estimated_cost_usd")
            or _attr(trace, "totalCost", "total_cost", default=0)
            or 0
        ),
        tool_call_count=int(raw_metrics.get("tool_call_count") or len(tools_called)),
        error_count=int(raw_metrics.get("error_count") or 0),
    )

    timestamp = _attr(trace, "timestamp", default=None)
    return EvalRecord(
        id=f"rec-{hashlib.sha256(trace_id.encode()).hexdigest()[:12]}" if trace_id else f"rec-{hashlib.sha256(user_input.encode()).hexdigest()[:12]}",
        source="trace",
        trace_id=trace_id or None,
        timestamp=timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or ""),
        input=user_input,
        actual_output=actual_output,
        tools_called=tools_called,
        retrieval_context=retrieval_context,
        classification=_as_dict(agentops.get("classification")),
        validation=_as_dict(agentops.get("validation")),
        errors=[str(e) for e in (agentops.get("errors") or [])],
        metrics=metrics,
        tags=[str(t) for t in (_attr(trace, "tags", default=[]) or [])],
        review=ReviewMeta(status="pending"),
    )


def _sampling_reason(record: EvalRecord) -> str:
    reasons: list[str] = []
    if record.errors:
        reasons.append("errors")
    if not record.validation.get("sufficient", True):
        reasons.append("validation_failed")
    if float(record.classification.get("confidence", 1.0) or 1.0) < 0.6:
        reasons.append("low_classification_confidence")
    if record.metrics.tool_call_count == 0:
        reasons.append("no_tool_used")
    if record.metrics.latency_ms > 15_000:
        reasons.append("high_latency")
    if not record.actual_output:
        reasons.append("empty_answer")
    return ",".join(reasons) or "random_sample"


def curate(
    records: list[EvalRecord],
    max_records: int = 50,
    exclude_inputs: set[str] | None = None,
    seed: int = 20260101,
) -> list[EvalRecord]:
    """重複排除 + 失敗事例優先のサンプリング.

    失敗・低信頼のケースを優先しつつ、成功ケースもランダムに混ぜて
    データセットが「難しい例」だけに偏らないようにする。
    """
    exclude_inputs = exclude_inputs or set()
    unique: dict[str, EvalRecord] = {}
    for record in records:
        if not record.is_usable():
            continue
        key = record.dedup_key
        if key in exclude_inputs or key in unique:
            continue
        record.review.sampling_reason = _sampling_reason(record)
        unique[key] = record

    interesting = [r for r in unique.values() if r.review.sampling_reason != "random_sample"]
    routine = [r for r in unique.values() if r.review.sampling_reason == "random_sample"]

    rng = random.Random(seed)  # noqa: S311 - 統計用途であり暗号用途ではない
    rng.shuffle(routine)

    quota_interesting = min(len(interesting), int(max_records * 0.7))
    selected = interesting[:quota_interesting]
    selected += routine[: max(0, max_records - len(selected))]
    if len(selected) < max_records:
        selected += interesting[quota_interesting : max_records - len(selected) + quota_interesting]
    return selected[:max_records]


def build_candidates(
    lookback_hours: int = 24,
    max_records: int = 50,
    limit: int = 500,
    environment: str | None = None,
    output_dir: Path = DATASET_CANDIDATES_DIR,
    golden_path: Path = GOLDEN_DATASET,
) -> tuple[Path, list[EvalRecord]]:
    """トレースを取得 → キュレーション → 候補 JSONL とレビュー用サマリを書き出す."""
    traces = fetch_traces(lookback_hours=lookback_hours, limit=limit, environment=environment)
    records = [record for record in (trace_to_record(t) for t in traces) if record is not None]

    existing = {record.dedup_key for record in read_jsonl(golden_path)}
    selected = curate(records, max_records=max_records, exclude_inputs=existing)

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    output_path = Path(output_dir) / f"{stamp}.jsonl"
    write_jsonl(selected, output_path)

    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(render_review_summary(selected, traces_count=len(traces)), encoding="utf-8")

    logger.info("候補 %d 件を %s に書き出しました", len(selected), output_path)
    return output_path, selected


def render_review_summary(records: list[EvalRecord], traces_count: int) -> str:
    """Pull Request 本文に貼るレビュー用サマリ (Markdown)."""
    lines = [
        "# 評価データセット候補レビュー",
        "",
        f"- 取得トレース数: **{traces_count}**",
        f"- 候補レコード数: **{len(records)}**",
        f"- 生成日時 (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## レビュー手順",
        "",
        "1. 各レコードの `input` / `actual_output` を確認する",
        "2. 理想回答が定義できる場合は `expected_output` を記入する",
        "3. 期待されるツール列を `expected_tools` に記入する (例: `[\"lookup_order\"]`)",
        "4. 分類の正解を `expected_category` に記入する",
        "5. 採用するレコードは `review.status` を `approved` に、除外するものは `rejected` にする",
        "6. 承認したレコードを `data/datasets/golden.jsonl` へ移動して PR をマージする",
        "",
        "## 候補一覧",
        "",
        "| # | 抽出理由 | カテゴリ | ツール | レイテンシー(ms) | 入力 (先頭 60 文字) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, record in enumerate(records, start=1):
        tools = ", ".join(tool.name for tool in record.tools_called) or "-"
        snippet = " ".join(record.input.split())[:60].replace("|", "\\|")
        lines.append(
            f"| {index} | {record.review.sampling_reason} | "
            f"{record.classification.get('category', '-')} | {tools} | "
            f"{record.metrics.latency_ms:.0f} | {snippet} |"
        )
    lines.append("")
    return "\n".join(lines)
