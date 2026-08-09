"""回帰テストの実行.

GEPA が生成した候補を本番へ出す前の最終ゲート。
最適化ジョブが出力した ``optimized_candidate.json`` と ``baseline_report.json``
(GitHub Artifacts 経由で受け取る) を使い、DeepEval で再採点して劣化を検知する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..agent.candidate import AgentCandidate
from ..dataset.schema import read_jsonl
from ..evaluation.judge import build_judge_model, load_evaluation_config
from ..evaluation.metrics import build_metric_specs
from ..evaluation.runner import EvaluationReport, evaluate_records, refresh_records
from ..paths import ARTIFACTS_DIR, CONFIG_DIR, GOLDEN_DATASET, PROJECT_ROOT

logger = logging.getLogger(__name__)


def load_regression_config() -> dict[str, Any]:
    path = CONFIG_DIR / "regression.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass
class RegressionOutcome:
    passed: bool
    overall_score: float
    baseline_score: float | None
    pass_rate: float
    violations: list[str] = field(default_factory=list)
    metric_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "baseline_score": self.baseline_score,
            "pass_rate": self.pass_rate,
            "violations": self.violations,
            "metric_deltas": self.metric_deltas,
        }


def _resolve_dataset(config: dict[str, Any]) -> Path:
    dataset_config = config.get("dataset", {}) or {}
    path = PROJECT_ROOT / str(dataset_config.get("path", "data/datasets/regression.jsonl"))
    if path.is_file():
        return path
    if dataset_config.get("fallback_to_golden", True):
        logger.info("%s が無いため golden.jsonl を使用します", path)
        return GOLDEN_DATASET
    raise FileNotFoundError(f"回帰テスト用データセットが見つかりません: {path}")


def run_regression(
    candidate_path: Path | None = None,
    baseline_report_path: Path | None = None,
    output_dir: Path = ARTIFACTS_DIR,
    config: dict[str, Any] | None = None,
) -> tuple[EvaluationReport, RegressionOutcome]:
    config = config or load_regression_config()
    gates = config.get("gates", {}) or {}
    agent_config = config.get("agent", {}) or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(_resolve_dataset(config))
    if not records:
        raise ValueError("回帰テスト用のレコードがありません")

    if candidate_path and Path(candidate_path).is_file():
        candidate = AgentCandidate.load_json(Path(candidate_path))
        # 固定メタ (allowlist / ツールカタログ) は常にリポジトリ側を正とする
        candidate.meta = AgentCandidate.from_repo().meta
    else:
        candidate = AgentCandidate.from_repo()

    metric_specs = build_metric_specs(load_evaluation_config(), build_judge_model())
    tools_mode = str(agent_config.get("tools_mode", "local"))
    concurrency = int(agent_config.get("concurrency", 4))

    refreshed = refresh_records(records, candidate, tools_mode, concurrency)
    report = evaluate_records(
        refreshed,
        label="regression",
        specs=metric_specs,
        push_scores=False,
        max_concurrency=concurrency,
        candidate_keys=sorted(candidate.components),
    )
    report.save(output_dir / "regression_report.json")
    (output_dir / "regression_report.md").write_text(report.render_markdown(), encoding="utf-8")

    baseline: dict[str, Any] | None = None
    if baseline_report_path and Path(baseline_report_path).is_file():
        baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))

    outcome = _apply_gates(report, baseline, gates)
    (output_dir / "regression_outcome.json").write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "regression_summary.md").write_text(
        render_regression_summary(report, outcome), encoding="utf-8"
    )
    return report, outcome


def _apply_gates(
    report: EvaluationReport, baseline: dict[str, Any] | None, gates: dict[str, Any]
) -> RegressionOutcome:
    violations: list[str] = []
    baseline_score = float(baseline["overall_score"]) if baseline else None
    deltas: dict[str, float] = {}

    if baseline is not None:
        max_overall_regression = float(gates.get("max_overall_regression", 0.01))
        delta = report.overall_score - float(baseline["overall_score"])
        if delta < -max_overall_regression:
            violations.append(
                f"総合スコアが {delta:+.4f} 劣化しました (許容 -{max_overall_regression})"
            )

        max_metric_regression = float(gates.get("max_metric_regression", 0.05))
        for key, before in (baseline.get("metric_averages") or {}).items():
            after = report.metric_averages.get(key)
            if after is None:
                continue
            deltas[key] = round(after - float(before), 4)
            if deltas[key] < -max_metric_regression:
                violations.append(
                    f"{key} が {deltas[key]:+.4f} 劣化しました (許容 -{max_metric_regression})"
                )

    min_pass_rate = float(gates.get("min_pass_rate", 0.7))
    if report.pass_rate < min_pass_rate:
        violations.append(f"合格率 {report.pass_rate:.1%} が下限 {min_pass_rate:.1%} を下回りました")

    for key, minimum in (gates.get("min_metric_scores") or {}).items():
        value = report.metric_averages.get(key)
        if value is not None and value < float(minimum):
            violations.append(f"{key} の平均 {value:.4f} が下限 {minimum} を下回りました")

    max_cost = gates.get("max_avg_cost_usd")
    if max_cost is not None and report.run_metrics.get("avg_cost_usd", 0.0) > float(max_cost):
        violations.append(
            f"平均コスト {report.run_metrics['avg_cost_usd']:.5f} USD が上限 {max_cost} を超えました"
        )

    max_latency = gates.get("max_avg_latency_ms")
    if max_latency is not None and report.run_metrics.get("avg_latency_ms", 0.0) > float(max_latency):
        violations.append(
            f"平均レイテンシー {report.run_metrics['avg_latency_ms']:.0f}ms が上限 {max_latency}ms を超えました"
        )

    return RegressionOutcome(
        passed=not violations,
        overall_score=report.overall_score,
        baseline_score=baseline_score,
        pass_rate=report.pass_rate,
        violations=violations,
        metric_deltas=deltas,
    )


def render_regression_summary(report: EvaluationReport, outcome: RegressionOutcome) -> str:
    lines = [
        "# 回帰テスト結果",
        "",
        f"- 判定: **{'PASS' if outcome.passed else 'FAIL'}**",
        f"- 実行日時 (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- 総合スコア: **{outcome.overall_score:.4f}**"
        + (f" (ベースライン {outcome.baseline_score:.4f})" if outcome.baseline_score else ""),
        f"- 合格率: **{outcome.pass_rate:.1%}**",
        f"- 対象レコード数: {report.dataset_size}",
        "",
    ]
    if outcome.metric_deltas:
        lines += ["## ベースラインとの差分", "", "| メトリクス | 差分 |", "| --- | --- |"]
        lines += [f"| {k} | {v:+.4f} |" for k, v in sorted(outcome.metric_deltas.items())]
        lines.append("")
    if outcome.violations:
        lines += ["## 違反したゲート", ""] + [f"- {v}" for v in outcome.violations] + [""]
    else:
        lines += ["すべてのゲートを通過しました。", ""]
    return "\n".join(lines)
