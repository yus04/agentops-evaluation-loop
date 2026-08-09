"""GEPA 最適化の実行エントリ."""

from __future__ import annotations

import inspect
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..agent.candidate import AgentCandidate
from ..dataset.schema import EvalRecord, read_jsonl
from ..evaluation.judge import build_judge_model, load_evaluation_config
from ..evaluation.metrics import build_metric_specs
from ..evaluation.runner import EvaluationReport, evaluate_records, refresh_records
from ..paths import ARTIFACTS_DIR, CONFIG_DIR, GOLDEN_DATASET
from .adapter import AgentGEPAAdapter, build_reflection_lm, component_guidance

logger = logging.getLogger(__name__)


def load_optimization_config() -> dict[str, Any]:
    path = CONFIG_DIR / "optimization.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass
class OptimizationOutcome:
    accepted: bool
    baseline_score: float
    optimized_score: float
    improvement: float
    changed_components: list[str] = field(default_factory=list)
    metric_deltas: dict[str, float] = field(default_factory=dict)
    rollouts: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "baseline_score": self.baseline_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "changed_components": self.changed_components,
            "metric_deltas": self.metric_deltas,
            "rollouts": self.rollouts,
            "notes": self.notes,
        }


def split_dataset(
    records: list[EvalRecord], train_ratio: float, max_records: int, seed: int
) -> tuple[list[EvalRecord], list[EvalRecord]]:
    usable = [record for record in records if record.is_usable()]
    if len(usable) < 4:
        raise ValueError(
            f"最適化には最低 4 件の評価データが必要です (現在 {len(usable)} 件)。"
            " データセットキュレーションのパイプラインを先に実行してください。"
        )
    rng = random.Random(seed)  # noqa: S311 - 実験の再現性目的
    shuffled = usable[:]
    rng.shuffle(shuffled)
    shuffled = shuffled[:max_records]

    split_at = max(2, int(len(shuffled) * train_ratio))
    train = shuffled[:split_at]
    val = shuffled[split_at:] or shuffled[-2:]
    return train, val


def _seed_components(candidate: AgentCandidate, selected: list[str]) -> dict[str, str]:
    if not selected:
        return candidate.to_gepa_candidate()
    missing = [key for key in selected if key not in candidate.components]
    if missing:
        raise ValueError(f"存在しないコンポーネントが指定されています: {missing}")
    return {key: candidate.components[key] for key in selected}


def _call_gepa(**kwargs: Any) -> Any:
    import gepa

    signature = inspect.signature(gepa.optimize)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        logger.warning("この gepa バージョンでは未対応の引数を無視します: %s", dropped)
    return gepa.optimize(**supported)


def run_optimization(
    dataset_path: Path = GOLDEN_DATASET,
    output_dir: Path = ARTIFACTS_DIR,
    config: dict[str, Any] | None = None,
) -> tuple[AgentCandidate, OptimizationOutcome]:
    """golden データセットを使って GEPA でエージェントを最適化する."""
    config = config or load_optimization_config()
    gepa_config = config.get("gepa", {}) or {}
    dataset_config = config.get("dataset", {}) or {}
    agent_config = config.get("agent", {}) or {}
    acceptance = config.get("acceptance", {}) or {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(dataset_path))
    train, val = split_dataset(
        records,
        train_ratio=float(dataset_config.get("train_ratio", 0.6)),
        max_records=int(dataset_config.get("max_records", 24)),
        seed=int(dataset_config.get("shuffle_seed", 20260101)),
    )
    logger.info("train=%d 件 / val=%d 件", len(train), len(val))

    seed_candidate = AgentCandidate.from_repo()
    eval_config = load_evaluation_config()
    metric_specs = build_metric_specs(eval_config, build_judge_model())

    tools_mode = str(agent_config.get("tools_mode", "local"))
    concurrency = int(agent_config.get("concurrency", 4))

    adapter = AgentGEPAAdapter(
        seed_candidate=seed_candidate,
        metric_specs=metric_specs,
        tools_mode=tools_mode,
        concurrency=concurrency,
    )

    # ---------------------------------------------------------------- ベースライン
    baseline_records = refresh_records(val, seed_candidate, tools_mode, concurrency)
    baseline_report = evaluate_records(
        baseline_records,
        label="baseline",
        specs=metric_specs,
        push_scores=False,
        max_concurrency=concurrency,
        candidate_keys=sorted(seed_candidate.components),
    )
    baseline_report.save(output_dir / "baseline_report.json")
    (output_dir / "baseline_report.md").write_text(
        baseline_report.render_markdown(), encoding="utf-8"
    )

    # ---------------------------------------------------------------- GEPA 実行
    selected = list(config.get("components") or [])
    seed_dict = _seed_components(seed_candidate, selected)

    guidance = "\n".join(
        f"- {key}: {component_guidance(key, seed_candidate)}" for key in sorted(seed_dict)
    )
    logger.info("最適化対象コンポーネント:\n%s", guidance)

    result = _call_gepa(
        seed_candidate=seed_dict,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=build_reflection_lm(),
        max_metric_calls=int(gepa_config.get("max_metric_calls", 60)),
        reflection_minibatch_size=int(gepa_config.get("reflection_minibatch_size", 3)),
        candidate_selection_strategy=str(
            gepa_config.get("candidate_selection_strategy", "pareto")
        ),
        use_merge=bool(gepa_config.get("use_merge", True)),
        seed=int(gepa_config.get("seed", 0)),
        display_progress_bar=bool(gepa_config.get("display_progress_bar", False)),
        run_dir=str(output_dir / "gepa_run"),
        raise_on_exception=False,
        track_best_outputs=False,
    )

    best_components = dict(getattr(result, "best_candidate", None) or seed_dict)
    optimized_candidate = seed_candidate.with_components(best_components)
    optimized_candidate.save_json(output_dir / "optimized_candidate.json")
    seed_candidate.save_json(output_dir / "baseline_candidate.json")

    # ---------------------------------------------------------------- 最適化後の評価
    optimized_records = refresh_records(val, optimized_candidate, tools_mode, concurrency)
    optimized_report = evaluate_records(
        optimized_records,
        label="optimized",
        specs=metric_specs,
        push_scores=False,
        max_concurrency=concurrency,
        candidate_keys=sorted(optimized_candidate.components),
    )
    optimized_report.save(output_dir / "optimized_report.json")
    (output_dir / "optimized_report.md").write_text(
        optimized_report.render_markdown(), encoding="utf-8"
    )

    outcome = _decide(
        baseline_report,
        optimized_report,
        seed_candidate,
        optimized_candidate,
        acceptance,
        adapter.rollouts,
    )
    (output_dir / "optimization_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "train_size": len(train),
                "val_size": len(val),
                "outcome": outcome.to_dict(),
                "baseline": {
                    "overall_score": baseline_report.overall_score,
                    "pass_rate": baseline_report.pass_rate,
                    "metric_averages": baseline_report.metric_averages,
                    "run_metrics": baseline_report.run_metrics,
                },
                "optimized": {
                    "overall_score": optimized_report.overall_score,
                    "pass_rate": optimized_report.pass_rate,
                    "metric_averages": optimized_report.metric_averages,
                    "run_metrics": optimized_report.run_metrics,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "optimization_summary.md").write_text(
        render_summary(baseline_report, optimized_report, outcome), encoding="utf-8"
    )
    return optimized_candidate, outcome


def _decide(
    baseline: EvaluationReport,
    optimized: EvaluationReport,
    seed_candidate: AgentCandidate,
    optimized_candidate: AgentCandidate,
    acceptance: dict[str, Any],
    rollouts: int,
) -> OptimizationOutcome:
    min_improvement = float(acceptance.get("min_absolute_improvement", 0.005))
    max_regression = float(acceptance.get("max_metric_regression", 0.05))

    improvement = round(optimized.overall_score - baseline.overall_score, 4)
    deltas = {
        key: round(optimized.metric_averages.get(key, 0.0) - value, 4)
        for key, value in baseline.metric_averages.items()
    }
    regressions = [key for key, delta in deltas.items() if delta < -max_regression]
    changed = seed_candidate.diff_keys(optimized_candidate)

    notes: list[str] = []
    accepted = True
    if not changed:
        accepted = False
        notes.append("コンポーネントに変更がありませんでした。")
    if improvement < min_improvement:
        accepted = False
        notes.append(
            f"総合スコアの改善 {improvement:+.4f} が受け入れ基準 {min_improvement} に達しませんでした。"
        )
    if regressions:
        accepted = False
        notes.append(f"許容幅を超えて劣化したメトリクスがあります: {regressions}")

    return OptimizationOutcome(
        accepted=accepted,
        baseline_score=baseline.overall_score,
        optimized_score=optimized.overall_score,
        improvement=improvement,
        changed_components=changed,
        metric_deltas=deltas,
        rollouts=rollouts,
        notes=notes,
    )


def render_summary(
    baseline: EvaluationReport, optimized: EvaluationReport, outcome: OptimizationOutcome
) -> str:
    lines = [
        "# GEPA 最適化サマリ",
        "",
        f"- 判定: **{'採用' if outcome.accepted else '不採用'}**",
        f"- 総合スコア: {outcome.baseline_score:.4f} → {outcome.optimized_score:.4f} "
        f"(**{outcome.improvement:+.4f}**)",
        f"- 合格率: {baseline.pass_rate:.1%} → {optimized.pass_rate:.1%}",
        f"- エージェント実行回数 (rollouts): {outcome.rollouts}",
        "",
        "## メトリクス別の変化",
        "",
        "| メトリクス | ベースライン | 最適化後 | 差分 |",
        "| --- | --- | --- | --- |",
    ]
    for key in sorted(set(baseline.metric_averages) | set(optimized.metric_averages)):
        before = baseline.metric_averages.get(key, 0.0)
        after = optimized.metric_averages.get(key, 0.0)
        lines.append(f"| {key} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")

    lines += ["", "## 実行メトリクスの変化", "", "| 指標 | ベースライン | 最適化後 |", "| --- | --- | --- |"]
    for key in sorted(set(baseline.run_metrics) | set(optimized.run_metrics)):
        lines.append(
            f"| {key} | {baseline.run_metrics.get(key, 0.0):.4f} | "
            f"{optimized.run_metrics.get(key, 0.0):.4f} |"
        )

    lines += ["", "## 変更されたコンポーネント", ""]
    lines += [f"- `{key}`" for key in outcome.changed_components] or ["- (なし)"]

    if outcome.notes:
        lines += ["", "## 備考", ""] + [f"- {note}" for note in outcome.notes]
    lines.append("")
    return "\n".join(lines)
