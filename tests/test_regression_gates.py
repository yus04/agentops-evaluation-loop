from __future__ import annotations

import pytest

pytest.importorskip("deepeval", reason="評価系の依存 (uv sync --extra eval) が必要です")

from agentops_loop.evaluation.runner import EvaluationReport  # noqa: E402
from agentops_loop.regression.run import _apply_gates  # noqa: E402


def make_report(
    overall: float = 0.85,
    pass_rate: float = 0.9,
    metrics: dict[str, float] | None = None,
    run_metrics: dict[str, float] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        label="test",
        created_at="2026-08-09T00:00:00+00:00",
        dataset_size=10,
        metric_averages=metrics
        or {
            "faithfulness": 0.9,
            "tool_selection": 0.85,
            "task_completion": 0.8,
            "policy_compliance": 0.9,
        },
        pass_rate=pass_rate,
        overall_score=overall,
        run_metrics=run_metrics or {"avg_cost_usd": 0.004, "avg_latency_ms": 5000.0},
    )


GATES = {
    "max_overall_regression": 0.01,
    "max_metric_regression": 0.05,
    "min_pass_rate": 0.7,
    "min_metric_scores": {"faithfulness": 0.8, "task_completion": 0.7},
    "max_avg_cost_usd": 0.05,
    "max_avg_latency_ms": 25000,
}


def test_gates_pass_when_no_regression() -> None:
    baseline = make_report().to_dict()

    outcome = _apply_gates(make_report(), baseline, GATES)

    assert outcome.passed
    assert outcome.violations == []


def test_overall_regression_is_detected() -> None:
    baseline = make_report(overall=0.90).to_dict()

    outcome = _apply_gates(make_report(overall=0.80), baseline, GATES)

    assert not outcome.passed
    assert any("総合スコア" in violation for violation in outcome.violations)


def test_metric_regression_is_detected() -> None:
    baseline = make_report().to_dict()
    degraded = make_report(
        metrics={
            "faithfulness": 0.82,
            "tool_selection": 0.60,
            "task_completion": 0.8,
            "policy_compliance": 0.9,
        }
    )

    outcome = _apply_gates(degraded, baseline, GATES)

    assert not outcome.passed
    assert any("tool_selection" in violation for violation in outcome.violations)


def test_absolute_minimum_is_enforced_without_baseline() -> None:
    weak = make_report(metrics={"faithfulness": 0.5, "task_completion": 0.5})

    outcome = _apply_gates(weak, None, GATES)

    assert not outcome.passed
    assert len(outcome.violations) == 2


def test_cost_explosion_is_detected() -> None:
    expensive = make_report(run_metrics={"avg_cost_usd": 0.2, "avg_latency_ms": 5000.0})

    outcome = _apply_gates(expensive, None, GATES)

    assert not outcome.passed
    assert any("平均コスト" in violation for violation in outcome.violations)
