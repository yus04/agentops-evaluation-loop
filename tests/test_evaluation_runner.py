from __future__ import annotations

import pytest

from agentops_loop.dataset.schema import EvalRecord, ReviewMeta
from agentops_loop.evaluation.metrics import MetricSpec
from agentops_loop.evaluation.runner import build_test_case, evaluate_records


class FailingMetric:
    def measure(self, _test_case) -> None:
        raise RuntimeError("judge unavailable")


def test_build_test_case_preserves_empty_tools_called() -> None:
    test_case = build_test_case(
        EvalRecord(
            id="no-tools-needed",
            input="注文番号がわからないため配送状況を確認できません",
            actual_output="注文 ID をお知らせください。",
            review=ReviewMeta(status="approved"),
        )
    )

    assert test_case.tools_called == []


def test_evaluate_records_requires_actual_output() -> None:
    record = EvalRecord(
        id="seed-1",
        input="質問",
        expected_output="期待回答",
        review=ReviewMeta(status="approved"),
    )

    with pytest.raises(ValueError, match=r"actual_output.*--rerun"):
        evaluate_records([record], specs=[])


def test_metric_execution_failure_scores_zero() -> None:
    record = EvalRecord(
        id="trace-1",
        input="質問",
        actual_output="回答",
        review=ReviewMeta(status="approved"),
    )
    spec = MetricSpec(
        key="failing",
        label="Failing",
        category="answer_quality",
        metric=FailingMetric(),  # type: ignore[arg-type]
        threshold=0.7,
        weight=1.0,
    )

    report = evaluate_records([record], specs=[spec], max_concurrency=1)

    assert report.overall_score == 0.0
    assert report.pass_rate == 0.0
    assert report.metric_averages == {"failing": 0.0}
    assert report.evaluations[0].metrics[0].error == "judge unavailable"