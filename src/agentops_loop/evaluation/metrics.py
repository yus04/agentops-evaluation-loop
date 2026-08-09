"""DeepEval メトリクス定義.

評価観点は 3 軸に整理する。

| 軸       | メトリクス                                            | 種別            |
| -------- | ----------------------------------------------------- | --------------- |
| 回答品質 | Answer Relevancy / Faithfulness / Answer Correctness   | LLM-as-a-Judge  |
| 行動品質 | Tool Correctness / Tool Selection (GEval カスタム)     | 決定的 + Judge  |
| 業務価値 | Task Completion / Policy Compliance / Operational Eff. | Judge + 決定的  |

すべてのメトリクスは score だけでなく **reason (スコア付けの理由)** を返す。
この reason が GEPA の reflection (改善候補生成) の入力になる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)

from .judge import LangChainJudge

logger = logging.getLogger(__name__)

try:  # deepeval >= 3.5 は SingleTurnParams、それ以前は LLMTestCaseParams
    from deepeval.test_case import SingleTurnParams as TestCaseParams
except ImportError:  # pragma: no cover
    from deepeval.test_case import LLMTestCaseParams as TestCaseParams  # type: ignore[assignment]


ANSWER_QUALITY = "answer_quality"
BEHAVIOR_QUALITY = "behavior_quality"
BUSINESS_VALUE = "business_value"


@dataclass
class MetricSpec:
    """メトリクスとその実行条件."""

    key: str
    label: str
    category: str
    metric: BaseMetric
    threshold: float
    weight: float
    #: このメトリクスを実行するために EvalRecord に必要なフィールド
    requires: tuple[str, ...] = field(default_factory=tuple)


class OperationalEfficiencyMetric(BaseMetric):
    """レイテンシー / コスト / ツール呼び出し数 / エラーから運用効率を採点するカスタムメトリクス.

    LLM を使わない決定的メトリクス。GEPA が「精度は上がったがコストが爆発した」
    候補を選ばないようにするためのガードとして総合スコアに組み込む。
    """

    def __init__(self, threshold: float = 0.6, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.evaluation_model = "deterministic"
        self.error: str | None = None
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

        self._target_latency = float(config.get("target_latency_ms", 8000))
        self._max_latency = float(config.get("max_latency_ms", 25000))
        self._target_cost = float(config.get("target_cost_usd", 0.01))
        self._max_cost = float(config.get("max_cost_usd", 0.05))
        self._target_tool_calls = float(config.get("target_tool_calls", 2))
        self._max_tool_calls = float(config.get("max_tool_calls", 4))

    @property
    def __name__(self) -> str:  # noqa: A003 - DeepEval が表示名として参照する
        return "Operational Efficiency"

    @staticmethod
    def _band(value: float, target: float, ceiling: float) -> float:
        if value <= target:
            return 1.0
        if value >= ceiling:
            return 0.0
        return round(1.0 - (value - target) / (ceiling - target), 4)

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        payload = (getattr(test_case, "additional_metadata", None) or {}).get("run_metrics", {})
        latency = float(payload.get("latency_ms", 0.0) or 0.0)
        cost = float(payload.get("estimated_cost_usd", 0.0) or 0.0)
        tool_calls = float(payload.get("tool_call_count", 0) or 0)
        errors = int(payload.get("error_count", 0) or 0)

        latency_score = self._band(latency, self._target_latency, self._max_latency)
        cost_score = self._band(cost, self._target_cost, self._max_cost)
        tool_score = self._band(tool_calls, self._target_tool_calls, self._max_tool_calls)
        error_score = 1.0 if errors == 0 else max(0.0, 1.0 - 0.25 * errors)

        self.score = round(
            0.3 * latency_score + 0.3 * cost_score + 0.2 * tool_score + 0.2 * error_score, 4
        )
        self.success = self.score >= self.threshold
        self.reason = (
            f"latency={latency:.0f}ms (score {latency_score}), "
            f"cost=${cost:.5f} (score {cost_score}), "
            f"tool_calls={tool_calls:.0f} (score {tool_score}), "
            f"errors={errors} (score {error_score})"
        )
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success


def _geval(
    name: str,
    criteria: str,
    params: list[Any],
    threshold: float,
    judge: LangChainJudge,
) -> GEval:
    return GEval(
        name=name,
        criteria=criteria.strip(),
        evaluation_params=params,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def build_metric_specs(
    config: dict[str, Any],
    judge: LangChainJudge,
) -> list[MetricSpec]:
    """設定ファイルから有効なメトリクス一覧を構築する."""
    metrics_config = config.get("metrics", {}) or {}
    specs: list[MetricSpec] = []

    def entry(key: str) -> dict[str, Any]:
        return metrics_config.get(key, {}) or {}

    def enabled(key: str) -> bool:
        return bool(entry(key).get("enabled", True))

    def threshold(key: str, default: float) -> float:
        return float(entry(key).get("threshold", default))

    def weight(key: str, default: float) -> float:
        return float(entry(key).get("weight", default))

    def requires(key: str) -> tuple[str, ...]:
        return tuple(entry(key).get("requires", []) or [])

    # ------------------------------------------------------------ 回答品質
    if enabled("answer_relevancy"):
        specs.append(
            MetricSpec(
                key="answer_relevancy",
                label="Answer Relevancy",
                category=ANSWER_QUALITY,
                metric=AnswerRelevancyMetric(
                    threshold=threshold("answer_relevancy", 0.7),
                    model=judge,
                    include_reason=True,
                    async_mode=False,
                ),
                threshold=threshold("answer_relevancy", 0.7),
                weight=weight("answer_relevancy", 1.0),
            )
        )

    if enabled("faithfulness"):
        specs.append(
            MetricSpec(
                key="faithfulness",
                label="Faithfulness",
                category=ANSWER_QUALITY,
                metric=FaithfulnessMetric(
                    threshold=threshold("faithfulness", 0.8),
                    model=judge,
                    include_reason=True,
                    async_mode=False,
                ),
                threshold=threshold("faithfulness", 0.8),
                weight=weight("faithfulness", 1.5),
                requires=("retrieval_context",),
            )
        )

    if enabled("answer_correctness"):
        specs.append(
            MetricSpec(
                key="answer_correctness",
                label="Answer Correctness",
                category=ANSWER_QUALITY,
                metric=_geval(
                    "Answer Correctness",
                    entry("answer_correctness").get("criteria", ""),
                    [
                        TestCaseParams.INPUT,
                        TestCaseParams.ACTUAL_OUTPUT,
                        TestCaseParams.EXPECTED_OUTPUT,
                    ],
                    threshold("answer_correctness", 0.7),
                    judge,
                ),
                threshold=threshold("answer_correctness", 0.7),
                weight=weight("answer_correctness", 1.0),
                requires=requires("answer_correctness") or ("expected_output",),
            )
        )

    # ------------------------------------------------------------ 行動品質
    if enabled("tool_correctness"):
        specs.append(
            MetricSpec(
                key="tool_correctness",
                label="Tool Correctness",
                category=BEHAVIOR_QUALITY,
                # スコア自体は決定的だが、DeepEval は初期化時に model を要求する
                metric=ToolCorrectnessMetric(
                    threshold=threshold("tool_correctness", 0.8), model=judge
                ),
                threshold=threshold("tool_correctness", 0.8),
                weight=weight("tool_correctness", 1.5),
                requires=requires("tool_correctness") or ("expected_tools",),
            )
        )

    if enabled("tool_selection"):
        specs.append(
            MetricSpec(
                key="tool_selection",
                label="Tool Selection",
                category=BEHAVIOR_QUALITY,
                metric=_geval(
                    "Tool Selection",
                    entry("tool_selection").get("criteria", ""),
                    [
                        TestCaseParams.INPUT,
                        TestCaseParams.ACTUAL_OUTPUT,
                        TestCaseParams.TOOLS_CALLED,
                    ],
                    threshold("tool_selection", 0.7),
                    judge,
                ),
                threshold=threshold("tool_selection", 0.7),
                weight=weight("tool_selection", 1.5),
            )
        )

    # ------------------------------------------------------------ 業務価値
    if enabled("task_completion"):
        specs.append(
            MetricSpec(
                key="task_completion",
                label="Task Completion",
                category=BUSINESS_VALUE,
                metric=_geval(
                    "Task Completion",
                    entry("task_completion").get("criteria", ""),
                    [
                        TestCaseParams.INPUT,
                        TestCaseParams.ACTUAL_OUTPUT,
                        TestCaseParams.TOOLS_CALLED,
                    ],
                    threshold("task_completion", 0.7),
                    judge,
                ),
                threshold=threshold("task_completion", 0.7),
                weight=weight("task_completion", 2.0),
            )
        )

    if enabled("policy_compliance"):
        specs.append(
            MetricSpec(
                key="policy_compliance",
                label="Policy Compliance",
                category=BUSINESS_VALUE,
                metric=_geval(
                    "Policy Compliance",
                    entry("policy_compliance").get("criteria", ""),
                    [
                        TestCaseParams.INPUT,
                        TestCaseParams.ACTUAL_OUTPUT,
                        TestCaseParams.RETRIEVAL_CONTEXT,
                    ],
                    threshold("policy_compliance", 0.8),
                    judge,
                ),
                threshold=threshold("policy_compliance", 0.8),
                weight=weight("policy_compliance", 1.0),
                requires=("retrieval_context",),
            )
        )

    if enabled("operational_efficiency"):
        specs.append(
            MetricSpec(
                key="operational_efficiency",
                label="Operational Efficiency",
                category=BUSINESS_VALUE,
                metric=OperationalEfficiencyMetric(
                    threshold=threshold("operational_efficiency", 0.6),
                    config=entry("operational_efficiency"),
                ),
                threshold=threshold("operational_efficiency", 0.6),
                weight=weight("operational_efficiency", 0.5),
            )
        )

    return specs
