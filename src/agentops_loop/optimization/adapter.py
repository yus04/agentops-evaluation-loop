"""GEPA アダプタ.

GEPA は candidate (``dict[str, str]``) を突然変異させながら Pareto 探索を行う。
本アダプタは以下を担う。

1. ``evaluate``               : candidate でエージェントを実行 → DeepEval で採点 → スコア返却
2. ``make_reflective_dataset``: DeepEval の **スコアと理由 (failure reason)** を
                                reflection LM 向けの教材へ整形

DeepEval の reason がそのまま GEPA の "Actionable Side Information" になる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from ..agent.candidate import AgentCandidate
from ..dataset.schema import EvalRecord
from ..evaluation.metrics import MetricSpec
from ..evaluation.runner import RecordEvaluation, evaluate_records, refresh_records
from ..llm import build_chat_model
from ..settings import ReflectionSettings, reflection_settings

logger = logging.getLogger(__name__)

try:  # gepa がインストールされている場合は公式の型を使う
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - gepa 未インストール時のフォールバック

    @dataclass
    class EvaluationBatch:  # type: ignore[no-redef]
        outputs: list[Any]
        scores: list[float]
        trajectories: list[Any] | None = None

    class GEPAAdapter:  # type: ignore[no-redef]
        """gepa 未インストール時のダミー基底クラス."""


def build_reflection_lm(settings: ReflectionSettings | None = None):
    """GEPA の reflection LM. `(prompt: str) -> str` の呼び出し可能オブジェクトを返す.

    エージェント用モデル・Judge 用モデルとは独立に設定できる。
    """
    settings = settings or reflection_settings()
    model = build_chat_model(
        model=settings.model,
        provider=settings.provider,
        temperature=settings.temperature,
        timeout=settings.request_timeout_s,
        max_retries=2,
    )

    def _call(prompt: str) -> str:
        response = model.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    return _call


@dataclass
class AgentGEPAAdapter(GEPAAdapter):
    """注文サポートエージェント用の GEPA アダプタ."""

    seed_candidate: AgentCandidate
    metric_specs: list[MetricSpec]
    tools_mode: str = "local"
    concurrency: int = 4
    #: 統計用 (何回エージェントを実行したか)
    rollouts: int = field(default=0, init=False)

    # ------------------------------------------------------------------ 評価

    def evaluate(
        self,
        batch: list[EvalRecord],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        agent_candidate = self.seed_candidate.with_components(candidate)

        refreshed = refresh_records(
            records=list(batch),
            candidate=agent_candidate,
            tools_mode=self.tools_mode,
            concurrency=self.concurrency,
        )
        self.rollouts += len(refreshed)

        report = evaluate_records(
            refreshed,
            label="gepa-rollout",
            specs=self.metric_specs,
            push_scores=False,
            max_concurrency=self.concurrency,
        )

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []

        by_id = {evaluation.record_id: evaluation for evaluation in report.evaluations}
        for record in refreshed:
            evaluation = by_id.get(record.id)
            score = evaluation.overall_score if evaluation else 0.0
            outputs.append(
                {
                    "final_answer": record.actual_output,
                    "tools_called": [tool.name for tool in record.tools_called],
                    "classification": record.classification,
                }
            )
            scores.append(float(score))
            if capture_traces:
                trajectories.append({"record": record, "evaluation": evaluation})

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    # ------------------------------------------------------------------ 反省用データ

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories = list(eval_batch.trajectories or [])
        dataset: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            items: list[dict[str, Any]] = []
            for trajectory in trajectories:
                record: EvalRecord = trajectory["record"]
                evaluation: RecordEvaluation | None = trajectory["evaluation"]
                items.append(
                    {
                        "Inputs": _render_inputs(record, component),
                        "Generated Outputs": _render_outputs(record),
                        "Feedback": _render_feedback(evaluation, component),
                    }
                )
            dataset[component] = items
        return dataset


# --------------------------------------------------------------------------- レンダリング

_COMPONENT_ROLE = {
    "instructions/classify": "問い合わせ分類ノードのシステムプロンプト",
    "instructions/select_tools": "MCP ツール選択ノードのシステムプロンプト",
    "instructions/validate": "ツール結果検証ノードのシステムプロンプト",
    "instructions/respond": "最終回答生成ノードのシステムプロンプト",
    "policy/tools": "有効なツール一覧・ツール説明文・ツール呼び出し上限を定義する YAML",
    "policy/models": "ノードごとのモデル選択と temperature を定義する YAML",
}


def component_guidance(component: str, seed: AgentCandidate) -> str:
    """reflection LM に渡す、コンポーネントの役割と制約の説明."""
    if component.startswith("skills/"):
        return (
            f"`{component}` は skills/{component.split('/', 1)[1]}.md に対応するスキル文書です。"
            "エージェントはカテゴリに応じてこの文書を参照し、手順・使用ツール・注意点を判断します。"
            "Markdown 見出し構造 (## 使うツール / ## 手順 / ## 回答に含める項目 / ## 注意) を保ってください。"
        )
    if component == "policy/tools":
        catalog = sorted((seed.meta.get("tool_catalog") or {}).keys())
        return (
            "`policy/tools` は YAML です。キーは `enabled` (ツール名の配列), "
            "`descriptions` (ツール名 -> 説明文), `max_tool_calls` (1-8 の整数) のみ使用できます。"
            f"使用可能なツール名は次に限られます: {catalog}。"
            "カタログ外の名前を書いても無視されます。必ず有効な YAML を出力してください。"
        )
    if component == "policy/models":
        allowed = list(seed.meta.get("allowed_models") or [])
        return (
            "`policy/models` は YAML です。キーは `per_node` (classify / select_tools / "
            "validate / respond -> モデル名) と `temperature` (0.0-1.0) のみです。"
            f"使用できるモデル名は次に限られます: {allowed}。"
            "安価なモデルで十分なノードは安価に、品質が要求されるノードは高性能モデルにして"
            "品質とコストのバランスを取ってください。必ず有効な YAML を出力してください。"
        )
    return _COMPONENT_ROLE.get(component, f"`{component}` はエージェント挙動を決めるテキストです。")


def _render_inputs(record: EvalRecord, component: str) -> str:
    parts = [f"## ユーザー入力\n{record.input}"]
    if record.classification:
        parts.append(f"## 分類結果\n{record.classification}")
    if component != "instructions/classify":
        tools = [
            f"- {tool.name}(args={tool.arguments}) -> {tool.error or tool.output}"
            for tool in record.tools_called
        ] or ["- (ツール呼び出しなし)"]
        parts.append("## ツール呼び出しと結果\n" + "\n".join(tools))
    if record.expected_output:
        parts.append(f"## 期待される回答 (人間がキュレーション)\n{record.expected_output}")
    if record.expected_tools:
        parts.append(f"## 期待されるツール\n{record.expected_tools}")
    return "\n\n".join(parts)


def _render_outputs(record: EvalRecord) -> str:
    return (
        f"## 最終回答\n{record.actual_output}\n\n"
        f"## 検証結果\n{record.validation}\n\n"
        f"## 実行メトリクス\n{record.metrics.model_dump()}\n\n"
        f"## エラー\n{record.errors or '(なし)'}"
    )


def _render_feedback(evaluation: RecordEvaluation | None, component: str) -> str:
    if evaluation is None:
        return "評価結果を取得できませんでした。"

    lines = [f"総合スコア: {evaluation.overall_score:.3f} (合格: {evaluation.passed})", ""]
    for metric in evaluation.metrics:
        if metric.skipped:
            continue
        status = "PASS" if metric.success else "FAIL"
        lines.append(
            f"- [{status}] {metric.label}: score={metric.score} "
            f"(threshold={metric.threshold})\n  理由: {metric.reason}"
        )

    hint = _improvement_hint(component)
    if hint:
        lines += ["", f"改善の着眼点: {hint}"]
    return "\n".join(lines)


def _improvement_hint(component: str) -> str:
    if component == "instructions/classify":
        return "分類の誤りは後続のスキル選択とツール選択を連鎖的に誤らせます。境界事例の判別条件を具体化してください。"
    if component == "instructions/select_tools":
        return "Tool Selection / Tool Correctness のスコアと理由に着目し、不足ツールと過剰ツールの両方を減らしてください。"
    if component == "instructions/validate":
        return "Faithfulness が低い場合は検証の甘さが原因です。根拠が取れていない主張を通さない条件を強めてください。"
    if component == "instructions/respond":
        return "Task Completion / Answer Relevancy / Policy Compliance の理由を読み、回答の構成と禁止事項を具体化してください。"
    if component.startswith("skills/"):
        return "このカテゴリの失敗事例に共通する手順の抜けを、スキル文書の手順・注意点として明文化してください。"
    if component == "policy/tools":
        return "ツール説明文が曖昧だと選択を誤ります。いつ使う/使わないかを説明文に書き込んでください。"
    if component == "policy/models":
        return "Operational Efficiency のコスト/レイテンシーと品質メトリクスのトレードオフを見てモデルを割り当ててください。"
    return ""
