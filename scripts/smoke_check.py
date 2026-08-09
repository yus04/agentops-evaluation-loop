"""LLM を呼ばない範囲でパイプライン全体の配線を検証するスモークテスト.

    OPENAI_API_KEY=dummy uv run python scripts/smoke_check.py

決定的メトリクス (Tool Correctness / Operational Efficiency) だけを有効にして、
評価ランナー → レポート生成 → GEPA reflective dataset までを一気通貫で流す。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from agentops_loop.agent.candidate import AgentCandidate  # noqa: E402
from agentops_loop.agent.tools import ToolRuntime  # noqa: E402
from agentops_loop.dataset.schema import RunMetrics, ToolCallRecord, read_jsonl  # noqa: E402
from agentops_loop.evaluation.judge import build_judge_model  # noqa: E402
from agentops_loop.evaluation.metrics import build_metric_specs  # noqa: E402
from agentops_loop.evaluation.runner import evaluate_records  # noqa: E402
from agentops_loop.optimization.adapter import AgentGEPAAdapter, EvaluationBatch  # noqa: E402
from agentops_loop.paths import CONFIG_DIR, REGRESSION_DATASET  # noqa: E402

LLM_METRICS = (
    "answer_relevancy",
    "faithfulness",
    "answer_correctness",
    "tool_selection",
    "task_completion",
    "policy_compliance",
)


async def check_tools(seed: AgentCandidate) -> None:
    runtime = ToolRuntime(
        policy=seed.tool_policy(), catalog=seed.meta["tool_catalog"], mode="local"
    )
    for name, args in [
        ("lookup_order", {"order_id": "ORD-1001"}),
        ("track_shipment", {"order_id": "ORD-1002"}),
        ("search_policy", {"query": "返金 日数"}),
    ]:
        output, error = await runtime.acall(name, args)
        assert error is None, error
        print(f"  tool {name:<26} -> found={output.get('found')}")


def main() -> None:
    seed = AgentCandidate.from_repo()
    print("1) candidate components:", len(seed.components))
    print("   tools:", seed.tool_policy().enabled)
    print("   models:", seed.model_policy().per_node)

    print("2) MCP バックエンド (local モード)")
    asyncio.run(check_tools(seed))

    records = read_jsonl(REGRESSION_DATASET)[:3]
    for record in records:
        record.actual_output = "ORD-1001 の合計金額は 12,800 円です。"
        record.tools_called = [
            ToolCallRecord(name=name, arguments={"order_id": "ORD-1001"}, output={"found": True})
            for name in (record.expected_tools or ["lookup_order"])
        ]
        record.retrieval_context = [f"{t.name} -> {t.output}" for t in record.tools_called]
        record.metrics = RunMetrics(
            latency_ms=4200,
            total_tokens=1200,
            estimated_cost_usd=0.006,
            tool_call_count=len(record.tools_called),
            error_count=0,
        )

    config = yaml.safe_load((CONFIG_DIR / "evaluation.yaml").read_text(encoding="utf-8"))
    for key in LLM_METRICS:
        config["metrics"][key]["enabled"] = False
    specs = build_metric_specs(config, build_judge_model())
    print("3) 有効メトリクス:", [spec.key for spec in specs])

    report = evaluate_records(records, label="smoke", specs=specs, push_scores=False)
    print(f"   overall={report.overall_score} pass_rate={report.pass_rate}")
    print("   metric_averages:", report.metric_averages)
    print("   run_metrics:", report.run_metrics)

    adapter = AgentGEPAAdapter(seed_candidate=seed, metric_specs=specs, tools_mode="local")
    batch = EvaluationBatch(
        outputs=[{}],
        scores=[report.overall_score],
        trajectories=[{"record": records[0], "evaluation": report.evaluations[0]}],
    )
    reflective = adapter.make_reflective_dataset(
        seed.to_gepa_candidate(), batch, ["policy/models", "skills/returns"]
    )
    print("4) reflective dataset:", list(reflective))
    print("   feedback 抜粋:", reflective["policy/models"][0]["Feedback"][:160].replace("\n", " "))

    print("\nSMOKE CHECK OK")


if __name__ == "__main__":
    main()
