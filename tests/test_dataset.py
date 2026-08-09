from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agentops_loop.dataset.build import curate, trace_to_record
from agentops_loop.dataset.promote import promote_candidates
from agentops_loop.dataset.schema import EvalRecord, ReviewMeta, read_jsonl, write_jsonl


def make_trace(user_input: str, *, errors: list[str] | None = None) -> dict:
    return {
        "id": f"trace-{abs(hash(user_input)) % 10**8}",
        "timestamp": datetime.now(UTC),
        "input": {"user_input": user_input},
        "output": {"final_answer": "回答本文"},
        "tags": ["production"],
        "metadata": {
            "agentops": {
                "classification": {"category": "order_status", "confidence": 0.9},
                "validation": {"sufficient": True},
                "errors": errors or [],
                "tool_executions": [
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "ORD-1001"},
                        "output": {"found": True},
                        "error": None,
                        "latency_ms": 12.0,
                    }
                ],
                "metrics": {
                    "latency_ms": 1200.0,
                    "total_tokens": 900,
                    "estimated_cost_usd": 0.004,
                    "tool_call_count": 1,
                    "error_count": len(errors or []),
                },
            }
        },
    }


def test_trace_to_record_extracts_business_attributes() -> None:
    record = trace_to_record(make_trace("ORD-1001 の状況は？"))

    assert record is not None
    assert record.input == "ORD-1001 の状況は？"
    assert record.actual_output == "回答本文"
    assert [tool.name for tool in record.tools_called] == ["lookup_order"]
    assert record.retrieval_context
    assert record.metrics.total_tokens == 900
    assert record.review.status == "pending"


def test_trace_without_user_input_is_skipped() -> None:
    trace = make_trace("x")
    trace["input"] = {}

    assert trace_to_record(trace) is None


def test_curate_deduplicates_and_prioritizes_failures() -> None:
    records = [
        trace_to_record(make_trace("同じ質問です")),
        trace_to_record(make_trace("同じ質問です")),
        trace_to_record(make_trace("失敗した質問", errors=["tool_error"])),
    ]
    selected = curate([r for r in records if r], max_records=10)

    assert len(selected) == 2
    reasons = {record.review.sampling_reason for record in selected}
    assert "errors" in reasons


def test_curate_excludes_already_curated_inputs() -> None:
    record = trace_to_record(make_trace("既出の質問"))
    assert record is not None

    selected = curate([record], max_records=10, exclude_inputs={record.dedup_key})

    assert selected == []


def test_promote_only_moves_approved_records(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    golden = tmp_path / "golden.jsonl"
    write_jsonl(
        [
            EvalRecord(id="a", input="承認", review=ReviewMeta(status="approved")),
            EvalRecord(id="b", input="保留", review=ReviewMeta(status="pending")),
            EvalRecord(id="c", input="却下", review=ReviewMeta(status="rejected")),
        ],
        candidates,
    )

    added, total = promote_candidates(candidates, golden)

    assert (added, total) == (1, 1)
    assert [record.id for record in read_jsonl(golden)] == ["a"]


def test_promote_is_idempotent(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    golden = tmp_path / "golden.jsonl"
    write_jsonl([EvalRecord(id="a", input="承認", review=ReviewMeta(status="approved"))], candidates)

    promote_candidates(candidates, golden)
    added, total = promote_candidates(candidates, golden)

    assert (added, total) == (0, 1)


def test_shipped_datasets_are_parseable() -> None:
    from agentops_loop.paths import GOLDEN_DATASET, REGRESSION_DATASET

    golden = read_jsonl(GOLDEN_DATASET)
    regression = read_jsonl(REGRESSION_DATASET)

    assert len(golden) >= 4
    assert len(regression) >= 4
    assert all(record.expected_output for record in golden)
