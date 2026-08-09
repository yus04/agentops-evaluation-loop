"""承認済みレコードを golden データセットへ昇格させる補助関数."""

from __future__ import annotations

from pathlib import Path

from ..paths import GOLDEN_DATASET
from .schema import EvalRecord, read_jsonl, write_jsonl


def promote_candidates(
    candidate_path: Path,
    golden_path: Path = GOLDEN_DATASET,
) -> tuple[int, int]:
    """候補 JSONL のうち ``review.status == "approved"`` を golden へ追記する.

    Returns:
        (追加件数, 追加後の総件数)
    """
    candidates = [r for r in read_jsonl(Path(candidate_path)) if r.review.status == "approved"]
    golden = read_jsonl(golden_path)
    existing = {record.dedup_key for record in golden}

    added: list[EvalRecord] = [c for c in candidates if c.dedup_key not in existing]
    merged = golden + added
    write_jsonl(merged, golden_path)
    return len(added), len(merged)
