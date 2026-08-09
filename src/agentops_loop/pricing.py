"""トークン消費からの概算コスト計算."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from .paths import CONFIG_DIR


@lru_cache(maxsize=1)
def _pricing() -> dict[str, Any]:
    path = CONFIG_DIR / "pricing.yaml"
    if not path.is_file():
        return {"pricing": {}, "default": {"input": 1.0, "output": 4.0}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def estimate_cost_usd(usage_entries: list[dict[str, Any]]) -> float:
    """``token_usage`` エントリ群から概算コスト (USD) を求める."""
    table = _pricing().get("pricing", {}) or {}
    default = _pricing().get("default", {"input": 1.0, "output": 4.0})
    total = 0.0
    for entry in usage_entries:
        rates = table.get(str(entry.get("model", "")), default)
        total += (int(entry.get("input_tokens", 0) or 0) / 1_000_000) * float(rates["input"])
        total += (int(entry.get("output_tokens", 0) or 0) / 1_000_000) * float(rates["output"])
    return round(total, 8)
