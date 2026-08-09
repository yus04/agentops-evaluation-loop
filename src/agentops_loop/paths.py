"""リポジトリ内の標準パス定義."""

from __future__ import annotations

import os
from pathlib import Path

# src/agentops_loop/paths.py -> リポジトリルート
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("AGENTOPS_PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

DATA_DIR = PROJECT_ROOT / "data"
SEED_DIR = DATA_DIR / "seed"
DATASETS_DIR = DATA_DIR / "datasets"
# 人間レビュー待ちのデータセット候補 (PR で提案される)
DATASET_CANDIDATES_DIR = DATASETS_DIR / "candidates"
# 人間レビュー済み (マージ済み) の評価データセット
GOLDEN_DATASET = DATASETS_DIR / "golden.jsonl"
# 最適化後の回帰テスト用データセット
REGRESSION_DATASET = DATASETS_DIR / "regression.jsonl"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def ensure_dirs() -> None:
    """出力先ディレクトリを作成する."""
    for directory in (DATASETS_DIR, DATASET_CANDIDATES_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
