"""GEPA の最適化対象となる「エージェント候補 (candidate)」の定義.

GEPA は `dict[str, str]` (コンポーネント名 -> テキスト) を進化させる。
そのため本プロジェクトでは以下 4 種類をすべて **テキスト** として表現し、
1 つの candidate 辞書にまとめる。

| コンポーネント種別 | キー例                       | 実体                       |
| ------------------ | ---------------------------- | -------------------------- |
| インストラクション | `instructions/classify`      | prompts/classify.md        |
| スキル             | `skills/order_status`        | skills/order_status.md     |
| ツール             | `policy/tools`               | config/agent.yaml: tools   |
| モデル選択         | `policy/models`              | config/agent.yaml: models  |

`policy/tools` と `policy/models` は YAML テキスト。LLM が壊れた YAML を生成しても
落ちないよう、パース失敗時はシード値へフォールバックする (`safe_load` のみ使用)。
モデル名は allowlist で検証し、任意の文字列がそのまま API 呼び出しに渡らないようにする。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..paths import CONFIG_DIR, PROMPTS_DIR, SKILLS_DIR

INSTRUCTION_PREFIX = "instructions/"
SKILL_PREFIX = "skills/"
TOOL_POLICY_KEY = "policy/tools"
MODEL_POLICY_KEY = "policy/models"

#: LangGraph のノード名 = インストラクションの名前
NODE_NAMES: tuple[str, ...] = ("classify", "select_tools", "validate", "respond")


class CandidateError(RuntimeError):
    """candidate の読み込み/検証に失敗した."""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() + "\n"


@dataclass(frozen=True)
class ToolPolicy:
    """ツールの有効化・説明文・呼び出し上限 (GEPA 最適化対象)."""

    enabled: tuple[str, ...]
    descriptions: dict[str, str] = field(default_factory=dict)
    max_tool_calls: int = 4

    def description_for(self, name: str, fallback: str) -> str:
        override = self.descriptions.get(name)
        return override.strip() if isinstance(override, str) and override.strip() else fallback


@dataclass(frozen=True)
class ModelPolicy:
    """ノードごとのモデル選択 (GEPA 最適化対象)."""

    per_node: dict[str, str]
    temperature: float = 0.0
    allowed_models: tuple[str, ...] = ()

    def for_node(self, node: str) -> str:
        return self.per_node[node]


@dataclass
class AgentCandidate:
    """エージェントの挙動を決めるテキスト集合."""

    components: dict[str, str]
    #: allowlist / ツールカタログなど、最適化対象ではない固定メタ情報
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 読み書き

    @classmethod
    def from_repo(
        cls,
        prompts_dir: Path = PROMPTS_DIR,
        skills_dir: Path = SKILLS_DIR,
        config_dir: Path = CONFIG_DIR,
    ) -> AgentCandidate:
        """リポジトリ上のファイル群から candidate を組み立てる."""
        components: dict[str, str] = {}

        for node in NODE_NAMES:
            path = prompts_dir / f"{node}.md"
            if not path.is_file():
                raise CandidateError(f"インストラクションが見つかりません: {path}")
            components[f"{INSTRUCTION_PREFIX}{node}"] = _read_text(path)

        for path in sorted(skills_dir.glob("*.md")):
            components[f"{SKILL_PREFIX}{path.stem}"] = _read_text(path)

        config_path = config_dir / "agent.yaml"
        if not config_path.is_file():
            raise CandidateError(f"エージェント設定が見つかりません: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        components[TOOL_POLICY_KEY] = yaml.safe_dump(
            config.get("tools", {}), allow_unicode=True, sort_keys=False
        )
        components[MODEL_POLICY_KEY] = yaml.safe_dump(
            config.get("models", {}), allow_unicode=True, sort_keys=False
        )

        meta = {
            "allowed_models": list(config.get("allowed_models", [])),
            "tool_catalog": config.get("tool_catalog", {}),
            "config_path": str(config_path),
        }
        return cls(components=components, meta=meta)

    @classmethod
    def from_dict(cls, components: dict[str, str], meta: dict[str, Any] | None = None) -> AgentCandidate:
        return cls(components=dict(components), meta=dict(meta or {}))

    @classmethod
    def load_json(cls, path: Path) -> AgentCandidate:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload["components"], payload.get("meta"))

    def save_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {"components": self.components, "meta": self.meta},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_to_repo(
        self,
        prompts_dir: Path = PROMPTS_DIR,
        skills_dir: Path = SKILLS_DIR,
        config_dir: Path = CONFIG_DIR,
    ) -> list[Path]:
        """最適化結果をリポジトリのファイルへ書き戻す (PR 化するため)."""
        written: list[Path] = []
        prompts_dir.mkdir(parents=True, exist_ok=True)
        skills_dir.mkdir(parents=True, exist_ok=True)

        for key, text in self.components.items():
            if key.startswith(INSTRUCTION_PREFIX):
                path = prompts_dir / f"{key[len(INSTRUCTION_PREFIX):]}.md"
            elif key.startswith(SKILL_PREFIX):
                path = skills_dir / f"{key[len(SKILL_PREFIX):]}.md"
            else:
                continue
            path.write_text(text.strip() + "\n", encoding="utf-8")
            written.append(path)

        config_path = config_dir / "agent.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["tools"] = self._safe_yaml(TOOL_POLICY_KEY)
        config["models"] = self._safe_yaml(MODEL_POLICY_KEY)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        written.append(config_path)
        return written

    # ------------------------------------------------------------------ アクセサ

    def instruction(self, node: str) -> str:
        return self.components[f"{INSTRUCTION_PREFIX}{node}"]

    @property
    def skill_names(self) -> list[str]:
        return sorted(
            key[len(SKILL_PREFIX) :] for key in self.components if key.startswith(SKILL_PREFIX)
        )

    def skill(self, name: str) -> str | None:
        return self.components.get(f"{SKILL_PREFIX}{name}")

    def skills_digest(self, names: list[str] | None = None) -> str:
        """指定スキル (省略時は全件) を 1 つのテキストへ連結する."""
        selected = names if names is not None else self.skill_names
        chunks = []
        for name in selected:
            body = self.skill(name)
            if body:
                chunks.append(f"### skill: {name}\n{body.strip()}")
        return "\n\n".join(chunks)

    def _safe_yaml(self, key: str) -> dict[str, Any]:
        raw = self.components.get(key, "")
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # ------------------------------------------------------------------ ポリシー

    def tool_policy(self, fallback: AgentCandidate | None = None) -> ToolPolicy:
        parsed = self._safe_yaml(TOOL_POLICY_KEY)
        catalog: dict[str, Any] = self.meta.get("tool_catalog") or {}
        known = set(catalog) or None

        enabled_raw = parsed.get("enabled")
        enabled = [str(name) for name in enabled_raw] if isinstance(enabled_raw, list) else []
        if known is not None:
            enabled = [name for name in enabled if name in known]
        if not enabled:
            if fallback is not None:
                return fallback.tool_policy()
            enabled = sorted(known or [])

        descriptions_raw = parsed.get("descriptions")
        descriptions = (
            {str(k): str(v) for k, v in descriptions_raw.items()}
            if isinstance(descriptions_raw, dict)
            else {}
        )

        try:
            max_tool_calls = int(parsed.get("max_tool_calls", 4))
        except (TypeError, ValueError):
            max_tool_calls = 4
        max_tool_calls = max(1, min(max_tool_calls, 8))

        return ToolPolicy(
            enabled=tuple(dict.fromkeys(enabled)),
            descriptions=descriptions,
            max_tool_calls=max_tool_calls,
        )

    def model_policy(self, fallback: AgentCandidate | None = None) -> ModelPolicy:
        parsed = self._safe_yaml(MODEL_POLICY_KEY)
        allowed = tuple(str(m) for m in (self.meta.get("allowed_models") or []))
        base = fallback.model_policy() if fallback is not None else None

        per_node: dict[str, str] = {}
        raw_nodes = parsed.get("per_node") if isinstance(parsed.get("per_node"), dict) else {}
        for node in NODE_NAMES:
            value = raw_nodes.get(node)
            candidate_model = str(value) if isinstance(value, str) and value.strip() else ""
            # allowlist に無いモデルは採用しない (不正なモデル名の注入を防ぐ)
            if allowed and candidate_model not in allowed:
                candidate_model = ""
            if not candidate_model and base is not None:
                candidate_model = base.for_node(node)
            if not candidate_model:
                candidate_model = allowed[0] if allowed else "gpt-5-mini"
            per_node[node] = candidate_model

        try:
            temperature = float(parsed.get("temperature", 0.0))
        except (TypeError, ValueError):
            temperature = 0.0
        temperature = max(0.0, min(temperature, 1.0))

        return ModelPolicy(per_node=per_node, temperature=temperature, allowed_models=allowed)

    # ------------------------------------------------------------------ GEPA I/F

    def to_gepa_candidate(self) -> dict[str, str]:
        return copy.deepcopy(self.components)

    def with_components(self, components: dict[str, str]) -> AgentCandidate:
        """GEPA が返したコンポーネント辞書を取り込んだ新しい candidate を返す."""
        merged = dict(self.components)
        merged.update({k: v for k, v in components.items() if k in merged})
        return AgentCandidate(components=merged, meta=copy.deepcopy(self.meta))

    def optimizable_keys(self) -> list[str]:
        return sorted(self.components)

    def diff_keys(self, other: AgentCandidate) -> list[str]:
        return sorted(
            key
            for key in set(self.components) | set(other.components)
            if self.components.get(key) != other.components.get(key)
        )
