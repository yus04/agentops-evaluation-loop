from __future__ import annotations

import yaml

from agentops_loop.agent.candidate import AgentCandidate


def test_from_repo_loads_all_component_types() -> None:
    candidate = AgentCandidate.from_repo()
    keys = candidate.optimizable_keys()

    assert "instructions/classify" in keys
    assert "instructions/respond" in keys
    assert any(key.startswith("skills/") for key in keys)
    assert "policy/tools" in keys
    assert "policy/models" in keys


def test_tool_policy_falls_back_on_broken_yaml() -> None:
    seed = AgentCandidate.from_repo()
    broken = seed.with_components({"policy/tools": "enabled: [unclosed"})

    policy = broken.tool_policy(fallback=seed)

    assert policy.enabled == seed.tool_policy().enabled


def test_tool_policy_drops_unknown_tools() -> None:
    seed = AgentCandidate.from_repo()
    mutated = seed.with_components(
        {
            "policy/tools": yaml.safe_dump(
                {"enabled": ["lookup_order", "rm_rf_everything"], "max_tool_calls": 99}
            )
        }
    )

    policy = mutated.tool_policy(fallback=seed)

    assert policy.enabled == ("lookup_order",)
    assert policy.max_tool_calls == 8  # 上限でクランプされる


def test_model_policy_rejects_models_outside_allowlist() -> None:
    seed = AgentCandidate.from_repo()
    mutated = seed.with_components(
        {
            "policy/models": yaml.safe_dump(
                {"per_node": {"respond": "evil-model:latest"}, "temperature": 9.9}
            )
        }
    )

    policy = mutated.model_policy(fallback=seed)

    assert policy.for_node("respond") in seed.meta["allowed_models"]
    assert policy.temperature == 1.0


def test_with_components_ignores_unknown_keys() -> None:
    seed = AgentCandidate.from_repo()
    mutated = seed.with_components({"totally/unknown": "x"})

    assert "totally/unknown" not in mutated.components


def test_diff_keys_detects_changes() -> None:
    seed = AgentCandidate.from_repo()
    mutated = seed.with_components({"instructions/respond": "変更後のプロンプト"})

    assert seed.diff_keys(mutated) == ["instructions/respond"]
