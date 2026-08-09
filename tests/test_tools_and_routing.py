from __future__ import annotations

import pytest

from agentops_loop.agent.candidate import AgentCandidate
from agentops_loop.agent.nodes import route_after_validation
from agentops_loop.agent.tools import ToolRuntime


@pytest.fixture
def runtime() -> ToolRuntime:
    seed = AgentCandidate.from_repo()
    return ToolRuntime(
        policy=seed.tool_policy(), catalog=seed.meta["tool_catalog"], mode="local"
    )


async def test_lookup_order_returns_order(runtime: ToolRuntime) -> None:
    output, error = await runtime.acall("lookup_order", {"order_id": "ORD-1001"})

    assert error is None
    assert output["found"] is True
    assert output["total_jpy"] == 12800


async def test_invalid_order_id_is_reported_not_raised(runtime: ToolRuntime) -> None:
    output, error = await runtime.acall("lookup_order", {"order_id": "'; DROP TABLE orders;--"})

    assert error is None
    assert output["found"] is False
    assert output["error"] == "invalid_order_id"


async def test_unknown_argument_is_stripped(runtime: ToolRuntime) -> None:
    output, error = await runtime.acall(
        "lookup_order", {"order_id": "ORD-1001", "__class__": "boom"}
    )

    assert error is None
    assert output["found"] is True


async def test_disabled_tool_is_rejected() -> None:
    seed = AgentCandidate.from_repo()
    policy = seed.tool_policy()
    restricted = ToolRuntime(
        policy=type(policy)(enabled=("lookup_order",), descriptions={}, max_tool_calls=2),
        catalog=seed.meta["tool_catalog"],
        mode="local",
    )

    output, error = await restricted.acall("escalate_to_human", {"reason": "x"})

    assert output is None
    assert error is not None
    assert "tool_disabled" in error


async def test_create_return_request_requires_eligibility(runtime: ToolRuntime) -> None:
    output, _ = await runtime.acall(
        "create_return_request", {"order_id": "ORD-1004", "reason": "不要になった"}
    )

    assert output["created"] is False
    assert output["error"] == "not_eligible"


def test_render_catalog_uses_policy_descriptions() -> None:
    seed = AgentCandidate.from_repo()
    policy = seed.tool_policy()
    overridden = type(policy)(
        enabled=("lookup_order",),
        descriptions={"lookup_order": "GEPA が書き換えた説明文"},
        max_tool_calls=2,
    )
    runtime = ToolRuntime(policy=overridden, catalog=seed.meta["tool_catalog"], mode="local")

    assert "GEPA が書き換えた説明文" in runtime.render_catalog()


@pytest.mark.parametrize(
    ("validation", "iteration", "expected"),
    [
        ({"sufficient": True}, 1, "respond"),
        ({"sufficient": False, "needs_user_input": True}, 1, "respond"),
        ({"sufficient": False, "retry_tools": True}, 1, "select_tools"),
        ({"sufficient": False, "retry_tools": True}, 5, "respond"),
    ],
)
def test_route_after_validation(validation: dict, iteration: int, expected: str) -> None:
    state = {"validation": validation, "iteration": iteration}

    assert route_after_validation(state, max_retries=2) == expected
