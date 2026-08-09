"""本番エージェントの HTTP サービス (コンテナのエントリポイント)."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .agent import AgentCandidate, OrderSupportAgent
from .observability import flush_tracing, init_tracing, shutdown_tracing
from .settings import agent_settings

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000


class SupportRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=10)


class SupportResponse(BaseModel):
    answer: str
    category: str
    priority: str
    tools_called: list[str]
    trace_id: str | None
    session_id: str
    latency_ms: float
    total_tokens: int
    estimated_cost_usd: float


def _verify_api_key(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
    expected = agent_settings().api_key
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    init_tracing()
    agent = OrderSupportAgent(AgentCandidate.from_repo())
    await agent.__aenter__()
    app.state.agent = agent
    try:
        yield
    finally:
        await agent.__aexit__(None, None, None)
        flush_tracing()
        shutdown_tracing()


app = FastAPI(title="Order Support Agent", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    agent: OrderSupportAgent | None = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not ready")
    return {
        "status": "ready",
        "tools_mode": agent.tool_runtime.mode,
        "tools_enabled": list(agent.context.tool_policy.enabled),
        "models": agent.context.model_policy.per_node,
    }


@app.post(
    "/v1/support",
    response_model=SupportResponse,
    dependencies=[Depends(_verify_api_key)],
)
async def support(request: SupportRequest) -> SupportResponse:
    agent: OrderSupportAgent | None = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not ready")

    result = await agent.arun(
        user_input=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        tags=request.tags,
    )
    return SupportResponse(
        answer=result.final_answer,
        category=str(result.classification.get("category", "other")),
        priority=str(result.classification.get("priority", "medium")),
        tools_called=result.tools_called,
        trace_id=result.trace_id,
        session_id=result.session_id or "",
        latency_ms=result.latency_ms,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )
