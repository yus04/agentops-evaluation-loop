"""注文サポート MCP サーバー (stdio / streamable-http).

起動例:
    uv run python -m agentops_loop.mcp_server.order_support            # stdio
    MCP_TRANSPORT=streamable-http uv run python -m agentops_loop.mcp_server.order_support
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings

from . import backend

FastMCPSettings.model_rebuild()
mcp = FastMCP("order-support")


@mcp.tool()
def lookup_order(order_id: str) -> dict[str, Any]:
    """注文 ID から注文の明細・金額・ステータス・購入日を取得する。

    Args:
        order_id: 注文 ID (例: ORD-1001)
    """
    return backend.lookup_order(order_id)


@mcp.tool()
def track_shipment(order_id: str) -> dict[str, Any]:
    """注文 ID から配送業者・追跡番号・現在地・配達予定日を取得する。

    Args:
        order_id: 注文 ID (例: ORD-1002)
    """
    return backend.track_shipment(order_id)


@mcp.tool()
def check_return_eligibility(order_id: str) -> dict[str, Any]:
    """注文 ID から返品可能かどうか、返品期限、条件を判定する。

    Args:
        order_id: 注文 ID
    """
    return backend.check_return_eligibility(order_id)


@mcp.tool()
def create_return_request(order_id: str, reason: str) -> dict[str, Any]:
    """返品申請を作成する。返品可否を確認した後にのみ使用する副作用のある操作。

    Args:
        order_id: 注文 ID
        reason: 返品理由
    """
    return backend.create_return_request(order_id, reason)


@mcp.tool()
def get_refund_status(order_id: str) -> dict[str, Any]:
    """注文 ID から返金処理の進捗と着金予定日を取得する。

    Args:
        order_id: 注文 ID
    """
    return backend.get_refund_status(order_id)


@mcp.tool()
def search_policy(query: str, limit: int = 3) -> dict[str, Any]:
    """返品・返金・配送・キャンセルなどの社内ポリシー文書を検索して根拠条文を返す。

    Args:
        query: 検索クエリ
        limit: 最大取得件数
    """
    return backend.search_policy(query, limit)


@mcp.tool()
def escalate_to_human(reason: str, priority: str = "medium") -> dict[str, Any]:
    """自動対応の範囲を超える場合に有人サポートへ引き継ぐ。

    Args:
        reason: 引き継ぎ理由 (ユーザーの要望と実施済みの確認内容)
        priority: low / medium / high
    """
    return backend.escalate_to_human(reason, priority)


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
