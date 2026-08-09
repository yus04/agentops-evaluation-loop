"""注文サポート業務ロジック (バックエンド).

MCP サーバー (`order_support.py`) と、ローカルツールモード (`agent/tools.py`) の
両方から使われる単一の実装。デモ用に JSON シードデータを参照する。
本番では基幹システム API 呼び出しに差し替える。
"""

from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache
from typing import Any

from ..paths import SEED_DIR

ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")


@lru_cache(maxsize=1)
def _seed() -> dict[str, Any]:
    path = SEED_DIR / "orders.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_order_id(order_id: str) -> str | None:
    value = (order_id or "").strip().upper()
    return value if ORDER_ID_PATTERN.match(value) else None


def _not_found(order_id: str) -> dict[str, Any]:
    return {
        "found": False,
        "error": "order_not_found",
        "message": f"注文 {order_id} は見つかりませんでした。注文 ID の形式は ORD-1234 です。",
    }


def _invalid_id(order_id: str) -> dict[str, Any]:
    return {
        "found": False,
        "error": "invalid_order_id",
        "message": f"注文 ID の形式が不正です: {order_id!r}. 正しい形式は ORD-1234 です。",
    }


def lookup_order(order_id: str) -> dict[str, Any]:
    """注文 ID から注文の明細・金額・ステータス・購入日を取得する."""
    normalized = _normalize_order_id(order_id)
    if normalized is None:
        return _invalid_id(order_id)
    order = _seed()["orders"].get(normalized)
    if order is None:
        return _not_found(normalized)
    return {"found": True, **order}


def track_shipment(order_id: str) -> dict[str, Any]:
    """注文 ID から配送業者・追跡番号・現在地・配達予定日を取得する."""
    normalized = _normalize_order_id(order_id)
    if normalized is None:
        return _invalid_id(order_id)
    if normalized not in _seed()["orders"]:
        return _not_found(normalized)
    shipment = _seed()["shipments"].get(normalized)
    if shipment is None:
        return {
            "found": False,
            "error": "shipment_not_created",
            "order_id": normalized,
            "message": "配送情報はまだ作成されていません (出荷前)。",
        }
    return {"found": True, "order_id": normalized, **shipment}


def check_return_eligibility(order_id: str) -> dict[str, Any]:
    """注文 ID から返品可能かどうか、返品期限、条件を判定する."""
    normalized = _normalize_order_id(order_id)
    if normalized is None:
        return _invalid_id(order_id)
    if normalized not in _seed()["orders"]:
        return _not_found(normalized)
    record = _seed()["returns"].get(normalized)
    if record is None:
        return {
            "found": True,
            "order_id": normalized,
            "eligible": False,
            "reason": "この注文は返品対象として登録されていません。サポートへご相談ください。",
        }
    return {"found": True, "order_id": normalized, **record}


def create_return_request(order_id: str, reason: str) -> dict[str, Any]:
    """返品申請を作成する (副作用あり). 返品可否を確認した後にのみ使用する."""
    normalized = _normalize_order_id(order_id)
    if normalized is None:
        return _invalid_id(order_id)
    eligibility = check_return_eligibility(normalized)
    if not eligibility.get("eligible"):
        return {
            "created": False,
            "order_id": normalized,
            "error": "not_eligible",
            "message": eligibility.get("reason", "返品対象外です。"),
        }
    return {
        "created": True,
        "order_id": normalized,
        "return_request_id": f"RMA-{uuid.uuid4().hex[:8].upper()}",
        "reason": (reason or "").strip()[:500],
        "next_step": "返送用ラベルをメールで送付します。7 日以内に発送してください。",
    }


def get_refund_status(order_id: str) -> dict[str, Any]:
    """注文 ID から返金処理の進捗と着金予定日を取得する."""
    normalized = _normalize_order_id(order_id)
    if normalized is None:
        return _invalid_id(order_id)
    if normalized not in _seed()["orders"]:
        return _not_found(normalized)
    refund = _seed()["refunds"].get(normalized)
    if refund is None:
        return {
            "found": False,
            "order_id": normalized,
            "error": "refund_not_found",
            "message": "この注文に対する返金レコードはありません。返品が完了しているかご確認ください。",
        }
    return {"found": True, "order_id": normalized, **refund}


def search_policy(query: str, limit: int = 3) -> dict[str, Any]:
    """ポリシー文書を検索して根拠となる条文を返す."""
    terms = [t for t in re.split(r"[\s、。,.]+", (query or "").lower()) if t]
    scored: list[tuple[int, dict[str, Any]]] = []
    for policy in _seed()["policies"]:
        haystack = f"{policy['title']} {policy['text']} {' '.join(policy['tags'])}".lower()
        score = sum(1 for term in terms if term and term in haystack)
        if score:
            scored.append((score, policy))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [policy for _, policy in scored[: max(1, min(limit, 5))]]
    return {
        "found": bool(top),
        "query": query,
        "results": top,
        "message": "" if top else "該当するポリシー条文が見つかりませんでした。",
    }


def escalate_to_human(reason: str, priority: str = "medium") -> dict[str, Any]:
    """自動対応の範囲を超える場合に有人サポートへ引き継ぐ."""
    normalized_priority = priority if priority in {"low", "medium", "high"} else "medium"
    sla = {"high": "1 時間以内", "medium": "1 営業日以内", "low": "2 営業日以内"}
    return {
        "escalated": True,
        "ticket_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
        "priority": normalized_priority,
        "reason": (reason or "").strip()[:1000],
        "response_sla": sla[normalized_priority],
    }


#: ツール名 -> 実装
TOOL_IMPLEMENTATIONS = {
    "lookup_order": lookup_order,
    "track_shipment": track_shipment,
    "check_return_eligibility": check_return_eligibility,
    "create_return_request": create_return_request,
    "get_refund_status": get_refund_status,
    "search_policy": search_policy,
    "escalate_to_human": escalate_to_human,
}
