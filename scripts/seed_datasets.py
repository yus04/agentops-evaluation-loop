"""初期データセットの生成スクリプト.

Langfuse にまだテレメトリが無い段階でも評価 → 最適化 → 回帰のループを
一通り動かせるよう、人手で書いたシードデータセットを生成する。

    uv run python scripts/seed_datasets.py

本番運用では `agentops dataset build` が生成した候補を PR レビューして
`data/datasets/golden.jsonl` を育てていく。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentops_loop.dataset.schema import (  # noqa: E402
    EvalRecord,
    ReviewMeta,
    write_jsonl,
)
from agentops_loop.paths import GOLDEN_DATASET, REGRESSION_DATASET  # noqa: E402

REVIEWED = ReviewMeta(status="approved", reviewer="seed", notes="初期シード (人手作成)")


def record(
    record_id: str,
    user_input: str,
    expected_output: str,
    expected_tools: list[str],
    expected_category: str,
    tags: list[str],
) -> EvalRecord:
    return EvalRecord(
        id=record_id,
        source="manual",
        input=user_input,
        actual_output="",
        expected_output=expected_output,
        expected_tools=expected_tools,
        expected_category=expected_category,
        tags=tags,
        review=REVIEWED,
    )


GOLDEN = [
    record(
        "seed-0001",
        "注文 ORD-1001 の内容と金額を教えてください。",
        "注文 ORD-1001 は 2026-07-18 のご注文で、合計 12,800 円 (ワイヤレスイヤホン A2 × 1) です。現在のステータスは配達済みです。",
        ["lookup_order"],
        "order_status",
        ["seed", "order_status"],
    ),
    record(
        "seed-0002",
        "ORD-1002 がまだ届きません。今どこにありますか？",
        "ORD-1002 は佐川急便で配送中で、現在は厚木中継センターにあります。お届け予定日は 2026-08-09、追跡番号は 5512-8877-0011 です。",
        ["track_shipment"],
        "shipping",
        ["seed", "shipping"],
    ),
    record(
        "seed-0003",
        "ORD-1004 を返品したいのですが可能ですか？",
        "ORD-1004 は返品期限 (2026-07-01) を超過しているため、通常の返品はお受けできません。初期不良の場合はメーカー保証の範囲で個別に対応いたしますので、症状をお知らせください。",
        ["check_return_eligibility", "search_policy"],
        "returns",
        ["seed", "returns", "negative"],
    ),
    record(
        "seed-0004",
        "ORD-1005 の返金はいつ振り込まれますか？",
        "ORD-1005 の返金は現在処理中で、15,400 円を 2026-08-15 までにクレジットカードへ返金予定です。カード会社の締め日により、明細への反映まで最大 14 日程度かかる場合があります。",
        ["get_refund_status", "search_policy"],
        "refund",
        ["seed", "refund"],
    ),
    record(
        "seed-0005",
        "注文したいのですが返品はいつまでできますか？",
        "返品は商品到着日から 30 日以内、未使用かつ外箱・付属品が揃っていることを条件にお受けしています。初期不良の場合は 30 日を過ぎていてもメーカー保証期間内であれば個別に対応いたします。",
        ["search_policy"],
        "returns",
        ["seed", "policy"],
    ),
    record(
        "seed-0006",
        "先月の注文が二重に請求されている気がします。確認してください。",
        "二重請求の可能性があるとのこと、担当部署へ引き継ぎました。1 時間以内に担当者よりご連絡いたします。お手数ですが、ご請求明細に表示されている日付と金額をお手元にご用意ください。",
        ["escalate_to_human"],
        "billing",
        ["seed", "billing", "escalation"],
    ),
    record(
        "seed-0007",
        "商品が届かないので状況を教えてください。",
        "状況を確認いたしますので、注文 ID (ORD-1234 の形式) をお知らせいただけますでしょうか。ご購入時の確認メールに記載されています。",
        [],
        "shipping",
        ["seed", "missing_argument"],
    ),
    record(
        "seed-0008",
        "ORD-1006 の配送状況を教えてください。届く予定日を過ぎています。",
        "ORD-1006 は住所不備のため配達が保留されています。お届け予定日 (2026-08-06) を過ぎているため、担当者へ引き継ぎました。正しいお届け先住所をご返信いただければ再配達の手配を進めます。",
        ["track_shipment", "escalate_to_human"],
        "shipping",
        ["seed", "shipping", "escalation"],
    ),
    record(
        "seed-0009",
        "ORD-1003 はまだ発送されていませんか？キャンセルしたいです。",
        "ORD-1003 は現在準備中 (processing) のため、まだ発送されていません。出荷準備が完了する前であればマイページからキャンセルいただけます。出荷後はキャンセルできず返品手続きとなります。",
        ["lookup_order", "search_policy"],
        "order_status",
        ["seed", "cancel"],
    ),
    record(
        "seed-0010",
        "ORD-1001 を返品したいです。手続きをお願いします。",
        "ORD-1001 は返品可能です (返品期限 2026-08-19)。返品申請を受け付けました。返送用ラベルをメールでお送りしますので、7 日以内にご発送ください。未使用かつ外箱・付属品が揃っていることが条件となります。",
        ["check_return_eligibility", "create_return_request"],
        "returns",
        ["seed", "returns", "side_effect"],
    ),
]

REGRESSION = [
    record(
        "reg-0001",
        "ORD-1001 の合計金額はいくらですか？",
        "ORD-1001 の合計金額は 12,800 円です (ワイヤレスイヤホン A2 × 1、2026-07-18 のご注文)。",
        ["lookup_order"],
        "order_status",
        ["regression", "order_status"],
    ),
    record(
        "reg-0002",
        "ORD-1002 の追跡番号を教えてください。",
        "ORD-1002 の追跡番号は 5512-8877-0011 (佐川急便) です。現在は厚木中継センターにあり、お届け予定日は 2026-08-09 です。",
        ["track_shipment"],
        "shipping",
        ["regression", "shipping"],
    ),
    record(
        "reg-0003",
        "ORD-1004 は返品できますか。",
        "ORD-1004 は返品期限 (2026-07-01) を超過しているため返品をお受けできません。初期不良の場合は個別に対応いたします。",
        ["check_return_eligibility"],
        "returns",
        ["regression", "returns"],
    ),
    record(
        "reg-0004",
        "返金にはどのくらい時間がかかりますか？",
        "返品商品の検品完了後、3 営業日以内に返金処理を行います。クレジットカードの場合はカード会社の締め日により反映まで最大 14 日程度、コンビニ払いの場合は口座確認後 5 営業日以内のお振り込みとなります。",
        ["search_policy"],
        "refund",
        ["regression", "policy"],
    ),
    record(
        "reg-0005",
        "注文番号がわからないのですが配送状況を知りたいです。",
        "配送状況の確認には注文 ID (ORD-1234 の形式) が必要です。ご購入時の確認メールに記載されていますので、お手数ですがご確認のうえお知らせください。",
        [],
        "shipping",
        ["regression", "missing_argument"],
    ),
    record(
        "reg-0006",
        "身に覚えのない請求があります。すぐ調べてください。",
        "身に覚えのないご請求について担当部署へ引き継ぎました。1 時間以内に担当者よりご連絡いたします。ご請求日と金額をお手元にご用意のうえお待ちください。",
        ["escalate_to_human"],
        "billing",
        ["regression", "billing", "escalation"],
    ),
    record(
        "reg-0007",
        "ORD-1005 の返金額を教えてください。",
        "ORD-1005 の返金額は 15,400 円です。現在処理中で、2026-08-15 までにクレジットカードへ返金予定です。",
        ["get_refund_status"],
        "refund",
        ["regression", "refund"],
    ),
    record(
        "reg-0008",
        "ORD-9999 の注文内容を教えてください。",
        "注文 ORD-9999 は見つかりませんでした。注文 ID は ORD-1234 の形式です。ご購入時の確認メールに記載の注文 ID をご確認ください。",
        ["lookup_order"],
        "order_status",
        ["regression", "not_found"],
    ),
]


def main() -> None:
    golden_count = write_jsonl(GOLDEN, GOLDEN_DATASET)
    regression_count = write_jsonl(REGRESSION, REGRESSION_DATASET)
    print(f"golden.jsonl     : {golden_count} records -> {GOLDEN_DATASET}")
    print(f"regression.jsonl : {regression_count} records -> {REGRESSION_DATASET}")


if __name__ == "__main__":
    main()
