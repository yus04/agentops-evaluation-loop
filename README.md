# AgentOps Evaluation Loop

LangGraph で構築した**注文サポートエージェント**に対して、
**観測 (Langfuse / OpenTelemetry) → 評価 (DeepEval) → 最適化 (GEPA) → 回帰テスト → デプロイ**
までを一周させる、エージェント改善ループのリファレンス実装です。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [ディレクトリ構成](#3-ディレクトリ構成)
4. [事前準備](#4-事前準備)
5. [ローカルでの利用手順](#5-ローカルでの利用手順)
6. [コンテナでの本番運用](#6-コンテナでの本番運用)
7. [GitHub Actions パイプライン](#7-github-actions-パイプライン)
8. [評価設計](#8-評価設計)
9. [GEPA による最適化設計](#9-gepa-による最適化設計)
10. [設定リファレンス](#10-設定リファレンス)
11. [トラブルシューティング](#11-トラブルシューティング)

---

## 1. プロジェクト概要

エージェントは「作って終わり」ではなく、本番の挙動を観測し、定量評価し、改善を反映し続ける
運用ループが必要です。本プロジェクトはそのループを次の OSS 構成で実装しています。

| フェーズ | 使用 OSS | 役割 |
| --- | --- | --- |
| エージェント | **LangGraph** | 分類 → ツール選択 → ツール実行 → 検証 → 回答生成のグラフ |
| ツール提供 | **MCP (FastMCP)** | 注文照会・配送追跡・返品・返金・ポリシー検索・エスカレーション |
| 観測 | **Langfuse + OpenTelemetry** | トレース (入力・分類・ツール入出力・検証・回答) とメトリクス (レイテンシー・トークン・コスト・エラー) |
| 評価 | **DeepEval** | LLM-as-a-Judge による**スコア + 理由**の生成 |
| 最適化 | **GEPA** | インストラクション / スキル / ツール / モデル選択の進化的最適化 |
| 回帰テスト | **DeepEval** | 最適化候補のデグレ検知ゲート |
| CI/CD | **GitHub Actions + Artifacts** | 4 本のパイプラインと 2 か所の人間レビュー (PR) |

### 全体像

```text
【オンライン運用フェーズ】

  ユーザー
     ↓
  本番 Agent (Docker コンテナ / FastAPI)
     ↓  OpenTelemetry
  Langfuse
     ↓
  Telemetry 蓄積 (Trace / Metrics / Scores)

=====================================================
【人間による確認フェーズ】 ※ 定期実行 (日次)
=====================================================

  01 Dataset Curation (GitHub Actions / cron)
     ↓  テレメトリ → 評価データセット候補 (JSONL)
  Pull Request
     ↓  人間によるレビュー・データキュレーション
     ↓  expected_output / expected_tools の付与
  PR マージ  → 評価データセット確定 (data/datasets/golden.jsonl)

=====================================================
【改善フェーズ (バッチ)】 ※ 02 → 03 → 04 はチェーン実行
=====================================================

  02 Batch Evaluation (DeepEval)
     ↓  Score + Failure Reason        ── GitHub Artifacts ──┐
  03 GEPA Optimization                                      │
     ↓  改善候補生成 (prompts / skills / tools / models)     │
     ↓                                ── GitHub Artifacts ──┤
  04 Regression Test (DeepEval)                             │
     ↓  ゲート通過                     ←────────────────────┘
  Pull Request
     ↓  人間によるレビュー
  マージ = デプロイ → 本番 Agent へ反映
```

> **なぜ 01 と 02 をチェーンしないのか**
> 01 と 02 の間には「データキュレーション」という人間の判断が必ず入るためです。
> 01 は PR を出して止まり、人間が PR をマージした時点 (`golden.jsonl` の変更) で 02 が起動します。
> 一方 02 → 03 → 04 は機械的な処理なので `workflow_run` でチェーンします。

---

## 2. アーキテクチャ

### 2.1 エージェント (LangGraph)

```text
ユーザー
  ↓
LangGraph Agent
  ├─ classify        問い合わせ分類 (category / priority / entities)
  ├─ select_tools    必要な MCP tool を選択 (ツール名 + 引数 + 理由)
  ├─ execute_tools   MCP tool 実行 (タイムアウト・引数サニタイズ・呼び出し上限)
  ├─ validate        tool 結果を検証 (根拠が揃っているか / 再取得が必要か)
  │                     └─ 不足なら select_tools へ戻る (最大 N 回)
  └─ respond         回答生成 (検証済みの事実のみを使用)
        ↓
  Langfuse trace, metrics
```

実装:

- グラフ定義: [src/agentops_loop/agent/graph.py](src/agentops_loop/agent/graph.py)
- ノード実装: [src/agentops_loop/agent/nodes.py](src/agentops_loop/agent/nodes.py)
- 状態定義: [src/agentops_loop/agent/state.py](src/agentops_loop/agent/state.py)
- MCP サーバー: [src/agentops_loop/mcp_server/order_support.py](src/agentops_loop/mcp_server/order_support.py)

### 2.2 最適化しやすさのための設計

エージェントの挙動を決める要素をすべて**テキストのコンポーネント**に外出しし、
`AgentCandidate` という 1 つの辞書 (`dict[str, str]`) で表現しています。
GEPA はこの辞書をそのまま進化させます。

| コンポーネント | 実体ファイル | 内容 |
| --- | --- | --- |
| `instructions/classify` | `prompts/classify.md` | 分類ノードのシステムプロンプト |
| `instructions/select_tools` | `prompts/select_tools.md` | ツール選択ノードのシステムプロンプト |
| `instructions/validate` | `prompts/validate.md` | 検証ノードのシステムプロンプト |
| `instructions/respond` | `prompts/respond.md` | 回答生成ノードのシステムプロンプト |
| `skills/<name>` | `skills/*.md` | カテゴリ別の対応手順 (スキルセット) |
| `policy/tools` | `config/agent.yaml` の `tools` | 有効ツール / ツール説明文 / 呼び出し上限 |
| `policy/models` | `config/agent.yaml` の `models` | ノードごとのモデル選択 / temperature |

**ガードレール**: `policy/*` は YAML テキストなので LLM が壊れた出力を書く可能性があります。
そのため [candidate.py](src/agentops_loop/agent/candidate.py) では

- `yaml.safe_load` のみ使用 (任意オブジェクトの生成を許さない)
- パース失敗時はシード値へフォールバック
- モデル名は `allowed_models` の allowlist で検証
- ツール名は `tool_catalog` に存在するものだけ採用
- `max_tool_calls` / `temperature` は範囲でクランプ

としており、最適化が壊れた設定を本番に持ち込めないようにしています。

### 2.3 観測 (Langfuse / OpenTelemetry)

Langfuse Python SDK (v3 以降) は内部が OpenTelemetry で実装されています。
`CallbackHandler` を LangGraph に渡すことで、ノード単位の span と LLM generation
(モデル名・トークン数・レイテンシー・コスト) が自動で記録されます。

これに加えて [tracing.py](src/agentops_loop/observability/tracing.py) が
**オフライン評価に必要な業務属性**をトレース metadata (`agentops.*`) として明示的に添付します。

| トレース属性 | 内容 |
| --- | --- |
| `input.user_input` | ユーザー入力 |
| `metadata.agentops.classification` | 分類結果 (category / priority / entities / confidence) |
| `metadata.agentops.tool_plan` | MCP tool 選択計画 (ツール名 / 引数 / 理由) |
| `metadata.agentops.tool_executions` | ツール入力・出力・エラー・レイテンシー |
| `metadata.agentops.validation` | 検証結果 (sufficient / issues / grounded_facts) |
| `output.final_answer` | 最終回答 |
| `metadata.agentops.metrics` | レイテンシー / トークン / 概算コスト / ツール呼び出し数 / エラー数 |

Langfuse が未設定の場合はすべて no-op になり、エージェントは動作し続けます。

---

## 3. ディレクトリ構成

```text
agentops-evaluation-loop/
├── README.md
├── pyproject.toml                     # uv プロジェクト定義 (extras: eval / optimize)
├── Dockerfile                         # 本番エージェント用コンテナ (評価系依存は含めない)
├── .env.example
│
├── config/                            # 設定 (一部は GEPA 最適化対象)
│   ├── agent.yaml                     #   ツールポリシー / モデルポリシー / allowlist
│   ├── evaluation.yaml                #   DeepEval メトリクス定義と閾値・重み
│   ├── optimization.yaml              #   GEPA 探索設定と受け入れ基準
│   ├── regression.yaml                #   回帰テストのゲート
│   └── pricing.yaml                   #   モデル単価 (コスト概算用)
│
├── prompts/                           # ★最適化対象: インストラクション
│   ├── classify.md
│   ├── select_tools.md
│   ├── validate.md
│   └── respond.md
│
├── skills/                            # ★最適化対象: スキルセット
│   ├── order_status.md
│   ├── shipping.md
│   ├── returns.md
│   ├── refund.md
│   ├── billing.md
│   └── escalation.md
│
├── data/
│   ├── seed/orders.json               # デモ用の業務データ (MCP バックエンド)
│   └── datasets/
│       ├── golden.jsonl               # 人間レビュー済みの評価データセット
│       ├── regression.jsonl           # 回帰テスト用データセット
│       └── candidates/                # 01 が生成するレビュー待ち候補
│
├── src/agentops_loop/
│   ├── cli.py                         # `agentops` CLI
│   ├── service.py                     # FastAPI (本番エンドポイント)
│   ├── settings.py                    # 設定 (エージェント / Judge / reflection を分離)
│   ├── llm.py                         # チャットモデルファクトリ
│   ├── pricing.py                     # コスト概算
│   ├── agent/                         # LangGraph エージェント
│   │   ├── candidate.py               #   ★最適化対象の抽象化
│   │   ├── graph.py / nodes.py / state.py / tools.py
│   ├── mcp_server/                    # MCP サーバー + 業務ロジック
│   ├── observability/tracing.py       # Langfuse / OTel
│   ├── dataset/                       # テレメトリ → 評価データセット
│   │   ├── schema.py / build.py / promote.py
│   ├── evaluation/                    # DeepEval
│   │   ├── judge.py / metrics.py / runner.py
│   ├── optimization/                  # GEPA
│   │   ├── adapter.py / run.py
│   └── regression/run.py              # 回帰テストゲート
│
├── scripts/
│   ├── seed_datasets.py            # 初期データセット生成
│   └── smoke_check.py              # LLM を呼ばずに配線を検証
├── tests/                             # ユニットテスト
└── .github/workflows/
    ├── 01-dataset-curation.yml        # cron (日次) → PR
    ├── 02-evaluation.yml              # golden.jsonl 更新で起動
    ├── 03-optimization.yml            # 02 成功でチェーン
    └── 04-regression.yml              # 03 成功でチェーン → PR
```

---

## 4. 事前準備

### 4.1 必要なもの

| 項目 | 内容 |
| --- | --- |
| Python | 3.11 〜 3.13 |
| [uv](https://docs.astral.sh/uv/) | パッケージ / 仮想環境管理 |
| Docker | 本番コンテナのビルド (任意) |
| LLM API キー | 既定は OpenAI。`init_chat_model` 経由なので他プロバイダにも変更可 |
| Langfuse プロジェクト | [Langfuse Cloud](https://cloud.langfuse.com) または self-host |

### 4.2 環境変数

`.env.example` をコピーして `.env` を作成します。

```bash
cp .env.example .env
```

重要な設定:

| 変数 | 説明 |
| --- | --- |
| `OPENAI_API_KEY` | LLM プロバイダの API キー |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse の認証情報 |
| `LANGFUSE_BASE_URL` | EU: `https://cloud.langfuse.com` / US: `https://us.cloud.langfuse.com` / JP: `https://jp.cloud.langfuse.com` |
| `AGENT_TOOLS_MODE` | `mcp` (本番) / `local` (CI・最適化ループ) |
| `JUDGE_MODEL` | **評価用モデル。エージェント用とは必ず別にする** |
| `REFLECTION_MODEL` | GEPA が改善候補を書くときのモデル |
| `AGENT_API_KEY` | 設定すると `/v1/support` に `X-API-Key` が必須になる |

> **モデル分離の理由**
> LLM-as-a-Judge をエージェントと同じモデルにすると、自分の出力形式に有利なバイアスが乗り、
> スコアが実運用品質と乖離します。本プロジェクトはエージェント / Judge / reflection の
> 3 系統を独立に設定できるようにしています ([settings.py](src/agentops_loop/settings.py))。

### 4.3 GitHub リポジトリ設定 (パイプラインを使う場合)

ワークフローは `${{ secrets.* }}` / `${{ vars.* }}` を **リポジトリレベル** で参照します。
Environment (環境) 配下に登録した場合はジョブに `environment:` の宣言が必要になるため、
特別な理由がなければ下記のリポジトリレベルに登録してください。

#### 4.3.1 Secrets の登録手順

1. GitHub のリポジトリページ → **Settings**
2. 左メニュー **Secrets and variables** → **Actions**
3. **Secrets** タブ (Repository secrets) → **New repository secret**
4. 下表の Name / Secret を入力して **Add secret**

| Name | 用途 | 必須 |
| --- | --- | --- |
| `OPENAI_API_KEY` | エージェント / Judge / reflection の LLM 呼び出し | ✅ (02・03・04) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse からのトレース取得 / スコア書き戻し | ✅ (01・02) |
| `LANGFUSE_SECRET_KEY` | 同上 | ✅ (01・02) |

> `GITHUB_TOKEN` は GitHub が自動発行するため登録不要です。

#### 4.3.2 Variables の登録手順

1. 同じ画面の **Variables** タブ → **New repository variable**
2. 下表の Name / Value を入力して **Add variable**

いずれも未設定の場合はワークフロー側の既定値が使われるため、**すべて任意**です。

| Name | 既定値 | 説明 |
| --- | --- | --- |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | US: `https://us.cloud.langfuse.com` / JP: `https://jp.cloud.langfuse.com` |
| `LANGFUSE_TRACING_ENVIRONMENT` | `production` | 01 が収集対象とする environment |
| `JUDGE_PROVIDER` | `openai` | 評価 (LLM-as-a-Judge) のプロバイダ |
| `JUDGE_MODEL` | `gpt-5` | 評価モデル。**エージェント用とは必ず別にする** |
| `REFLECTION_PROVIDER` | `openai` | GEPA reflection LM のプロバイダ |
| `REFLECTION_MODEL` | `gpt-5` | GEPA が改善候補を書くモデル |

#### 4.3.3 ワークフロー別の必要変数

| ワークフロー | 必要な Secrets | 参照する Variables |
| --- | --- | --- |
| 01 Dataset Curation | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT` |
| 02 Evaluation | `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `LANGFUSE_BASE_URL`, `JUDGE_PROVIDER`, `JUDGE_MODEL` |
| 03 Optimization | `OPENAI_API_KEY` | `JUDGE_*`, `REFLECTION_*` |
| 04 Regression | `OPENAI_API_KEY` | `JUDGE_PROVIDER`, `JUDGE_MODEL` |

#### 4.3.4 Azure OpenAI (Microsoft Foundry) を使う場合

ワークフローは OpenAI / Azure OpenAI の両方に対応済みです。Azure を使う場合は
`OPENAI_API_KEY` の代わりに下記を登録してください (ワークフローの編集は不要)。

| 種別 | Name | 例 |
| --- | --- | --- |
| Secret | `AZURE_OPENAI_API_KEY` | Azure ポータルのキー |
| Variable | `AZURE_OPENAI_ENDPOINT` | `https://<resource-name>.openai.azure.com/` |
| Variable | `OPENAI_API_VERSION` | `2025-04-01-preview` |
| Variable | `AGENT_PROVIDER` | `azure_openai` |
| Variable | `JUDGE_PROVIDER` | `azure_openai` |
| Variable | `REFLECTION_PROVIDER` | `azure_openai` |
| Variable | `JUDGE_MODEL` / `REFLECTION_MODEL` | Azure の**デプロイ名** |

> `JUDGE_MODEL` / `REFLECTION_MODEL` には、Azure OpenAI の「モデル名」ではなく
> **デプロイ名** を指定してください。

#### 4.3.5 リポジトリ権限

01 と 04 は PR を自動作成します。既定では `GITHUB_TOKEN` に PR 作成権限が無いため、
以下を有効にしないと **候補ブランチの push までは成功するが PR 作成で失敗します**。

1. リポジトリの **Settings** を開く
2. 左メニュー **Actions** → **General**
3. ページ下部の **Workflow permissions** セクションまでスクロール
4. **Allow GitHub Actions to create and approve pull requests** にチェックを入れる
5. **Save** をクリック

> 失敗時のログには `Create or update the pull request` ステップで
> `GitHub Actions is not permitted to create or approve pull requests` が出ます。

---

## 5. ローカルでの利用手順

### 5.1 セットアップ

```bash
# 依存インストール (agent + eval + optimize)
uv sync --all-extras

# 初期データセットの生成 (Langfuse にテレメトリが無い状態でも一周できる)
uv run python scripts/seed_datasets.py
```

### 5.2 エージェントを 1 回動かす

```bash
# MCP サーバーを子プロセスとして起動して実行 (本番と同じ経路)
uv run agentops chat "注文 ORD-1002 がまだ届きません。今どこにありますか？"

# MCP を使わずローカル実装で実行 (高速・CI 向け)
uv run agentops chat "ORD-1001 を返品したいです" --tools-mode local
```

実行すると Langfuse にトレースが 1 本作られます。

### 5.3 HTTP サービスとして起動

```bash
uv run agentops serve --port 8080

curl -s http://localhost:8080/v1/support \
  -H 'Content-Type: application/json' \
  -d '{"message":"ORD-1005 の返金はいつですか？","user_id":"u-001"}' | jq
```

### 5.4 テレメトリから評価データセット候補を作る

```bash
uv run agentops dataset build --lookback-hours 24 --max-records 50
# -> data/datasets/candidates/YYYYMMDD.jsonl  (候補)
# -> data/datasets/candidates/YYYYMMDD.md     (レビュー用サマリ)
```

候補 JSONL を人間がレビューし、次を記入します。

```jsonc
{
  "id": "rec-xxxx",
  "input": "ORD-1002 がまだ届きません",
  "actual_output": "…本番エージェントの回答…",
  "expected_output": "…理想の回答 (任意)…",      // ← 記入
  "expected_tools": ["track_shipment"],           // ← 記入
  "expected_category": "shipping",                // ← 記入
  "review": { "status": "approved" }              // ← approved / rejected
}
```

承認済みレコードを golden へ昇格:

```bash
uv run agentops dataset promote data/datasets/candidates/<YYYYMMDD>.jsonl
uv run agentops dataset stats
```

### 5.5 評価する

```bash
# 記録済み出力をそのまま評価 (本番トレースの事後評価)
uv run agentops evaluate --dataset data/datasets/golden.jsonl

# 現在のエージェント設定で再実行してから評価 (最適化前の実力測定)
uv run agentops evaluate --dataset data/datasets/golden.jsonl --rerun --tools-mode local
```

`artifacts/evaluation_report.json` (スコア + 理由) と `artifacts/evaluation_report.md` が出力されます。
`--push-scores` を付けると Langfuse のトレースにスコアが書き戻され、ダッシュボードで時系列比較できます。

### 5.6 GEPA で最適化する

```bash
uv run agentops optimize --dataset data/datasets/golden.jsonl --output artifacts
```

出力:

| ファイル | 内容 |
| --- | --- |
| `artifacts/optimized_candidate.json` | 最適化後のコンポーネント一式 |
| `artifacts/baseline_report.json` | 最適化前の評価結果 |
| `artifacts/optimized_report.json` | 最適化後の評価結果 |
| `artifacts/optimization_summary.md` | 差分サマリ (採用可否・メトリクス変化・変更コンポーネント) |

### 5.7 回帰テストして反映する

```bash
uv run agentops regression \
  --candidate artifacts/optimized_candidate.json \
  --baseline artifacts/baseline_report.json

# 合格したらリポジトリのファイルへ書き戻す
uv run agentops apply-candidate artifacts/optimized_candidate.json
git diff   # prompts/ skills/ config/agent.yaml の差分を確認
```

回帰テストが不合格の場合、コマンドは終了コード 1 で終わります (CI ゲートとして機能)。

### 5.8 テスト・静的解析

```bash
uv run pytest -q
uv run ruff check .

# LLM API を呼ばずにパイプライン全体の配線を確認する
OPENAI_API_KEY=dummy uv run python scripts/smoke_check.py
```

`tests/test_agent_graph.py` は LLM をスタブに差し替えて LangGraph の
分岐・ツール実行・呼び出し上限・無効ツールのフィルタを検証します (API キー不要)。

---

## 6. コンテナでの本番運用

本番コンテナには**評価 (deepeval) / 最適化 (gepa) の依存を含めません**。
それらはバッチ (GitHub Actions) の責務です。

```bash
docker build -t order-support-agent:latest .

docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY" \
  -e LANGFUSE_SECRET_KEY="$LANGFUSE_SECRET_KEY" \
  -e LANGFUSE_BASE_URL="https://cloud.langfuse.com" \
  -e AGENT_TOOLS_MODE=mcp \
  -e AGENT_ENVIRONMENT=production \
  order-support-agent:latest
```

コンテナの特徴:

- マルチステージビルド (uv でビルド → `python:3.12-slim` へ venv のみコピー)
- **非 root ユーザー** (`agent`, uid 10001) で実行
- `HEALTHCHECK` で `/healthz` を監視
- `prompts/` `skills/` `config/` を同梱するため、**GEPA 最適化の成果 (マージされた PR) が
  そのまま次のイメージビルドで本番に反映される**

エンドポイント:

| メソッド | パス | 説明 |
| --- | --- | --- |
| `GET` | `/healthz` | 死活監視 |
| `GET` | `/readyz` | 有効ツール・モデル割当の確認 |
| `POST` | `/v1/support` | 問い合わせ対応 (`AGENT_API_KEY` 設定時は `X-API-Key` 必須) |

---

## 7. GitHub Actions パイプライン

| # | ワークフロー | トリガー | 出力 (GitHub Artifacts) |
| --- | --- | --- | --- |
| 01 | [Dataset Curation](.github/workflows/01-dataset-curation.yml) | **cron 日次** (18:00 UTC = 03:00 JST) + 手動 | `dataset-candidates` + **PR 作成** |
| 02 | [Batch Evaluation](.github/workflows/02-evaluation.yml) | `main` の `golden.jsonl` 更新 (= 01 の PR マージ) + 手動 | `evaluation-report` |
| 03 | [GEPA Optimization](.github/workflows/03-optimization.yml) | **02 の成功時** (`workflow_run`) | `optimized-candidate` |
| 04 | [Regression Test & Deploy PR](.github/workflows/04-regression.yml) | **03 の成功時** (`workflow_run`) | `regression-report` + **PR 作成** |

### データの受け渡し

パイプライン間のデータはすべて **GitHub Artifacts** 経由です。
`workflow_run` で起動したワークフローは、`actions/download-artifact` の
`run-id` + `github-token` で前段の成果物を取得します。

```yaml
- uses: actions/download-artifact@v4
  with:
    name: evaluation-report
    path: artifacts
    run-id: ${{ github.event.workflow_run.id }}
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

```text
01 ──[dataset-candidates]──▶ PR ──(人間レビュー/マージ)──▶ 02
02 ──[evaluation-report]───▶ 03 ──[optimized-candidate]──▶ 04 ──[regression-report]──▶ PR
```

### 人間レビューが入る 2 か所

1. **データキュレーション PR** (01 の出力) — 何を「正解」とするかの合意形成
2. **デプロイ PR** (04 の出力) — 最適化されたプロンプト / スキル / ツール / モデルの妥当性確認

04 は次の条件をすべて満たしたときのみ PR を作成します。

- GEPA の最適化候補が受け入れ基準を満たしている (`optimization_summary.json` の `accepted == true`)
- DeepEval の回帰テストがすべてのゲートを通過している

---

## 8. 評価設計

評価は 3 つの観点に整理しています。定義は
[config/evaluation.yaml](config/evaluation.yaml)、実装は
[metrics.py](src/agentops_loop/evaluation/metrics.py) です。

| 観点 | メトリクス | 種別 | 見ているもの |
| --- | --- | --- | --- |
| **回答品質** | Answer Relevancy | DeepEval 標準 | 質問に対する回答の関連性 |
| | Faithfulness | DeepEval 標準 | **根拠性** — ツール出力に裏づけのない主張 (ハルシネーション) の検知 |
| | Answer Correctness | GEval (カスタム) | 理想回答との事実整合 (`expected_output` がある場合のみ) |
| **行動品質** | Tool Correctness | DeepEval 標準 (決定的) | `expected_tools` との一致 |
| | **Tool Selection** | **GEval (カスタム)** | ツール選択の妥当性・引数の正しさ・副作用ツールの前提確認 |
| **業務価値** | Task Completion | GEval (カスタム) | 1 往復でユーザーの課題が解決したか (タスク完了率) |
| | Policy Compliance | GEval (カスタム) | 業務ルール順守 (根拠のない約束をしない・内部情報を漏らさない) |
| | Operational Efficiency | **独自 BaseMetric (非 LLM)** | レイテンシー / コスト / ツール呼び出し数 / エラー |

カスタムエバリュエーターの例 (`config/evaluation.yaml` の `metrics.tool_selection.criteria` から生成):

```python
GEval(
    name="Tool Selection",
    criteria="""
    Evaluate whether the agent selected
    the correct tool to complete the task.
    ...
    """,
    evaluation_params=[INPUT, ACTUAL_OUTPUT, TOOLS_CALLED],
    model=judge_model,   # ← エージェントとは別モデル
)
```

`Operational Efficiency` は LLM を使わない決定的メトリクスです。
GEPA が「精度は上がったがコストとレイテンシーが爆発した」候補を選ばないようにするための
ガードとして総合スコアに組み込んでいます。

すべてのメトリクスは **score と reason (スコア付けの理由)** を返し、
`artifacts/evaluation_report.json` に記録されます。この reason が次段の GEPA の入力になります。

### 総合スコアの算出

```text
overall_score = Σ(metric_score × weight) / Σ(weight)
```

`weight` は `config/evaluation.yaml` で調整します (例: Task Completion 2.0、Faithfulness 1.5)。

---

## 9. GEPA による最適化設計

GEPA (Genetic-Pareto) は、評価スコアだけでなく**失敗理由のテキスト**を LLM に読ませて
「なぜ失敗したか」を診断し、テキストパラメータを改善していく最適化手法です。

本プロジェクトのアダプタ ([adapter.py](src/agentops_loop/optimization/adapter.py)) は次の 2 つを実装しています。

```python
class AgentGEPAAdapter(GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        # 1. candidate でエージェントを再実行
        # 2. DeepEval で採点
        # 3. スコアを返す
        ...

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # DeepEval の score + reason を reflection LM 向けの教材へ整形
        # -> "Inputs" / "Generated Outputs" / "Feedback"
        ...
```

### 最適化ループ

```text
golden.jsonl ──split──▶ trainset / valset
                          │
      ┌───────────────────┴──────────────────────────┐
      │ 1. Pareto フロンティアから candidate を選択    │
      │ 2. minibatch でエージェント実行 (rollout)      │
      │ 3. DeepEval で採点 → score + failure reason   │
      │ 4. reflection LM が原因を診断し改善案を生成    │
      │ 5. スコアが改善したら候補プールへ追加          │
      └───────────────────┬──────────────────────────┘
                          ▼
              optimized_candidate.json
```

### 最適化対象と、渡される制約

`make_reflective_dataset` の Feedback には、コンポーネントごとの
**役割説明・制約・改善の着眼点**が併記されます (`component_guidance`)。例:

- `policy/models`: 「使用できるモデル名は `["gpt-5-mini", "gpt-5", ...]` に限られます。
  安価なモデルで十分なノードは安価に、品質が要求されるノードは高性能モデルにしてください」
- `policy/tools`: 「ツール説明文が曖昧だと選択を誤ります。いつ使う/使わないかを説明文に書いてください」
- `instructions/validate`: 「Faithfulness が低い場合は検証の甘さが原因です」

### 受け入れ基準

`config/optimization.yaml` の `acceptance` を満たさない候補は PR になりません。

```yaml
acceptance:
  min_absolute_improvement: 0.005   # 総合スコアの最低改善幅
  max_metric_regression: 0.05       # 個別メトリクスの劣化許容幅
```

---

## 10. 設定リファレンス

| ファイル | 用途 | 主なキー |
| --- | --- | --- |
| [config/agent.yaml](config/agent.yaml) | エージェント挙動 (★一部が最適化対象) | `tools.enabled`, `tools.descriptions`, `models.per_node`, `allowed_models`, `tool_catalog` |
| [config/evaluation.yaml](config/evaluation.yaml) | 評価メトリクス | `judge.model`, `metrics.*.threshold`, `metrics.*.weight`, `metrics.*.criteria` |
| [config/optimization.yaml](config/optimization.yaml) | GEPA 探索 | `gepa.max_metric_calls`, `dataset.train_ratio`, `components`, `acceptance.*` |
| [config/regression.yaml](config/regression.yaml) | 回帰ゲート | `gates.max_metric_regression`, `gates.min_metric_scores`, `gates.max_avg_cost_usd` |
| [config/pricing.yaml](config/pricing.yaml) | コスト概算単価 | `pricing.<model>.input/output` |

### CLI 一覧

```bash
uv run agentops chat "<message>"                     # エージェントを 1 回実行
uv run agentops serve --port 8080                    # HTTP サービス起動
uv run agentops mcp-server                           # MCP サーバー単体起動 (stdio)
uv run agentops dataset build [--lookback-hours 24]  # テレメトリ → 候補生成
uv run agentops dataset promote <candidate.jsonl>    # approved を golden へ昇格
uv run agentops dataset stats                        # データセット統計
uv run agentops evaluate [--rerun] [--push-scores]   # DeepEval 評価
uv run agentops optimize                             # GEPA 最適化
uv run agentops apply-candidate <candidate.json>     # 候補をリポジトリへ書き戻し
uv run agentops regression --candidate ... --baseline ...  # 回帰テスト
```

---

## 11. トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `Langfuse は未設定のためトレースを送信しません` | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` が未設定。設定してもエージェントは動作します |
| `MCP サーバーの起動に失敗したため local モードへフォールバックします` | MCP プロセスを起動できない環境。`AGENT_TOOLS_MODE=local` を明示すると警告が消えます |
| `最適化には最低 4 件の評価データが必要です` | `golden.jsonl` の件数不足。`scripts/seed_datasets.py` または 01 パイプラインでデータを増やしてください |
| `評価対象のレコードがありません` | 全レコードが `review.status: rejected` になっている可能性があります |
| GEPA 実行時に引数エラー | `gepa` のバージョン差は `run.py` の `_call_gepa` が吸収します。ログの「未対応の引数を無視します」を確認してください |
| PR が作成されない (Actions) | Settings → Actions → General の「Allow GitHub Actions to create and approve pull requests」を有効化 |
| 評価コストが高い | `config/optimization.yaml` の `gepa.max_metric_calls` と `dataset.max_records`、`config/evaluation.yaml` の Judge モデルを見直してください |

---

## ライセンス

MIT
