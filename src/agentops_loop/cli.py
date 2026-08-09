"""CLI エントリポイント (`uv run agentops ...`)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

from .paths import ARTIFACTS_DIR, DATASET_CANDIDATES_DIR, GOLDEN_DATASET, ensure_dirs

app = typer.Typer(add_completion=False, help="AgentOps 評価ループ CLI")
dataset_app = typer.Typer(help="評価データセットの生成・キュレーション")
app.add_typer(dataset_app, name="dataset")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------- エージェント


@app.command()
def chat(
    message: str = typer.Argument(..., help="ユーザーからの問い合わせ本文"),
    tools_mode: str = typer.Option("mcp", help="mcp | local"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """エージェントを 1 回実行する (トレースは Langfuse へ送信される)."""
    _setup_logging(verbose)
    from .agent import OrderSupportAgent
    from .observability import flush_tracing, init_tracing, shutdown_tracing

    async def _run() -> None:
        init_tracing()
        async with OrderSupportAgent(tools_mode=tools_mode) as agent:
            result = await agent.arun(message, tags=["cli"])
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        flush_tracing()
        shutdown_tracing()

    asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="バインドアドレス"),  # noqa: S104
    port: int = typer.Option(8080),
) -> None:
    """本番エージェントの HTTP サービスを起動する."""
    import uvicorn

    uvicorn.run("agentops_loop.service:app", host=host, port=port, log_level="info")


@app.command("mcp-server")
def mcp_server() -> None:
    """注文サポート MCP サーバーを起動する (stdio)."""
    from .mcp_server.order_support import main

    main()


# --------------------------------------------------------------------------- データセット


@dataset_app.command("build")
def dataset_build(
    lookback_hours: int = typer.Option(24, help="取得対象のトレース期間"),
    max_records: int = typer.Option(50, help="生成する候補レコード数の上限"),
    limit: int = typer.Option(500, help="取得するトレース数の上限"),
    environment: str | None = typer.Option(None, help="対象の Langfuse environment"),
    output_dir: Path = typer.Option(DATASET_CANDIDATES_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Langfuse のテレメトリから評価データセット候補を生成する."""
    _setup_logging(verbose)
    ensure_dirs()
    from .dataset.build import build_candidates

    path, records = build_candidates(
        lookback_hours=lookback_hours,
        max_records=max_records,
        limit=limit,
        environment=environment,
        output_dir=output_dir,
    )
    typer.echo(f"候補 {len(records)} 件を書き出しました: {path}")
    if not records:
        typer.echo("新規候補がありません。")


@dataset_app.command("promote")
def dataset_promote(
    candidate_path: Path = typer.Argument(..., help="候補 JSONL のパス"),
    golden_path: Path = typer.Option(GOLDEN_DATASET),
) -> None:
    """候補のうち review.status=approved のレコードを golden へ昇格させる."""
    _setup_logging()
    from .dataset.promote import promote_candidates

    added, total = promote_candidates(candidate_path, golden_path)
    typer.echo(f"{added} 件を追加しました (合計 {total} 件): {golden_path}")


@dataset_app.command("stats")
def dataset_stats(dataset_path: Path = typer.Option(GOLDEN_DATASET)) -> None:
    """データセットの統計を表示する."""
    from .dataset.schema import read_jsonl

    records = read_jsonl(dataset_path)
    categories: dict[str, int] = {}
    for record in records:
        key = str(record.classification.get("category", "unknown"))
        categories[key] = categories.get(key, 0) + 1
    typer.echo(
        json.dumps(
            {
                "path": str(dataset_path),
                "total": len(records),
                "with_expected_output": sum(1 for r in records if r.expected_output),
                "with_expected_tools": sum(1 for r in records if r.expected_tools),
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# --------------------------------------------------------------------------- 評価 / 最適化


@app.command()
def evaluate(
    dataset_path: Path = typer.Option(GOLDEN_DATASET, "--dataset"),
    label: str = typer.Option("golden-evaluation"),
    output_dir: Path = typer.Option(ARTIFACTS_DIR, "--output"),
    rerun: bool = typer.Option(
        False, help="現在のエージェント設定で再実行してから評価する (既定は記録済み出力を評価)"
    ),
    tools_mode: str = typer.Option("local"),
    concurrency: int = typer.Option(4),
    push_scores: bool = typer.Option(False, help="スコアを Langfuse へ書き戻す"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """DeepEval でデータセットを採点し、スコアと理由をレポート出力する."""
    _setup_logging(verbose)
    ensure_dirs()
    from .agent.candidate import AgentCandidate
    from .dataset.schema import read_jsonl
    from .evaluation.runner import evaluate_records, refresh_records
    from .observability import init_tracing

    if push_scores:
        init_tracing()

    records = read_jsonl(dataset_path)
    if not records:
        typer.secho(f"データセットが空です: {dataset_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    candidate = AgentCandidate.from_repo()
    if rerun:
        records = refresh_records(records, candidate, tools_mode, concurrency)

    try:
        report = evaluate_records(
            records,
            label=label,
            push_scores=push_scores,
            max_concurrency=concurrency,
            candidate_keys=sorted(candidate.components),
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    output_dir = Path(output_dir)
    report.save(output_dir / "evaluation_report.json")
    (output_dir / "evaluation_report.md").write_text(report.render_markdown(), encoding="utf-8")
    typer.echo(report.render_markdown())


@app.command()
def optimize(
    dataset_path: Path = typer.Option(GOLDEN_DATASET, "--dataset"),
    output_dir: Path = typer.Option(ARTIFACTS_DIR, "--output"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """GEPA でエージェント (プロンプト / スキル / ツール / モデル) を最適化する."""
    _setup_logging(verbose)
    ensure_dirs()
    from .optimization.run import run_optimization

    _, outcome = run_optimization(dataset_path=dataset_path, output_dir=output_dir)
    typer.echo(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
    if not outcome.accepted:
        typer.secho("最適化候補は受け入れ基準を満たしませんでした。", fg=typer.colors.YELLOW)


@app.command("apply-candidate")
def apply_candidate(
    candidate_path: Path = typer.Argument(..., help="optimized_candidate.json のパス"),
) -> None:
    """最適化された候補を prompts/ · skills/ · config/agent.yaml へ書き戻す."""
    _setup_logging()
    from .agent.candidate import AgentCandidate

    seed = AgentCandidate.from_repo()
    candidate = AgentCandidate.load_json(candidate_path)
    candidate.meta = seed.meta
    written = candidate.write_to_repo()
    for path in written:
        typer.echo(f"updated: {path}")


@app.command()
def regression(
    candidate_path: Path | None = typer.Option(None, "--candidate"),
    baseline_report: Path | None = typer.Option(None, "--baseline"),
    output_dir: Path = typer.Option(ARTIFACTS_DIR, "--output"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """最適化後の候補に対して回帰テストを実行する (失敗時は非 0 で終了)."""
    _setup_logging(verbose)
    ensure_dirs()
    from .regression.run import run_regression

    _, outcome = run_regression(
        candidate_path=candidate_path,
        baseline_report_path=baseline_report,
        output_dir=output_dir,
    )
    typer.echo(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
    if not outcome.passed:
        typer.secho("回帰テストに失敗しました。", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("回帰テストに合格しました。", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
