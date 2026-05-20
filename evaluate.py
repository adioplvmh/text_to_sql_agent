"""
Full evaluation runner for the Text-to-SQL agent.

Usage:
    poetry run python evaluate.py
    poetry run python evaluate.py --level 3       # only L3 queries
    poetry run python evaluate.py --limit 20      # first 20 queries
    poetry run python evaluate.py --output results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track
from rich.table import Table

from text_to_sql_adk.core.tools.schema_inspector import get_schema_context
from text_to_sql_adk.core.tools.sql_executor import execute_sql
from text_to_sql_adk.core.tools.sql_evaluator import evaluate_sql

log = logging.getLogger(__name__)
console = Console()

DATASET_PATH = Path(__file__).parent / "data" / "dataset.jsonl"


def load_dataset(
    level_filter: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            rec = json.loads(line.strip())
            if level_filter and rec["level"] != level_filter:
                continue
            records.append(rec)
            if limit and len(records) >= limit:
                break
    return records


def run_agent_on_query(schema_context: str, record: dict) -> tuple[str, dict, dict]:
    """
    Run the Text-to-SQL agent on a single record.
    Returns (generated_sql, gen_result, eval_metrics).
    """
    from text_to_sql_adk.core.tools.sql_executor import execute_sql
    from text_to_sql_adk.core.tools.sql_evaluator import evaluate_sql

    # For evaluation purposes we call the generation agent directly via the
    # google-adk synchronous API. In full agentic mode use app.stream_query().
    try:
        from text_to_sql_adk.agent import create_text_to_sql_agent
        # Direct synchronous call — simpler for batch evaluation
        import vertexai
        from vertexai.agent_engines import AdkApp
        agent = create_text_to_sql_agent()
        adk_app = AdkApp(agent=agent)

        generated_sql = None
        response_text = ""

        async def _run():
            nonlocal generated_sql, response_text
            async for event in adk_app.async_stream_query(
                user_id="evaluator",
                message=record["nl_query"],
            ):
                if hasattr(event, "text"):
                    response_text += event.text or ""
                # Try to extract SQL from tool call events
                if hasattr(event, "tool_calls"):
                    for tc in event.tool_calls:
                        if tc.get("name") == "execute_sql_tool":
                            generated_sql = tc.get("args", {}).get("sql_query")

        asyncio.run(_run())

        # Fall back: extract first code block from response text
        if not generated_sql:
            import re
            m = re.search(r"```(?:sql)?\n(.+?)```", response_text, re.DOTALL)
            if m:
                generated_sql = m.group(1).strip()

    except Exception as exc:
        log.warning("Agent call failed for %s: %s", record["id"], exc)
        generated_sql = None

    if not generated_sql:
        generated_sql = ""

    gen_result = execute_sql(generated_sql) if generated_sql else {
        "success": False, "columns": [], "rows": [], "row_count": 0,
        "error_message": "No SQL generated", "execution_time_ms": 0.0,
    }
    ref_result = execute_sql(record["sql_query"])

    metrics = evaluate_sql(
        generated_sql=generated_sql,
        reference_sql=record["sql_query"],
        generated_result=gen_result,
        reference_result=ref_result,
        difficulty_level=record["level"],
    )

    return generated_sql, gen_result, metrics


def build_report(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    exact_matches = sum(1 for r in results if r["evaluation"]["execution_match"])
    avg_semantic = sum(r["evaluation"]["semantic_score"] for r in results) / total
    avg_overall = sum(r["evaluation"]["overall_score"] for r in results) / total

    by_level: dict[int, dict] = {}
    for r in results:
        lvl = r["level"]
        by_level.setdefault(lvl, {"total": 0, "exact": 0, "semantic_sum": 0.0})
        by_level[lvl]["total"] += 1
        if r["evaluation"]["execution_match"]:
            by_level[lvl]["exact"] += 1
        by_level[lvl]["semantic_sum"] += r["evaluation"]["semantic_score"]

    for lvl, stats in by_level.items():
        stats["exact_match_rate"] = round(stats["exact"] / stats["total"], 3)
        stats["avg_semantic"] = round(stats["semantic_sum"] / stats["total"], 3)
        del stats["semantic_sum"]

    return {
        "total_queries": total,
        "overall_exact_match_rate": round(exact_matches / total, 3),
        "overall_semantic_score": round(avg_semantic, 3),
        "overall_score": round(avg_overall, 3),
        "by_level": by_level,
        "records": results,
    }


def print_summary_table(report: dict):
    table = Table(title="Text-to-SQL Evaluation Results", show_lines=True)
    table.add_column("Level", style="bold cyan")
    table.add_column("Queries", justify="right")
    table.add_column("Exact Match", justify="right")
    table.add_column("Avg Semantic", justify="right")

    for lvl in sorted(report["by_level"]):
        s = report["by_level"][lvl]
        em = f"{s['exact_match_rate']*100:.1f}%"
        sem = f"{s['avg_semantic']:.3f}"
        table.add_row(f"L{lvl}", str(s["total"]), em, sem)

    table.add_section()
    table.add_row(
        "ALL",
        str(report["total_queries"]),
        f"{report['overall_exact_match_rate']*100:.1f}%",
        f"{report['overall_semantic_score']:.3f}",
        style="bold",
    )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL agent on H&M dataset")
    parser.add_argument("--level", type=int, default=None, help="Filter to a specific difficulty level (1-6)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of queries to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    args = parser.parse_args()

    console.print("[bold blue]Text-to-SQL Agent Evaluation[/bold blue]")
    console.print(f"Dataset: {DATASET_PATH}")

    # Load schema context once
    console.print("Loading schema context…")
    try:
        schema_context = get_schema_context()
    except Exception as exc:
        console.print(f"[red]Could not load schema — is the DB running? {exc}[/red]")
        sys.exit(1)

    records = load_dataset(level_filter=args.level, limit=args.limit)
    console.print(f"Loaded {len(records)} queries to evaluate.")

    results = []
    for record in track(records, description="Evaluating…"):
        generated_sql, gen_result, metrics = run_agent_on_query(schema_context, record)
        results.append(
            {
                "query_id": record["id"],
                "template_id": record["template_id"],
                "level": record["level"],
                "category": record["category"],
                "nl_query": record["nl_query"],
                "reference_sql": record["sql_query"],
                "generated_sql": generated_sql,
                "gen_row_count": gen_result.get("row_count", 0),
                "evaluation": metrics,
            }
        )

    report = build_report(results)
    print_summary_table(report)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
