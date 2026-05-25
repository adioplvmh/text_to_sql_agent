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
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
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


async def _call_agent_async(runner, nl_query: str) -> str:
    """Call the agent with a fresh session and return the generated SQL."""
    from google.genai import types as genai_types

    # Fresh session per query — no history bleed-over
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name="text_to_sql_eval",
        user_id="evaluator",
        session_id=session_id,
    )

    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=nl_query)]
    )
    captured_sql = ""
    response_text = ""

    try:
        async for event in runner.run_async(
            user_id="evaluator",
            session_id=session_id,
            new_message=content,
        ):
            if event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        args = dict(fc.args) if fc.args else {}
                        sql_arg = (
                            args.get("sql_query")
                            or args.get("query")
                            or args.get("sql")
                            or ""
                        )
                        if sql_arg and sql_arg.strip().upper().startswith("SELECT"):
                            captured_sql = sql_arg.strip()

            if event.is_final_response():
                if event.content and event.content.parts:
                    response_text = event.content.parts[0].text or ""
    except Exception as exc:
        log.warning("Agent stream error for query %r: %s", nl_query[:60], exc)

    if captured_sql:
        return captured_sql

    m = re.search(r"```(?:sql)?\n(.+?)```", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(SELECT\s.+?)(?:\n\n|$)", response_text, re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return ""


async def run_agent_on_query_async(runner, record: dict) -> tuple[str, dict, dict]:
    """Run the agent on one record and return (generated_sql, gen_result, metrics)."""
    try:
        generated_sql = await asyncio.wait_for(
            _call_agent_async(runner, record["nl_query"]),
            timeout=120,
        )
    except asyncio.TimeoutError:
        log.warning("Timeout for query %s", record.get("id"))
        generated_sql = ""
    except Exception as exc:
        log.warning("Agent call failed for %s: %s", record.get("id"), exc)
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
    avg_rouge_l = sum(r["evaluation"].get("rouge_l", 0) for r in results) / total
    avg_emb_sim = sum(r["evaluation"].get("embedding_similarity", 0) for r in results) / total

    by_level: dict[int, dict] = {}
    for r in results:
        lvl = r["level"]
        by_level.setdefault(lvl, {
            "total": 0, "exact": 0,
            "semantic_sum": 0.0, "rouge_l_sum": 0.0, "emb_sim_sum": 0.0,
            "col_prec_sum": 0.0, "col_rec_sum": 0.0, "row_overlap_sum": 0.0,
        })
        by_level[lvl]["total"] += 1
        if r["evaluation"]["execution_match"]:
            by_level[lvl]["exact"] += 1
        by_level[lvl]["semantic_sum"] += r["evaluation"]["semantic_score"]
        by_level[lvl]["rouge_l_sum"] += r["evaluation"].get("rouge_l", 0)
        by_level[lvl]["emb_sim_sum"] += r["evaluation"].get("embedding_similarity", 0)
        by_level[lvl]["col_prec_sum"] += r["evaluation"].get("column_precision", 0)
        by_level[lvl]["col_rec_sum"] += r["evaluation"].get("column_recall", 0)
        by_level[lvl]["row_overlap_sum"] += r["evaluation"].get("row_overlap", 0)

    for lvl, s in by_level.items():
        n = s["total"]
        s["exact_match_rate"] = round(s["exact"] / n, 3)
        s["avg_semantic"] = round(s["semantic_sum"] / n, 3)
        s["avg_rouge_l"] = round(s["rouge_l_sum"] / n, 3)
        s["avg_embedding_similarity"] = round(s["emb_sim_sum"] / n, 3)
        s["avg_column_precision"] = round(s["col_prec_sum"] / n, 3)
        s["avg_column_recall"] = round(s["col_rec_sum"] / n, 3)
        s["avg_row_overlap"] = round(s["row_overlap_sum"] / n, 3)
        for k in ["semantic_sum", "rouge_l_sum", "emb_sim_sum",
                  "col_prec_sum", "col_rec_sum", "row_overlap_sum"]:
            del s[k]

    return {
        "total_queries": total,
        "overall_exact_match_rate": round(exact_matches / total, 3),
        "overall_semantic_score": round(avg_semantic, 3),
        "overall_score": round(avg_overall, 3),
        "overall_rouge_l": round(avg_rouge_l, 3),
        "overall_embedding_similarity": round(avg_emb_sim, 3),
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


async def async_main(args):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from text_to_sql_adk.agent import create_text_to_sql_agent

    console.print("[bold blue]Text-to-SQL Agent Evaluation[/bold blue]")
    console.print(f"Dataset: {DATASET_PATH}")

    console.print("Loading schema context…")
    try:
        schema_context = get_schema_context()
    except Exception as exc:
        console.print(f"[red]Could not load schema — is the DB running? {exc}[/red]")
        sys.exit(1)

    console.print("Initialising agent…")
    agent = create_text_to_sql_agent()
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="text_to_sql_eval", session_service=svc)

    records = load_dataset(level_filter=args.level, limit=args.limit)
    total = len(records)
    console.print(f"Loaded {total} queries to evaluate.")

    if args.progress_file:
        Path(args.progress_file).write_text(json.dumps({"done": 0, "total": total}))

    results = []
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating…", total=total)
        for i, record in enumerate(records, start=1):
            progress.update(task, description=f"[{i}/{total}] {record['id']} …")
            generated_sql, gen_result, metrics = await run_agent_on_query_async(runner, record)
            results.append({
                "query_id": record["id"],
                "template_id": record["template_id"],
                "level": record["level"],
                "category": record["category"],
                "nl_query": record["nl_query"],
                "reference_sql": record["sql_query"],
                "generated_sql": generated_sql,
                "gen_row_count": gen_result.get("row_count", 0),
                "evaluation": metrics,
            })
            progress.advance(task)
            if args.progress_file:
                Path(args.progress_file).write_text(json.dumps({"done": i, "total": total}))

    report = build_report(results)
    print_summary_table(report)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[green]Results saved to {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL agent on H&M dataset")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--progress-file", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()


from text_to_sql_adk.core.tools.schema_inspector import get_schema_context
from text_to_sql_adk.core.tools.sql_executor import execute_sql
from text_to_sql_adk.core.tools.sql_evaluator import evaluate_sql

log = logging.getLogger(__name__)
console = Console()

DATASET_PATH = Path(__file__).parent / "data" / "dataset.jsonl"

