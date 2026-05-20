"""SQL executor tool — runs a SQL query against the H&M PostgreSQL DB."""
from __future__ import annotations

import logging
import time
import re

from sqlalchemy import create_engine, text

from text_to_sql_adk.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


def _normalise_sql(sql: str) -> str:
    """Convert BigQuery-style backtick-quoted identifiers to PostgreSQL double-quotes."""
    # Replace `table_name` with "table_name"
    normalised = re.sub(r'`([^`]+)`', r'"\1"', sql)
    # Strip trailing semicolon (sqlalchemy text() doesn't need it)
    normalised = normalised.rstrip().rstrip(";")
    return normalised


def execute_sql(sql_query: str, max_rows: int = 50) -> dict:
    """
    Execute a SQL query against the H&M PostgreSQL database.

    Args:
        sql_query: The SQL string to execute (BigQuery backtick syntax is auto-converted).
        max_rows: Maximum rows to return (default 50).

    Returns:
        dict with keys: success, columns, rows, row_count, error_message, execution_time_ms.
    """
    normalised = _normalise_sql(sql_query)

    # Safety guard — only SELECT statements
    first_word = normalised.strip().split()[0].upper()
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error_message": f"Only SELECT/WITH queries are allowed. Got: {first_word}",
            "execution_time_ms": 0.0,
        }

    engine = _get_engine()
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(normalised))
            columns = list(result.keys())
            all_rows = result.fetchall()
            rows = [list(r) for r in all_rows[:max_rows]]
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(all_rows),
                "error_message": None,
                "execution_time_ms": round(elapsed, 2),
            }
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.warning("SQL execution error: %s", exc)
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error_message": str(exc),
            "execution_time_ms": round(elapsed, 2),
        }


def results_to_markdown(execution_result: dict) -> str:
    """Convert an execution result dict to a Markdown table string."""
    if not execution_result.get("success"):
        return f"**Error:** {execution_result.get('error_message', 'unknown error')}"

    columns = execution_result.get("columns", [])
    rows = execution_result.get("rows", [])

    if not rows:
        return "*(no rows returned)*"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows
    ]
    return "\n".join([header, separator] + body_lines)
