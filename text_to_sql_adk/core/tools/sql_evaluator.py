"""SQL structural evaluator — compares generated vs reference SQL results."""
from __future__ import annotations

import re
from typing import Any


def _normalise_result(rows: list[list[Any]]) -> set[tuple]:
    """Convert rows to a frozenset of sorted tuples for order-independent comparison."""
    return {tuple(str(v).strip().lower() if v is not None else "null" for v in row) for row in rows}


def _extract_clauses(sql: str) -> dict[str, bool]:
    """Return a dict of SQL clause presence flags."""
    upper = sql.upper()
    return {
        "has_join": bool(re.search(r"\bJOIN\b", upper)),
        "has_group_by": bool(re.search(r"\bGROUP\s+BY\b", upper)),
        "has_having": bool(re.search(r"\bHAVING\b", upper)),
        "has_cte": bool(re.search(r"\bWITH\b", upper) and re.search(r"\bAS\s*\(", upper)),
        "has_window": bool(re.search(r"\bOVER\s*\(", upper)),
        "has_agg": bool(re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", upper)),
        "has_subquery": bool(re.search(r"\(\s*SELECT", upper)),
    }


def _tables_in_sql(sql: str) -> set[str]:
    """Extract table names mentioned in a SQL string (rough heuristic)."""
    pattern = r'(?:FROM|JOIN)\s+"?(\w+)"?'
    return {m.lower() for m in re.findall(pattern, sql, flags=re.IGNORECASE)}


def evaluate_sql(
    generated_sql: str,
    reference_sql: str,
    generated_result: dict,
    reference_result: dict,
    difficulty_level: int = 1,
) -> dict:
    """
    Compare a generated SQL query to the reference (ground truth).

    Args:
        generated_sql: The SQL produced by the agent.
        reference_sql: The ground-truth SQL from the dataset.
        generated_result: Execution result dict from execute_sql().
        reference_result: Execution result dict from execute_sql().
        difficulty_level: 1-6, used to weight scoring.

    Returns:
        dict with evaluation metrics.
    """
    # ── Execution match ───────────────────────────────────────────────────────
    gen_ok = generated_result.get("success", False)
    ref_ok = reference_result.get("success", False)

    execution_match = False
    if gen_ok and ref_ok:
        gen_set = _normalise_result(generated_result.get("rows", []))
        ref_set = _normalise_result(reference_result.get("rows", []))
        execution_match = gen_set == ref_set

    # ── Structural score ──────────────────────────────────────────────────────
    gen_clauses = _extract_clauses(generated_sql)
    ref_clauses = _extract_clauses(reference_sql)

    clause_matches = sum(
        gen_clauses[k] == ref_clauses[k] for k in ref_clauses
    )
    structural_score = clause_matches / max(len(ref_clauses), 1)

    # Tables match
    gen_tables = _tables_in_sql(generated_sql)
    ref_tables = _tables_in_sql(reference_sql)
    tables_match = gen_tables == ref_tables

    # Joins correct (only if reference uses JOIN)
    joins_correct = (
        gen_clauses["has_join"] == ref_clauses["has_join"]
        if ref_clauses["has_join"]
        else True
    )

    # ── Semantic score (heuristic, will be overridden by LLM agent) ──────────
    # Weighted: execution match is the gold standard
    if execution_match:
        semantic_score = 1.0
    elif gen_ok and ref_ok:
        # Partial: same columns, different rows
        gen_cols = set(generated_result.get("columns", []))
        ref_cols = set(reference_result.get("columns", []))
        col_overlap = len(gen_cols & ref_cols) / max(len(ref_cols), 1)
        semantic_score = 0.3 + 0.4 * col_overlap + 0.3 * structural_score
    elif gen_ok:
        semantic_score = 0.2 * structural_score
    else:
        semantic_score = 0.0

    # ── Overall score ─────────────────────────────────────────────────────────
    overall_score = round(
        0.5 * float(execution_match)
        + 0.3 * semantic_score
        + 0.2 * structural_score,
        3,
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if execution_match:
        verdict = "CORRECT"
    elif overall_score >= 0.5:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"

    # ── Feedback ─────────────────────────────────────────────────────────────
    feedback_parts: list[str] = []
    if not gen_ok:
        feedback_parts.append(
            f"Generated SQL failed to execute: {generated_result.get('error_message')}"
        )
    if not tables_match:
        feedback_parts.append(
            f"Table mismatch — expected {ref_tables}, got {gen_tables}."
        )
    if ref_clauses["has_join"] and not gen_clauses["has_join"]:
        feedback_parts.append("Missing JOIN clause(s).")
    if ref_clauses["has_cte"] and not gen_clauses["has_cte"]:
        feedback_parts.append("Missing CTE (WITH clause).")
    if ref_clauses["has_window"] and not gen_clauses["has_window"]:
        feedback_parts.append("Missing window function.")
    if ref_clauses["has_having"] and not gen_clauses["has_having"]:
        feedback_parts.append("Missing HAVING clause.")
    if execution_match:
        feedback_parts.append("Result sets match exactly. ✓")

    feedback = " | ".join(feedback_parts) if feedback_parts else "No major structural issues."

    return {
        "execution_match": execution_match,
        "structural_score": round(structural_score, 3),
        "semantic_score": round(semantic_score, 3),
        "overall_score": overall_score,
        "verdict": verdict,
        "feedback": feedback,
        "tables_match": tables_match,
        "joins_correct": joins_correct,
    }
