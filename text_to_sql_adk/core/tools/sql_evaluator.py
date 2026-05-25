"""SQL structural evaluator — compares generated vs reference SQL results.

Metrics (in order of cost):
  Tier 1 — free / deterministic
    execution_match      : result-set exact match (order-insensitive)
    structural_score     : clause-presence overlap (JOIN/GROUP BY/CTE…)
    column_precision     : fraction of generated cols that appear in reference cols
    column_recall        : fraction of reference cols found in generated cols
    row_overlap          : Jaccard similarity of normalised result-row sets
    rouge_l              : ROUGE-L F1 on SQL token sequences (from notebook)
  Tier 2 — Vertex AI embedding (same model as maia_evaluation notebook)
    embedding_similarity : cosine similarity of SQL strings via text-embedding-004
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ── lazy imports so cold-starts are fast ─────────────────────────────────────
def _rouge_scorer():
    try:
        from rouge_score import rouge_scorer as rs
        return rs.RougeScorer(["rougeL"], use_stemmer=False)
    except ImportError:
        return None


def _vertex_embedding_model():
    try:
        from vertexai.language_models import TextEmbeddingModel
        return TextEmbeddingModel.from_pretrained("text-embedding-004")
    except Exception as exc:
        log.warning("Vertex embedding model unavailable: %s", exc)
        return None


def _normalise_result(rows: list[list[Any]]) -> set[tuple]:
    """Convert rows to a frozenset of sorted tuples for order-independent comparison."""
    return {tuple(str(v).strip().lower() if v is not None else "null" for v in row) for row in rows}


# ── Tier-1 helpers ────────────────────────────────────────────────────────────

def _rouge_l(generated_sql: str, reference_sql: str) -> float:
    """ROUGE-L F1 on SQL token sequences (same approach as maia_evaluation notebook)."""
    scorer = _rouge_scorer()
    if scorer is None or not generated_sql or not reference_sql:
        return 0.0
    try:
        result = scorer.score(reference_sql, generated_sql)
        return float(result["rougeL"].fmeasure)
    except Exception:
        return 0.0


def _column_precision_recall(generated_result: dict, reference_result: dict) -> tuple[float, float]:
    """Precision / recall of output columns against reference columns."""
    gen_cols = {c.lower() for c in generated_result.get("columns", [])}
    ref_cols = {c.lower() for c in reference_result.get("columns", [])}
    if not ref_cols:
        return (1.0, 1.0) if not gen_cols else (0.0, 1.0)
    recall = len(gen_cols & ref_cols) / len(ref_cols)
    precision = len(gen_cols & ref_cols) / len(gen_cols) if gen_cols else 0.0
    return round(precision, 3), round(recall, 3)


def _row_overlap(generated_result: dict, reference_result: dict) -> float:
    """Jaccard similarity of normalised row sets — soft execution match."""
    gen_rows = _normalise_result(generated_result.get("rows", []))
    ref_rows = _normalise_result(reference_result.get("rows", []))
    if not gen_rows and not ref_rows:
        return 1.0
    if not gen_rows or not ref_rows:
        return 0.0
    intersection = len(gen_rows & ref_rows)
    union = len(gen_rows | ref_rows)
    return round(intersection / union, 3)


# ── Tier-2 helper ─────────────────────────────────────────────────────────────

def _embedding_similarity(generated_sql: str, reference_sql: str) -> float:
    """Cosine similarity between SQL strings via Vertex AI text-embedding-004.
    Same model used in maia_evaluation/general_multicategory_eval_v3.ipynb.
    Returns 0.0 on any error (e.g. no Vertex credentials in CI).
    """
    if not generated_sql or not reference_sql:
        return 0.0
    model = _vertex_embedding_model()
    if model is None:
        return 0.0
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        gen_emb = model.get_embeddings([generated_sql])[0].values
        ref_emb = model.get_embeddings([reference_sql])[0].values
        score = cos_sim(
            np.array(gen_emb).reshape(1, -1),
            np.array(ref_emb).reshape(1, -1),
        )[0][0]
        return round(float(score), 4)
    except Exception as exc:
        log.warning("Embedding similarity failed: %s", exc)
        return 0.0


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
    compute_embeddings: bool = True,
) -> dict:
    """
    Compare a generated SQL query to the reference (ground truth).

    Args:
        generated_sql: The SQL produced by the agent.
        reference_sql: The ground-truth SQL from the dataset.
        generated_result: Execution result dict from execute_sql().
        reference_result: Execution result dict from execute_sql().
        difficulty_level: 1-6, used to weight scoring.
        compute_embeddings: Whether to call Vertex AI for embedding similarity
                            (set False in unit tests / offline runs).

    Returns:
        dict with evaluation metrics including ROUGE-L, embedding similarity,
        column precision/recall, and row overlap (from maia_evaluation notebook).
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

    # ── Tier-1 new metrics ────────────────────────────────────────────────────
    rouge_l = _rouge_l(generated_sql, reference_sql)
    col_precision, col_recall = _column_precision_recall(generated_result, reference_result)
    row_overlap = _row_overlap(generated_result, reference_result) if (gen_ok and ref_ok) else 0.0

    # ── Semantic score ────────────────────────────────────────────────────────
    if execution_match:
        semantic_score = 1.0
    elif gen_ok and ref_ok:
        col_overlap = len(
            {c.lower() for c in generated_result.get("columns", [])} &
            {c.lower() for c in reference_result.get("columns", [])}
        ) / max(len(reference_result.get("columns", [])), 1)
        # Richer formula: adds row_overlap and rouge_l contributions
        semantic_score = (
            0.25 * col_overlap
            + 0.25 * structural_score
            + 0.25 * row_overlap
            + 0.25 * rouge_l
        )
        semantic_score = min(semantic_score + 0.3, 1.0)  # base offset for both running
    elif gen_ok:
        semantic_score = 0.2 * structural_score + 0.1 * rouge_l
    else:
        semantic_score = 0.0

    # ── Tier-2: embedding similarity (Vertex AI text-embedding-004) ───────────
    emb_sim = (
        _embedding_similarity(generated_sql, reference_sql)
        if compute_embeddings and generated_sql
        else 0.0
    )

    # ── Overall score ─────────────────────────────────────────────────────────
    overall_score = round(
        0.40 * float(execution_match)
        + 0.20 * semantic_score
        + 0.15 * structural_score
        + 0.15 * rouge_l
        + 0.10 * emb_sim,
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
        # ── Core ─────────────────────────────────────────────────────────────
        "execution_match": execution_match,
        "overall_score": overall_score,
        "verdict": verdict,
        "feedback": feedback,
        # ── Structural ───────────────────────────────────────────────────────
        "structural_score": round(structural_score, 3),
        "tables_match": tables_match,
        "joins_correct": joins_correct,
        # ── Tier-1 new ───────────────────────────────────────────────────────
        "rouge_l": round(rouge_l, 4),
        "column_precision": col_precision,
        "column_recall": col_recall,
        "row_overlap": row_overlap,
        # ── Semantic composite ────────────────────────────────────────────────
        "semantic_score": round(semantic_score, 3),
        # ── Tier-2 embedding (Vertex AI text-embedding-004) ───────────────────
        "embedding_similarity": emb_sim,
    }
