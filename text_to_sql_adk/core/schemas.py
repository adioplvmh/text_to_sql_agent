"""Pydantic schemas for agent inputs / outputs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Schema Inspection ────────────────────────────────────────────────────────

class SchemaInspectionInput(BaseModel):
    tables: list[str] = Field(
        default_factory=list,
        description="Specific tables to inspect; empty list means all tables.",
    )


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool


class TableSchema(BaseModel):
    table_name: str
    columns: list[ColumnInfo]
    row_count: int
    sample_values: dict[str, list[Any]] = Field(default_factory=dict)


class SchemaInspectionOutput(BaseModel):
    schemas: list[TableSchema]
    summary: str


# ── SQL Generation ────────────────────────────────────────────────────────────

class SQLGenerationInput(BaseModel):
    nl_query: str = Field(description="Natural language question to convert to SQL.")
    schema_context: str = Field(description="Database schema DDL and sample values.")
    difficulty_level: int = Field(default=1, ge=1, le=6)
    previous_attempt: str | None = Field(
        default=None,
        description="Previous SQL attempt (if retrying after an error).",
    )
    error_message: str | None = Field(
        default=None,
        description="Error from the previous execution attempt.",
    )


class SQLGenerationOutput(BaseModel):
    generated_sql: str = Field(description="The generated PostgreSQL SQL query.")
    reasoning: str = Field(description="Step-by-step reasoning for the SQL.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1.")
    tables_used: list[str] = Field(default_factory=list)


# ── SQL Execution ─────────────────────────────────────────────────────────────

class SQLExecutionInput(BaseModel):
    sql_query: str
    max_rows: int = Field(default=50)


class SQLExecutionOutput(BaseModel):
    success: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    error_message: str | None = None
    execution_time_ms: float = 0.0


# ── SQL Evaluation ────────────────────────────────────────────────────────────

class SQLEvaluationInput(BaseModel):
    nl_query: str
    generated_sql: str
    reference_sql: str
    generated_result: SQLExecutionOutput
    reference_result: SQLExecutionOutput
    difficulty_level: int = Field(default=1, ge=1, le=6)


class SQLEvaluationOutput(BaseModel):
    execution_match: bool = Field(
        description="True if both queries return the same result set."
    )
    structural_score: float = Field(
        ge=0.0, le=1.0,
        description="How closely the generated SQL matches the reference structurally.",
    )
    semantic_score: float = Field(
        ge=0.0, le=1.0,
        description="LLM-based semantic correctness of the generated SQL.",
    )
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: str = Field(description="CORRECT | PARTIAL | INCORRECT")
    feedback: str = Field(description="Detailed feedback on the generated SQL.")
    tables_match: bool = False
    joins_correct: bool = False


# ── Research / Deep Analysis ──────────────────────────────────────────────────

class ResearchInput(BaseModel):
    nl_query: str
    schema_context: str
    difficulty_level: int = Field(default=1, ge=1, le=6)


class ResearchOutput(BaseModel):
    query_intent: str
    relevant_tables: list[str]
    join_strategy: str | None = None
    aggregation_strategy: str | None = None
    filter_conditions: list[str] = Field(default_factory=list)
    sql_hints: list[str] = Field(default_factory=list)


# ── Evaluation Report ─────────────────────────────────────────────────────────

class EvaluationRecord(BaseModel):
    query_id: str
    template_id: str
    level: int
    nl_query: str
    reference_sql: str
    generated_sql: str
    evaluation: SQLEvaluationOutput
    retries: int = 0


class EvaluationReport(BaseModel):
    total_queries: int
    by_level: dict[int, dict[str, Any]] = Field(default_factory=dict)
    overall_exact_match_rate: float
    overall_semantic_score: float
    records: list[EvaluationRecord] = Field(default_factory=list)
