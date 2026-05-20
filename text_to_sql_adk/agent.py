"""
Text-to-SQL Deep-Research Agent for H&M Fashion Data.

Architecture (Google ADK):
  Orchestrator
    ├── SchemaInspectionAgent   (inspect DB schema & samples)
    ├── ResearchAgent           (deep query plan for complex NL)
    ├── SQLGenerationAgent      (NL → SQL, with retry)
    └── SQLEvaluationAgent      (LLM-based semantic scoring)

  FunctionTools (direct DB access):
    inspect_schema_tool
    execute_sql_tool
    evaluate_sql_tool
"""
from __future__ import annotations

# ── Arize OTEL must be set up before any google.adk import ───────────────────
from text_to_sql_adk import tracing_setup  # noqa: F401

import json
import logging
from pathlib import Path

from google import adk
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.apps import App
from google.adk.tools.agent_tool import AgentTool

from text_to_sql_adk.core.config import get_model
from text_to_sql_adk.core.models import DEFAULT_MODELS, ModelsConfig, get_model_for_agent
from text_to_sql_adk.core.prompts.orchestration import ORCHESTRATION_PROMPT
from text_to_sql_adk.core.prompts.sql_generation import (
    SQL_GENERATION_INSTRUCTION,
    RETRY_HINT_TEMPLATE,
)
from text_to_sql_adk.core.prompts.research import RESEARCH_INSTRUCTION
from text_to_sql_adk.core.prompts.sql_evaluation import SQL_EVALUATION_INSTRUCTION
from text_to_sql_adk.core.tools.schema_inspector import get_schema_context
from text_to_sql_adk.core.tools.sql_executor import execute_sql, results_to_markdown
from text_to_sql_adk.core.tools.sql_evaluator import evaluate_sql

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapped function tools (called directly by the orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_schema_tool(tables: list[str] | None = None) -> str:
    """
    Retrieve the H&M database schema (DDL + sample values).

    Args:
        tables: optional list of table names to inspect (default: all).

    Returns:
        Schema context string ready for LLM consumption.
    """
    return get_schema_context(tables)


def execute_sql_tool(sql_query: str, max_rows: int = 50) -> str:
    """
    Execute a PostgreSQL SELECT query against the H&M database.

    Args:
        sql_query: The SQL string (PostgreSQL syntax).
        max_rows: Maximum rows to return.

    Returns:
        JSON string with keys: success, columns, rows, row_count, error_message, execution_time_ms.
    """
    result = execute_sql(sql_query, max_rows=max_rows)
    return json.dumps(result, default=str)


def verify_sql_tool(sql_query: str) -> str:
    """
    Validate a PostgreSQL SELECT query by executing it with LIMIT 1.
    Use this to check syntax and runtime errors BEFORE returning the SQL.
    Does NOT return full results — only success/failure status.

    Args:
        sql_query: The SQL string to validate (PostgreSQL syntax).

    Returns:
        JSON string with keys: success, error_message, execution_time_ms.
        On success, 'success' is true and 'error_message' is null.
        On failure, 'success' is false and 'error_message' describes the problem.
    """
    # Wrap in a subquery with LIMIT 1 to minimise DB load during validation
    validation_sql = f"SELECT * FROM ({sql_query}) AS _validation_subquery LIMIT 1"
    result = execute_sql(validation_sql, max_rows=1)
    return json.dumps({
        "success": result["success"],
        "error_message": result.get("error_message"),
        "execution_time_ms": result.get("execution_time_ms"),
    }, default=str)


def evaluate_sql_tool(
    generated_sql: str,
    reference_sql: str,
    difficulty_level: int = 1,
) -> str:
    """
    Structurally compare *generated_sql* to *reference_sql*.

    Executes both queries and compares result sets.

    Args:
        generated_sql: The SQL produced by the agent.
        reference_sql: The ground-truth SQL from the benchmark dataset.
        difficulty_level: 1-6.

    Returns:
        JSON string with evaluation metrics.
    """
    gen_result = execute_sql(generated_sql)
    ref_result = execute_sql(reference_sql)
    metrics = evaluate_sql(
        generated_sql=generated_sql,
        reference_sql=reference_sql,
        generated_result=gen_result,
        reference_result=ref_result,
        difficulty_level=difficulty_level,
    )
    return json.dumps(metrics, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agents
# ─────────────────────────────────────────────────────────────────────────────

def create_schema_inspection_agent(model_name: str) -> LlmAgent:
    """Agent that inspects the live database schema and returns a DDL summary."""
    return LlmAgent(
        name="SchemaInspectionAgent",
        model=get_model(model_name),
        instruction="""You are a database schema specialist.
Use the inspect_schema_tool to fetch the latest DDL and sample values from the
H&M PostgreSQL database, then return a clean, well-formatted schema summary
that can be used by a SQL generation agent.""",
        tools=[FunctionTool(inspect_schema_tool)],
        description="Fetches live schema DDL and sample catalogue values from the H&M DB.",
    )


def create_research_agent(model_name: str) -> LlmAgent:
    """Deep-research agent that analyses complex NL queries before SQL generation."""
    return LlmAgent(
        name="ResearchAgent",
        model=get_model(model_name),
        instruction=RESEARCH_INSTRUCTION,
        description=(
            "Analyses a natural-language question and produces a structured "
            "query plan (tables, joins, aggregations, filters) to guide SQL generation."
        ),
    )


def create_sql_generation_agent(model_name: str) -> LlmAgent:
    """Agent that generates PostgreSQL SQL from a natural-language question."""
    return LlmAgent(
        name="SQLGenerationAgent",
        model=get_model(model_name),
        instruction=SQL_GENERATION_INSTRUCTION,
        tools=[FunctionTool(verify_sql_tool)],
        description=(
            "Converts a natural-language question to a PostgreSQL query, "
            "validates syntax with verify_sql_tool, self-corrects on failure."
        ),
    )


def create_sql_evaluation_agent(model_name: str) -> LlmAgent:
    """LLM-based semantic SQL evaluator."""
    return LlmAgent(
        name="SQLEvaluationAgent",
        model=get_model(model_name),
        instruction=SQL_EVALUATION_INSTRUCTION,
        tools=[FunctionTool(evaluate_sql_tool)],
        description=(
            "Semantically scores a generated SQL query against the ground-truth "
            "reference, returning a structured evaluation with feedback."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

def create_text_to_sql_agent(models: ModelsConfig | None = None) -> adk.Agent:
    """
    Build the complete Text-to-SQL deep-research agent.

    The orchestrator uses direct FunctionTools (schema inspection + SQL execution)
    and sub-agents wrapped as AgentTools for research and evaluation.

    Args:
        models: Optional custom model configuration.

    Returns:
        Configured ADK Agent ready for use.
    """

    config = models or DEFAULT_MODELS

    # Sub-agents as tools
    research_tool = AgentTool(
        agent=create_research_agent(get_model_for_agent("research", config))
    )
    sql_generation_tool = AgentTool(
        agent=create_sql_generation_agent(get_model_for_agent("sql_generation", config))
    )
    sql_evaluation_tool = AgentTool(
        agent=create_sql_evaluation_agent(get_model_for_agent("sql_evaluation", config))
    )

    # Orchestrator
    orchestrator = adk.Agent(
        name="TextToSQLOrchestrator",
        model=get_model(get_model_for_agent("orchestration", config)),
        instruction=ORCHESTRATION_PROMPT,
        tools=[
            FunctionTool(inspect_schema_tool),
            FunctionTool(execute_sql_tool),
            FunctionTool(evaluate_sql_tool),
            research_tool,
            sql_generation_tool,
            sql_evaluation_tool,
        ],
        description=(
            "Deep-research Text-to-SQL agent for H&M fashion data. "
            "Takes natural-language questions and returns accurate PostgreSQL queries "
            "with execution results and quality evaluation."
        ),
    )

    return orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# ADK App entry-point
# ─────────────────────────────────────────────────────────────────────────────

retry_plugin = ReflectAndRetryToolPlugin(
    max_retries=2,
    throw_exception_if_retry_exceeded=False,
)

app = App(
    name="text_to_sql_adk",
    root_agent=create_text_to_sql_agent(),
    plugins=[retry_plugin],
)
