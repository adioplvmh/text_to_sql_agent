"""Prompt for the LLM-based semantic SQL evaluation agent."""

SQL_EVALUATION_INSTRUCTION = """
You are an expert SQL evaluator for H&M fashion retail analytics.

## Task
Score the **Generated SQL** against the **Reference SQL** for the given question.

## Natural Language Question
{nl_query}

## Reference SQL (ground truth)
```sql
{reference_sql}
```

## Generated SQL (to evaluate)
```sql
{generated_sql}
```

## Execution Results Comparison
Reference returned {ref_row_count} rows.
Generated returned {gen_row_count} rows.
Execution match: {execution_match}

## Scoring Rubric
Return a JSON object with:
- semantic_score: float 0.0-1.0
  * 1.0  — semantically identical to reference (same logic, may differ in style)
  * 0.7+ — correct intent, minor issues (different column alias, extra columns)
  * 0.4+ — partially correct (right tables/joins but wrong filter or aggregation)
  * 0.1+ — some correct elements but fundamentally wrong
  * 0.0  — completely wrong
- feedback: one paragraph explaining the score
- suggestions: list of specific improvements

Respond with ONLY the JSON, no markdown fences.
"""
