# Text-to-SQL Agent — H&M Fashion Data

A **deep-research Text-to-SQL evaluation agent** built with [Google ADK](https://google.github.io/adk-docs/) and Poetry.

## Architecture

```
TextToSQLOrchestrator (ADK Agent)
  ├─ inspect_schema_tool       → live DDL + sample values from PostgreSQL
  ├─ execute_sql_tool          → execute generated SQL, return results
  ├─ evaluate_sql_tool         → structural compare generated vs reference
  ├─ ResearchAgent             → deep query analysis for L3-L6 questions
  ├─ SQLGenerationAgent        → NL → PostgreSQL (with self-correction)
  └─ SQLEvaluationAgent        → LLM-based semantic scoring
```

The agent follows a **deep-research workflow**:
1. Inspect live schema (DDL + sample catalogue values)
2. Research the query intent (tables, joins, aggregation strategy)
3. Generate SQL (retry on error up to 2×)
4. Execute and return results as a Markdown table
5. Evaluate against the ground-truth (execution match + semantic score)

---

## Database

Real H&M fashion data loaded from `lvprojects/data_generator/data/generated/` into the local **pgvector-container** PostgreSQL:

| Container | Host | Port | DB |
|-----------|------|------|----|
| `pgvector-container` | localhost | 6024 | `hm_fashion` |

### Source files

| File | Rows | Description |
|------|------|-------------|
| `products.csv` | ~70k | Full product catalogue |
| `clients.csv` | ~1.4M | Customer demographics (sampled to 100k) |
| `transactions.csv` | ~18.5M | Purchase history (sampled to 500k) |

### Table schema

**`articles`** — product catalogue (69,723 rows)

| Column | Type | Example |
|--------|------|---------|
| `article_id` | INTEGER PK | `108775015` |
| `prod_name` | TEXT | `"Jersey top with narrow shoulder straps"` |
| `product_type_name` | TEXT | `"Vest top"` |
| `product_group_name` | TEXT | `"Garment Upper body"` |
| `colour_group_name` | TEXT | `"Black"` |
| `index_group_name` | TEXT | `"Women"` |
| `garment_group_name` | TEXT | `"Jersey Basic"` |
| `product_composition` | TEXT | `"100% Organic Cotton"` |
| `price_euro` | FLOAT | `420.0` |
| `is_limited_edition` | BOOLEAN | `false` |

**`customers`** — client demographics (100,000 rows, sampled)

| Column | Type | Example |
|--------|------|---------|
| `customer_id` | TEXT PK | `"00007d2de826..."` |
| `nationality` | TEXT | `"France"` |
| `country` | TEXT | `"France"` |
| `segment` | TEXT | `"Active"` |
| `total_spending_euro` | FLOAT | `5385.0` |
| `global_contactable` | BOOLEAN | `false` |
| `email_contactable` | BOOLEAN | `true` |
| `mobile_contactable` | BOOLEAN | `false` |
| `latest_transaction_date` | DATE | `2020-03-21` |

**`transactions`** — purchase history (500,000 rows, sampled)

| Column | Type | Example |
|--------|------|---------|
| `id` | SERIAL PK | — |
| `t_dat` | DATE | `2018-09-20` |
| `customer_id` | TEXT FK | `"00007d2de826..."` |
| `article_id` | INTEGER FK | `108775015` |
| `price` | FLOAT | `770.0` |
| `quantity` | FLOAT | `1.0` |
| `transaction_type` | TEXT | `"online"` / `"offline"` |
| `boutique_zone` | TEXT | `"Southern Europe"` |
| `boutique_type` | TEXT | `"Online"` |
| `currency` | TEXT | `"EUR"` |

> **Note on `article_id`:** the source CSV uses zero-padded 10-digit strings (e.g. `0108775015`). `load_data.py` strips the leading zero and stores them as integers. This matches the product image file paths in `adk_test/data/images/`.

---

## Benchmark Dataset

`data/dataset.jsonl` — **277 NL→SQL pairs** across 6 difficulty levels.

### ⚠️ Schema compatibility notice

The benchmark was originally authored against a **synthetic schema** that had different columns in `customers` and `articles`. After migrating to real data, **~64 of 277 reference SQL queries reference columns that no longer exist**:

| Old column (removed) | Was in table | Replacement in real schema |
|----------------------|--------------|---------------------------|
| `detail_desc` | `articles` | `prod_name` (product description) |
| `graphical_appearance_name` | `articles` | — (not in real data) |
| `perceived_colour_master_name` | `articles` | `colour_group_name` |
| `club_member_status` | `customers` | `segment` |
| `fashion_news_frequency` | `customers` | `email_contactable` / `mobile_contactable` |
| `age` | `customers` | — (not in real data) |
| `postal_code` | `customers` | `country` / `nationality` |
| `sales_channel_id` | `transactions` | `transaction_type` (`"online"` / `"offline"`) |

**Impact on `evaluate.py`:** queries that use removed columns will produce an execution error during reference SQL execution. The evaluator will mark them `INCORRECT` with `error_message` set. The remaining ~213 queries are fully valid against the new schema.

**To regenerate the benchmark** with updated reference SQL aligned to the real schema, run a script that rewrites affected queries using the column mappings above (not yet implemented — contributions welcome).

### Difficulty levels

| Level | Count | Pattern | Example |
|-------|-------|---------|---------|
| L1 | 56 | Single-table, 1–2 filters | `"Show me all Black Dress items"` |
| L2 | 48 | Single-table, 3–4 filters | `"Find Blue Top items with a Stripe pattern"` |
| L3 | 24 | Aggregation (GROUP BY, COUNT, AVG) | `"Top 10 most common colours"` |
| L4 | 75 | 2-table JOIN | `"Which Jacket items were sold the most?"` |
| L5 | 49 | 3-table JOIN + filters | `"Red Sneakers bought online in Southern Europe"` |
| L6 | 25 | CTEs, window functions, HAVING | `"Top 3 product types per boutique zone"` |

### JSONL format

```json
{
  "id": "q0001",
  "template_id": "L1_001",
  "level": 1,
  "category": "article_lookup",
  "nl_query": "Show me all Off White Leggings/Tights items",
  "sql_query": "SELECT article_id, prod_name, colour_group_name, product_type_name FROM articles WHERE colour_group_name = 'Off White' AND product_type_name = 'Leggings/Tights'",
  "tables_used": ["articles"],
  "params": {"colour": "Off White", "product_type": "Leggings/Tights"}
}
```

---

## Setup

### Prerequisites
- Python ≥ 3.10
- [Poetry](https://python-poetry.org/) ≥ 2.0
- Docker (`pgvector-container` running on port 6024)
- GCP project with Vertex AI enabled
- Service account JSON with Vertex AI permissions

### 1 — Install dependencies

```bash
poetry install
```

### 2 — Configure environment

```bash
cp .env.example .env
# Fill in: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
#          GOOGLE_APPLICATION_CREDENTIALS, ARIZE_* keys
```

### 3 — Load real H&M data into PostgreSQL

Data is read from `lvprojects/data_generator/data/generated/`:

```bash
poetry run python scripts/load_data.py
# Force full reload (drops & recreates tables):
poetry run python scripts/load_data.py --force
```

This loads:
- All **~70k products** from `products.csv`
- **100k customers** sampled from `clients.csv`
- **500k transactions** streamed + filtered from `transactions.csv`

### 4 — Run a single query locally

```bash
poetry run python test_locally.py "Show me all Black Dress items"
poetry run python test_locally.py "What are the top 5 best-selling product types?"
poetry run python test_locally.py "Which online transactions were made in Southern Europe?"
```

### 5 — Launch the Streamlit UI

```bash
# Full UI with product image gallery
poetry run streamlit run streamlit_app_v2.py --server.port 8504
```

The app shows an inline horizontal product image strip (sourced from `adk_test/data/images/`) for any query that returns `article_id` values.

### 6 — Run the evaluation benchmark

```bash
# All 277 queries (note: ~64 reference SQLs use old schema columns)
poetry run python evaluate.py

# Only schema-safe levels (L1–L3 are mostly unaffected)
poetry run python evaluate.py --level 1
poetry run python evaluate.py --level 3

# Quick smoke test
poetry run python evaluate.py --limit 10 --output results_smoke.json
```

Results are saved as JSON and printed as a Rich table:

```
┌───────┬─────────┬─────────────┬──────────────┐
│ Level │ Queries │ Exact Match │ Avg Semantic │
├───────┼─────────┼─────────────┼──────────────┤
│ L1    │      56 │        TBD  │          TBD │
│ L2    │      48 │        TBD  │          TBD │
│ L3    │      24 │        TBD  │          TBD │
│ L4    │      75 │        TBD  │          TBD │
│ L5    │      49 │        TBD  │          TBD │
│ L6    │      25 │        TBD  │          TBD │
│ ALL   │     277 │        TBD  │          TBD │
└───────┴─────────┴─────────────┴──────────────┘
```

---

## Project Structure

```
text_to_sql_agent/
├── pyproject.toml
├── .env                        ← credentials & config (not committed)
├── .env.example
├── service_account.json        ← GCP service account (not committed)
├── evaluate.py                 ← full benchmark runner
├── test_locally.py             ← interactive single-query test
├── streamlit_app_v2.py         ← Streamlit UI with product image gallery
├── data/
│   └── dataset.jsonl           ← 277 NL→SQL benchmark pairs
│                                  ⚠️ ~64 reference SQLs use old schema columns
├── scripts/
│   └── load_data.py            ← loads real CSVs into PostgreSQL
│                                  source: data_generator/data/generated/
└── text_to_sql_adk/
    ├── agent.py                ← ADK orchestrator + sub-agents + App
    ├── tracing_setup.py        ← Arize AX OTEL instrumentation
    └── core/
        ├── config.py           ← DB URL, Vertex AI config
        ├── models.py           ← per-agent model assignments
        ├── schemas.py          ← Pydantic I/O schemas
        ├── prompts/
        │   ├── orchestration.py
        │   ├── sql_generation.py
        │   ├── research.py
        │   └── sql_evaluation.py
        └── tools/
            ├── schema_inspector.py  ← live DDL introspection via SQLAlchemy
            ├── sql_executor.py      ← SQL runner + BQ→PG syntax normaliser
            └── sql_evaluator.py     ← structural + execution match scoring
```

---

## Query Flow — Step by Step

What happens from the moment you type a query in Streamlit to images appearing on screen.

### Phase 0 — App startup (once per session)

`streamlit_app_v2.py` initialises `st.session_state` with:
- A `Runner` wrapping `create_text_to_sql_agent()` — builds the full ADK hierarchy (orchestrator + 3 sub-agents + 3 tools)
- An `InMemorySessionService` holding the conversation history
- `ReflectAndRetryToolPlugin` (max 2 retries) attached at the `App` level

### Phase 1 — Message dispatched to the ADK Runner

`st.chat_input` captures your text → `run_agent(user_input)` → `asyncio.run(_run_async(...))`.  
The Runner wraps it as a `genai_types.Content(role="user")` and sends it into the session.

### Phase 2 — TextToSQLOrchestrator (Gemini 2.5 Pro) takes over

The orchestrator reads `ORCHESTRATION_PROMPT` which mandates this 6-step sequence:

**Step 1 — `inspect_schema_tool`** (direct FunctionTool → PostgreSQL)
```
inspect_schema_tool(tables=None)
  → get_schema_context()
  → SQLAlchemy inspect() on localhost:6024/hm_fashion
  → returns DDL for articles / customers / transactions
    with real column names from load_data.py:
    articles:     article_id, prod_name, product_type_name, colour_group_name,
                  index_group_name, garment_group_name, price_euro, ...
    customers:    customer_id, nationality, country, segment,
                  total_spending_euro, email_contactable, ...
    transactions: t_dat, customer_id, article_id, price, quantity,
                  transaction_type ("online"/"offline"), boutique_zone, ...
```

**Step 2 — `ResearchAgent`** (AgentTool → Gemini 2.5 Flash)

Receives the user question + schema. Runs `RESEARCH_INSTRUCTION` and returns a structured JSON plan:
```json
{
  "query_intent": "rank product types by total sales",
  "relevant_tables": ["transactions", "articles"],
  "join_strategy": "transactions.article_id = articles.article_id",
  "aggregations": ["SUM(price * quantity)"],
  "filters": [],
  "difficulty": "L4"
}
```

**Step 3 — `SQLGenerationAgent`** (AgentTool → Gemini 2.5 Flash)

Receives the question + schema + research plan. Runs `SQL_GENERATION_INSTRUCTION` (PostgreSQL rules, L1–L6 difficulty guidelines). Optionally calls `execute_sql_tool` on itself to self-validate before returning. Returns a PostgreSQL query using the **real column names** (`price_euro`, `transaction_type`, `boutique_zone`, etc.).

**Step 4 — `execute_sql_tool`** (direct FunctionTool → PostgreSQL)
```
execute_sql_tool(sql_query)
  → execute_sql() → SQLAlchemy text() → localhost:6024/hm_fashion
  → returns JSON: { success, columns, rows, row_count,
                    execution_time_ms, error_message }
```
Rows come from the real data: 69,723 articles · 100k customers · 500k transactions.

**Step 5 (conditional) — Retry via `ReflectAndRetryToolPlugin`**

If execution failed, the plugin injects the error back into context and loops to Step 3. Up to **2 retries**.

**Step 6 (benchmark only) — `SQLEvaluationAgent`**

Only used in `evaluate.py`. Calls `evaluate_sql_tool(generated_sql, reference_sql)` to compare result sets against `dataset.jsonl` ground-truth.  
⚠️ ~64 benchmark queries reference old synthetic columns (`detail_desc`, `age`, `club_member_status`, `sales_channel_id`) that no longer exist — those will error at this step.

### Phase 3 — Response rendered in Streamlit

`_run_async` collects the final event → markdown text with SQL block.

In `streamlit_app_v2.py`:
1. `_extract_sql(response_text)` — regex pulls the SQL from the response
2. `execute_sql(sql, max_rows=100)` — re-runs it to render a live dataframe
3. `_extract_article_ids(result)` — checks if `article_id` column is in the result
4. If yes → `_enrich_articles(article_ids[:20])`:
   - Loads `product_metadata.parquet` for product names & prices
   - Reads JPEG from `adk_test/data/images/{aid[:3]}/{aid_padded}.jpg`
   - Base64-encodes each image
5. `_product_strip(cards)` — renders the horizontal scrollable image strip via `st.markdown()`

### Phase 4 — OTEL traces sent to Arize AX

Every tool call boundary emits a span via `GoogleADKInstrumentor` (initialised in `tracing_setup.py`, imported before any ADK import). Spans are batched and sent to `https://otlp.eu-west-1a.arize.com/v1/traces`.

---

## Observability

Traces are sent to **Arize AX** via OpenTelemetry:

```
ARIZE_COLLECTOR_ENDPOINT=https://otlp.eu-west-1a.arize.com/v1
ARIZE_PROJECT_NAME=text-to-sql-agent
```

Every tool call, sub-agent invocation, and final response is captured as a span. View at [app.arize.com](https://app.arize.com).
