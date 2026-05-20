"""
Text-to-SQL Agent — Streamlit UI
==================================
Run:
    poetry run streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path

# ── Arize OTEL — must be imported before any google.adk import ───────────────
if "_arize_text_to_sql_instrumented" not in sys.modules:
    from text_to_sql_adk import tracing_setup  # noqa: F401

import streamlit as st
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=True)

# Fix relative GOOGLE_APPLICATION_CREDENTIALS
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds and not os.path.isabs(_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_HERE / _creds)

# ── ADK imports ───────────────────────────────────────────────────────────────
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from text_to_sql_adk.agent import create_text_to_sql_agent
from text_to_sql_adk.core.tools.sql_executor import execute_sql, results_to_markdown

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="H&M Text-to-SQL Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .sql-box {
        background: #0d1117; color: #c9d1d9;
        border-radius: 8px; padding: 14px 18px;
        font-family: 'Menlo', 'Monaco', monospace; font-size: 0.82rem;
        white-space: pre-wrap; overflow-x: auto;
        border: 1px solid #30363d; margin: 8px 0 12px 0;
    }
    .badge-correct  { background:#d4edda; color:#155724; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-partial  { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-incorrect{ background:#f8d7da; color:#721c24; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .metric-row { display:flex; gap:16px; margin:8px 0; flex-wrap:wrap; }
    .metric-pill {
        background:#f0f2f5; border-radius:20px;
        padding:4px 12px; font-size:0.78rem; color:#444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Example queries by level ──────────────────────────────────────────────────
EXAMPLE_QUERIES = {
    "L1 — Simple filter": [
        "Show me all Black Dress items",
        "Find all White Shirt articles",
        "List all Blue Sneakers",
    ],
    "L2 — Multi-filter": [
        "Find Blue Top items with a Stripe pattern",
        "Show Ladieswear Blazer items that are Solid colour",
        "Find Black Cardigan items in the Premium section",
    ],
    "L3 — Aggregation": [
        "What are the top 10 most common colours in the catalogue?",
        "How many articles are there per product type?",
        "What is the average price per sales channel?",
    ],
    "L4 — JOIN": [
        "Which Jacket items were sold the most?",
        "What are the top 5 best-selling product types by transaction count?",
        "Which articles have never been purchased?",
    ],
    "L5 — 3-table JOIN": [
        "What Red Sneaker items did customers aged 25 to 35 buy in Winter?",
        "Which Dress items did active club members buy online in Summer?",
        "What items did customers over 50 buy most frequently?",
    ],
    "L6 — CTE / Window": [
        "For each product type, what was the top-selling colour?",
        "What is the repurchase rate for Blouse items?",
        "Rank the top 3 article types per sales channel by revenue.",
    ],
}


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []           # {role, content, meta?}
    if "runner" not in st.session_state:
        agent = create_text_to_sql_agent()
        svc = InMemorySessionService()
        runner = Runner(agent=agent, app_name="text_to_sql", session_service=svc)
        asyncio.run(
            svc.create_session(
                app_name="text_to_sql",
                user_id="user",
                session_id=st.session_state.session_id,
            )
        )
        st.session_state.runner = runner
        st.session_state.svc = svc


_init_state()


# ── Agent runner ──────────────────────────────────────────────────────────────

async def _run_async(user_message: str) -> tuple[str, list[dict]]:
    """Stream the agent and collect final text + tool trace."""
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=user_message)]
    )
    final_text = ""
    tool_trace: list[dict] = []

    async for event in st.session_state.runner.run_async(
        user_id="user",
        session_id=st.session_state.session_id,
        new_message=content,
    ):
        # Collect tool calls / responses for the trace panel
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_trace.append(
                        {
                            "type": "call",
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                    )
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    resp = fr.response if fr.response else {}
                    tool_trace.append({"type": "response", "name": fr.name, "response": resp})

        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""

    return final_text, tool_trace


def run_agent(user_message: str) -> tuple[str, list[dict]]:
    return asyncio.run(_run_async(user_message))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_sql(text: str) -> str | None:
    """Pull the first SQL code block out of the agent response."""
    m = re.search(r"```(?:sql)?\n(.+?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _verdict_badge(verdict: str) -> str:
    cls = {"CORRECT": "badge-correct", "PARTIAL": "badge-partial"}.get(
        verdict, "badge-incorrect"
    )
    return f'<span class="{cls}">{verdict}</span>'


def _render_tool_trace(trace: list[dict]):
    """Render the agent tool call chain in an expander."""
    if not trace:
        return
    with st.expander("🔍 Agent trace", expanded=False):
        for step in trace:
            if step["type"] == "call":
                args_str = str(step.get("args", {}))[:300]
                st.markdown(f"→ **`{step['name']}`** `{args_str}`")
            else:
                resp = step.get("response", {})
                result_str = str(resp.get("result", resp))[:500]
                st.caption(f"↩ `{step['name']}`: {result_str}")


def _render_result_table(sql: str):
    """Execute the SQL and render results as a dataframe."""
    if not sql:
        return
    import pandas as pd
    res = execute_sql(sql, max_rows=100)
    if res["success"] and res["rows"]:
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        st.dataframe(df, use_container_width=True)
        st.caption(
            f"⏱ {res['execution_time_ms']:.1f} ms · "
            f"{res['row_count']} row{'s' if res['row_count'] != 1 else ''} total"
        )
    elif res["success"]:
        st.info("Query returned 0 rows.")
    else:
        st.error(f"Execution error: {res['error_message']}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ H&M Text-to-SQL")
    st.caption("Powered by Google ADK · Gemini 2.5 · PostgreSQL")
    st.divider()

    st.markdown("### 💡 Example queries")
    for level, queries in EXAMPLE_QUERIES.items():
        with st.expander(level, expanded=False):
            for q in queries:
                if st.button(q, key=f"ex_{q}", use_container_width=True):
                    st.session_state["pending_query"] = q
                    st.rerun()

    st.divider()
    st.markdown("### 🗄️ Database")
    st.markdown(
        """
| Table | Rows |
|-------|------|
| `articles` | 5,000 |
| `customers` | 10,000 |
| `transactions` | 50,000 |
        """
    )

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        for k in ["messages", "runner", "svc", "session_id"]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    # Arize AX tracing link
    _arize_project = os.getenv("ARIZE_PROJECT_NAME", "text-to-sql-agent")
    st.markdown(
        f"[![Arize AX](https://img.shields.io/badge/Arize%20AX-Traces-blue)]"
        f"(https://app.arize.com)  \n"
        f"📡 Tracing → **{_arize_project}**"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🛍️ H&M Fashion Analytics — Text-to-SQL Agent")
st.caption(
    "Ask any question about H&M fashion data in plain English. "
    "The agent will generate SQL, execute it, and return the results."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            # SQL block
            if meta.get("sql"):
                with st.expander("📄 Generated SQL", expanded=True):
                    st.code(meta["sql"], language="sql")
            # Result table
            if meta.get("sql"):
                _render_result_table(meta["sql"])
            # Tool trace
            _render_tool_trace(meta.get("trace", []))


# ── Input handling ────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
user_input = pending or st.chat_input("Ask about H&M fashion data…")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        status = st.status("⚙️ Running agent…", expanded=True)
        with status:
            st.write("🔎 Inspecting schema…")
            response_text, trace = run_agent(user_input)
            # Count tool calls to give better status
            tool_calls = [s["name"] for s in trace if s["type"] == "call"]
            st.write(f"✅ Done — {len(tool_calls)} tool call(s): {', '.join(dict.fromkeys(tool_calls))}")
        status.update(label="✅ Agent finished", state="complete", expanded=False)

        # Main response
        st.markdown(response_text)

        # Extract SQL and render results
        sql = _extract_sql(response_text)
        if sql:
            with st.expander("📄 Generated SQL", expanded=True):
                st.code(sql, language="sql")
            _render_result_table(sql)

        # Tool trace
        _render_tool_trace(trace)

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "meta": {"sql": sql, "trace": trace},
        }
    )
    st.rerun()
