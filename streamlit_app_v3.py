"""
Text-to-SQL Agent V3 — Streamlit UI with Eval Progress + Dashboard
===================================================================
New in V3 (on top of V2):
  - Live progress bar while evaluation is running (polls a temp progress file)
  - Eval Dashboard tab in the main area: bar charts per level for exact-match
    rate, average semantic score, and per-query verdict breakdown
  - Tabs layout: 💬 Chat  |  📊 Eval Dashboard

Run:
    poetry run streamlit run streamlit_app_v3.py --server.port 8505
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time as _time
import uuid
from pathlib import Path

import threading

# ── Dedicated event loop — persisted across Streamlit hot-reloads ─────────────
# On every hot-reload Streamlit re-executes this file, which would normally
# create a brand-new _ADK_LOOP. Meanwhile get_model() is @lru_cache and keeps
# returning the same VertexGemini instance whose aiohttp.ClientSession is bound
# to the OLD loop → "Future attached to a different loop".
# Storing the loop in sys.modules makes it a true process-level singleton.
_ADK_WORKER_KEY = "_text_to_sql_adk_worker"
if _ADK_WORKER_KEY not in sys.modules:
    _ADK_LOOP: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    _adk_thread = threading.Thread(
        target=_ADK_LOOP.run_forever, daemon=True, name="adk-worker"
    )
    _adk_thread.start()
    sys.modules[_ADK_WORKER_KEY] = (_ADK_LOOP, _adk_thread)  # type: ignore
else:
    _ADK_LOOP, _adk_thread = sys.modules[_ADK_WORKER_KEY]  # type: ignore

# ── Arize OTEL — must be imported before any google.adk import ───────────────
if "_arize_text_to_sql_instrumented" not in sys.modules:
    from text_to_sql_adk import tracing_setup  # noqa: F401

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=True)
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds and not os.path.isabs(_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_HERE / _creds)

# ── ADK imports ───────────────────────────────────────────────────────────────
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from text_to_sql_adk.agent import create_text_to_sql_agent
from text_to_sql_adk.core.tools.sql_executor import execute_sql

# ── Image / metadata config ───────────────────────────────────────────────────
_IMAGES_BASE = Path("/Users/DIOPAB/Downloads/lvprojects/adk_test/data/images")
_META_PATH   = Path("/Users/DIOPAB/Downloads/lvprojects/adk_test/data/product_metadata.parquet")

# product_metadata.parquet now stores real EUR prices from products.csv
# (price column = actual EUR, no conversion factor needed)
PRICE_FACTOR = 1


@st.cache_data(show_spinner=False)
def _load_metadata() -> pd.DataFrame:
    if _META_PATH.exists():
        return pd.read_parquet(_META_PATH).set_index("article_id")
    return pd.DataFrame()


def _image_path(article_id: int) -> Path:
    aid = str(article_id).zfill(10)
    return _IMAGES_BASE / aid[:3] / f"{aid}.jpg"


def _b64_image(path: Path) -> str | None:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def _enrich_articles(article_ids: list[int]) -> list[dict]:
    """Return product card dicts with image + metadata for a list of article_ids."""
    meta = _load_metadata()
    cards = []
    for aid in article_ids[:20]:   # cap at 20 cards
        card: dict = {"article_id": aid}
        if len(meta) and aid in meta.index:
            row = meta.loc[aid]
            card["prod_name"]    = row.get("prod_name", "—")
            card["product_type"] = row.get("product_type_name", "")
            card["colour"]       = row.get("colour_group_name", "")
            card["index_group"]  = row.get("index_group_name", "")
            # avg_price now holds real EUR from products.csv; guard against NaN
            raw_price = row.get("avg_price")
            try:
                price_val = float(raw_price)
                card["price_eur"] = round(price_val, 2) if price_val == price_val else None  # NaN check
            except (TypeError, ValueError):
                card["price_eur"] = None
            card["n_purchases"] = int(row.get("n_purchases", 0) or 0)
        else:
            card["prod_name"] = f"Article {aid}"
        card["img_b64"] = _b64_image(_image_path(aid))
        cards.append(card)
    return cards


def _extract_article_ids(result: dict) -> list[int]:
    """Pull article_id values from a SQL execution result."""
    cols = [c.lower() for c in result.get("columns", [])]
    if "article_id" not in cols:
        return []
    idx = cols.index("article_id")
    ids = []
    for row in result.get("rows", []):
        try:
            ids.append(int(row[idx]))
        except (ValueError, TypeError):
            pass
    return ids


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H&M Text-to-SQL V3",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ── Example queries ───────────────────────────────────────────────────────────
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
        "Top 5 best-selling product types by transaction count",
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
async def _create_runner_async(session_id: str):
    """Create agent + runner entirely inside _ADK_LOOP so the google-genai
    aiohttp.ClientSession binds to _ADK_LOOP from the very first use."""
    agent = create_text_to_sql_agent()
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="text_to_sql_v2", session_service=svc)
    await svc.create_session(
        app_name="text_to_sql_v2",
        user_id="user",
        session_id=session_id,
    )
    return runner, svc


def _init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "runner" not in st.session_state:
        # Everything created inside _ADK_LOOP — aiohttp session binds to it
        future = asyncio.run_coroutine_threadsafe(
            _create_runner_async(st.session_state.session_id),
            _ADK_LOOP,
        )
        runner, svc = future.result(timeout=60)
        st.session_state.runner = runner
        st.session_state.svc = svc

_init_state()

# ── Agent runner ──────────────────────────────────────────────────────────────
async def _run_async(user_message: str, runner, session_id: str) -> tuple[str, list[dict]]:
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=user_message)]
    )
    final_text = ""
    tool_trace: list[dict] = []

    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=content,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_trace.append({"type": "call", "name": fc.name,
                                       "args": dict(fc.args) if fc.args else {}})
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    tool_trace.append({"type": "response", "name": fr.name,
                                       "response": fr.response or {}})
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""

    return final_text, tool_trace


def run_agent(msg: str) -> tuple[str, list[dict]]:
    runner = st.session_state.runner
    session_id = st.session_state.session_id
    future = asyncio.run_coroutine_threadsafe(
        _run_async(msg, runner, session_id), _ADK_LOOP
    )
    return future.result(timeout=300)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql)?\n(.+?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _product_strip(cards: list[dict]) -> None:
    """Render an inline horizontal scrollable product strip (like recommendation_agent_v10)."""
    cards_html = ""
    for c in cards:
        if c.get("img_b64"):
            img_html = (
                f"<img src='data:image/jpeg;base64,{c['img_b64']}' "
                f"style='width:160px;height:190px;object-fit:cover;"
                f"border-radius:8px;display:block;margin-bottom:8px;'/>"
            )
        else:
            img_html = (
                "<div style='width:160px;height:190px;background:#f0f0f0;"
                "border-radius:8px;display:flex;align-items:center;"
                "justify-content:center;color:#bbb;font-size:2rem;"
                "margin-bottom:8px;'>🖼️</div>"
            )

        price_str = f"€{c['price_eur']:,.0f}" if c.get("price_eur") else ""
        purchases = f"{c['n_purchases']:,} sold" if c.get("n_purchases") else ""
        ptype     = c.get("product_type", "")
        colour    = c.get("colour", "")
        meta_line = " · ".join(filter(None, [ptype, colour]))

        cards_html += f"""
        <div style='min-width:180px;max-width:180px;background:#fafafa;
                    border:1px solid #e0e0e0;border-radius:12px;
                    padding:10px;flex-shrink:0;'>
            {img_html}
            <div style='font-weight:700;font-size:0.82rem;
                        white-space:nowrap;overflow:hidden;
                        text-overflow:ellipsis;margin-bottom:2px;'>{c.get('prod_name','—')}</div>
            <div style='color:#888;font-size:0.73rem;
                        white-space:nowrap;overflow:hidden;
                        text-overflow:ellipsis;margin-bottom:6px;'>{meta_line}</div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-weight:700;font-size:0.85rem;color:#1a1a1a;'>{price_str}</span>
                <span style='color:#aaa;font-size:0.70rem;'>{purchases}</span>
            </div>
        </div>"""

    n = len(cards)
    prices = [c["price_eur"] for c in cards if c.get("price_eur")]
    price_info = ""
    if prices:
        price_info = f" &nbsp;·&nbsp; min <b>€{min(prices):,.0f}</b> &nbsp;avg <b>€{sum(prices)/len(prices):,.0f}</b> &nbsp;max <b>€{max(prices):,.0f}</b>"

    st.markdown(
        f"<div style='color:#555;font-size:0.82rem;padding:4px 0 6px 0;'>"
        f"🖼️ <b>{n} article{'s' if n!=1 else ''} found</b>{price_info}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style='overflow-x:auto;display:flex;flex-direction:row;
                    gap:14px;padding:14px 4px 18px 4px;
                    border-top:1px solid #e0e0e0;margin-top:4px;'>
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tool_trace(trace: list[dict]):
    if not trace:
        return
    with st.expander("🔍 Agent trace", expanded=False):
        for step in trace:
            if step["type"] == "call":
                args_str = str(step.get("args", {}))[:300]
                st.markdown(f"→ **`{step['name']}`** `{args_str}`")
            else:
                resp = step.get("response", {})
                st.caption(f"↩ `{step['name']}`: {str(resp.get('result', resp))[:400]}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ H&M Text-to-SQL V3")
    st.caption("SQL · Images · Eval Dashboard · Google ADK · Gemini 2.5 · Arize AX")
    st.divider()

    st.markdown("### 💡 Example queries")
    for level, queries in EXAMPLE_QUERIES.items():
        with st.expander(level, expanded=False):
            for q in queries:
                if st.button(q, key=f"ex_{q}", width="stretch"):
                    st.session_state["pending_query"] = q
                    st.rerun()

    st.divider()
    st.markdown("### 🧪 Benchmark Evaluation")
    st.caption("Run `evaluate.py` against `data/dataset.jsonl` (277 queries, L1–L6).")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_level = st.selectbox(
            "Level filter",
            options=["All", "L1", "L2", "L3", "L4", "L5", "L6"],
            key="eval_level",
            label_visibility="collapsed",
        )
    with eval_col2:
        eval_limit = st.number_input(
            "Max queries",
            min_value=1, max_value=277, value=10, step=5,
            key="eval_limit",
            label_visibility="collapsed",
        )

    eval_output = st.text_input(
        "Output file",
        value="results_eval.json",
        key="eval_output",
        label_visibility="collapsed",
    )

    if st.button("🚀 Run Evaluation", width="stretch", type="primary"):
        import subprocess, sys as _sys, tempfile
        _progress_file = Path(tempfile.mktemp(suffix="_eval_progress.json"))
        cmd = [
            _sys.executable, str(Path(__file__).parent / "evaluate.py"),
            "--limit", str(int(eval_limit)),
            "--output", eval_output,
            "--progress-file", str(_progress_file),
        ]
        if eval_level != "All":
            cmd += ["--level", eval_level[1:]]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        st.session_state["eval_proc"] = proc
        st.session_state["eval_progress_file"] = str(_progress_file)
        st.session_state["eval_running"] = True
        st.session_state["eval_total"] = int(eval_limit)
        st.rerun()

    # ── Live progress polling ────────────────────────────────────────────
    if st.session_state.get("eval_running"):
        proc = st.session_state.get("eval_proc")
        pfile = Path(st.session_state.get("eval_progress_file", ""))
        total_q = st.session_state.get("eval_total", 1)

        done, total_q = 0, total_q
        if pfile.exists():
            try:
                pg = json.loads(pfile.read_text())
                done = pg.get("done", 0)
                total_q = pg.get("total", total_q)
            except Exception:
                pass

        pct = done / max(total_q, 1)
        st.markdown(f"**Evaluating… {done}/{total_q} queries**")
        st.progress(pct)

        if proc and proc.poll() is not None:  # finished
            stdout, stderr = proc.communicate()
            st.session_state["eval_running"] = False
            try:
                pfile.unlink(missing_ok=True)
            except Exception:
                pass
            if proc.returncode == 0:
                st.success("✅ Evaluation complete! Open the 📊 Eval Dashboard tab.")
                st.session_state["eval_results_file"] = eval_output
            else:
                st.error("❌ Evaluation failed")
                with st.expander("📋 Error output", expanded=True):
                    st.code((stderr or stdout or "no output")[-3000:], language=None)
        else:
            _time.sleep(1)
            st.rerun()

    st.divider()
    st.markdown("### 🗄️ Database")
    st.markdown("""
| Table | Rows |
|-------|------|
| `articles` | 69,723 |
| `customers` | 100,000 |
| `transactions` | 500,000 |
    """)
    st.markdown("### 🖼️ Images")
    total_imgs = sum(len(f) for _, _, f in os.walk(_IMAGES_BASE)) if _IMAGES_BASE.exists() else 0
    st.markdown(f"`{total_imgs:,}` product photos available")

    st.divider()
    _arize_project = os.getenv("ARIZE_PROJECT_NAME", "text-to-sql-agent")
    st.markdown(
        f"📡 **Arize AX** tracing → `{_arize_project}`  \n"
        f"[Open dashboard →](https://app.arize.com)"
    )

    st.divider()
    if st.button("🗑️ New conversation", width="stretch"):
        for k in ["messages", "runner", "svc", "session_id"]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ── Main layout: tabs ────────────────────────────────────────────────────────
st.title("🛍️ H&M Fashion Analytics — Text-to-SQL V3")
st.caption(
    "Ask any fashion question in plain English. "
    "The agent generates SQL, executes it, and shows product images. "
    "Run an evaluation then switch to the 📊 Eval Dashboard tab."
)

_tab_chat, _tab_dashboard = st.tabs(["💬 Chat", "📊 Eval Dashboard"])

# ── Eval Dashboard ────────────────────────────────────────────────────────────
with _tab_dashboard:
    _res_file_key = st.session_state.get("eval_results_file", "results_eval.json")
    _res_path = Path(__file__).parent / _res_file_key

    # File picker to load any results file
    _custom = st.text_input("Results JSON file", value=_res_file_key, key="dashboard_file")
    if _custom:
        _res_path = Path(__file__).parent / _custom

    if not _res_path.exists():
        st.info("No evaluation results found. Run an evaluation from the sidebar first.")
    else:
        try:
            _report = json.loads(_res_path.read_text())
        except Exception as _e:
            st.error(f"Could not parse results file: {_e}")
            _report = {}

        if _report:
            # ── Top KPI cards ────────────────────────────────────────────────
            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric("Total queries", _report.get("total_queries", 0))
            _k2.metric("Exact match rate",
                       f"{_report.get('overall_exact_match_rate', 0):.1%}")
            _k3.metric("Avg semantic score",
                       f"{_report.get('overall_semantic_score', 0):.3f}")
            _k4.metric("Overall score",
                       f"{_report.get('overall_score', 0):.3f}")

            st.divider()

            # ── Per-level bar charts ─────────────────────────────────────────
            _by_level = _report.get("by_level", {})
            if _by_level:
                _lvl_df = pd.DataFrame([
                    {
                        "Level": f"L{lvl}",
                        "Queries": s["total"],
                        "Exact match %": round(s["exact_match_rate"] * 100, 1),
                        "Avg semantic": round(s["avg_semantic"], 3),
                        "Avg ROUGE-L": round(s.get("avg_rouge_l", 0), 3),
                        "Avg emb sim": round(s.get("avg_embedding_similarity", 0), 3),
                    }
                    for lvl, s in sorted(_by_level.items(), key=lambda x: int(x[0]))
                ])

                _bc1, _bc2 = st.columns(2)
                with _bc1:
                    st.markdown("**Exact Match Rate (%) by Level**")
                    st.bar_chart(_lvl_df.set_index("Level")["Exact match %"],
                                 color="#4CAF50", height=280)
                with _bc2:
                    st.markdown("**Avg Semantic Score by Level**")
                    st.bar_chart(_lvl_df.set_index("Level")["Avg semantic"],
                                 color="#2196F3", height=280)

                _bc3, _bc4 = st.columns(2)
                with _bc3:
                    st.markdown("**Avg ROUGE-L (SQL token overlap) by Level**")
                    st.bar_chart(_lvl_df.set_index("Level")["Avg ROUGE-L"],
                                 color="#9C27B0", height=240)
                with _bc4:
                    st.markdown("**Avg Embedding Similarity (text-embedding-004) by Level**")
                    st.bar_chart(_lvl_df.set_index("Level")["Avg emb sim"],
                                 color="#FF5722", height=240)

                st.dataframe(_lvl_df, hide_index=True, use_container_width=True)

            st.divider()

            # ── Per-query verdict breakdown ──────────────────────────────────
            _records = _report.get("records", [])
            if _records:
                st.markdown("**Per-query results**")
                _rec_df = pd.DataFrame([
                    {
                        "ID": r["query_id"],
                        "Level": f"L{r['level']}",
                        "Category": r.get("category", ""),
                        "Question": r["nl_query"],
                        "Verdict": r["evaluation"].get("verdict", ""),
                        "Exact ✓": "✅" if r["evaluation"].get("execution_match") else "❌",
                        "Semantic": round(r["evaluation"].get("semantic_score", 0), 3),
                        "ROUGE-L": round(r["evaluation"].get("rouge_l", 0), 3),
                        "Col Prec": round(r["evaluation"].get("column_precision", 0), 3),
                        "Col Rec": round(r["evaluation"].get("column_recall", 0), 3),
                        "Row ∩/∪": round(r["evaluation"].get("row_overlap", 0), 3),
                        "Emb Sim": round(r["evaluation"].get("embedding_similarity", 0), 3),
                        "Overall": round(r["evaluation"].get("overall_score", 0), 3),
                    }
                    for r in _records
                ])

                # Verdict distribution bar
                _verd_counts = _rec_df["Verdict"].value_counts().reset_index()
                _verd_counts.columns = ["Verdict", "Count"]
                st.markdown("**Verdict distribution**")
                st.bar_chart(_verd_counts.set_index("Verdict")["Count"],
                             color="#FF9800", height=220)

                _filter_lvl = st.multiselect(
                    "Filter by level",
                    options=sorted(_rec_df["Level"].unique()),
                    default=sorted(_rec_df["Level"].unique()),
                    key="dashboard_lvl_filter",
                )
                _filtered = _rec_df[_rec_df["Level"].isin(_filter_lvl)] if _filter_lvl else _rec_df
                st.dataframe(
                    _filtered,
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Question": st.column_config.TextColumn(width="large"),
                        "Verdict": st.column_config.TextColumn(width="small"),
                    },
                )

with _tab_chat:
    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                if meta.get("sql"):
                    with st.expander("📄 Generated SQL", expanded=False):
                        st.code(meta["sql"], language="sql")
                if meta.get("result"):
                    res = meta["result"]
                    if res.get("success") and res.get("rows"):
                        df_res = pd.DataFrame(res["rows"], columns=res["columns"])
                        st.dataframe(df_res, width="stretch", height=200)
                        st.caption(
                            f"⏱ {res['execution_time_ms']:.0f} ms · "
                            f"{res['row_count']} row{'s' if res['row_count']!=1 else ''}"
                        )
                    elif res.get("success"):
                        st.info("Query returned 0 rows.")
                    else:
                        st.error(res.get("error_message", "Execution error"))
                _render_tool_trace(meta.get("trace", []))
                if meta.get("cards"):
                    _product_strip(meta["cards"])

    # ── Input ─────────────────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_query", None)
    user_input = pending or st.chat_input("Ask about H&M fashion data…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status = st.status("⚙️ Running agent…", expanded=True)
            with status:
                st.write("🔎 Inspecting schema…")
                response_text, trace = run_agent(user_input)
                tool_names = list(dict.fromkeys(
                    s["name"] for s in trace if s["type"] == "call"
                ))
                st.write(f"✅ Done — tools: {', '.join(tool_names)}")
            status.update(label="✅ Agent finished", state="complete", expanded=False)

            st.markdown(response_text)

            sql = _extract_sql(response_text)
            result = None
            article_ids: list[int] = []

            if sql:
                with st.expander("📄 Generated SQL", expanded=True):
                    st.code(sql, language="sql")

                result = execute_sql(sql, max_rows=100)
                if result.get("success") and result.get("rows"):
                    df_res = pd.DataFrame(result["rows"], columns=result["columns"])
                    st.dataframe(df_res, width="stretch", height=220)
                    st.caption(
                        f"⏱ {result['execution_time_ms']:.0f} ms · "
                        f"{result['row_count']} row{'s' if result['row_count']!=1 else ''}"
                    )
                    article_ids = _extract_article_ids(result)
                elif result.get("success"):
                    st.info("Query returned 0 rows.")
                else:
                    st.error(result.get("error_message", "Execution error"))

            _render_tool_trace(trace)

            cards: list[dict] = []
            if article_ids:
                with st.spinner(f"Loading {min(len(article_ids),20)} product images…"):
                    cards = _enrich_articles(article_ids)
                _product_strip(cards)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "meta": {"sql": sql, "result": result, "trace": trace, "cards": cards},
        })
        st.rerun()
