"""
Arize AX OTEL Tracing Setup for Text-to-SQL Agent
===================================================
IMPORT THIS MODULE BEFORE ANY google.adk IMPORT.

Sets up:
  1. gRPC SSL fix (certifi CA bundle) + truststore system CA injection
  2. Arize AX OTEL registration via arize.otel.register()
  3. GoogleADKInstrumentor — auto-captures all ADK agent/tool/LLM spans

Usage:
    from text_to_sql_adk import tracing_setup  # noqa — must be first
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

# ── Load .env early so credentials are available ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env", override=True)

_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds and not os.path.isabs(_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_PROJECT_ROOT / _creds)

# ── SSL fix 1: certifi CA bundle for gRPC ────────────────────────────────────
# gRPC uses its own TLS stack that ignores Python's ssl module.
import certifi

os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ── SSL fix 2: truststore injects macOS/system keychain into Python ssl ───────
# This fixes CERTIFICATE_VERIFY_FAILED for HTTPS exporters (requests / urllib3).
import truststore
truststore.inject_into_ssl()

# ── Arize AX credentials ──────────────────────────────────────────────────────
ARIZE_SPACE_ID           = os.getenv("ARIZE_SPACE_ID", "")
ARIZE_API_KEY            = os.getenv("ARIZE_API_KEY", "")
ARIZE_PROJECT_NAME       = os.getenv("ARIZE_PROJECT_NAME", "text-to-sql-agent")
# Endpoint must include the /traces path
ARIZE_COLLECTOR_ENDPOINT = os.getenv(
    "ARIZE_COLLECTOR_ENDPOINT", "https://otlp.eu-west-1a.arize.com/v1"
).rstrip("/") + "/traces"

# ── Instrument once — use sys.modules as a persistent flag ───────────────────
# This prevents double-instrumentation on Streamlit rerenders.
_FLAG = "_arize_text_to_sql_instrumented"

if _FLAG not in sys.modules:
    sys.modules[_FLAG] = True  # type: ignore

    if not ARIZE_SPACE_ID or not ARIZE_API_KEY:
        log.warning(
            "⚠️  Arize AX credentials not set (ARIZE_SPACE_ID / ARIZE_API_KEY). "
            "Agent will run without OTEL tracing."
        )
    else:
        try:
            from arize.otel import register, Transport
            from openinference.instrumentation.google_adk import GoogleADKInstrumentor

            tracer_provider = register(
                space_id=ARIZE_SPACE_ID,
                api_key=ARIZE_API_KEY,
                project_name=ARIZE_PROJECT_NAME,
                transport=Transport.HTTP,
                endpoint=ARIZE_COLLECTOR_ENDPOINT,
            )

            GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

            log.info(
                "✅ Arize AX + GoogleADKInstrumentor activated — project: %s",
                ARIZE_PROJECT_NAME,
            )
            print(
                f"✅ Arize AX tracing active — project: {ARIZE_PROJECT_NAME}\n"
                f"   Endpoint: {ARIZE_COLLECTOR_ENDPOINT}\n"
                f"   Dashboard: https://app.arize.com"
            )

        except Exception as exc:
            log.warning("⚠️  Arize instrumentation failed: %s", exc)
            print(f"⚠️  Arize tracing skipped: {exc}")
