"""Core configuration — DB connection and Vertex AI model factory.

Mirrors the GCP setup used in recommendation_agent_v10:
  - GOOGLE_CLOUD_PROJECT  / GOOGLE_CLOUD_LOCATION  (env vars)
  - GOOGLE_APPLICATION_CREDENTIALS pointing to service_account.json
  - GOOGLE_GENAI_USE_VERTEXAI=1
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=True)  # always override shell env

# If GOOGLE_APPLICATION_CREDENTIALS is a relative path, resolve it against project root
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds and not os.path.isabs(_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_PROJECT_ROOT / _creds)

# ── GCP ──────────────────────────────────────────────────────────────────────
GCP_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "grp-prd-lvmhai-jat1")
GCP_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ── PostgreSQL (pgvector-container) ──────────────────────────────────────────
PG_HOST: str = os.getenv("PG_HOST", "localhost")
PG_PORT: int = int(os.getenv("PG_PORT", "6024"))
PG_USER: str = os.getenv("PG_USER", "langchain")
PG_PASSWORD: str = os.getenv("PG_PASSWORD", "langchain")
PG_DB: str = os.getenv("PG_DB", "hm_fashion")

DATABASE_URL: str = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)


def _init_vertexai() -> None:
    """
    Configure Vertex AI via environment variables only.
    Avoids calling vertexai.init() which triggers a cloudresourcemanager DNS
    lookup that can fail in restricted network environments.
    The google-genai / google-adk client reads GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI and
    GOOGLE_APPLICATION_CREDENTIALS from the environment automatically.
    """
    # These are already set from .env via load_dotenv; set them explicitly
    # so any library that reads them directly is covered.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", GCP_LOCATION)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")


# Initialise on import
_init_vertexai()


from google.adk.models import Gemini  # noqa: E402 — after vertexai.init
from google.genai import Client       # noqa: E402


class VertexGemini(Gemini):
    """Gemini wrapper with lazy Vertex AI client initialisation."""

    _internal_client: Client | None = None

    @property
    def api_client(self) -> Client:  # type: ignore[override]
        if self._internal_client is None:
            self._internal_client = Client(
                vertexai=True,
                project=GCP_PROJECT,
                location=GCP_LOCATION,
            )
        return self._internal_client


@lru_cache(maxsize=16)
def get_model(model_name: str = "gemini-2.5-flash") -> VertexGemini:
    """Return a cached VertexGemini instance for *model_name*."""
    return VertexGemini(model=model_name)
