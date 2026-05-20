"""Model configuration for each agent in the workflow."""
from __future__ import annotations

from typing import TypedDict


class ModelsConfig(TypedDict):
    schema_inspection: str
    sql_generation: str
    sql_evaluation: str
    research: str
    orchestration: str


DEFAULT_MODELS: ModelsConfig = {
    "schema_inspection": "gemini-2.5-flash",
    "sql_generation": "gemini-2.5-pro",
    "sql_evaluation": "gemini-2.5-flash",
    "research": "gemini-2.5-flash",
    "orchestration": "gemini-2.5-pro",
}


def get_model_for_agent(agent_name: str, models: ModelsConfig | None = None) -> str:
    config = models or DEFAULT_MODELS
    if agent_name not in DEFAULT_MODELS:
        raise KeyError(f"Unknown agent '{agent_name}'. Valid: {list(DEFAULT_MODELS)}")
    return config.get(agent_name, DEFAULT_MODELS[agent_name])
