"""Configuration: load API keys from .env and create the shared LLM.

Key lookup order (first real value wins, placeholders are ignored):
    1. Variables already set in your shell (export OPENAI_API_KEY=...)
    2. Multi_Travel_Agent/.env           (this project)
    3. <course root>/.env                (fallback, shared by other notes)
"""

import os
from pathlib import Path

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent      # Multi_Travel_Agent/
COURSE_ROOT = PROJECT_DIR.parent.parent                   # Agentic AI basics Notes/

REQUIRED_KEYS = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"]


def _is_placeholder(value) -> bool:
    return not value or value.strip().lower().startswith("your_")


def _load_env() -> None:
    sources = [dotenv_values(PROJECT_DIR / ".env"), dotenv_values(COURSE_ROOT / ".env")]
    for source in sources:
        for key, value in source.items():
            if _is_placeholder(os.environ.get(key)) and not _is_placeholder(value):
                os.environ[key] = value


_load_env()


def missing_keys() -> list[str]:
    """Return the required API keys that are not configured."""
    return [k for k in REQUIRED_KEYS if _is_placeholder(os.environ.get(k))]


MODEL_NAME = os.getenv("TRAVEL_MODEL", "gpt-4o-mini")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# One shared LLM for all agents (low temperature = more factual / consistent)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
