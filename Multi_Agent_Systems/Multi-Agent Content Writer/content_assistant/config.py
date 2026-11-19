"""Central configuration: environment variables, model settings, and the LLM factory."""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# --- LLM settings -----------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ROUTER_TEMPERATURE = 0.0
WRITER_TEMPERATURE = 0.7
GENERAL_TEMPERATURE = 0.3

# --- Tool settings ----------------------------------------------------------
# Hard safety net for the expensive tool, enforced in code on top of the prompt.
# After this many research_tool calls in a single turn, the SEO agent is
# re-bound with only internet_search_tool.
MAX_RESEARCH_CALLS_PER_TURN = int(os.getenv("MAX_RESEARCH_CALLS_PER_TURN", "2"))

# Max graph steps for one request (protects against runaway tool loops).
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "25"))

# --- Route names (single source of truth) -----------------------------------
ROUTE_SEO = "seo_blog_writer"
ROUTE_X = "x_blog_writer"
ROUTE_GENERAL = "general"
VALID_ROUTES = (ROUTE_SEO, ROUTE_X, ROUTE_GENERAL)


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.3):
    """Return a cached chat model instance for the given temperature."""
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or the Streamlit sidebar."
        )
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", OPENAI_MODEL), temperature=temperature)
