"""Agent tools.

- research_tool        : deep research (Tavily "advanced") - heavy & costly, call 1-2x.
- internet_search_tool : quick lookup  (Tavily "basic")    - light & cheap, call freely.

Without TAVILY_API_KEY both tools return mock data, so the app runs offline.
"""

import json
import os

from langchain_core.tools import tool


def _tavily(query: str, depth: str) -> str:
    """Shared Tavily call. depth = "advanced" (research) or "basic" (search)."""
    from tavily import TavilyClient

    resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query=query, search_depth=depth, max_results=5, include_answer=True
    )
    results = [{"title": r["title"], "url": r["url"], "snippet": r["content"][:500]} for r in resp["results"]]
    return json.dumps({"answer": resp.get("answer"), "results": results}, indent=2)


@tool
def research_tool(query: str) -> str:
    """Deep research: finds in-depth articles, papers, statistics and references.

    EXPENSIVE - call at most once or twice per blog, with one comprehensive query.
    Do NOT use it for SEO keywords, hashtags or trends.
    """
    if os.getenv("TAVILY_API_KEY"):
        return _tavily(query, "advanced")
    return json.dumps({
        "summary": f"{query}: rapid adoption, 20-30% efficiency gains reported, "
                   "main challenges are cost, data quality and skills.",
        "articles": [
            {"title": f"The State of {query} in 2026", "url": "https://www.mckinsey.com/insights/example"},
            {"title": f"{query}: A Systematic Review", "url": "https://arxiv.org/abs/2601.12345"},
            {"title": f"A Practical Guide to {query}", "url": "https://hbr.org/2026/01/example"},
        ],
    }, indent=2)


@tool
def internet_search_tool(query: str) -> str:
    """Quick, cheap web lookup for SEO keywords, trending topics and hashtags.

    Fine to call multiple times, e.g. "<topic> SEO keywords" or "<topic> trending hashtags".
    """
    if os.getenv("TAVILY_API_KEY"):
        return _tavily(query, "basic")
    tag = "".join(w.capitalize() for w in query.split()[:3])
    return json.dumps({
        "keywords": [query, f"what is {query}", f"{query} benefits", f"{query} best practices", f"{query} 2026"],
        "trending": [f"{query} breakthroughs this week", f"{query} myths vs reality"],
        "hashtags": [f"#{tag}", "#Tech", "#Innovation", "#AI"],
    }, indent=2)


SEO_TOOLS = [research_tool, internet_search_tool]
X_TOOLS = [internet_search_tool]
