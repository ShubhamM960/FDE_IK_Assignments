"""Multi-agent travel planner built with LangGraph + LangChain.

Typical use:
    from travel_planner import build_graph
    graph = build_graph()
    graph.invoke({"user_query": "Best time to visit Jodhpur?"},
                 config={"configurable": {"thread_id": "1"}})
"""

from .graph import build_graph

__all__ = ["build_graph"]
