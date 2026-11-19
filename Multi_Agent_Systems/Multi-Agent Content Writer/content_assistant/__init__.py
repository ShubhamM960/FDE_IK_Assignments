"""Smart Content Assistant - a LangGraph multi-agent system.

Modules
-------
config      settings, route names, LLM factory
state       AgentState (messages + route)
prompts     system prompts (incl. tool cost policy)
tools       research_tool (deep, costly) & internet_search_tool (quick, cheap)
nodes/      router, seo_blog_writer, x_blog_writer, general + should_continue fns
graph       graph wiring + MemorySaver checkpointer
"""

from content_assistant.graph import build_graph, make_config

__all__ = ["build_graph", "make_config"]
