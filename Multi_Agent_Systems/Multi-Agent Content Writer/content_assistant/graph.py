"""Graph wiring: nodes, conditional edges, tool loops and the MemorySaver checkpointer."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from content_assistant.config import RECURSION_LIMIT, ROUTE_GENERAL, ROUTE_SEO, ROUTE_X
from content_assistant.nodes import (
    general_node,
    route_decision,
    router_node,
    seo_blog_writer_node,
    should_continue_seo,
    should_continue_x,
    x_blog_writer_node,
)
from content_assistant.state import AgentState
from content_assistant.tools import SEO_TOOLS, X_TOOLS


def build_graph(checkpointer=None):
    """Build and compile the multi-agent graph.

    START -> router -(route)-> seo_blog_writer <-> seo_tools -> END
                            -> x_blog_writer   <-> x_tools   -> END
                            -> general                       -> END
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("router", router_node)
    builder.add_node(ROUTE_SEO, seo_blog_writer_node)
    builder.add_node("seo_tools", ToolNode(SEO_TOOLS))
    builder.add_node(ROUTE_X, x_blog_writer_node)
    builder.add_node("x_tools", ToolNode(X_TOOLS))
    builder.add_node(ROUTE_GENERAL, general_node)

    # Entry + 3-way routing
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_decision,
        {ROUTE_SEO: ROUTE_SEO, ROUTE_X: ROUTE_X, ROUTE_GENERAL: ROUTE_GENERAL},
    )

    # Agent <-> tool loops
    builder.add_conditional_edges(ROUTE_SEO, should_continue_seo, {"seo_tools": "seo_tools", END: END})
    builder.add_edge("seo_tools", ROUTE_SEO)

    builder.add_conditional_edges(ROUTE_X, should_continue_x, {"x_tools": "x_tools", END: END})
    builder.add_edge("x_tools", ROUTE_X)

    builder.add_edge(ROUTE_GENERAL, END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def make_config(thread_id: str) -> dict:
    """Config that must be passed on EVERY call so the checkpointer finds the thread."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
