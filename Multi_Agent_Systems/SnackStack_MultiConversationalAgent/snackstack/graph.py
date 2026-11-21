"""
graph.py — Build and compile the SnackStack LangGraph StateGraph.

Graph topology (conditional edges shown as dashed):
    START → orchestrator ─ ─ ─ ┬─ menu_agent_node ─ ─ ─ menu_tools_node ─┐
                                │       ↑________________________ ________┘
                                │       └─ ─ ─ ─ → synthesizer_node → END
                                │
                                └─ order_agent_node ─ ─ ─ order_tools_node ─┐
                                        ↑___________________________________┘
                                        └─ ─ ─ ─ → synthesizer_node → END

    Orchestrator dispatches agents conditionally via Send().
    Each agent uses a conditional edge to route to its tools node
    or directly to the synthesizer.

HITL:
    order_agent_node uses interrupt() to pause execution when the user
    query is missing an order ID / tracking ID / email.  The caller
    resumes with Command(resume=<user_input>).
"""
from langgraph.graph import StateGraph, START, END
from snackstack.state import StackState
from langgraph.checkpoint.memory import MemorySaver
from snackstack.config import llm
from snackstack.logger import setup_logger
from snackstack.agents.orchestrator import orchestrator_node
from snackstack.agents.menu_agent import menu_agent_node
from snackstack.agents.order_agent import order_agent_node
from snackstack.agents.synthesizer import synthesizer_node

#─── Logger ───────────────────────────────────────────────────
logger = setup_logger("graph")

#─── Build Graph ───────────────────────────────────────────────────
def build_graph():
    graph_builder  = StateGraph(StackState)
    #── Graph Nodes ───────────────────────────────────────────────────
    graph_builder.add_node("orchestrator", orchestrator_node)
    graph_builder.add_node("menu_agent", menu_agent_node)
    graph_builder.add_node("order_agent", order_agent_node)
    graph_builder.add_node("synthesizer", synthesizer_node)

    #── Graph Edges ───────────────────────────────────────────────────
    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_edge("synthesizer", END)

    #── Memory ───────────────────────────────────────────────────
    memory = MemorySaver()
    
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


graph = build_graph()
