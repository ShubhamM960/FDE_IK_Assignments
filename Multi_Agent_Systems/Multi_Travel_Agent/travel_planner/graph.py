"""Wire the nodes into a LangGraph StateGraph.

    START
      |
    orchestrator ──(Send fan-out, parallel)──┬──> hotel_flight_agent_node <──> hotel_flight_tool_node
                                             └──> travel_scout_agent_node <──> travel_scout_tool_node
                                                          |
                                             synthesizer (defer=True, waits for all branches)
                                                          |
                                                         END

Quick test from the Multi_Travel_Agent/ folder:
    python -m travel_planner.graph "Best time to visit Jodhpur?"
"""

import sys
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    HOTEL_FLIGHT_AGENT,
    HOTEL_FLIGHT_TOOLS,
    ORCHESTRATOR,
    SYNTHESIZER,
    TRAVEL_SCOUT_AGENT,
    TRAVEL_SCOUT_TOOLS,
    dispatch_to_agents,
    hotel_flight_agent_node,
    hotel_flight_tool_node,
    orchestrator,
    should_continue_hotel_flight,
    should_continue_travel_scout,
    synthesizer_node,
    travel_scout_agent_node,
    travel_scout_tool_node,
)
from .state import TravelPlannerState


def build_graph(checkpointer=None):
    """Build and compile the travel planner graph."""
    builder = StateGraph(TravelPlannerState)

    # ---- Nodes ----
    builder.add_node(ORCHESTRATOR, orchestrator)
    builder.add_node(HOTEL_FLIGHT_AGENT, hotel_flight_agent_node)
    builder.add_node(HOTEL_FLIGHT_TOOLS, hotel_flight_tool_node)
    builder.add_node(TRAVEL_SCOUT_AGENT, travel_scout_agent_node)
    builder.add_node(TRAVEL_SCOUT_TOOLS, travel_scout_tool_node)
    # defer=True -> wait until ALL parallel branches are done, then run once
    builder.add_node(SYNTHESIZER, synthesizer_node, defer=True)

    # ---- Edges ----
    builder.add_edge(START, ORCHESTRATOR)
    builder.add_conditional_edges(
        ORCHESTRATOR, dispatch_to_agents, [HOTEL_FLIGHT_AGENT, TRAVEL_SCOUT_AGENT, SYNTHESIZER]
    )

    # Flight/hotel ReAct loop
    builder.add_conditional_edges(
        HOTEL_FLIGHT_AGENT, should_continue_hotel_flight, [HOTEL_FLIGHT_TOOLS, SYNTHESIZER]
    )
    builder.add_edge(HOTEL_FLIGHT_TOOLS, HOTEL_FLIGHT_AGENT)

    # Travel scout ReAct loop
    builder.add_conditional_edges(
        TRAVEL_SCOUT_AGENT, should_continue_travel_scout, [TRAVEL_SCOUT_TOOLS, SYNTHESIZER]
    )
    builder.add_edge(TRAVEL_SCOUT_TOOLS, TRAVEL_SCOUT_AGENT)

    builder.add_edge(SYNTHESIZER, END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


if __name__ == "__main__":
    graph = build_graph()
    query = " ".join(sys.argv[1:]) or "Best time to visit Jodhpur?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\nQ: {query}\n")
    for update in graph.stream({"user_query": query}, config=config, stream_mode="updates"):
        for node in update:
            print(f"  -> {node}")
    print("\n" + graph.get_state(config).values.get("final_answer", ""))
