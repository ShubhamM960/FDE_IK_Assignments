"""(Optional) Alternative approach using LangChain `create_agent`.

Here each agent is a self-contained ReAct agent, and a `main_agent` router calls
the sub-agents *as tools*. This is simpler than the LangGraph version but:
  - sub-agents run one after another (no parallel fan-out)
  - you can't see / control each step as a graph node

The Streamlit app uses the LangGraph version (graph.py). Use this file to compare:

    from travel_planner.agents import main_agent
    out = main_agent.invoke({"messages": [{"role": "user", "content": "Best time to visit Jodhpur?"}]})
    print(out["messages"][-1].content)
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from .config import llm
from .prompts import HOTEL_FLIGHT_SEARCH_PROMPT, MAIN_AGENT_PROMPT, TRAVEL_SCOUT_PROMPT
from .tools import hotel_flight_tools, travel_scout_tools

# ---- Sub-agents -----------------------------------------------------------
hotel_flight_search_agent = create_agent(
    model=llm,
    system_prompt=HOTEL_FLIGHT_SEARCH_PROMPT,
    tools=hotel_flight_tools,
)

travel_scout_agent = create_agent(
    model=llm,
    system_prompt=TRAVEL_SCOUT_PROMPT,
    tools=travel_scout_tools,
)


def _ask(agent, query: str) -> str:
    """Run a sub-agent and return only its final text answer."""
    out = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return out["messages"][-1].content


# ---- Sub-agents wrapped as tools ------------------------------------------
@tool
def call_travel_flight_agent(query: str) -> str:
    """Search flights and/or hotels. Include cities, dates and number of travellers."""
    return _ask(hotel_flight_search_agent, query)


@tool
def call_travel_scout_agent(query: str) -> str:
    """Travel guidance (weather, visas, tips) and day-by-day itineraries."""
    return _ask(travel_scout_agent, query)


# ---- Router agent ---------------------------------------------------------
main_agent = create_agent(
    model=llm,
    system_prompt=MAIN_AGENT_PROMPT,
    tools=[call_travel_flight_agent, call_travel_scout_agent],
)
