"""Graph nodes: orchestrator -> agent/tool loops -> synthesizer.

Each agent is implemented as TWO nodes that form a ReAct loop:
    <agent>_node       : LLM decides -> answer OR request tool calls
    <agent>_tool_node  : runs the requested tools, feeds results back
A router function (`should_continue_*`) decides whether to loop or finish.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

from .config import llm
from .prompts import HOTEL_FLIGHT_SEARCH_PROMPT, ROUTER_PROMPT, SYNTHESIS_PROMPT, TRAVEL_SCOUT_PROMPT
from .state import ClassificationResult, TravelPlannerState
from .tools import hotel_flight_tools, travel_scout_tools

# ---- Node names (single source of truth, also used in graph.py) ------------
ORCHESTRATOR = "orchestrator"
HOTEL_FLIGHT_AGENT = "hotel_flight_agent_node"
HOTEL_FLIGHT_TOOLS = "hotel_flight_tool_node"
TRAVEL_SCOUT_AGENT = "travel_scout_agent_node"
TRAVEL_SCOUT_TOOLS = "travel_scout_tool_node"
SYNTHESIZER = "synthesizer"

MAX_TOOL_ROUNDS = 5  # safety limit per agent so a confused LLM can't loop forever

# ---- LLMs that know about their tools ---------------------------------------
hotel_flight_llm = llm.bind_tools(hotel_flight_tools)
travel_scout_llm = llm.bind_tools(travel_scout_tools)
hotel_flight_tools_by_name = {t.name: t for t in hotel_flight_tools}
travel_scout_tools_by_name = {t.name: t for t in travel_scout_tools}


# ===========================================================================
# Shared helpers (both agents follow the same pattern)
# ===========================================================================
def _agent_step(state, history_key, system_prompt, model, agent_name):
    """One LLM turn. Records a final result when the LLM stops calling tools."""
    history = state.get(history_key, [])
    first_turn = not history

    query = state["user_query"]
    if state.get("focus"):
        query += f"\n\nFocus: {state['focus']}"

    messages = [SystemMessage(content=system_prompt)]
    messages += [HumanMessage(content=query)] if first_turn else history

    tool_rounds = sum(isinstance(m, ToolMessage) for m in history)
    # Out of tool budget -> use the plain LLM so it must answer now
    result = (llm if tool_rounds >= MAX_TOOL_ROUNDS else model).invoke(messages)

    update = {history_key: ([HumanMessage(content=query)] if first_turn else []) + [result]}
    if not result.tool_calls:
        update["agent_results"] = [{
            "agent_name": agent_name,
            "focus": state.get("focus", ""),
            "result": result.content,
        }]
    return update


def _tool_step(state, history_key, tools_by_name):
    """Execute every tool call requested by the last AI message."""
    last = state[history_key][-1]
    outputs = []
    for call in last.tool_calls:
        tool_fn = tools_by_name.get(call["name"])
        try:
            out = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool: {call['name']}"
        except Exception as e:  # never crash the graph because of one tool
            out = f"Tool error: {e}"
        outputs.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    return {history_key: outputs}


def _route(state, history_key, tool_node):
    """If the last AI message asked for tools -> run them, else -> synthesizer."""
    history = state.get(history_key, [])
    if history and getattr(history[-1], "tool_calls", None):
        return tool_node
    return SYNTHESIZER


# ===========================================================================
# Flight / Hotel agent
# ===========================================================================
def hotel_flight_agent_node(state: TravelPlannerState):
    return _agent_step(state, "search_messages", HOTEL_FLIGHT_SEARCH_PROMPT,
                       hotel_flight_llm, "flight_hotel_search_agent")


def hotel_flight_tool_node(state: TravelPlannerState):
    return _tool_step(state, "search_messages", hotel_flight_tools_by_name)


def should_continue_hotel_flight(state: TravelPlannerState):
    return _route(state, "search_messages", HOTEL_FLIGHT_TOOLS)


# ===========================================================================
# Travel scout agent
# ===========================================================================
def travel_scout_agent_node(state: TravelPlannerState):
    return _agent_step(state, "planner_messages", TRAVEL_SCOUT_PROMPT,
                       travel_scout_llm, "travel_scout_agent")


def travel_scout_tool_node(state: TravelPlannerState):
    return _tool_step(state, "planner_messages", travel_scout_tools_by_name)


def should_continue_travel_scout(state: TravelPlannerState):
    return _route(state, "planner_messages", TRAVEL_SCOUT_TOOLS)


# ===========================================================================
# Orchestrator (router) + fan-out
# ===========================================================================
def orchestrator(state: TravelPlannerState):
    """Classify the request into one or more agent tasks (structured output)."""
    router = llm.with_structured_output(ClassificationResult)
    result = router.invoke([
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ])
    return {"tasks": result.tasks, "requires_synthesis": result.requires_synthesis}


_NODE_FOR_SOURCE = {
    "flight_hotel_search_agent_node": HOTEL_FLIGHT_AGENT,
    "travel_scout_agent_node": TRAVEL_SCOUT_AGENT,
}


def dispatch_to_agents(state: TravelPlannerState):
    """Fan-out with `Send`: one branch per task, each gets its own sub-query.

    The Send payload is the *branch's* input; the parent state's `user_query`
    is untouched, so the synthesizer still sees the original question.
    """
    tasks = state.get("tasks") or []
    if not tasks:  # not travel related -> go straight to synthesizer
        return [Send(SYNTHESIZER, {"user_query": state["user_query"]})]
    return [
        Send(_NODE_FOR_SOURCE[t.source], {"user_query": t.user_query, "focus": t.focus})
        for t in tasks
    ]


# ===========================================================================
# Synthesizer (runs ONCE after all branches finish - `defer=True` in graph.py)
# ===========================================================================
def synthesizer_node(state: TravelPlannerState):
    results = state.get("agent_results", [])

    if not results:
        answer = ("I can only help with travel questions - flights, hotels, itineraries, "
                  "weather, visas and travel tips. Please rephrase your request.")
    elif len(results) == 1:
        answer = results[0]["result"]
    else:
        formatted = ("\n\n" + "=" * 50 + "\n\n").join(
            f"**{r['agent_name'].upper()}**\nFocus: {r.get('focus') or 'N/A'}\n\n{r['result']}"
            for r in results
        )
        answer = llm.invoke([HumanMessage(content=SYNTHESIS_PROMPT.format(
            query=state["user_query"], results=formatted,
        ))]).content

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
