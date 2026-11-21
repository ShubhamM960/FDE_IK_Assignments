from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from snackstack.config import llm
from snackstack.state import AgentsInputState
from snackstack.tools.menu_tools import search_menu_catalog
from snackstack.logger import setup_logger
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from typing import Annotated, TypedDict
import operator
from snackstack.agents.prompts import MENU_AGENT_PROMPT
logger = setup_logger("menu_agent")

# Each loop = 1 model step + 1 tools step. Cap it so the LLM can't loop forever.
# FIX: was used in the except-block but never defined -> NameError.
MAX_TOOL_ITERATIONS = 5

#─── menu agent tools ───────────────────────────────────────────────────
MENU_TOOLS = [search_menu_catalog]

#─── menu agent llm ───────────────────────────────────────────────────
menu_agent_llm = llm.bind_tools(MENU_TOOLS)


#─── menu agent tools by name ───────────────────────────────────────────────────
menu_agent_tools_by_name = {tool.name: tool for tool in MENU_TOOLS}

#─── menu agent state ───────────────────────────────────────────────────
class MenuAgentState(TypedDict): 
    """ State for menu agent """
    messages: Annotated[list[AnyMessage], operator.add]
    

#─── menu agent graph ───────────────────────────────────────────────────
# FIX: renamed to build_menu_agent_graph — `menu_agent_graph = menu_agent_graph()` inside
# menu_agent_node made the name local -> UnboundLocalError (same bug as the order agent).
def build_menu_agent_graph():
    menu_builder = StateGraph(MenuAgentState)
    menu_builder.add_node("menu_agent_model", menu_agent_model)
    menu_builder.add_node("menu_agent_tools", menu_agent_tools)

    menu_builder.add_edge(START, "menu_agent_model")
    menu_builder.add_conditional_edges("menu_agent_model", should_continue, {"tools": "menu_agent_tools", "end": END})
    menu_builder.add_edge("menu_agent_tools", "menu_agent_model")
    return menu_builder.compile()


#─── menu agent node / function from  main grap───────────────────────────────────────────────────
def menu_agent_node(state: AgentsInputState) -> Command:
    """Menu agent is responsible for searching the menu catalog for items matching the query."""
    context = state.get("messages", [])[-6:]
    history = "\n".join(f"{m.type}: {m.content}" for m in context)
    logger.info(f"query asked is : {state['user_query']}")
    try:
        result = menu_agent_graph.invoke(
            {'messages' : [SystemMessage(content= MENU_AGENT_PROMPT),
            HumanMessage(content = f"Given context : {history} . Query asked : {state['user_query']}") ]},
            {"recursion_limit": 2 * MAX_TOOL_ITERATIONS + 1},  # FIX: cap the tool loop
        )
        response = result["messages"][-1].content
    except GraphRecursionError:
        # FIX: `result` didn't exist here (NameError below); set `response` instead.
        logger.warning("[menu_agent] hit MAX_TOOL_ITERATIONS=%d", MAX_TOOL_ITERATIONS)
        response = "Sorry, I couldn't find that on the menu right now. Could you rephrase?"

    update = {"menu_response": {"source": "menu_agent", "response": response}}
    return Command(update=update, goto="synthesizer")

####### menu agent sub graph functions #########################################################
def should_continue(state: MenuAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def menu_agent_model(state: MenuAgentState) -> dict:
    """Menu agent model is responsible for generating the menu agent response from the llm"""
    result = menu_agent_llm.invoke(state['messages'])
    logger.info("[menu_agent:model] tool_calls=%s", bool(result.tool_calls))
    return {"messages": [result]}

def menu_agent_tools(state: MenuAgentState) -> dict:
    """Menu agent tools are responsible for executing the menu agent tools."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[menu_agent:tools] %s(%s)", name, args)
        out = menu_agent_tools_by_name[name].invoke(args) if name in menu_agent_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


# Compiled once at import, AFTER the node functions above are defined.
menu_agent_graph = build_menu_agent_graph()

