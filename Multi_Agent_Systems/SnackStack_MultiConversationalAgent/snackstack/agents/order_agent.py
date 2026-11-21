from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, interrupt
from snackstack.config import llm
from snackstack.state import AgentsInputState
from snackstack.tools import get_order_status
from snackstack.logger import setup_logger
from langchain_core.messages import (AnyMessage, SystemMessage, HumanMessage, ToolMessage)
from typing import Annotated, TypedDict
import operator
import re
from snackstack.agents.prompts import ORDER_AGENT_PROMPT

logger = setup_logger("order_agent")

# Each loop = 1 model step + 1 tools step. Cap it so the LLM can't loop forever.
MAX_TOOL_ITERATIONS = 3

#─── order agent tools ───────────────────────────────────────────────────
# FIX: renamed — the tools-node function below had the same name and replaced this list.
ORDER_TOOLS = [get_order_status]

#─── order agent llm ───────────────────────────────────────────────────
order_agent_llm = llm.bind_tools(ORDER_TOOLS)

#─── order agent tools by name ───────────────────────────────────────────────────
order_agent_tools_by_name = {tool.name: tool for tool in ORDER_TOOLS}  # FIX: `=` -> `:`

#─── order agent state ───────────────────────────────────────────────────
class OrderAgentState(TypedDict): 
    """ State for order agent """
    messages: Annotated[list[AnyMessage], operator.add]

#─── order agent graph ───────────────────────────────────────────────────
def order_agent_graph() -> StateGraph:
    order_builder = StateGraph(OrderAgentState)
    order_builder.add_node("order_agent_model", order_agent_model)
    order_builder.add_node("order_agent_tools", ORDER_TOOLS)

    order_builder.add_edge(START, "order_agent_model")
    order_builder.add_conditional_edges("order  _agent_model" , should_continue, {"tools": "order_agent_tools", "end": END})
    order_builder.add_edge("order_agent_tools", "order_agent_model")
    
    return order_builder.compile()
#── order agent sub graph functions ───────────────────────────────────────────────────
def should_continue(state: OrderAgentState) -> str:
    last = state["messages"][-1]
    # FIX: must return the path-map keys ("tools" / "end") used in add_conditional_edges.
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

def order_agent_model(state: OrderAgentState) -> dict:
    """Order agent model is responsible for generating the order agent response from the llm"""
    result = order_agent_llm.invoke(state['messages'])
    logger.info("[order_agent:model] tool_calls=%s", bool(result.tool_calls))
    return {"messages": [result]}   

def order_agent_tools(state: OrderAgentState) -> dict:
    """Order agent tools are responsible for executing the order agent tools."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[order_agent:tools] %s(%s)", name, args)
        out = order_agent_tools_by_name[name].invoke(args) if name in order_agent_tools_by_name else f"Unknown tool: {name}"
        logger.info(f"Order agent tools output is: {out}")
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}

#─── order agent graph ───────────────────────────────────────────────────
def build_order_agent_graph():
    order_builder = StateGraph(OrderAgentState)
    order_builder.add_node("order_agent_model", order_agent_model)
    # FIX: the node must be the *function* order_agent_tools, not the ORDER_TOOLS list.
    order_builder.add_node("order_agent_tools", order_agent_tools)

    order_builder.add_edge(START, "order_agent_model")
    order_builder.add_conditional_edges("order_agent_model", should_continue, {"tools": "order_agent_tools", "end": END})
    order_builder.add_edge("order_agent_tools", "order_agent_model")
    return order_builder.compile()

# Compiled once at import (not on every query).
order_agent_graph = build_order_agent_graph()

#───Regex patterns for order id, tracking id and email ───────────────────────────────────────────────────
ORDER_ID_RE    = re.compile(r"\bORD-\d+\b", re.IGNORECASE)                 # ORD-201
TRACKING_ID_RE = re.compile(r"\bSS\d+TRK\b", re.IGNORECASE)                # SS201TRK
EMAIL_RE       = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")                # priya@example.com

def extract_identifier(text: str) -> str | None:
    """Try to extract an order ID, tracking ID, or email from text."""
    logger.info(f"Extract identifier for order_agent is: {text}")
    for pattern in (ORDER_ID_RE, TRACKING_ID_RE, EMAIL_RE):
        match = pattern.search(text)
        if match:
            return match.group()
    return None

#─── order agent node / function from  main grap───────────────────────────────────────────────────
def order_agent_node(state: AgentsInputState) -> Command:
    """Order agent is responsible for searching the order catalog for items matching the query."""
    context = state.get("messages", [])[-6:]
    user_query = state.get("user_query", "")
    logger.info("Order Agent  query=%r", user_query)
    
    # ── HITL: must run BEFORE any LLM call, because the node re-runs on resume ──
    last_user_msg = context[-1].content if context else ""
    identifier = extract_identifier(user_query) or extract_identifier(last_user_msg)
    if identifier:
        logger.info("Order Agent  identifier Found=%r", identifier)
    else:
        logger.info("Order Agent  identifier Not Found, --Interrupting for user input")
        user_input = interrupt(
            "Please provide a valid order ID, tracking ID, or email.\n"  # FIX: missing "\n"
            "  • Order ID    (e.g. ORD-201)\n"
            "  • Tracking ID (e.g. SS201TRK)\n"
            "  • Email       (e.g. priya@example.com)"
        )
        logger.info("User provided input=%r", user_input)
        # Use the extracted ID if present, else pass the raw text (tool will say "not found").
        identifier = extract_identifier(user_input) or str(user_input).strip()

    history = "\n".join(f"{m.type}: {m.content}" for m in context)
    try:
        # FIX: the prompt expects a "Lookup key: ..." line, it was never sent.
        result = order_agent_graph.invoke(
            {'messages': [SystemMessage(content=ORDER_AGENT_PROMPT),
                          HumanMessage(content=f"Conversation so far:\n{history}\n\n"
                                               f"Query asked: {user_query}\nLookup key: {identifier}")]},
            {"recursion_limit": 2 * MAX_TOOL_ITERATIONS + 1},
        )
        # FIX: `result[-1]` -> result is a dict; the answer is the last message.
        response = result["messages"][-1].content
    except GraphRecursionError:
        logger.warning("[order_agent] hit MAX_TOOL_ITERATIONS=%d", MAX_TOOL_ITERATIONS)
        response = "Sorry, I couldn't fetch your order status right now. Please try again."

    update = {"order_response": {"source": "order_agent", "response": response}}
    return Command(update=update, goto="synthesizer")


