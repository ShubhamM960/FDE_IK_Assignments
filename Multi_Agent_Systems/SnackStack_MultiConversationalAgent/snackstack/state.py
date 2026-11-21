"""
Shared state for the SnackStack LangGraph workflow.

Each node (orchestrator, menu agent, order agent, synthesizer) reads from this
state and returns only the fields it changed; LangGraph merges them in.
"""

from typing import Annotated, Literal, Optional, TypedDict, Dict, List, Any
from pydantic import Field, BaseModel
from langchain_core.messages import AnyMessage
import operator

class AgentToCall(BaseModel):
    """Agent results are the results of the agent's work."""
    # FIX: `source: str = Literal[...]` made Literal the *default value*, not the type,
    # so the LLM could return any string (e.g. "menu") and Send() would fail.
    source: Literal["menu_agent", "order_agent"] = Field(description="Agent that should handle this task")
    user_query: str = Field(
        description="The part of the user's query this agent should answer. "
                    "Copy any order ID, tracking ID or email exactly."
    )

class AgentTaskOutputFormat(BaseModel):
    """Agent task output format is the output format of the agent's work."""
    tasks: List[AgentToCall] = Field(description="The tasks to be executed by the agent")
    requires_synthesis: bool = Field(description="Whether the agent's results need to be synthesized")
    reasoning: str = Field(description="The reasoning behind the agent's decision")

class StackState(TypedDict):
    """
    Data passed between agents during one conversation turn.

    Flow:
        user_query -> orchestrator sets route
                   -> menu agent and/or order agent fill their responses
                   -> synthesizer combines them into final_answer
    """

    # Full chat history (user + assistant messages).
    # `operator.add` appends new messages instead of overwriting the list,
    # so the agents remember earlier turns of the conversation.
    messages: Annotated[list[AnyMessage], operator.add]

    # The user's latest question, e.g. "What snacks are vegan?" or "Order 2 samosas".
    user_query: str

    # Decided by the orchestrator: which agent(s) should handle the query.
    #   "menu"  -> questions about items, prices, ingredients
    #   "order" -> placing, checking, or cancelling an order
    #   "both"  -> the query needs both agents
    route_agent: str

    # Answer produced by the menu agent (None if it was not called).
    # FIX: `Annotated[Dict]` is invalid (Annotated needs metadata) -> TypeError at import.
    menu_response: Optional[Dict[str, Any]]

    # Answer produced by the order agent (None if it was not called).
    order_response: Optional[Dict[str, Any]]

    # FIX: orchestrator writes "tasks" (plural); the key must exist in the state
    # or LangGraph raises InvalidUpdateError. Stored as plain dicts (AgentToCall.model_dump()).
    tasks: List[Dict[str, Any]]

    requires_synthesis: bool
    # Final reply shown to the user, written by the synthesizer
    # from menu_response and/or order_response.
    # FIX: this is the LLM's text reply, so it is a str (not a list of dicts).
    final_answer: str

class AgentsInputState(TypedDict):
    """Payload delivered to an agent worker node via Send().
    NOTE: This is intentionally flat — no nested Pydantic objects —
    so that Send() serialisation works without surprises.
    """
    messages: Annotated[list[AnyMessage], operator.add]
    user_query : str
    route_agent : str
