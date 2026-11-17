"""Graph state + Pydantic schemas.

`TravelPlannerState` is the shared "memory" every node reads from and writes to.
Fields annotated with `operator.add` are *reducers*: when several nodes (or
parallel branches) return that key, LangGraph appends the lists instead of
overwriting them.
"""

import operator
from typing import Annotated, List, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

AgentSource = Literal["flight_hotel_search_agent_node", "travel_scout_agent_node"]


class AgentTask(BaseModel):
    """A single agent task with a targeted query."""

    source: AgentSource = Field(description="Which agent should handle this task")
    user_query: str = Field(description="Sub-query containing only the part relevant to this agent")
    focus: str = Field(default="", description="Short phrase: what the agent should prioritise")


class ClassificationResult(BaseModel):
    """Structured output of the router (orchestrator) LLM call."""

    tasks: List[AgentTask] = Field(
        default_factory=list,
        description="Agents to invoke with their targeted queries (empty if not travel-related)",
    )
    requires_synthesis: bool = Field(
        default=False,
        description="True when more than one agent is used and results must be merged",
    )


class TravelPlannerState(MessagesState):
    """Shared state for the whole graph (inherits `messages` from MessagesState)."""

    # Input: the user's question (each parallel branch receives its own sub-query)
    user_query: str
    focus: str

    # Router output
    tasks: List[AgentTask]
    requires_synthesis: bool

    # Results from each agent - appended safely by parallel branches
    agent_results: Annotated[List[dict], operator.add]

    # Final answer shown to the user
    final_answer: str

    # Private message histories for each agent's tool loop
    search_messages: Annotated[List[AnyMessage], operator.add]   # flight/hotel agent
    planner_messages: Annotated[List[AnyMessage], operator.add]  # travel scout agent
