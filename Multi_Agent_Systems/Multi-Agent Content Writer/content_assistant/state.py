"""Shared graph state."""

from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Full conversation history. `add_messages` appends (never overwrites), and the
    # checkpointer persists it per thread_id, which is what gives us memory.
    messages: Annotated[List[BaseMessage], add_messages]
    # Raw text of the current request (set by the caller on every invoke).
    user_input: str
    # One of: seo_blog_writer / x_blog_writer / general (set by the router).
    route: str
