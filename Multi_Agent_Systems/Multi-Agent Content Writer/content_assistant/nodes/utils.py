"""Message-history helpers shared by the agent nodes."""

from typing import List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def last_human_index(messages: Sequence[BaseMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def is_final_ai(msg: BaseMessage) -> bool:
    return isinstance(msg, AIMessage) and not msg.tool_calls and bool(msg.content)


def conversation_history(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """User messages + final AI answers only (tool-call chatter stripped)."""
    return [m for m in messages if isinstance(m, HumanMessage) or is_final_ai(m)]


def agent_context(messages: Sequence[BaseMessage], max_prior: int = 10) -> List[BaseMessage]:
    """Context for a tool-calling agent.

    Prior turns are condensed to user/AI messages (cheaper: old tool outputs are not
    re-sent), while the CURRENT turn is kept in full so the tool loop can continue
    (AI tool_calls -> ToolMessages -> next decision).
    """
    idx = last_human_index(messages)
    prior = conversation_history(messages[:idx])[-max_prior:]
    return prior + list(messages[idx:])


def count_tool_calls(messages: Sequence[BaseMessage], tool_name: str) -> int:
    """How many times `tool_name` was called in the current turn."""
    idx = last_human_index(messages)
    return sum(
        1
        for m in messages[idx:]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
        if tc["name"] == tool_name
    )
