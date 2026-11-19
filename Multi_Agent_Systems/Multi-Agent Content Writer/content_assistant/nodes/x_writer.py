"""X/Twitter Writer agent: only internet_search_tool bound, same loop pattern."""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END

from content_assistant.config import WRITER_TEMPERATURE, get_llm
from content_assistant.nodes.utils import agent_context
from content_assistant.prompts import X_WRITER_PROMPT
from content_assistant.state import AgentState
from content_assistant.tools import X_TOOLS

AGENT_NAME = "x_blog_writer"


def x_blog_writer_node(state: AgentState) -> dict:
    llm = get_llm(WRITER_TEMPERATURE).bind_tools(X_TOOLS)
    response: AIMessage = llm.invoke(
        [SystemMessage(content=X_WRITER_PROMPT)] + agent_context(state["messages"])
    )
    response.name = AGENT_NAME
    return {"messages": [response]}


def should_continue_x(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "x_tools"
    return END
