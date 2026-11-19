"""SEO Blog Writer agent: both tools bound, autonomous tool loop."""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END

from content_assistant.config import MAX_RESEARCH_CALLS_PER_TURN, WRITER_TEMPERATURE, get_llm
from content_assistant.nodes.utils import agent_context, count_tool_calls
from content_assistant.prompts import SEO_BLOG_PROMPT
from content_assistant.state import AgentState
from content_assistant.tools import SEO_TOOLS, internet_search_tool

AGENT_NAME = "seo_blog_writer"


def seo_blog_writer_node(state: AgentState) -> dict:
    messages = state["messages"]
    research_calls = count_tool_calls(messages, "research_tool")

    system = SEO_BLOG_PROMPT
    if research_calls >= MAX_RESEARCH_CALLS_PER_TURN:
        # Code-level guardrail on top of the prompt: the expensive tool is removed.
        tools = [internet_search_tool]
        system += "\n\nResearch budget used up: research_tool is no longer available."
    else:
        tools = SEO_TOOLS
        if research_calls:
            system += f"\n\nYou have already called research_tool {research_calls}x for this request."

    llm = get_llm(WRITER_TEMPERATURE).bind_tools(tools)
    response: AIMessage = llm.invoke([SystemMessage(content=system)] + agent_context(messages))
    response.name = AGENT_NAME
    return {"messages": [response]}


def should_continue_seo(state: AgentState) -> str:
    """Loop to the tool node while the agent keeps requesting tools, else finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "seo_tools"
    return END
