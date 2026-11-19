"""General handler: answers everything else using the full conversation memory."""

from langchain_core.messages import SystemMessage

from content_assistant.config import GENERAL_TEMPERATURE, get_llm
from content_assistant.nodes.utils import conversation_history, last_human_index
from content_assistant.prompts import GENERAL_PROMPT
from content_assistant.state import AgentState

AGENT_NAME = "general"


def general_node(state: AgentState) -> dict:
    messages = state["messages"]
    history = conversation_history(messages)  # every user msg + every final AI answer

    response = get_llm(GENERAL_TEMPERATURE).invoke([SystemMessage(content=GENERAL_PROMPT)] + history)
    response.name = AGENT_NAME

    # Save both the user and AI messages. The user message already carries an id
    # (assigned when the router added it), so `add_messages` updates it in place
    # rather than duplicating it.
    user_msg = messages[last_human_index(messages)]
    return {"messages": [user_msg, response]}
