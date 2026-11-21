
from snackstack.state import StackState 
from snackstack.logger import setup_logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from snackstack.config import llm
from snackstack.agents.prompts import SYNTHESIZER_PROMPT

logger = setup_logger("synthesizer")

def synthesizer_node(state: StackState) -> dict:
    """Synthesizer agent is responsible for synthesizing the responses from the agents."""

    logger.info("Synthesize the response >>>")
    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    order_response = state.get("order_response") or {}
    menu_response = state.get("menu_response") or {}
    combined_response = ""

    # FIX: was `if ... elif ...`, so when BOTH agents ran the menu answer was dropped.
    if menu_response:
        combined_response += f"[Menu Agent]\n{menu_response.get('response', '')}\n\n"
    if order_response:
        combined_response += f"[Order Agent]\n{order_response.get('response', '')}\n\n"  # FIX: unclosed f-string brace

    if not combined_response:
        answer = "Sorry, I could not process your request. Could you try again?"
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    prompt = [SystemMessage(content=SYNTHESIZER_PROMPT.format(combined_response=combined_response)), HumanMessage(content=user_query)]
    results = llm.invoke(prompt)
    logger.info("Synthesizer response is ready")
    return {
        "final_answer": results.content,
        "messages": [results]
    }
