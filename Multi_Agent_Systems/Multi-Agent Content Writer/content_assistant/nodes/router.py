"""Router node: classifies the request and persists the user input into messages."""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from content_assistant.config import (
    ROUTE_GENERAL, ROUTE_SEO, ROUTE_X, ROUTER_TEMPERATURE, VALID_ROUTES, get_llm,
)
from content_assistant.nodes.utils import conversation_history
from content_assistant.prompts import ROUTER_PROMPT
from content_assistant.state import AgentState


def _parse_route(text: str) -> str:
    """Normalise the LLM output to exactly one valid route word."""
    lowered = (text or "").strip().lower()
    first = lowered.split()[0] if lowered else ""
    word = re.sub(r"[^a-z_]", "", first)
    if word in VALID_ROUTES:
        return word
    for route in VALID_ROUTES:
        if route in lowered:
            return route
    return ROUTE_GENERAL


def _keyword_fallback(user_input: str) -> str:
    text = user_input.lower()
    if re.search(r"\b(tweet|twitter|x post|thread|hashtag)", text):
        return ROUTE_X
    if re.search(r"\b(blog|article|seo|long[- ]form)", text):
        return ROUTE_SEO
    return ROUTE_GENERAL


def router_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "").strip()
    human = HumanMessage(content=user_input)

    # Give the classifier a little recent context so follow-ups route correctly.
    recent = conversation_history(state.get("messages", []))[-4:]
    context = "\n".join(
        f"{'USER' if isinstance(m, HumanMessage) else 'ASSISTANT(' + (m.name or 'ai') + ')'}: "
        f"{str(m.content)[:200]}"
        for m in recent
    )
    prompt = f"Recent conversation:\n{context or '(none)'}\n\nLatest request: {user_input}"

    try:
        result = get_llm(ROUTER_TEMPERATURE).invoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=prompt)]
        )
        route = _parse_route(str(result.content))
    except Exception:
        route = _keyword_fallback(user_input)

    # Persist the user's input into the message history + record the route.
    return {"messages": [human], "route": route}


def route_decision(state: AgentState) -> str:
    """Conditional-edge function: returns the next node name."""
    return state.get("route", ROUTE_GENERAL)
