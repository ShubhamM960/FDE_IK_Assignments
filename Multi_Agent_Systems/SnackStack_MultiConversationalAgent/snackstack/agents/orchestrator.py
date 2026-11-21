from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, Send
from snackstack.config import llm
from snackstack.logger import setup_logger
from snackstack.state import StackState, AgentTaskOutputFormat
from snackstack.agents.prompts import ORCHESTRATOR_PROMPT

logger = setup_logger("orchestrator")

# OPTIMIZATION: build the structured-output runnable once, not on every call.
router_llm = llm.with_structured_output(AgentTaskOutputFormat)


def orchestrator_node(state: StackState) -> Command:
    """Orchestrator agent is responsible for orchestrating the flow of the conversation."""
    user_query = state.get("user_query", "")
    messages = state.get("messages", [])  # FIX: `messages` is used below but was never defined
    if not user_query and messages:
        # FIX: `messages[-6:]` is a list of message objects, not text; use the last message's content.
        user_query = messages[-1].content

    logger.info("Orchestrator  query=%r", user_query)

    # FIX: pass recent history so vague follow-ups ("anything works") route correctly,
    # as the prompt asks. Exclude the current message (it is sent separately).
    history = "\n".join(f"{m.type}: {m.content}" for m in messages[-7:-1])
    prompt = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=f"Conversation history:\n{history or '(none)'}\n\nLatest message: {user_query}"),
    ]
    try:
        agent_to_call = router_llm.invoke(prompt)
        tasks = agent_to_call.tasks
        requires_synthesis = agent_to_call.requires_synthesis
        logger.info("Orchestrator  route=%s reason=%s", [t.source for t in tasks], agent_to_call.reasoning)
    except Exception:
        logger.exception("Routing failed, defaulting to menu_agent")
        tasks, requires_synthesis = [], False

    targets: list[Send] = []
    #loop over `tasks` — `agent_to_call` doesn't exist when the LLM call failed.
    seen = set()
    for task in tasks:
        if task.source in seen:
            continue
        seen.add(task.source)
        current_state = {
            'messages': state["messages"],
            'user_query': task.user_query,
            'route_agent': task.source,
        }
        targets.append(Send(task.source, current_state))

    if not targets:
        targets = [Send("menu_agent", {"messages": messages, "user_query": user_query, "route_agent": "menu_agent"})]
        tasks = []

    return Command(
        update={
            "tasks": [t.model_dump() for t in tasks],  # plain dicts serialise cleanly in MemorySaver
            "requires_synthesis": requires_synthesis,
            "user_query": user_query,
            # FIX: clear last turn's answers, otherwise MemorySaver keeps them and the
            # synthesizer would re-use an old order/menu reply in a new turn.
            "menu_response": None,
            "order_response": None,
        },
        goto=targets,
    )