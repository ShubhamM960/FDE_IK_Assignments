# Development Challenges & Fixes: Multi-conversational Agent

A reference list of problems you may run into when building a multi-conversational agent with LangGraph, and how to fix each one. The examples follow this project's design: **orchestrator → product/support agents → synthesizer**.

---

## 1. The agent forgets earlier turns

**Symptom:** The user asks "Where is ORD102?", then "When will *it* arrive?", and the agent asks "Which order?"

**Cause:** Each call to `graph.invoke()` starts from empty state, or a new `thread_id` is created on every turn.

**Fix:** Compile the graph with a checkpointer and reuse the same `thread_id` for the whole session.

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": st.session_state.session_id}}  # create once per session
graph.invoke({"messages": [HumanMessage(user_input)]}, config)
```

**Tip:** `MemorySaver` only keeps memory while the process runs. For production, use `SqliteSaver` or `PostgresSaver`.

---

## 2. The router sends messages to the wrong agent

**Symptom:** "My headphones arrived broken" goes to Product Discovery instead of Support.

**Fixes:**
- Make the router return **structured output** (a Pydantic model), not free text, so you never have to parse its reply.
- Put clear rules and a few examples in the router prompt, including mixed cases.
- Set a safe default for when it's unsure. This project defaults to `product_agent`.
- Pass the last few messages to the router, not just the latest one. "What about the blue one?" can't be classified on its own.

```python
class RouteDecision(BaseModel):
    tasks: list[Literal["product_agent", "support_agent"]]
    requires_synthesis: bool

router = llm.with_structured_output(RouteDecision)
```

---

## 3. State gets overwritten when agents run in parallel

**Symptom:** You fan out with `Send` to both agents and only one answer survives, or LangGraph raises `InvalidUpdateError: Can receive only one value per step`.

**Cause:** Both branches write to the same state key, and that key has no rule for combining values.

**Fix:** Add a reducer to the shared key so the results get added together.

```python
import operator
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent_outputs: Annotated[list[dict], operator.add]  # each branch adds its result here
```

---

## 4. Human-in-the-loop resumes in the wrong way

**Symptom:** The support agent asks for an order ID. The user replies, but the graph starts over from the beginning or loops asking again.

**Cause:** The user's reply was sent as a new `invoke` instead of resuming the paused one.

**Fix:** Check whether the graph is paused, then resume it with `Command(resume=...)` on the same thread.

```python
from langgraph.types import Command, interrupt

# inside the node
order_id = interrupt("Could you share your order ID or email?")

# in the app
state = graph.get_state(config)
if state.next:  # graph is paused at an interrupt
    graph.invoke(Command(resume=user_input), config)
else:
    graph.invoke({"messages": [HumanMessage(user_input)]}, config)
```

**Also:** Any code that runs before `interrupt()` in a node runs **again** when the graph resumes. Keep side effects (like creating a ticket) after the interrupt.

---

## 5. Agents loop forever on tools, or make up tool results

**Symptom:** The ReAct loop calls `search_products` 10 times, or the agent invents an order status.

**Fixes:**
- Set `recursion_limit` in the config: `{"recursion_limit": 15}`.
- Cap the number of calls to expensive tools per turn.
- Return clear, structured tool results, including for failures (`{"found": false, "reason": "Order not found"}`). This stops the LLM from guessing.
- Add a rule to the prompt: *"Only state facts returned by tools."*

---

## 6. The context window fills up and costs grow

**Symptom:** Long sessions get slow and expensive, or you get a context-length error.

**Fixes:**
- Trim the history before calling the LLM (`trim_messages` with a token limit).
- Summarize old turns into a single system message.
- Don't re-send large tool outputs (such as full RAG chunks or search results) on later turns. Keep only the agent's final answer.

---

## 7. RAG returns irrelevant products

**Symptom:** "Noise-cancelling headphones" returns speakers.

**Fixes:**
- Embed useful text such as name, category, features and description, not just the product name.
- Store category and price as metadata and filter on them (`where={"category": "audio"}`).
- Tune `k` and add a similarity-score threshold so weak matches are dropped.
- Build the Chroma index **once** at startup and cache it. In Streamlit, use `@st.cache_resource`.

---

## 8. The synthesizer merges answers badly

**Symptom:** The answer is repeated twice, contradicts itself, or reads like two separate replies.

**Fixes:**
- Skip the synthesizer when only one agent ran (check `requires_synthesis`).
- Give it a strict prompt: *"Combine into one reply, remove duplicates, keep order details exact."*
- Handle the empty case. If the orchestrator failed and there are no outputs, return a polite fallback.

---

## 9. Streamlit reruns break state

**Symptom:** Every button click rebuilds the graph, loses the chat, or creates a new session.

**Fixes:**
- Use `@st.cache_resource` for the graph, LLM and vector store.
- Keep `session_id` and chat history in `st.session_state`, and only set them if they don't already exist.
- Run the app from the project root (`streamlit run src/app.py`) so `src` imports work.

---

## 10. Voice mode problems (Whisper + TTS)

**Symptoms:** Nothing is recorded, transcripts are empty, or replies are slow.

**Fixes:**
- Check OS microphone permissions and the default input device.
- Skip calling the LLM if the transcript is empty or only a few characters.
- Make the voice imports lazy so text mode still works without `sounddevice` installed.
- Keep replies short for TTS, or ask for a short "spoken version" of the answer.

---

## 11. Hard-to-debug failures

**Symptom:** "It gave a weird answer" and you can't tell which node caused it.

**Fixes:**
- Turn on **LangSmith** tracing (`LANGCHAIN_TRACING_V2=true`) to see every node, prompt and tool call.
- Use `graph.stream(..., stream_mode="updates")` to print what each node changes.
- Wrap each node in `try/except` and fall back gracefully, so one failing agent doesn't crash the whole turn.

---

## 12. Testing

**Symptom:** Fixing one prompt breaks another flow.

**Fix:** Keep a small set of regression conversations and replay it after each change. For example:
- A greeting
- A product search
- An order lookup with a missing ID (checks the interrupt)
- A mixed product + support question (checks parallel fan-out)
- An escalation request

Check the routes chosen and the key facts in each reply.

---

## Quick reference

| # | Challenge | Main fix |
|---|---|---|
| 1 | Agent forgets earlier turns | Checkpointer + same `thread_id` for the session |
| 2 | Wrong routing | Structured output, prompt examples, safe default, recent history |
| 3 | Parallel state overwritten | Reducers (`operator.add`) on shared keys |
| 4 | Interrupt resumes wrongly | `Command(resume=...)` when `state.next` is set |
| 5 | Tool loops / made-up facts | `recursion_limit`, tool caps, structured tool results |
| 6 | Context too long | Trim/summarize history, drop old tool outputs |
| 7 | Poor RAG results | Richer embedded text, metadata filters, score threshold |
| 8 | Bad merged answers | Skip when single agent, strict synthesizer prompt |
| 9 | Streamlit loses state | `@st.cache_resource` + `st.session_state` |
| 10 | Voice issues | Mic permissions, empty-transcript guard, lazy imports |
| 11 | Hard to debug | LangSmith tracing, `stream_mode="updates"` |
| 12 | Regressions | Replay a fixed set of test conversations |
