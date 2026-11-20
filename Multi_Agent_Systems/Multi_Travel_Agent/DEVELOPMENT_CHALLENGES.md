# Development Challenges & Fixes: Multi Travel Agent

Real problems you can hit while building this project (LangGraph + SerpAPI + Tavily + Deep Agents), and how the code handles each one.

```
START → orchestrator ─┬─ flight_hotel_agent ⇄ tools ─┬─→ synthesizer → END
                      └─ travel_scout_agent ⇄ tools ─┘
```

---

## 1. Parallel agents overwrite each other's results

- **Problem:** Both agents run in parallel through `Send`. If they write to a plain `agent_results: list` key, LangGraph raises `InvalidUpdateError` (two values in one step), or only one answer survives.
- **Fix:** Give shared keys a reducer so the lists are joined instead of replaced (`state.py`).

```python
agent_results: Annotated[List[dict], operator.add]
```

---

## 2. Tool histories from two agents get mixed

- **Problem:** If both ReAct loops use the shared `messages` list, one agent sees the other agent's `ToolMessage`s. OpenAI then rejects the request, because every `ToolMessage` must follow the `AIMessage` that requested it (matching `tool_call_id`).
- **Fix:** Give each agent its **own private history key**, and keep `messages` for the final answer only (`state.py`).

```python
search_messages: Annotated[List[AnyMessage], operator.add]   # flight/hotel agent
planner_messages: Annotated[List[AnyMessage], operator.add]  # travel scout agent
```

---

## 3. The synthesizer runs too early or more than once

- **Problem:** The two branches take different numbers of tool rounds, so they reach the synthesizer at different times. A normal node would run once per branch and merge incomplete results.
- **Fix:** Add the synthesizer with `defer=True`, so it waits for **all** branches and runs once (`graph.py`).

```python
builder.add_node(SYNTHESIZER, synthesizer_node, defer=True)
```

- **Extra:** If only one agent ran, the synthesizer returns that answer as-is and skips an extra LLM call (`nodes.py`).

---

## 4. Each agent needs its own sub-query, but the synthesizer needs the original

- **Problem:** "Plan 3 days in Paris and find hotels for 2 adults" should send only the hotel part to the hotel agent. But if you overwrite `user_query` in the shared state, the synthesizer loses the original question.
- **Fix:** Put the sub-query in the **`Send` payload**. That payload is the branch's input only, so the parent `user_query` doesn't change (`nodes.py → dispatch_to_agents`).

```python
Send(_NODE_FOR_SOURCE[t.source], {"user_query": t.user_query, "focus": t.focus})
```

- **Related:** The router can drop details (dates, number of travellers) when it splits a request. `ROUTER_PROMPT` has an explicit rule ("Keep ALL details the user gave...") plus worked examples.

---

## 5. Non-travel questions leave the graph with nowhere to go

- **Problem:** For "Write me a poem" the router returns an empty `tasks` list. A conditional edge that returns `[]` schedules no node, so you get no answer at all.
- **Fix:** If there are no tasks, send straight to the synthesizer. With no results, it returns a polite "travel questions only" message (`nodes.py`).

```python
if not tasks:
    return [Send(SYNTHESIZER, {"user_query": state["user_query"]})]
```

---

## 6. Old results leak into the next question

- **Problem:** `agent_results`, `search_messages` and `planner_messages` use `operator.add`, and `MemorySaver` keeps them per thread. If the same `thread_id` is reused, question 2's synthesizer also merges question 1's results.
- **Fix:** Use a **fresh `thread_id` for every question** (`app.py → run()`).
- **Trade-off:** There's no follow-up memory. If an agent asks a clarifying question, the user has to re-ask with all the details (listed under Limitations in the README).

---

## 7. An agent loops on tool calls forever

- **Problem:** A confused LLM keeps calling `flight_search` with slightly different arguments and never answers.
- **Fix:** Count the `ToolMessage`s in the agent's history. Once `MAX_TOOL_ROUNDS` (5) is reached, call the plain `llm` (no tools bound), which can only reply with text (`nodes.py → _agent_step`).

```python
result = (llm if tool_rounds >= MAX_TOOL_ROUNDS else model).invoke(messages)
```

---

## 8. One failing tool crashes the whole graph

- **Problem:** A SerpAPI timeout, a bad argument, or a tool name the LLM made up raises an exception and kills the run.
- **Fix:** Handle errors at two levels:
  - Inside each tool, wrap the call in `try/except` and return `{"message": "..."}` as JSON, so the LLM can explain the error (`tools.py`).
  - In `_tool_step`, catch any leftover exception and handle unknown tool names (`nodes.py`).

```python
out = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool: {call['name']}"
```

---

## 9. SerpAPI rejects the request (bad dates or airport codes)

- **Problem:** Google Flights needs IATA codes (`DEL`, `LHR`) and `YYYY-MM-DD` dates, but users write "Delhi to London next Friday". Past dates and wrong codes return an `error` key instead of results.
- **Fix:**
  - The Pydantic `args_schema` describes the exact format for each field, which helps the LLM fill it correctly (`FlightSearchInput`, `HotelSearchInput`).
  - The tool checks for `"error" in results` and passes the message back to the LLM.
  - The prompts list the required fields, so the agent asks a clarifying question instead of guessing.
- **Known limit:** For cities with several airports (London, New York, Tokyo), the LLM may pick the wrong airport.

---

## 10. Large API responses waste tokens

- **Problem:** Raw SerpAPI JSON is large (images, tokens, booking links). Sending it all back to the LLM is slow and expensive.
- **Fix:** Keep only the **top 10** options and only the fields the agent needs (price, duration, stops, segments; or name, rating, price, amenities) before returning JSON (`tools.py`).

---

## 11. Setup problems: missing keys, package name, import errors

- **Problem 1:** `TavilySearch(...)` checks its API key when the object is created, so a missing key crashes at **import time**, before the UI can show a helpful message.
  - **Fix:** Pass a dummy value (`"not-set"`). The sidebar uses `missing_keys()` to show ❌ instead.
- **Problem 2:** `.env` files often still contain placeholders like `your_key_here`, which look like real values.
  - **Fix:** `_is_placeholder()` in `config.py` treats them as missing.
- **Problem 3:** `from serpapi import GoogleSearch` fails because the pip package is named `google-search-results`.
  - **Fix:** Put the right name in `requirements.txt`, and import SerpAPI **lazily** inside `_serpapi_search` so the app still starts without it.
- **Problem 4:** `python travel_planner/graph.py` fails with "attempted relative import".
  - **Fix:** Run it as a module from the project folder: `python -m travel_planner.graph "..."`.

---

## 12. Slow responses with no feedback

- **Problem:** A combined request can make many LLM and API calls. The Deep Agent itinerary is the slowest part, and the user just sees a frozen screen.
- **Fix:**
  - Stream with `stream_mode="updates"` and show a label for each node as it runs (`NODE_LABELS` in `app.py`).
  - Cache the compiled graph with `@st.cache_resource` so it isn't rebuilt on every Streamlit rerun.
  - The Deep Agent is wrapped as a tool (`call_itinerary_research_agent`) that returns only its final message, so its internal steps don't fill the Travel Scout's context.

---

## Quick reference

| # | Challenge | Where it's handled |
|---|---|---|
| 1 | Parallel writes overwrite each other | `operator.add` reducer in `state.py` |
| 2 | Tool histories mixed between agents | Separate `search_messages` / `planner_messages` |
| 3 | Synthesizer runs early or twice | `defer=True` in `graph.py` |
| 4 | Sub-query vs original query | `Send` payload in `dispatch_to_agents` |
| 5 | Non-travel query, no next node | Fallback `Send` to synthesizer |
| 6 | Old results leak into new question | Fresh `thread_id` per question in `app.py` |
| 7 | Endless tool loop | `MAX_TOOL_ROUNDS` + plain `llm` |
| 8 | Tool exception crashes graph | `try/except` in tools and `_tool_step` |
| 9 | Bad dates / IATA codes | Pydantic `args_schema` + error passthrough + prompt rules |
| 10 | Large API payloads | Top-10 results, selected fields only |
| 11 | Keys, package name, imports | `missing_keys()`, dummy Tavily key, lazy import, `python -m` |
| 12 | Slow, silent UI | Streamed node labels, `@st.cache_resource` |
