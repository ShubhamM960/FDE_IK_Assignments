# SnackStack: Development Challenges & Fixes

The problems we hit while building SnackStack (**orchestrator → menu / order agents → synthesizer**), the cause of each, and how we fixed it.

---

## 1. The app would not even start (syntax / import errors)

**Symptom:** `SyntaxError` / `ImportError` before the first prompt.

**Causes & fixes:**
| File | Problem | Fix |
|------|---------|-----|
| `config.py` | `OpenAI(api_key=api_key")`, a stray quote | Removed the quote |
| `graph.py` | `from snackstack.agents import { ... }` | Use `( ... )`. Braces are not valid import syntax |
| `state.py`, agents | `Class X(TypedDict)` | Python keywords are lowercase: `class` |
| agents | `{tool.name = tool for tool in tools}` | Dict comprehensions use `:` → `{tool.name: tool ...}` |
| `synthesizer.py` | `from snackstack.state import` (empty) and an unclosed `{menu_response"` f-string | Completed the import, closed the brace |
| `rag.py` | `def exist_vectorstore(path str : str = ...)` | `path: str = ...` |
| `order_agent.py` | `re.compile` used without `import re` | Added the import |

**Lesson:** run `python -m pyflakes snackstack` (or let the IDE lint) after every phase. It catches all of these in a second.

---

## 2. Circular import between `agents/__init__.py` and `menu_agent.py`

**Symptom:** `ImportError: cannot import name 'MENU_AGENT_PROMPT' from partially initialized module 'snackstack.agents'`.

**Cause:** `menu_agent.py` did `from snackstack.agents import MENU_AGENT_PROMPT`. While `agents/__init__.py` is still importing `menu_agent`, the package has not reached the line that imports the prompts yet.

**Fix:** import from the concrete module: `from snackstack.agents.prompts import MENU_AGENT_PROMPT`. The same applies to `menu_tools.py`, which imported `snackstack.rag` instead of `snackstack.tools.rag`.

---

## 3. Wrong import path for `OpenAIEmbeddings`

**Symptom:** `ImportError: cannot import name 'OpenAIEmbeddings' from 'langchain_core.embeddings'`.

**Cause:** `langchain_core` only defines the abstract `Embeddings` interface. The OpenAI implementation ships in the provider package.

**Fix:** `from langchain_openai import OpenAIEmbeddings`.

---

## 4. Orchestrator could route to an agent that doesn't exist

**Symptom:** `Send()` fails with an unknown node, or the LLM returns `"menu"` instead of `"menu_agent"`.

**Cause:** `source: str = Literal["menu_agent", "order_agent"]` makes the Literal the **default value**, not the **type**. The schema the LLM sees is just "any string".

**Fix:**
```python
source: Literal["menu_agent", "order_agent"] = Field(description="Agent that should handle this task")
```
Now `with_structured_output()` constrains the LLM to exactly those two values.

---

## 5. `InvalidUpdateError` from the orchestrator

**Symptom:** `InvalidUpdateError: Invalid update for channel 'tasks'`.

**Cause:** the orchestrator returned `{"tasks": ...}` but `StackState` declared `task` (singular). LangGraph only accepts keys that exist in the state schema. Also, `Annotated[Dict]` (without metadata) is not a valid type and crashed at import.

**Fix:** state keys match what nodes return (`tasks`, `menu_response: Optional[Dict]`, `order_response: Optional[Dict]`, `final_answer: str`). The tasks are stored as plain dicts (`model_dump()`) so `MemorySaver` serialises them without warnings.

---

## 6. `NameError` when building the `Send()` payload

**Symptom:** `NameError: name 'messages' is not defined`.

**Cause:** dictionary keys were written as bare names: `{messages: ..., user_query: ...}`. Python treats them as variables.

**Fix:** use string keys: `{"messages": ..., "user_query": ..., "route_agent": ...}`. We also used `messages = state.get("messages", [])` once at the top of the node.

---

## 7. Same agent dispatched twice → parallel write conflict

**Symptom:** `InvalidUpdateError: At key 'menu_response': Can receive only one value per step`.

**Cause:** the LLM sometimes lists `menu_agent` twice. Two parallel `Send()` branches then write the same non-reducer key in one super-step.

**Fix:** de-duplicate tasks in the orchestrator with a `seen` set before creating `Send()` objects. (The menu and order agents write to *different* keys, so running both in parallel is safe.)

---

## 8. Empty route → synthesizer with nothing to say

**Symptom:** for an unclear query, the reply was "Sorry, I could not process your request."

**Cause:** when the LLM returned no tasks, the code did `Send("synthesizer", {})`, so no agent ever ran.

**Fix:** the spec says *default to menu_agent*: `targets = [Send("menu_agent", {...})]`. Routing errors (API failures) are caught and use the same default.

---

## 9. Old answers leaking into new turns

**Symptom:** user asks "Track ORD-201", then "Show vegan dishes". The reply mentions the Butter Chicken order again.

**Cause:** `MemorySaver` persists **all** state keys across turns, not just `messages`. `order_response` from turn 1 was still there in turn 2, and the synthesizer used it.

**Fix:** the orchestrator resets `menu_response` and `order_response` to `None` in its `Command(update=...)` at the start of every turn.

---

## 10. Synthesizer dropped the menu answer for combined queries

**Symptom:** "Pizza options + track ORD-203" only answered the order part.

**Cause:** `if order_response: ... elif menu_response: ...`. With `elif`, the second branch is skipped whenever the first is true.

**Fix:** two independent `if` blocks, so both responses are added to `combined_response`.

---

## 11. Tool-calling sub-graph crashed on the first tool call

**Symptoms:** `KeyError: 'menu_agent_tools'`, `UnboundLocalError: menu_agent_graph`, `AttributeError: 'dict' object has no attribute 'content'`.

**Causes & fixes:**
- `should_continue` returned `"menu_agent_tools"` / `END`, but the path map keys were `"tools"` / `"end"`. **Fix:** return the path-map keys.
- `menu_agent_graph = menu_agent_graph()` inside the function makes the name local, so Python fails before the call. **Fix:** build the sub-graph once at module level (`menu_agent_graph = build_menu_agent_graph()`). This is also an optimisation: it is no longer recompiled on every query.
- The tools list and the tools-node function had the **same name** (`menu_agent_tools`). The function silently replaced the list. **Fix:** renamed the list to `MENU_TOOLS` / `ORDER_TOOLS`.
- `[result][-1].content` / `result[-1].content`: `graph.invoke()` returns a **dict**. **Fix:** `result["messages"][-1].content`.
- Typo `"order  _agent_model"` in `add_conditional_edges` → "unknown node". **Fix:** corrected the name.

---

## 12. Tool-calling infinite loop

**Symptom:** the agent keeps calling `search_menu_catalog` and never answers (or hits LangGraph's default recursion limit of 25 with an ugly traceback).

**Fix:** cap the loop with `MAX_TOOL_ITERATIONS = 5`. The sub-graph is invoked with `{"recursion_limit": 2 * MAX_TOOL_ITERATIONS + 1}` (each iteration = model step + tools step). `GraphRecursionError` is caught and a friendly fallback message is returned.

---

## 13. RAG search always returned nothing

**Symptom:** every menu query came back with "No menu items found".

**Causes:**
1. `if not vector_store:` was meant to detect a missing DB, but a `Chroma` object is **always truthy**, so `build_vectorstore()` never ran and we searched an empty collection.
2. The "exists" and "build" functions used different collection names (`"SNACKSTACK DB"` vs `"snackstack db"`). Chroma also rejects names with spaces.
3. `build_vectorstore()` referenced an undefined `menu_items`.

**Fix:** check `vector_store._collection.count() == 0`, use one shared `COLLECTION_NAME = "snackstack_menu"` and an absolute `DB_PATH`, and pass `get_menu`. Stable `ids` prevent duplicate documents on rebuild.

---

## 14. Text splitter broke each dish apart (and crashed)

**Symptom:** `ValueError: Got a larger chunk overlap (200) than chunk size (200)`. After that was fixed, results were missing prices or tags.

**Cause:** `chunk_size=200, chunk_overlap=200`. The overlap must be smaller than the chunk. A dish (~300 chars) was also split so that "Butter Chicken" and "price: 379" ended up in different chunks.

**Fix:** `chunk_size=500, chunk_overlap=50`, so one dish = one chunk. For a catalog this small, splitting is not needed at all.

---

## 15. Chroma rejected the metadata

**Symptom:** `ValueError: Expected metadata value to be a str, int, float or bool, got list`.

**Cause:** `dietary_tags` is a list (`["Vegan", "GF"]`).

**Fix:** store it as a string: `", ".join(tags) or "None"`. We also added `price` to the metadata.

---

## 16. "Indian food under 300" returned wrong dishes

**Symptom:** a ₹379 Butter Chicken was suggested, or the brief's expected answer listed Aglio e Olio (which is **Italian**).

**Cause:** embeddings capture *meaning*, not arithmetic. "Under 300" is a hard constraint that similarity search cannot enforce. With `k=3`, matching dishes can be pushed out by near matches.

**Fix:** retrieve `k=4` and keep the price in every document, so the LLM filters by price and cuisine. The menu prompt says *"respect every constraint"*. The correct answer from our data is **Paneer Tikka (₹199)** and **Mango Lassi (₹99)**.
(Optional upgrade: add `max_price` / `cuisine` parameters to the tool and use Chroma `filter={"price": {"$lte": 300}}`.)

---

## 17. `search_menu_catalog` / `get_order_status` crashed

**Symptoms:** `NameError: result`, `UnboundLocalError: results`, `NameError: tracking_id`, `UnboundLocalError: match_order`. "ORD-201" was also "not found".

**Fixes:**
- Menu tool: check `if not docs`, and initialise `results = ""` before `+=`.
- Order tool: compare `order["tracking_id"]` and `order["customer_email"]` with the passed `key` (case-insensitive). Set `match_order = ORDERS.get(oid) if oid else None` so it is always defined.
- `normalise_order_id` produced `ORD201`, but the dictionary keys are `ORD-201`. It now returns `ORD-{number}`, so `ord201`, `ORD 201` and `ORD-201` all work.

---

## 18. Human-in-the-Loop asked for an ID the user had already given

**Symptom:** "Pizza options and track ORD-203" still triggered "Please provide a valid order ID…".

**Cause:** the orchestrator rewrites the sub-query for each agent and sometimes drops the ID.

**Fix:** the order agent checks the sub-query **and** the user's original message (`context[-1]`) for an identifier before calling `interrupt()`. The orchestrator prompt and schema also say "copy any order ID / tracking ID / email exactly".

---

## 19. HITL resume behaved oddly

**Symptoms:** the resumed run re-asked the question but `main.py` didn't handle it, and the LLM never received the ID that the user typed.

**Causes & fixes:**
- **The node re-runs from the top on resume.** Everything before `interrupt()` executes again, so the HITL check is placed *before* any LLM call. Otherwise you pay for (and may duplicate) LLM calls.
- `main.py` resumed only once (`for task in snapshot.tasks`). **Fix:** a `while True` loop that re-reads `graph.get_state()` until there are no pending interrupts.
- The prompt told the LLM to use a `"Lookup key: ..."` line that was never sent. **Fix:** the human message now includes `Lookup key: {identifier}`.
- `interrupt()` requires a checkpointer. The graph is compiled with `checkpointer=MemorySaver()` and every call passes the same `thread_id`.

---

## 20. `graph.py` crashed at import

**Symptom:** `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id`.

**Cause:** a leftover debug line `graph.get_state_history()` at module level, called without a config.

**Fix:** removed it. `destinations=(...)` was also added to nodes that route with `Command`/`Send`, so `graph.get_graph().draw_mermaid()` shows the real edges.

---

## 21. Text mode crashed on machines without audio hardware

**Symptom:** `OSError: PortAudio library not found` even when running plain `snackstack`.

**Cause:** `main.py` always imports the voice package, and `speaker.py` imported `sounddevice` unguarded. `recorder.py` only caught `ImportError`, but a missing PortAudio raises `OSError`.

**Fix:** both modules use `try: import sounddevice ... except (ImportError, OSError)` and skip playback/recording gracefully.

---

## 22. Voice output didn't play

**Symptom:** TTS file created, but `soundfile` failed to read it.

**Cause:** OpenAI TTS returns **MP3** by default. Older `libsndfile` builds cannot decode MP3.

**Fix:** request `response_format="wav"`, save as `.wav`, and use `write_to_file` (`stream_to_file` is deprecated).

---

## 23. One API error killed the whole session

**Symptom:** a network hiccup or rate limit raised an exception and exited the REPL.

**Fix:** `run_text_loop` wraps `assistant.ask()` in `try/except`, logs the traceback to `logs/snackstack.log` and replies "Sorry, something went wrong".

---

## Optimisations applied

- Sub-graphs (`menu_agent_graph`, `order_agent_graph`) and `llm.with_structured_output(...)` are built **once at import**, not on every request.
- The persisted Chroma store is reused across runs. It is rebuilt only when the collection is empty.
- Agents and the orchestrator get only the **last 6 messages** of history, which keeps prompts small and cheap while still handling follow-ups.
- The tool loop is capped (`MAX_TOOL_ITERATIONS`) to bound latency and cost.
- `.gitignore` now excludes `.env`, `snackstack_db/` and `logs/`. `.env.example` documents `OPENAI_API_KEY`.
