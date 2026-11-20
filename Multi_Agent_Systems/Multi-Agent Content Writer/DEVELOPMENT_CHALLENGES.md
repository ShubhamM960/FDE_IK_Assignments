# Development Challenges & Fixes: Smart Content Assistant (Multi-Agent Content Writer)

Real problems you can hit while building this project (LangGraph + Tavily + `MemorySaver`), and how the code handles each one.

```
START → router ─┬─ seo_blog_writer ⇄ seo_tools → END
                ├─ x_blog_writer   ⇄ x_tools   → END
                └─ general                     → END
```

---

## 1. The assistant forgets earlier turns

- **Problem:** "What was my last request?" gets "I don't know", because every `graph.stream()` call starts with empty state.
- **Fix:** Use three things together:
  - `messages: Annotated[List[BaseMessage], add_messages]` appends to the history instead of replacing it (`state.py`).
  - Compile the graph with a `MemorySaver` checkpointer (`graph.py`).
  - Pass the **same config on every call**, via `make_config(thread_id)`.
- **Streamlit catch:** If the graph is built at the top of `app.py`, every rerun creates a new `MemorySaver` and memory is lost. `@st.cache_resource` on `get_graph()` keeps one graph per server process.

---

## 2. The router returns messy text instead of a route name

- **Problem:** The prompt says "one word only", but the LLM sometimes replies `Route: seo_blog_writer.` or `**x_blog_writer**`. Using that string as a conditional edge key fails.
- **Fix:**
  - Set router temperature to `0.0` (`config.py`).
  - Clean the output with `_parse_route()`: lowercase it, strip punctuation, match it against `VALID_ROUTES`, and default to `general` (`router.py`).
  - Keep route names in one place (`ROUTE_SEO`, `ROUTE_X`, `ROUTE_GENERAL`) so the prompt, router and graph edges always match.

---

## 3. The router LLM call fails

- **Problem:** A network error, rate limit or missing key during routing would end the turn with an exception.
- **Fix:** Wrap the call in `try/except` and fall back to a **regex keyword router** (`router.py → _keyword_fallback`).

```python
if re.search(r"\b(tweet|twitter|x post|thread|hashtag)", text):
    return ROUTE_X
if re.search(r"\b(blog|article|seo|long[- ]form)", text):
    return ROUTE_SEO
return ROUTE_GENERAL
```

---

## 4. Follow-up requests go to the wrong agent

- **Problem:** On its own, "Make it shorter" or "Now do one about healthcare" looks like a general question. It should go back to the writer that handled the previous turn.
- **Fix:** Give the router the **last 4 user/assistant messages** (each cut to 200 characters), labelled with the agent name (`ASSISTANT(x_blog_writer)`). This adds context without much extra cost (`router.py`).

---

## 5. The expensive research tool gets called too often

- **Problem:** `research_tool` uses Tavily's `advanced` search, which costs more. The LLM may ignore "call at most twice" in the prompt and keep calling it.
- **Fix:** Add a **code-level guardrail** on top of the prompt (`seo_writer.py`):
  - `count_tool_calls(messages, "research_tool")` counts calls **since the last human message**, so it only covers the current turn.
  - Once `MAX_RESEARCH_CALLS_PER_TURN` (default 2) is reached, the LLM is re-bound **without** `research_tool` and the system prompt says the budget is used up.

```python
if research_calls >= MAX_RESEARCH_CALLS_PER_TURN:
    tools = [internet_search_tool]
```

---

## 6. Token cost grows every turn

- **Problem:** With full memory, every new request re-sends all earlier Tavily results (up to 5 results × 500 characters per call), even though they aren't needed any more.
- **Fix:** `agent_context()` builds a trimmed context (`nodes/utils.py`):
  - **Earlier turns:** only user messages and final AI answers, limited to the last 10 (`max_prior=10`).
  - **Current turn:** kept **in full**.
- **Why the current turn must stay in full:** OpenAI requires every `ToolMessage` to follow the `AIMessage` with the matching `tool_calls`. Removing them mid-loop breaks the request.

---

## 7. Runaway agent ⇄ tool loops

- **Problem:** The writer keeps asking for "one more search" and never writes the post.
- **Fix:**
  - Set `recursion_limit` (default 25, configurable through an env variable) in `make_config()`. LangGraph stops the run with `GraphRecursionError` instead of looping forever.
  - The `should_continue_*` functions end the turn as soon as the last `AIMessage` has no `tool_calls`.
  - In the UI, the error is caught and shown with `st.exception`, so the app doesn't crash.

---

## 8. User messages get duplicated in history

- **Problem:** The router saves the user's `HumanMessage`, and `general_node` returns it again with the AI reply. Without care, the history would contain the same user message twice.
- **Fix:** `add_messages` merges by **message ID**. The router's message already has an ID, so returning the same object updates it in place instead of adding a copy (`general.py`).
- **Lesson:** Return the existing message object, not a new `HumanMessage(content=...)` with the same text. A new object gets a new ID and will be duplicated.

---

## 9. Developing without a Tavily key

- **Problem:** Without a Tavily key, you can't test routing, tool loops or memory, and every test run costs API credits.
- **Fix:**
  - Both tools return **mock JSON** with the same shape when `TAVILY_API_KEY` isn't set (`tools.py`).
  - `TavilyClient` is imported inside `_tavily()`, so the package is only needed for live mode.
  - The sidebar shows which mode is active (`🌐 live Tavily` / `🧪 mock data`).
- **Watch out:** The mock URLs and statistics are fake. Never publish content written in mock mode.

---

## 10. An API key entered in the UI has no effect

- **Problem:** `get_llm()` uses `@lru_cache`, so the `ChatOpenAI` instance created before the key was set stays cached. Entering a key in the sidebar changes nothing until restart.
- **Fix:** When the key changes, update `os.environ` and call `get_llm.cache_clear()` (`app.py`).
- **Also:** `get_llm()` raises a clear `EnvironmentError` when the key is missing, instead of a vague OpenAI auth error.

---

## 11. Not knowing which agent answered

- **Problem:** In the UI, and in the router's context, every reply is just "AI". You can't tell which agent wrote it or debug routing.
- **Fix:** Each node sets `response.name = AGENT_NAME`. The UI uses it for labels (`AGENT_LABELS`), and the router uses it in its context string (see #4).
- **Visibility:** `stream_mode="updates"` lets the UI and CLI show each step as it happens: the chosen route, every tool call with its arguments, and the size of each tool result.

---

## 12. Tweets go over 280 characters

- **Problem:** LLMs can't count characters reliably, so a "short" tweet with emojis and hashtags often goes over 280.
- **Current handling:** This is enforced by the prompt only (`X_WRITER_PROMPT`: "Each tweet MUST be under 280 characters"), plus a reminder in the UI.
- **Possible improvement (not yet in the code):** Check `len(tweet)` after generation and ask the LLM to shorten any tweet that's too long.

---

## Quick reference

| # | Challenge | Where it's handled |
|---|---|---|
| 1 | No memory across turns | `add_messages` + `MemorySaver` + `make_config(thread_id)` + `@st.cache_resource` |
| 2 | Messy router output | `_parse_route()`, temperature 0, shared route names |
| 3 | Router LLM failure | `_keyword_fallback()` regex |
| 4 | Follow-ups misrouted | Last 4 messages passed to the router |
| 5 | Expensive tool overused | `count_tool_calls` + re-bind without `research_tool` |
| 6 | Growing token cost | `agent_context()`: condensed past, full current turn |
| 7 | Runaway loops | `recursion_limit` + `should_continue_*` |
| 8 | Duplicate user messages | `add_messages` ID-based merge |
| 9 | No Tavily key | Mock tool outputs + lazy import |
| 10 | UI key ignored | `get_llm.cache_clear()` |
| 11 | Unknown agent / no visibility | `response.name` + `stream_mode="updates"` |
| 12 | Tweets over 280 chars | Prompt rule (code check is a possible improvement) |
