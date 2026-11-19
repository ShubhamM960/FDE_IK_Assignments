Assignment completed each week . 

Week -0 : 
Basic agent using Langchain_pandas

Week-1 : 
LLM library / LLM provider agent : IT support agent 
    Based on the memory it identify when to restart the system . when to esclate to human when we not to do any thing.

Week- 2 : RAG Basics : Chunking , Embeddings and Vector Store

---

## Multi-Agent Systems

All three projects in [`Multi_Agent_Systems/`](Multi_Agent_Systems/) use **LangGraph** and follow the same basic idea:

1. A **router / orchestrator** LLM reads the user's message and decides which specialist agent(s) should handle it.
2. Each **specialist agent** has its own prompt and tools (search, database lookup, web research, etc.) and runs a ReAct loop: think → call a tool → read the result → answer.
3. When more than one agent answers, a **synthesizer** merges their replies into one response.

Each project has a Streamlit UI and its own detailed README (setup, architecture, troubleshooting).

| Project | What it is | Agents | Key tech |
|---|---|---|---|
| [Multi-conversational Agent (AxiomCart Assistant)](Multi_Agent_Systems/Multi_conversational%20Agent/README.md) | Customer helper for an online store, usable by text or voice | Orchestrator, Product Discovery, Sales Support, Synthesizer | LangGraph `Send` fan-out, `interrupt()` (human-in-the-loop), ChromaDB RAG, Whisper + TTS |
| [Multi Travel Agent](Multi_Agent_Systems/Multi_Travel_Agent/README.md) | AI travel planner for flights, hotels, itineraries and travel tips | Router, Flight & Hotel, Travel Scout, Itinerary Deep Agent, Synthesizer | SerpAPI (Google Flights/Hotels), Tavily search, Deep Agents |
| [Multi-Agent Content Writer (Smart Content Assistant)](Multi_Agent_Systems/Multi-Agent%20Content%20Writer/README.md) | Writes SEO blogs and X (Twitter) posts backed by web research | Router, SEO Blog Writer, X Writer, General | Tavily research, cost-controlled tool use, `MemorySaver` memory |

### 1. Multi-conversational Agent: AxiomCart Assistant

A shopping assistant for a sample online store called AxiomCart.

- **What it does:** answers product questions ("Show me noise-cancelling headphones"), checks order status ("Where is ORD102?"), and opens escalation tickets when a customer asks for a human.
- **How it works:** the orchestrator sorts each message into product questions, support questions, or both. Mixed questions go to both agents **in parallel**, and the synthesizer combines the replies.
  - **Product Discovery Agent:** does a semantic search over the product catalog (stored in ChromaDB). It also handles greetings and small talk.
  - **Sales Support Agent:** looks up orders and creates escalation tickets. If the order ID or email is missing, it **pauses the graph to ask the customer** and continues once they answer.
- **Multimodal:** you can type, or talk through a microphone (Whisper speech-to-text) and hear the replies spoken back (OpenAI TTS).
- **Memory:** earlier turns in a session are kept, so follow-up questions work.

```
START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                      └─ support_agent ──↗
```

### 2. Multi Travel Agent

An AI travel planner. You ask a travel question in plain English and get one combined answer.

- **What it does:** searches flights and hotels, builds day-by-day itineraries, and gives general travel advice (best time to visit, weather, visas, safety, packing). It turns down questions that aren't about travel.
- **How it works:** the router splits the request into focused sub-tasks and sends them to the right agents in parallel.
  - **Flight & Hotel Agent:** live flight and hotel searches through SerpAPI (Google Flights / Google Hotels). It suggests Value / Balanced / Comfort bundles.
  - **Travel Scout Agent:** answers general travel questions with Tavily web search and cites its sources.
  - **Itinerary Deep Agent:** used by the Travel Scout as a tool when a detailed day-by-day plan is needed.
  - **Synthesizer:** waits until every branch is done, then merges the results.
- **Example:** "Plan 3 days in Paris and find hotels for 2 adults" → both agents run in parallel → one merged plan.
- Also has a single-file version for reference (`single file Travel agent/multi_Travel_agent.py`).

```
START → orchestrator ─┬─ flight_hotel_agent ⇄ tools ─┬─→ synthesizer → END
                      └─ travel_scout_agent ⇄ tools ─┘
```

### 3. Multi-Agent Content Writer: Smart Content Assistant

A writing assistant that sends each request to the right kind of writer.

- **What it does:** writes long-form SEO blog posts, short X (Twitter) posts, or simply answers general questions.
- **How it works:** the router picks **one** route per message.
  - **SEO Blog Writer:** researches the topic in depth (Tavily advanced search), then runs quick searches for keywords and trends before writing.
  - **X Writer:** does a quick web search and writes short posts for social media.
  - **General:** handles everything else, such as "What was my last request?"
- **Cost control:** the expensive research tool is limited to a few calls per turn, and old tool outputs aren't re-sent to the LLM.
- **Memory:** conversation history is kept per `thread_id`. If no Tavily key is set, it falls back to mock data.

```
START → router ─┬─ seo_blog_writer ⇄ seo_tools → END
                ├─ x_blog_writer   ⇄ x_tools   → END
                └─ general                     → END
```

### Concepts you'll see across these projects

| Concept | Meaning |
|---|---|
| **Router / Orchestrator** | An LLM with structured output that decides which agent(s) handle a request. It doesn't answer the question itself. |
| **Fan-out (`Send`)** | Runs several agents in parallel, each with its own sub-task. |
| **ReAct loop** | Agent ⇄ tool cycle that repeats until the agent has enough information to answer. |
| **Synthesizer** | Merges several agents' answers into one reply. |
| **Human-in-the-loop (`interrupt`)** | Pauses the graph to ask the user for missing details, then picks up where it stopped. |
| **Checkpointer (`MemorySaver`)** | Keeps conversation state per `thread_id` (in memory only). |
