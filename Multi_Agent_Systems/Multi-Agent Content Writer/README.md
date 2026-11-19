# Smart Content Assistant (LangGraph multi-agent)

```
START → router ─┬─ seo_blog_writer ⇄ seo_tools (research_tool, internet_search_tool) → END
                ├─ x_blog_writer   ⇄ x_tools   (internet_search_tool)                → END
                └─ general                                                          → END
```

## Package layout

| Module | Purpose |
|---|---|
| `content_assistant/config.py` | Env settings, route names, LLM factory, research budget |
| `content_assistant/state.py` | `AgentState` (`messages` with `add_messages`, `user_input`, `route`) |
| `content_assistant/prompts.py` | Router / SEO / X / General system prompts (tool-cost policy lives here) |
| `content_assistant/tools.py` | `research_tool` (Tavily advanced = costly), `internet_search_tool` (Tavily basic = cheap), mock fallbacks |
| `content_assistant/nodes/` | `router`, `seo_writer`, `x_writer`, `general`, `should_continue_*`, helpers |
| `content_assistant/graph.py` | Graph wiring + `MemorySaver` checkpointer, `make_config(thread_id)` |
| `content_assistant/cli.py` | Terminal chat |
| `app.py` | Streamlit UI |

## Setup

Python 3.9+ (3.10+ recommended for the latest LangGraph releases).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY (TAVILY_API_KEY optional -> mock data otherwise)
```

## Run

```bash
streamlit run app.py            # UI
python -m content_assistant.cli # CLI
```

## Usage in code

```python
from content_assistant import build_graph, make_config

graph = build_graph()
config = make_config("user-42")   # pass the same thread_id on every call
graph.invoke({"user_input": "Write an SEO blog on AI in healthcare"}, config)
graph.invoke({"user_input": "What was my last request?"}, config)
```

## Cost control

- The prompt tells the SEO agent: `research_tool` first, once (twice max), then `internet_search_tool` for keywords/trends.
- In code, `seo_writer.py` drops `research_tool` from the bound tools after `MAX_RESEARCH_CALLS_PER_TURN` calls.
- Earlier turns' tool outputs are not re-sent to the LLM; only the user and final AI messages are.
