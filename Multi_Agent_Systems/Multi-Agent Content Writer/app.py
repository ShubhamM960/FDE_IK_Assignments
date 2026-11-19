"""Streamlit UI for the Smart Content Assistant.

Run:  streamlit run app.py
"""

import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from content_assistant import build_graph, make_config
from content_assistant.config import get_llm
from content_assistant.nodes.utils import conversation_history

AGENT_LABELS = {
    "seo_blog_writer": "📝 SEO Blog Writer",
    "x_blog_writer": "🐦 X/Twitter Writer",
    "general": "💬 General",
}

st.set_page_config(page_title="Smart Content Assistant", page_icon="✍️", layout="wide")


# The graph (and therefore the MemorySaver) is created once per server process,
# so conversations survive Streamlit reruns and are shared across threads/tabs.
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# ---------------------------------------------------------------- session state
if "threads" not in st.session_state:
    st.session_state.threads = [str(uuid.uuid4())[:8]]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.threads[0]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("✍️ Content Assistant")

    with st.expander("🔑 API keys", expanded=not os.getenv("OPENAI_API_KEY")):
        openai_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        tavily_key = st.text_input(
            "Tavily API key (optional)", value=os.getenv("TAVILY_API_KEY", ""), type="password",
            help="Without it, the research/search tools return mock data.",
        )
        if openai_key and openai_key != os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = openai_key
            get_llm.cache_clear()
        if tavily_key != os.getenv("TAVILY_API_KEY", ""):
            os.environ["TAVILY_API_KEY"] = tavily_key

    st.caption("Tools mode: " + ("🌐 live Tavily" if os.getenv("TAVILY_API_KEY") else "🧪 mock data"))

    st.subheader("Conversations")
    if st.button("➕ New conversation", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.threads.insert(0, new_id)
        st.session_state.thread_id = new_id
        st.rerun()

    st.session_state.thread_id = st.radio(
        "Thread",
        st.session_state.threads,
        index=st.session_state.threads.index(st.session_state.thread_id),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "**Agents**\n"
        "- 📝 SEO Blog Writer — `research_tool` + `internet_search_tool`\n"
        "- 🐦 X Writer — `internet_search_tool`\n"
        "- 💬 General — conversation memory\n\n"
        "**Try:**\n"
        "- *Write an SEO blog on AI agents in healthcare*\n"
        "- *Tweet about remote work trends*\n"
        "- *What was my last request?*"
    )

config = make_config(st.session_state.thread_id)

# ---------------------------------------------------------------- history
st.header("Smart Content Assistant")
st.caption(f"Thread `{st.session_state.thread_id}` · memory persisted with LangGraph MemorySaver")

snapshot = graph.get_state(config)
for msg in conversation_history(snapshot.values.get("messages", [])):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.caption(AGENT_LABELS.get(msg.name, msg.name or "assistant"))
            st.markdown(msg.content)

# ---------------------------------------------------------------- new turn
prompt = st.chat_input("Ask for a blog post, a tweet, or anything else…")
if prompt:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please add your OpenAI API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Routing request…", expanded=True)
        final = None
        tool_counts = {}
        try:
            for update in graph.stream({"user_input": prompt}, config, stream_mode="updates"):
                for node, data in update.items():
                    if node == "router":
                        label = AGENT_LABELS.get(data["route"], data["route"])
                        status.write(f"🧭 Router → **{label}**")
                        status.update(label=f"{label} is working…")
                        continue
                    for msg in (data or {}).get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1
                                icon = "🔬" if tc["name"] == "research_tool" else "🔎"
                                status.write(f"{icon} `{tc['name']}` ← `{tc['args']}`")
                        elif isinstance(msg, ToolMessage):
                            status.write(f"✅ `{msg.name}` returned {len(str(msg.content))} chars")
                            status.code(str(msg.content)[:1500], language="json")
                        elif isinstance(msg, AIMessage):
                            final = msg
            summary = ", ".join(f"{k}×{v}" for k, v in tool_counts.items()) or "no tools"
            status.update(label=f"Done · {summary}", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Error", state="error")
            st.exception(exc)

        if final is not None:
            st.caption(AGENT_LABELS.get(final.name, final.name or "assistant"))
            st.markdown(final.content)
            if final.name == "x_blog_writer":
                st.caption("Tip: each option shows its character count (limit 280).")
