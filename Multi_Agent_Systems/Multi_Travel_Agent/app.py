"""Streamlit UI for the Multi Travel Agent.

Run from inside the Multi_Travel_Agent/ folder:
    streamlit run app.py
"""

import uuid

import streamlit as st

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

from travel_planner.config import MODEL_NAME, REQUIRED_KEYS, missing_keys  # noqa: E402


@st.cache_resource(show_spinner="Loading travel agents...")
def get_graph():
    """Build the graph once and reuse it across reruns/sessions."""
    from travel_planner import build_graph
    return build_graph()


NODE_LABELS = {
    "orchestrator": "🧭 Router is analysing your request",
    "hotel_flight_agent_node": "🛫 Flight & Hotel agent is thinking",
    "hotel_flight_tool_node": "🔎 Searching Google Flights / Hotels",
    "travel_scout_agent_node": "🗺️ Travel Scout agent is thinking",
    "travel_scout_tool_node": "🌐 Researching the web / building itinerary",
    "synthesizer": "🧩 Composing the final answer",
}

EXAMPLES = [
    "Best time to visit Jodhpur?",
    "Flights from Delhi to London on 2026-12-10 for 2 adults",
    "Plan 2 days in Agra for a history & culture lover",
    "Plan 3 days in Paris and find hotels from 2026-12-10 to 2026-12-13 for 2 adults",
]

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("✈️ AI Travel Planner")
    st.caption("Router → Flight/Hotel agent + Travel Scout → Synthesizer")

    st.subheader("API keys")
    missing = missing_keys()
    for key in REQUIRED_KEYS:
        st.write(f"{'❌' if key in missing else '✅'} `{key}`")
    st.caption(f"Model: `{MODEL_NAME}`")

    show_steps = st.toggle("Show agent steps", value=True)

    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])

# ---------------------------------------------------------------- Main area
st.header("Your AI Travel Assistant")
st.write("Ask about **flights, hotels, itineraries, weather, visas** or general travel tips.")

if "OPENAI_API_KEY" in missing:
    st.error("`OPENAI_API_KEY` is missing. Add it to `Multi_Travel_Agent/.env` and restart the app.")
    st.stop()
if missing:
    st.warning(
        f"Missing: {', '.join(missing)}. Related features will fail "
        "(SerpAPI → flights/hotels, Tavily → web research)."
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def run(query: str) -> str:
    """Stream the graph and show each node as it runs."""
    graph = get_graph()
    # A fresh thread per question so accumulated lists (agent_results, ...) start empty
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    status = st.status("Working on it...", expanded=True) if show_steps else None
    answer = ""

    for update in graph.stream({"user_query": query}, config=config, stream_mode="updates"):
        for node, out in update.items():
            if status:
                status.write(NODE_LABELS.get(node, node))
                if node == "orchestrator" and out:
                    for t in out.get("tasks", []):
                        status.caption(f"→ `{t.source}`: {t.user_query}")
            if out and out.get("final_answer"):
                answer = out["final_answer"]

    if status:
        status.update(label="Done ✅", state="complete", expanded=False)
    return answer or "Sorry, I couldn't produce an answer."


prompt = st.chat_input("e.g. Plan a 5-day trip to Tokyo in April for a foodie")
prompt = st.session_state.pop("pending", None) or prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = run(prompt)
        except Exception as e:
            answer = f"⚠️ Something went wrong: `{type(e).__name__}: {e}`"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# Run the file :
# =================
# Step by step Guilde
    #   cd "Multi_Agent_Systems/Multi_Travel_Agent"
    #   pip install -r requirements.txt
     
    #Run with UI / web
        # command :   streamlit run app.py
    
    
    
    # # or, without the UI:
        #command :  python -m travel_planner.graph "Best time to visit Jodhpur?"