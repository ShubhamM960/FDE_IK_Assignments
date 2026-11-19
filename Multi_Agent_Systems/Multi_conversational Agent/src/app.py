"""
AxiomCart Streamlit UI.

Run from the project root:
    uv run streamlit run src/app.py
    # or:  streamlit run src/app.py

How the UI drives the assistant
-------------------------------
• One ``AxiomCartAssistant`` is created per browser session and kept in
  ``st.session_state["assistant"]``. It is created with
  ``enable_voice=False`` so it never touches the *server's* microphone.
• Because the CLI's ``query()`` blocks on ``input()`` during HITL, the UI
  uses the non-blocking ``start_turn()`` / ``resume_turn()`` methods instead.
  When the support agent pauses with ``interrupt()``, the question is shown
  in the chat and the user's next message is sent via ``resume_turn()``.
• Optional voice: browser microphone → ``VoiceRecorder.transcribe`` (Whisper)
  and replies → ``VoiceSpeaker.synthesise`` (OpenAI TTS) played in the page.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make `import src.…` work when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="AxiomCart Assistant", page_icon="🛒", layout="centered")


# ── Load the backend (config → RAG → graph) once per process ────
@st.cache_resource(show_spinner="Starting AxiomCart (building product index)…")
def load_backend():
    """Import the heavy modules once. config.py calls sys.exit() when the
    API key is missing, so SystemExit is caught and shown in the UI."""
    try:
        from src.data import ORDER_DATABASE, PRODUCT_CATALOG
        from src.main import AxiomCartAssistant
        from src.voice import VOICE_OPTIONS, VoiceRecorder, VoiceSpeaker
    except SystemExit:
        return None
    return {
        "Assistant": AxiomCartAssistant,
        "VoiceRecorder": VoiceRecorder,
        "VoiceSpeaker": VoiceSpeaker,
        "VOICE_OPTIONS": VOICE_OPTIONS,
        "ORDER_DATABASE": ORDER_DATABASE,
        "PRODUCT_CATALOG": PRODUCT_CATALOG,
    }


backend = load_backend()
if backend is None:
    st.error(
        "**OPENAI_API_KEY is missing.** Copy `.env.example` to `.env`, add your "
        "key, then restart Streamlit."
    )
    st.stop()


# ── Session state ───────────────────────────────────────────────
def new_session() -> None:
    """Create a fresh assistant (= new thread_id = empty memory)."""
    st.session_state.assistant = backend["Assistant"](enable_voice=False)
    st.session_state.chat = []            # [{"role", "content", "audio"?}]
    st.session_state.pending_question = None  # set while graph is interrupted
    st.session_state.mic_key = 0


if "assistant" not in st.session_state:
    new_session()


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛒 AxiomCart")
    st.caption(f"Session thread: `{st.session_state.assistant.thread_id[:12]}…`")

    if st.button("🔄 New conversation", use_container_width=True):
        new_session()
        st.rerun()

    st.subheader("Voice")
    has_audio_input = hasattr(st, "audio_input")
    use_mic = st.toggle(
        "Speak instead of typing",
        value=False,
        disabled=not has_audio_input,
        help=None if has_audio_input else "Requires Streamlit ≥ 1.40 (st.audio_input).",
    )
    speak_replies = st.toggle("Read replies aloud", value=False)
    voices = backend["VOICE_OPTIONS"]
    voice = st.selectbox(
        "TTS voice",
        options=list(voices),
        index=list(voices).index("nova"),
        format_func=lambda v: f"{v} — {voices[v]}",
        disabled=not speak_replies,
    )

    with st.expander("Sample orders"):
        for oid, o in backend["ORDER_DATABASE"].items():
            st.markdown(f"**{oid}** · {o['product']} · _{o['status']}_  \n`{o['customer_email']}`")

    with st.expander("Catalog"):
        for p in backend["PRODUCT_CATALOG"]:
            st.markdown(f"**{p['name']}** — ₹{p['price']:,}")


# ── Helpers ─────────────────────────────────────────────────────
def tts(text: str) -> str | None:
    if not speak_replies:
        return None
    path = backend["VoiceSpeaker"](voice=voice, speed=1.1).synthesise(text)
    return path or None


def transcribe(audio_file) -> str:
    """Browser WAV → numpy → Whisper (reuses VoiceRecorder.transcribe)."""
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(audio_file.getvalue()), dtype="float32")
    return backend["VoiceRecorder"](sample_rate=sample_rate).transcribe(data)


def handle_user_message(text: str) -> None:
    """Route the message: resume a paused graph, or start a new turn."""
    assistant = st.session_state.assistant
    st.session_state.chat.append({"role": "user", "content": text})

    with st.spinner("Agents are working…"):
        try:
            if st.session_state.pending_question is not None:
                turn = assistant.resume_turn(text)
            else:
                turn = assistant.start_turn(text)
        except Exception as exc:  # surface backend errors in the chat
            st.session_state.pending_question = None
            st.session_state.chat.append(
                {"role": "assistant", "content": f"⚠️ Something went wrong: `{exc}`"}
            )
            return

    if turn["status"] == "needs_input":
        st.session_state.pending_question = turn["content"]
        content = f"🔄 **Agent asks:** {turn['content']}"
    else:
        st.session_state.pending_question = None
        content = turn["content"]

    st.session_state.chat.append(
        {"role": "assistant", "content": content, "audio": tts(turn["content"])}
    )


# ── Main chat area ──────────────────────────────────────────────
st.title("AxiomCart Assistant")
st.caption("Ask about products, track an order, or request a human agent.")

if not st.session_state.chat:
    st.info(
        "Try: *“Show me noise-cancelling headphones”*, *“Where is my order?”*, "
        "or *“My order ORD102 is late — any other headphones?”*"
    )

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

placeholder = (
    "Answer the agent's question…" if st.session_state.pending_question else "Type your message…"
)

user_text: str | None = None

if use_mic:
    audio = st.audio_input("Record your message", key=f"mic_{st.session_state.mic_key}")
    if audio is not None:
        with st.spinner("Transcribing…"):
            user_text = transcribe(audio)
        st.session_state.mic_key += 1  # reset the recorder for the next turn
        if not user_text:
            st.warning("I didn't catch that. Could you record again?")

typed = st.chat_input(placeholder)
if typed:
    user_text = typed.strip()

if user_text:
    handle_user_message(user_text)
    st.rerun()
