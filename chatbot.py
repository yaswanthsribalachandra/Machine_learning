import streamlit as st
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.3-70b-instruct"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

# ---------------- UI HEADER ----------------
st.title("🤖 AI Chat Assistant")
st.caption("Powered by NVIDIA LLaMA 3.3 70B")

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------
user_input = st.chat_input("Type your message...")

# ---------------- STOP BUTTON ----------------
col1, col2 = st.columns([6,1])
with col2:
    if st.button("⛔ Stop"):
        st.session_state.stop_generation = True

# ---------------- HANDLE INPUT ----------------
if user_input and user_input.strip() != "":
    st.session_state.stop_generation = False

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response placeholder
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
                temperature=0.3,
                top_p=0.7,
                max_tokens=2048,
                stream=True
            )

            for chunk in stream:
                if st.session_state.stop_generation:
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            response_placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

## ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    # -------- MODEL DISPLAY --------
    st.markdown("### 🤖 Model")
    st.text(f"{MODEL}")

    st.markdown("---")

    # -------- PARAMETERS --------
    st.markdown("### 🎛️ Generation Settings")

    st.session_state.temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.5,
        value=0.3,
        step=0.05,
        help="Lower = more deterministic, Higher = more creative"
    )

    st.session_state.top_p = st.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Controls diversity via probability mass"
    )

    st.session_state.max_tokens = st.slider(
        "Max Tokens (Response Length)",
        min_value=128,
        max_value=4096,
        value=2048,
        step=128,
        help="Maximum tokens generated in response"
    )

    st.session_state.presence_penalty = st.slider(
        "Presence Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Encourages new topics"
    )

    st.session_state.frequency_penalty = st.slider(
        "Frequency Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Reduces repetition"
    )

    st.markdown("---")

    # -------- ACTIONS --------
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Best Practices")
    st.markdown("""
- **0.2–0.4** → factual answers  
- **0.5–0.8** → balanced  
- **0.9+** → creative  
- Keep `top_p` around **0.7–0.9**
""")
