from pathlib import Path

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FRIDAY AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")

SYSTEM_PROMPT = """
You are FRIDAY, a polished, intelligent AI assistant.
Speak naturally, clearly, and professionally.
Explain technical topics with practical examples.
For VMware and IT questions, provide accurate, structured explanations
suitable for an IT engineer.
Be transparent about limitations and never claim to have performed actions
you did not perform.
"""

# Streamlit Cloud secrets take precedence over local environment values.
def get_setting(name, default=""):
    value = os.getenv(name, default)
    try:
        if name in st.secrets:
            value = st.secrets[name]
    except (FileNotFoundError, KeyError, TypeError):
        pass
    return str(value or "").strip()


def unique_env_models(*variable_names):
    """Read model names from .env/secrets, preserving order and removing duplicates."""
    result = []
    for variable_name in variable_names:
        model = get_setting(variable_name)
        if model and model not in result:
            result.append(model)
    return result


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODELS = unique_env_models(
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "GEMINI_BACKUP_MODEL",
) or [DEFAULT_GEMINI_MODEL]

OLLAMA_MODELS = unique_env_models(
    "OLLAMA_MODEL",
    "OLLAMA_CODER_MODEL",
) or [DEFAULT_OLLAMA_MODEL]

# ============================================================
# HEADER
# ============================================================

st.title("🤖 Friday.Ai")
st.caption("Your personal assistant for IT, VMware, Python and everyday questions.")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.radio(
        "Choose AI provider",
        options=["Gemini (cloud)", "Ollama (local)"],
        help="Gemini uses your API key and internet. Ollama runs models on this laptop.",
    )

    if provider == "Gemini (cloud)":
        selected_model = st.selectbox(
            "Gemini model",
            options=GEMINI_MODELS,
            help="These model names are loaded from GEMINI_MODEL, GEMINI_FALLBACK_MODEL and GEMINI_BACKUP_MODEL in your .env file.",
        )
        st.caption("Requires internet access and a valid Gemini API key.")
    else:
        selected_model = st.selectbox(
            "Ollama model",
            options=OLLAMA_MODELS,
            help="These model names are loaded from OLLAMA_MODEL and OLLAMA_CODER_MODEL in your .env file.",
        )
        st.caption("Runs locally on the laptop. Ollama must be running.")

    with st.expander("Connection checks"):
        if provider == "Gemini (cloud)":
            if GEMINI_API_KEY:
                st.success("Gemini API key found (value hidden).")
            else:
                st.error("GEMINI_API_KEY was not found in .env or Streamlit secrets.")
        else:
            st.write(f"Ollama URL: `{OLLAMA_URL}`")
            if st.button("Test Ollama connection", use_container_width=True):
                try:
                    check = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                    check.raise_for_status()
                    available = [
                        item.get("name", "")
                        for item in check.json().get("models", [])
                    ]
                    st.success("Connected to Ollama.")
                    if available:
                        st.write("Models installed in Ollama:")
                        for name in available:
                            st.write(f"- {name}")
                    else:
                        st.info("Ollama is running, but no models were listed.")
                except requests.RequestException as exc:
                    st.error(
                        "Could not reach Ollama. Check that it is running "
                        "and that OLLAMA_URL is correct."
                    )
                    print(f"Ollama connection check failed: {type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("About FRIDAY")
    st.write(
        "Choose a model above, then ask a question below. "
        "The conversation stays in this browser session."
    )
    st.caption("For local mobile access, keep this app and Ollama running on your laptop.")

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# MODEL CLIENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def make_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def call_gemini(model, history):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to your local .env file "
            "or Streamlit Cloud secrets."
        )

    client = make_gemini_client(GEMINI_API_KEY)
    # Send prior turns as context so follow-up questions make sense.
    contents = []
    for message in history:
        role = "user" if message["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=message["content"])],
        ))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty response.")
    return answer


def call_ollama(model, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item["role"] in {"user", "assistant"}
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=(5, 180),
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


# ============================================================
# CHAT HISTORY AND CHAT UI
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the selected provider/model clearly above the conversation.
provider_label = "Gemini" if provider == "Gemini (cloud)" else "Ollama"
st.info(f"**Selected model:** {provider_label} · `{selected_model}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask FRIDAY anything…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"FRIDAY is thinking with {selected_model}…"):
            try:
                if provider == "Gemini (cloud)":
                    answer = call_gemini(selected_model, st.session_state.messages)
                else:
                    answer = call_ollama(selected_model, st.session_state.messages)

                st.markdown(answer)

            except Exception as exc:
                print(
                    f"{provider_label} request failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if provider == "Gemini (cloud)":
                    answer = (
                        "Sorry, Boss. I couldn't get a response from Gemini. "
                        "Check your API key, model availability, quota and internet connection."
                    )
                else:
                    answer = (
                        "Sorry, Boss. I couldn't get a response from Ollama. "
                        "Check that Ollama is running, the selected model is installed, "
                        "and the Ollama URL is correct."
                    )
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Tip: switch providers in the sidebar whenever you want. "
    "Gemini uses the cloud; Ollama uses your laptop's resources."
)
