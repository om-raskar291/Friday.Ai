
from pathlib import Path
import os

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
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FALLING SNOW EFFECT
# ============================================================


# ============================================================
# FALLING SNOW + PREMIUM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F7F8FA;
        color: #1F2937;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 250, 0.88);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    h1, h2, h3 {
        color: #111827;
        letter-spacing: -0.035em;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #D1D5DB;
        padding: 0.55rem 0.9rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
    }

    /* Subtle falling snow */
    .friday-snow {
        position: fixed;
        inset: 0;
        pointer-events: none;
        overflow: hidden;
        z-index: 10;
    }

    .friday-snow span {
        position: absolute;
        top: -30px;
        color: #D7EAFE;
        opacity: 0.35;
        font-size: 15px;
        animation: friday-fall linear infinite;
    }

    @keyframes friday-fall {
        from {
            transform: translateY(-5vh) rotate(0deg);
        }
        to {
            transform: translate(35px, 110vh) rotate(360deg);
        }
    }

    .friday-snow span:nth-child(1) { left: 5%; animation-duration: 12s; }
    .friday-snow span:nth-child(2) { left: 15%; animation-duration: 9s; animation-delay: -4s; }
    .friday-snow span:nth-child(3) { left: 25%; animation-duration: 14s; animation-delay: -7s; }
    .friday-snow span:nth-child(4) { left: 35%; animation-duration: 10s; animation-delay: -2s; }
    .friday-snow span:nth-child(5) { left: 45%; animation-duration: 13s; animation-delay: -6s; }
    .friday-snow span:nth-child(6) { left: 55%; animation-duration: 8s; animation-delay: -3s; }
    .friday-snow span:nth-child(7) { left: 65%; animation-duration: 15s; animation-delay: -9s; }
    .friday-snow span:nth-child(8) { left: 75%; animation-duration: 11s; animation-delay: -5s; }
    .friday-snow span:nth-child(9) { left: 85%; animation-duration: 9s; animation-delay: -7s; }
    .friday-snow span:nth-child(10) { left: 95%; animation-duration: 14s; animation-delay: -1s; }

    @media (prefers-reduced-motion: reduce) {
        .friday-snow {
            display: none;
        }
    }
    </style>

    <div class="friday-snow" aria-hidden="true">
        <span>❄</span><span>❅</span><span>❄</span>
        <span>❅</span><span>❄</span><span>❅</span>
        <span>❄</span><span>❅</span><span>❄</span>
        <span>❅</span>
    </div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# LOAD CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
GROQ_API_URL = "https://api.groq.com/openai/v1"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://127.0.0.1:11434"
).strip().rstrip("/")

SYSTEM_PROMPT = """
You are FRIDAY, a polished, intelligent AI assistant.
Speak naturally, clearly, and professionally.
Explain technical topics with practical examples.
For VMware and IT questions, provide accurate, structured explanations
suitable for an IT engineer.
Be transparent about limitations and never claim to have performed actions
you did not perform.
"""


def get_setting(name, default=""):
    value = os.getenv(name, default)

    try:
        if name in st.secrets:
            value = st.secrets[name]
    except (FileNotFoundError, KeyError, TypeError):
        pass

    return str(value or "").strip()


def unique_env_models(*variable_names):
    """Read model names from .env/secrets and remove duplicates."""
    result = []

    for variable_name in variable_names:
        model = get_setting(variable_name)

        if model and model not in result:
            result.append(model)

    return result


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GROQ_API_KEY = get_setting("GROQ_API_KEY")

GROQ_MODELS = unique_env_models(
    "GROQ_MODEL",
    "GROQ_REASONING_MODEL",
) or DEFAULT_GROQ_MODELS

GEMINI_MODELS = unique_env_models(
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "GEMINI_BACKUP_MODEL",
) or [DEFAULT_GEMINI_MODEL]

# Include all four models installed in your local Ollama setup.
# Any additional model names configured in .env or Streamlit secrets
# are also included automatically.
DEFAULT_OLLAMA_MODELS = [
    "llama3.2:3b",
    "qwen2.5-coder:1.5b",
    "deepseek-r1:1.5b",
    "gemma4:e2b",
]

configured_ollama_models = unique_env_models(
    "OLLAMA_MODEL",
    "OLLAMA_CODER_MODEL",
    "OLLAMA_REASONING_MODEL",
    "OLLAMA_GENERAL_MODEL",
)

OLLAMA_MODELS = list(dict.fromkeys(
    DEFAULT_OLLAMA_MODELS + configured_ollama_models
))


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Friday.Ai")

st.caption(
    "Your personal assistant for everyday questions and queries"
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.radio(
        "Choose AI provider",
        options=["Gemini (cloud)", "Groq (cloud)", "Ollama (local)"],
        help=(
            "Gemini and Groq use cloud APIs and need internet access. "
            "Ollama runs models locally on this laptop."
        ),
    )

    if provider == "Gemini (cloud)":
        selected_model = st.selectbox(
            "Gemini model",
            options=GEMINI_MODELS,
            help=(
                "Model names come from GEMINI_MODEL, "
                "GEMINI_FALLBACK_MODEL and GEMINI_BACKUP_MODEL."
            ),
        )

        st.caption("Requires internet access and a valid Gemini API key.")

    elif provider == "Groq (cloud)":
        selected_model = st.selectbox(
            "Groq model",
            options=GROQ_MODELS,
            help=(
                "Configured using GROQ_MODEL and GROQ_REASONING_MODEL "
                "in your .env file or Streamlit secrets."
            ),
        )
        st.caption("Requires internet access and a valid Groq API key.")

    else:
        selected_model = st.selectbox(
            "Ollama model",
            options=OLLAMA_MODELS,
            help=(
                "Choose one of your four installed local models. "
                "Additional model names can be configured in .env or secrets."
            ),
        )

        st.caption("Runs locally on the laptop. Ollama must be running.")

    with st.expander("Connection checks"):
        if provider == "Gemini (cloud)":
            if GEMINI_API_KEY:
                st.success("Gemini API key found (value hidden).")
            else:
                st.error(
                    "GEMINI_API_KEY was not found in .env or Streamlit secrets."
                )

        elif provider == "Groq (cloud)":
            if GROQ_API_KEY:
                st.success("Groq API key found (value hidden).")
            else:
                st.error(
                    "GROQ_API_KEY was not found in .env or Streamlit secrets."
                )

            if st.button("Test Groq connection", use_container_width=True):
                if not GROQ_API_KEY:
                    st.error("Add GROQ_API_KEY to your local .env file first.")
                else:
                    try:
                        check = requests.get(
                            f"{GROQ_API_URL}/models",
                            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                            timeout=(10, 20),
                        )
                        check.raise_for_status()
                        st.success("Connected to Groq. API key was accepted.")
                        available = [
                            item.get("id", "")
                            for item in check.json().get("data", [])
                        ]
                        if available:
                            st.caption("Models returned by Groq (availability may vary by account):")
                            st.write(", ".join(available[:30]))
                    except requests.RequestException as exc:
                        detail = ""
                        if getattr(exc, "response", None) is not None:
                            detail = (exc.response.text or "").strip()[:1000]
                        st.error("Groq connection failed. Check your API key, internet connection, and account access.")
                        st.code(detail or f"{type(exc).__name__}: {exc}", language="text")
                    except (ValueError, AttributeError) as exc:
                        st.error(f"Groq responded, but the response could not be read: {exc}")

        else:
            st.write(f"Ollama URL: `{OLLAMA_URL}`")

            if st.button(
                "Test Ollama connection",
                use_container_width=True,
            ):
                try:
                    check = requests.get(
                        f"{OLLAMA_URL}/api/tags",
                        timeout=5,
                    )
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
                        st.info(
                            "Ollama is running, but no models were listed."
                        )

                except requests.RequestException as exc:
                    st.error(
                        "Could not reach Ollama. Check that it is running "
                        "and that OLLAMA_URL is correct."
                    )

                    print(
                        f"Ollama connection check failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

    st.divider()

    st.subheader("About FRIDAY")

    st.write(
        "Choose a model above, then ask a question below. "
        "The conversation stays in this browser session."
    )

    st.caption(
        "For local mobile access, keep this app and Ollama running on your laptop."
    )

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
            "Gemini API key is missing. Add GEMINI_API_KEY to your "
            "local .env file or Streamlit Cloud secrets."
        )

    client = make_gemini_client(GEMINI_API_KEY)

    contents = []

    for message in history:
        role = "user" if message["role"] == "user" else "model"

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(text=message["content"])
                ],
            )
        )

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


def call_groq(model, history):
    """Call Groq's OpenAI-compatible chat completions API."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Groq API key is missing. Add GROQ_API_KEY to your local .env "
            "file or Streamlit Cloud secrets."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
        if item.get("role") in {"user", "assistant"}
        and item.get("content")
    )

    try:
        response = requests.post(
            f"{GROQ_API_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
            },
            timeout=(10, 180),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Groq timed out while generating a response for '{model}'. Try again."
        ) from exc
    except requests.HTTPError as exc:
        body = (response.text or "").strip()[:1500]
        raise RuntimeError(
            f"Groq returned HTTP {response.status_code} for model '{model}'. "
            f"Response: {body or str(exc)}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Groq. Check internet access. Details: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"Groq returned an unexpected response: {(response.text or '')[:1500]}"
        ) from exc

    if not answer:
        raise RuntimeError(f"Groq returned an empty response for '{model}'.")

    return answer


def call_ollama(model, history):
    """Send the conversation to the local Ollama API with clear diagnostics."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.extend(
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
        if item.get("role") in {"user", "assistant"}
        and item.get("content")
    )

    # Check that the selected model is actually available on this Ollama host.
    try:
        tags_response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        tags_response.raise_for_status()
        installed_models = {
            item.get("name", "")
            for item in tags_response.json().get("models", [])
        }
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            f"Check that Ollama is running. Details: {type(exc).__name__}: {exc}"
        ) from exc
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"Ollama responded at {OLLAMA_URL}, but /api/tags returned "
            f"unexpected data: {exc}"
        ) from exc

    if model not in installed_models:
        installed = ", ".join(sorted(installed_models)) or "(no models listed)"
        raise RuntimeError(
            f"Selected model '{model}' was not found at {OLLAMA_URL}. "
            f"Models reported by Ollama: {installed}"
        )

    # Keep the request payload model-aware. Gemma 4 can spend a long time
    # producing a separate reasoning trace for simple questions, so disable
    # thinking for that model and cap the generated output to a sensible size.
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if model == "gemma4:e2b":
        chat_payload["think"] = False
        chat_payload["options"] = {"num_predict": 800}

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=chat_payload,
            # Allow a slow first load, but avoid an unbounded wait.
            timeout=(10, 600),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out while generating a response for '{model}'. "
            "The model may still be loading or generating. Try again, or "
            "check Ollama's console."
        ) from exc
    except requests.HTTPError as exc:
        body = (response.text or "").strip()[:1500]
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code} for model '{model}'. "
            f"Response: {body or str(exc)}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not send the chat request to Ollama at {OLLAMA_URL}. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Ollama returned a non-JSON response: {(response.text or '')[:1000]}"
        ) from exc

    answer = payload.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError(
            f"Ollama returned no answer for '{model}'. Response: {str(payload)[:1500]}"
        )

    return answer


# ============================================================
# CHAT HISTORY AND CHAT UI
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


provider_label = {
    "Gemini (cloud)": "Gemini",
    "Groq (cloud)": "Groq",
    "Ollama (local)": "Ollama",
}[provider]

st.info(
    f"**Selected model:** {provider_label} · `{selected_model}`"
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask FRIDAY anything…")


if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            f"FRIDAY is thinking with {selected_model}…"
        ):
            try:
                if provider == "Gemini (cloud)":
                    answer = call_gemini(
                        selected_model,
                        st.session_state.messages,
                    )
                elif provider == "Groq (cloud)":
                    answer = call_groq(
                        selected_model,
                        st.session_state.messages,
                    )
                else:
                    answer = call_ollama(
                        selected_model,
                        st.session_state.messages,
                    )

                st.markdown(answer)

            except Exception as exc:
                error_detail = f"{type(exc).__name__}: {exc}"
                print(f"{provider_label} request failed: {error_detail}")

                if provider == "Gemini (cloud)":
                    answer = (
                        "Sorry, Boss. I couldn't get a response from Gemini. "
                        "Check your API key, model availability, quota "
                        "and internet connection."
                    )
                    st.error(answer)
                    st.caption(f"Technical details: {error_detail}")
                elif provider == "Groq (cloud)":
                    answer = (
                        "Sorry, Boss. I couldn't get a response from Groq. "
                        "Check your API key, model access, usage limits, "
                        "and internet connection."
                    )
                    st.error(answer)
                    st.code(error_detail, language="text")
                else:
                    answer = (
                        "Sorry, Boss. I couldn't get a response from Ollama. "
                        "See the technical details below to identify the cause."
                    )
                    st.error(answer)
                    st.code(error_detail, language="text")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tip: switch providers in the sidebar whenever you want. "
    "Gemini and Groq use the cloud; Ollama uses your laptop's resources."
)
