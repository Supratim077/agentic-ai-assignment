import os
import streamlit as st
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import sympy

st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 Autonomous Agent")

# Get API key from environment or Streamlit Secrets
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please configure GEMINI_API_KEY in the app settings.")
    st.stop()

client = genai.Client(api_key=api_key)

# Tools
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results]) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"

def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        return f"Result: {sympy.sympify(expression)}"
    except Exception as e:
        return f"Calculation error: {e}"

tools = [search_web, calculate]

# Session memory
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(
            tools=tools,
            system_instruction="You are an autonomous AI agent. Use `search_web` for recent info and `calculate` for math.",
        ),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question or assign a task..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = st.session_state.chat.send_message(prompt)
                st.markdown(res.text)
                st.session_state.messages.append({"role": "assistant", "content": res.text})
            except Exception as e:
                st.error(f"Error: {e}")