import os
import sys
import time
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import sympy

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("\n[ERROR] GEMINI_API_KEY not found in .env file.")
    print("Please make sure your .env file has: GEMINI_API_KEY=AIzaSy...")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# =============================================================
# 1. TOOL DEFINITIONS
# =============================================================

def search_web(query: str) -> str:
    """Search the web for up-to-date facts, current events, or documentation."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No web results found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {e}"

def save_note(filename: str, content: str) -> str:
    """Save text content directly to a local workspace file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{filename}' created and saved successfully."
    except Exception as e:
        return f"File write error: {e}"

def read_note(filename: str) -> str:
    """Read and return the text content of a local workspace file."""
    try:
        if not os.path.exists(filename):
            return f"Error: File '{filename}' not found."
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"File read error: {e}"

def calculate(expression: str) -> str:
    """Safely evaluates mathematical expressions, equations, or scientific calculations."""
    try:
        result = sympy.sympify(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Math calculation error: {e}"

agent_tools = [search_web, save_note, read_note, calculate]

system_instruction = (
    "You are an autonomous AI agent with tool access. "
    "Use `search_web` when real-time facts or external information are needed. "
    "Use `save_note` to write content or task outputs to disk. "
    "Use `read_note` when asked to read a workspace file. "
    "Use `calculate` for any mathematical, statistical, or algebraic computations. "
    "For general queries, answer directly."
)

# Active models to try in sequence if one hits a temporary capacity spike
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.8-pro", "gemini-flash-latest"]

# =============================================================
# 2. FAILOVER & RETRY ENGINE
# =============================================================

def create_chat(model_name: str):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            tools=agent_tools,
            system_instruction=system_instruction,
        ),
    )

def send_with_failover(prompt: str) -> str:
    for model in MODELS_TO_TRY:
        # Give each model up to 2 attempts with a backoff pause
        for attempt in range(2):
            try:
                active_chat = create_chat(model)
                response = active_chat.send_message(prompt)
                return response.text
            except Exception as err:
                err_str = str(err)
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    wait_time = (attempt + 1) * 3
                    print(f"  [{model} busy (503). Retrying in {wait_time}s...]")
                    time.sleep(wait_time)
                elif "404" in err_str:
                    # Model not available on this tier; immediately skip to next model
                    break
                else:
                    raise err

        print(f"  [{model} unavailable. Trying next model...]")

    raise RuntimeError("All Google server endpoints are currently busy. Please wait a minute and retry.")

# =============================================================
# 3. INTERACTIVE CONSOLE
# =============================================================

print("=" * 55)
print("Agent online! Tools: search_web | save_note | read_note | calculate")
print("Type 'quit' or 'exit' to stop.")
print("=" * 55)

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Shutting down. Goodbye!")
            break

        reply = send_with_failover(user_input)
        print(f"\nAgent:\n{reply}")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print(f"\n[Error]: {e}")