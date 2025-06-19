from openai import OpenAI
from dotenv import load_dotenv
import os, sys

load_dotenv()              # pulls OPENAI_API_KEY into the process
client = OpenAI()          # uses the key automatically

def ask_chat(prompt: str,
             model: str = "gpt-4o-mini",
             temperature: float = 0.7) -> str:
    """Send a single-turn chat completion and return the reply text."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    print("🔮  Ask away!  (type 'quit' to exit)\n")
    for line in sys.stdin:                # tiny REPL
        prompt = line.strip()
        if prompt.lower() in {"quit", "exit"}:
            break
        try:
            print(ask_chat(prompt), "\n")
        except Exception as err:
            print(f"⚠️  {err}\n")
