"""Manual test harness: talk to Luther from the terminal before any web layer exists.
Run with: python scripts/chat_cli.py
Prints which sources got retrieved for each turn, so you can judge retrieval quality
alongside the response itself. Type 'exit' to quit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.luther import ask_luther  # noqa: E402


def main():
    history = []
    print("Talking to Luther. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        result = ask_luther(question, history=history)

        sources = ", ".join(f"{p['metadata']['source']} ({p['distance']:.2f})" for p in result["retrieved"])
        print(f"\n[retrieved: {sources}]\n")
        print(f"Luther: {result['response']}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["response"]})


if __name__ == "__main__":
    main()
