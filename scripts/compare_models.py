"""Side-by-side comparison: retrieve once, generate from both the Anthropic (Haiku)
and Ollama (local) backends against the identical retrieved context, so the comparison
isolates model quality rather than retrieval differences.

Usage: python scripts/compare_models.py "your question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.luther import BACKENDS, _base_prompt, _context_block  # noqa: E402
from app.retrieval import retrieve  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_models.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    passages = retrieve(question)
    messages = [{"role": "user", "content": question}]

    print(f"Question: {question}")
    print(f"Retrieved {len(passages)} passages: "
          + ", ".join(f"{p['metadata']['source']} ({p['metadata']['year']})" for p in passages))

    for name, backend in BACKENDS.items():
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        try:
            result = backend(passages, messages)
            print(result["response"])
            print(f"\n[tokens: {result['input_tokens']} in / {result['output_tokens']} out]")
        except Exception as e:
            print(f"ERROR calling {name}: {e}")


if __name__ == "__main__":
    main()
