"""Core chat logic: retrieve relevant passages, assemble the system prompt, and call
Claude Haiku to generate Luther's response. This module is the shared core used by
both the CLI test harness (scripts/chat_cli.py) and the FastAPI app (app/main.py)."""
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from app.retrieval import retrieve

load_dotenv()

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

_base_system_prompt = None
_anthropic_client = None


def _client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _base_prompt() -> str:
    global _base_system_prompt
    if _base_system_prompt is None:
        _base_system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return _base_system_prompt


def _format_context(passages: list[dict]) -> str:
    if not passages:
        return "(No relevant passages were found in your writings for this question.)"
    blocks = []
    for p in passages:
        meta = p["metadata"]
        blocks.append(f"[{meta['source']}, {meta['year']}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def ask_luther(question: str, history: list[dict] | None = None) -> dict:
    """Returns {"response": str, "retrieved": list[dict]} for a single user question.

    history, if given, is a list of {"role": "user"|"assistant", "content": str} from
    earlier turns in the same session -- passed through to Claude for conversational
    continuity, but retrieval is always keyed off the latest question only (v1; a
    fancier version might rewrite the query using conversation context).

    The system prompt is split into two blocks: the static persona/theology/guardrail
    text (identical on every call -- marked cache_control so Anthropic caches it and
    charges ~90% less for it on repeat calls within the 5-minute cache window) and the
    per-query retrieved passages (always different, never cached).
    """
    passages = retrieve(question)

    system_blocks = [
        {
            "type": "text",
            "text": _base_prompt(),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "## Passages from your own writings, relevant to this question\n\n"
                + _format_context(passages)
            ),
        },
    ]

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    response = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_blocks,
        messages=messages,
    )
    text = "".join(block.text for block in response.content if block.type == "text")

    return {"response": text, "retrieved": passages}
